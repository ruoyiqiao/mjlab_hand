"""Evaluate a juggling checkpoint (play config: paper start, no reference-state init).

Protocol: N envs, the first complete episode of each env (10 s cap) is counted.
Reports juggling duration (time until drop / stall / over-hold, capped at 10 s),
fraction juggling the full 10 s (paper criterion), valid catches and throws per
episode, catches per second, termination reasons, Wilson 95% CIs.

Usage: python eval_juggle.py <task> <checkpoint> <out.json> [--num-envs 512] [--seed 2001]
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict

import torch


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=2001)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    import mjlab_hand  # noqa: F401

    torch.manual_seed(a.seed)
    cfg = load_env_cfg(a.task, play=True)
    cfg.scene.num_envs = a.num_envs
    cfg.seed = a.seed
    raw = ManagerBasedRlEnv(cfg, device=a.device)
    rl_cfg = load_rl_cfg(a.task)
    env = RslRlVecEnvWrapper(raw, clip_actions=rl_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(env, asdict(rl_cfg), device=a.device)
    runner.load(a.ckpt, load_cfg={"actor": True}, strict=True, map_location=a.device)
    policy = runner.get_inference_policy(device=a.device)
    t = raw.command_manager.get_term("juggle")
    n = a.num_envs
    active = torch.ones(n, dtype=torch.bool, device=a.device)
    dur = torch.zeros(n, device=a.device)
    catches = torch.zeros(n, device=a.device)
    throws = torch.zeros(n, device=a.device)
    reason = ["running"] * n
    names = list(raw.termination_manager.active_terms)
    obs, _ = env.reset()
    steps = int(raw.max_episode_length) + 2
    for _ in range(steps):
        if not active.any():
            break
        with torch.no_grad():
            act = policy(obs)
        pre_c, pre_t = t.catches.clone().float(), t.throws.clone().float()
        pre_j = t.ep_time.clone()
        obs, rew, dones, extras = env.step(act)
        d = dones.bool() & active
        nd = active & ~d
        catches = torch.where(nd, t.catches.float(), catches)
        throws = torch.where(nd, t.throws.float(), throws)
        dur = torch.where(nd, t.ep_time, dur)
        if d.any():
            terms = {k: raw.termination_manager.get_term(k) for k in names}
            for i in d.nonzero().flatten().tolist():
                rs = [k for k, v in terms.items() if bool(v[i])]
                reason[i] = "+".join(rs)
                # values at the terminal step (tracker already reset -> use pre-step + this step)
                catches[i] = pre_c[i]
                throws[i] = pre_t[i]
                dur[i] = pre_j[i] + raw.step_dt
            active &= ~d
    cap = float(cfg.episode_length_s)
    full = [(reason[i] == "time_out") or (reason[i] == "running") for i in range(n)]
    k_full = sum(full)
    res = {
        "task": a.task,
        "checkpoint": a.ckpt,
        "seed": a.seed,
        "num_envs": n,
        "protocol": "play config (paper start), first episode per env, 10 s cap",
        "full_duration": {"k": k_full, "n": n, "rate": k_full / n, "ci95": wilson(k_full, n)},
        "duration_s_mean": float(dur.mean()),
        "duration_s_median": float(dur.median()),
        "catches_per_episode_mean": float(catches.mean()),
        "throws_per_episode_mean": float(throws.mean()),
        "catches_per_s": float(catches.sum() / dur.clamp(min=1e-6).sum()),
        "termination_reasons": {r: reason.count(r) for r in set(reason)},
        "cap_s": cap,
    }
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
