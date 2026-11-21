"""Evaluate a Baoding / PlateSpin checkpoint (play config, deterministic actions, CPU by default).

Usage: python palm_eval.py <task> <checkpoint> [--out out.json] [--num-envs 64] [--episodes 2] [--seed 2001]

Every env runs ``--episodes`` complete episodes. Reports per episode the turns in the commanded direction
(split by direction), the drop rate (an episode that ends before its time limit) and, for PlateSpin, the
share of steps on which 0 / 1 / 2 / 3+ fingertips carry the plate.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("checkpoint")
    ap.add_argument("--out", default="")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2001)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    from mjlab_hand.registration import register_tasks

    register_tasks()
    torch.manual_seed(a.seed)
    cfg = load_env_cfg(a.task, play=True)
    cfg.scene.num_envs = a.num_envs
    cfg.seed = a.seed
    raw = ManagerBasedRlEnv(cfg, device=a.device)
    rl_cfg = load_rl_cfg(a.task)
    env = RslRlVecEnvWrapper(raw, clip_actions=rl_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(env, asdict(rl_cfg), device=a.device)
    runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location=a.device)
    policy = runner.get_inference_policy(device=a.device)

    t = raw.command_manager.get_term("spin")
    plate = t.sp.task == "plate"
    n, horizon = a.num_envs, int(raw.max_episode_length)
    turns = torch.zeros(n)
    steps = torch.zeros(n)
    finished = torch.zeros(n, dtype=torch.long)
    direction = t.direction.clone()
    episodes, carried = [], []
    obs = env.get_observations()
    while (finished < a.episodes).any():
        with torch.no_grad():
            obs, _, dones, _ = env.step(policy(obs))
        turns += t.progress.cpu() / (2 * math.pi)
        steps += 1
        if plate:
            carried.append(t.tip_contacts().sum(-1).cpu()[finished < a.episodes])
        for e in torch.nonzero(dones.cpu()).flatten().tolist():
            if finished[e] < a.episodes:
                episodes.append((float(direction[e]), float(turns[e]), bool(steps[e] < horizon)))
            finished[e] += 1
            turns[e], steps[e] = 0.0, 0.0
        direction = torch.where(dones.cpu().bool(), t.direction.cpu(), direction)
    env.close()

    d = np.array(episodes)
    res = {
        "task": a.task,
        "checkpoint": a.checkpoint,
        "seed": a.seed,
        "episodes": len(d),
        "drop_rate": float(d[:, 2].mean()),
        "turns_per_episode": float(d[:, 1].mean()),
        "turns_ccw": float(d[d[:, 0] > 0, 1].mean()) if (d[:, 0] > 0).any() else None,
        "turns_cw": float(d[d[:, 0] < 0, 1].mean()) if (d[:, 0] < 0).any() else None,
        "wrong_way_episodes": float((d[:, 1] < 0).mean()),
    }
    if plate:
        c = torch.cat(carried).numpy()
        res["fingertips_carrying"] = {
            k: float(np.mean(v))
            for k, v in (("0", c == 0), ("1", c == 1), ("2", c == 2), ("3+", c >= 3))
        }
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
