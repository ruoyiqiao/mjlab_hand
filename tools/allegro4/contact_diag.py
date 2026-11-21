"""Per-fingertip contact statistics of a single-hand policy (CPU by default; no rendering, no GPU context).

Usage: python contact_diag.py <task_id> <checkpoint> [--num-envs 32] [--steps 400] [--device cpu]

Reports, over all env-steps while the object is not yet lifted and while it is lifted: the fraction of steps each
fingertip touches the object, thumb + >=1 other tip ("opposition"), any hand contact, and for throw tasks the stage flags.
Use it to tell "the hand grasps but the tips are not the contact points" from "the hand never opposes the thumb".
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("ckpt")
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    from mjlab_hand.registration import register_tasks

    register_tasks()
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    torch.manual_seed(a.seed)
    cfg = load_env_cfg(a.task, play=True)
    cfg.scene.num_envs = a.num_envs
    cfg.seed = a.seed
    raw = ManagerBasedRlEnv(cfg, device=a.device)
    rl = load_rl_cfg(a.task)
    env = RslRlVecEnvWrapper(raw, clip_actions=rl.clip_actions)
    runner = MjlabOnPolicyRunner(env, asdict(rl), device=a.device)
    runner.load(a.ckpt, load_cfg={"actor": True}, strict=True, map_location=a.device)
    policy = runner.get_inference_policy(device=a.device)
    t = next(iter(raw.command_manager._terms.values()))
    names = list(raw.scene["hand_tip_contact"].primary_names)
    acc = {
        k: {"n": 0, "tips": torch.zeros(len(names)), "opp": 0.0, "any": 0.0}
        for k in ("pre_lift", "lifted")
    }
    stage = {"grasped": 0.0, "released": 0.0, "in_bucket": 0.0, "n": 0}
    obs, _ = env.reset()
    for _ in range(a.steps):
        with torch.no_grad():
            obs, _, _, _ = env.step(policy(obs))
        tf = t.tip_found[:, 0].float().cpu()
        th = tf[:, t._thumb_cols[0]] if t._thumb_cols else tf[:, -1]
        opp = (th > 0) & ((tf.sum(-1) - th) >= 1)
        anyc = t.hand_contact[:, 0].float().cpu()
        lifted = t.lifted.cpu()
        for k, m in (("pre_lift", ~lifted), ("lifted", lifted)):
            if m.any():
                s = acc[k]
                s["n"] += int(m.sum())
                s["tips"] += tf[m].sum(0)
                s["opp"] += float(opp[m].float().sum())
                s["any"] += float(anyc[m].sum())
        if hasattr(t, "grasped"):
            stage["grasped"] += float(t.grasped.float().mean())
            stage["released"] += float(t.released.float().mean())
            stage["in_bucket"] += float(t.in_bucket.float().mean())
            stage["n"] += 1
    out = {"task": a.task, "tips": names}
    for k, s in acc.items():
        n = max(s["n"], 1)
        out[k] = {
            "steps": s["n"],
            "tip_contact_frac": [round(float(x) / n, 3) for x in s["tips"]],
            "opposition_frac": round(s["opp"] / n, 3),
            "any_hand_contact_frac": round(s["any"] / n, 3),
        }
    if stage["n"]:
        out["stage_mean"] = {k: round(v / stage["n"], 4) for k, v in stage.items() if k != "n"}
    print("CONTACT", json.dumps(out))


if __name__ == "__main__":
    main()
