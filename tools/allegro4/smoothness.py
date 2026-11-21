"""Finger smoothness of a trained policy, for any task and any hand.

Usage: python smoothness.py <task_id> <checkpoint> [--num-envs 256] [--steps 600] [--seed 2001] [--out s.json]

Runs the policy and, per hand entity (every entity with a ``base_x`` joint), reports over all finger joints:
  speed_rms      RMS joint velocity (rad/s)
  accel_rms      RMS joint acceleration from velocity differences (rad/s^2), the jerk proxy
  tv_per_step    mean |q_t - q_{t-1}| summed over finger joints (rad per control step, total variation)
  sign_flips     fraction of steps where a finger joint's velocity changes sign (chatter)
Steps in which an env was reset are skipped. Task success is NOT measured here (use the task's evaluator on the same checkpoint).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("ckpt")
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=2001)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
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

    hands = {}
    for name, ent in raw.scene.entities.items():
        names = list(ent.joint_names)
        if "base_x" in names:
            ids = [i for i, n in enumerate(names) if not n.startswith("base_")]
            hands[name] = ids
    if not hands:
        sys.exit("no entity with a base_x joint")

    acc = {h: dict(n=0, v2=0.0, a2=0.0, tv=0.0, flip=0.0) for h in hands}
    obs, _ = env.reset()
    prev_q = {h: None for h in hands}
    prev_v = {h: None for h in hands}
    prev_ep = raw.episode_length_buf.clone()
    for _ in range(a.steps):
        with torch.no_grad():
            act = policy(obs)
        obs, _, _, _ = env.step(act)
        fresh = raw.episode_length_buf > prev_ep  # env was not reset this step
        prev_ep = raw.episode_length_buf.clone()
        for h, ids in hands.items():
            d = raw.scene[h].data
            q, v = d.joint_pos[:, ids].clone(), d.joint_vel[:, ids].clone()
            if prev_q[h] is not None:
                m = fresh & (raw.episode_length_buf > 1)
                if m.any():
                    dt = raw.step_dt
                    s = acc[h]
                    s["n"] += int(m.sum())
                    s["v2"] += float((v[m] ** 2).mean(1).sum())
                    s["a2"] += float((((v[m] - prev_v[h][m]) / dt) ** 2).mean(1).sum())
                    s["tv"] += float((q[m] - prev_q[h][m]).abs().sum(1).sum())
                    s["flip"] += float(((v[m] * prev_v[h][m]) < 0).float().mean(1).sum())
            prev_q[h], prev_v[h] = q, v
    out = {}
    for h, s in acc.items():
        n = max(s["n"], 1)
        out[h] = dict(
            speed_rms=(s["v2"] / n) ** 0.5,
            accel_rms=(s["a2"] / n) ** 0.5,
            tv_per_step=s["tv"] / n,
            sign_flips=s["flip"] / n,
            samples=s["n"],
            finger_joints=len(hands[h]),
        )
    print("SMOOTH", json.dumps({"task": a.task, "ckpt": a.ckpt, "hands": out}))
    if a.out:
        json.dump({"task": a.task, "ckpt": a.ckpt, "hands": out}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
