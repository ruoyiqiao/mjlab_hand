"""Render policy rollouts of the three-ball juggling tasks (mp4 + keyframe strip).

Usage: python render_three_ball.py <task_id> <checkpoint> <out_dir> [--envs 0,1] [--steps 700] [--every 2]
           [--view j3front] [--num-envs 8] [--seed 2001] [--set key=value ...]

Do not call the command term's update() here: the environment already steps it once per control step.
Works with the current sources and with sources frozen before the rename (run directories).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--envs", default="0,1")
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--every", type=int, default=2, help="render every n-th control step")
    ap.add_argument("--view", default="j3front")
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2001)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from smoke_env import Snapshotter

    from mjlab_hand.registration import register_tasks

    register_tasks()
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    torch.manual_seed(a.seed)
    cfg = load_env_cfg(a.task, play=True)
    cfg.scene.num_envs = a.num_envs
    cfg.seed = a.seed
    spec = next(iter(cfg.commands.values())).spec
    for kv in a.set:
        k, v = kv.split("=", 1)
        old = getattr(spec, k)
        setattr(
            spec,
            k,
            type(old)(v) if not isinstance(old, bool) else v.lower() in ("1", "true", "yes"),
        )
    raw = ManagerBasedRlEnv(cfg, device=a.device)
    rl = load_rl_cfg(a.task)
    env = RslRlVecEnvWrapper(raw, clip_actions=rl.clip_actions)
    runner = MjlabOnPolicyRunner(env, asdict(rl), device=a.device)
    runner.load(a.ckpt, load_cfg={"actor": True}, strict=True, map_location=a.device)
    policy = runner.get_inference_policy(device=a.device)
    snap = Snapshotter(raw, width=800, height=600)
    envs = [int(x) for x in a.envs.split(",")]
    frames = {i: [] for i in envs}
    obs, _ = env.reset()
    for s in range(a.steps):
        with torch.no_grad():
            act = policy(obs)
        obs, _, _, _ = env.step(act)
        if s % a.every == 0:
            for i in envs:
                frames[i].append(snap.render(i, a.view))
    from PIL import Image

    fps = int(round(1.0 / (raw.step_dt * a.every)))
    for i in envs:
        strip = [Image.fromarray(f) for f in frames[i][:: max(1, len(frames[i]) // 8)]]
        w, h = strip[0].size
        sheet = Image.new("RGB", (4 * w, ((len(strip) + 3) // 4) * h))
        for k, im in enumerate(strip):
            sheet.paste(im, ((k % 4) * w, (k // 4) * h))
        sheet.save(os.path.join(a.out, f"env{i}_{a.view}_strip.png"))
        import imageio.v2 as imageio

        imageio.mimsave(os.path.join(a.out, f"env{i}_{a.view}.mp4"), frames[i], fps=fps)
    print("rendered", len(frames[envs[0]]), "frames per env at", fps, "fps ->", a.out)


if __name__ == "__main__":
    main()
