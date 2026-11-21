"""Roll out a checkpoint (or zero/random actions) and record video + keyframes.

Usage: python rollout_video.py <task_id> <checkpoint|zero|random> <out_dir>
           [--envs 0,1,2,3] [--steps 300] [--view persp] [--num-envs 16] [--seed 1001]

Writes <out_dir>/env<i>_<view>.mp4 (if imageio is available), a keyframe strip PNG
per env (every --strip-every steps), and rollout.json with per-step task state of
the recorded envs (object height, lifted, contacts, successes, goal error).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--envs", default="0,1,2,3")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--view", default="persp")
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--strip-every", type=int, default=25)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from smoke_env import Snapshotter, goal_vis

    import mjlab_hand  # noqa: F401

    torch.manual_seed(a.seed)
    cfg = load_env_cfg(a.task, play=True)
    cfg.scene.num_envs = a.num_envs
    cfg.seed = a.seed
    raw = ManagerBasedRlEnv(cfg, device=a.device)
    rl_cfg = load_rl_cfg(a.task)
    env = RslRlVecEnvWrapper(raw, clip_actions=rl_cfg.clip_actions)
    policy = None
    if a.ckpt not in ("zero", "random"):
        runner = MjlabOnPolicyRunner(env, asdict(rl_cfg), device=a.device)
        runner.load(a.ckpt, load_cfg={"actor": True}, strict=True, map_location=a.device)
        policy = runner.get_inference_policy(device=a.device)
    cmd = next(iter(raw.command_manager.active_terms))
    t = raw.command_manager.get_term(cmd)
    snap = Snapshotter(raw, width=480, height=360)
    envs = [int(x) for x in a.envs.split(",")]
    frames = {i: [] for i in envs}
    log = {i: [] for i in envs}
    obs, _ = env.reset()
    adim = raw.action_manager.total_action_dim
    for s in range(a.steps):
        with torch.no_grad():
            if policy is not None:
                act = policy(obs)
            elif a.ckpt == "zero":
                act = torch.zeros(a.num_envs, adim, device=a.device)
            else:
                act = torch.rand(a.num_envs, adim, device=a.device) * 2 - 1
        obs, rew, dones, extras = env.step(act)
        t.update()
        for i in envs:
            frames[i].append(snap.render(i, a.view, goal_vis(raw, cmd, i)))
            if not hasattr(t, "obj"):  # juggling
                log[i].append(
                    {
                        "step": s,
                        "balls_z": [round(float(z), 3) for z in t.ball_pos()[i, :, 2]],
                        "contact": t.contact[i].tolist(),
                        "catches": int(t.catches[i]),
                        "throws": int(t.throws[i]),
                        "done": bool(dones[i]),
                    }
                )
                continue
            p = t.obj.data.root_link_pos_w[i]
            log[i].append(
                {
                    "step": s,
                    "obj_dz_table": round(float(p[2] - t.table_top()[i]), 4),
                    "lifted": bool(t.lifted[i]),
                    "contact": t.hand_contact[i].tolist(),
                    "pos_err": round(float(t.pos_err[i]), 4),
                    "ang_err": round(float(t.ang_err[i]), 3),
                    "successes": int(t.successes[i]),
                    "done": bool(dones[i]),
                    "action_abs_mean": round(float(act[i].abs().mean()), 3),
                }
            )
    from PIL import Image

    for i in envs:
        strip = [Image.fromarray(f) for f in frames[i][:: a.strip_every]]
        w, h = strip[0].size
        cols = min(len(strip), 6)
        rows = (len(strip) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * w, rows * h))
        for k, im in enumerate(strip):
            sheet.paste(im, ((k % cols) * w, (k // cols) * h))
        sheet.save(os.path.join(a.out, f"env{i}_{a.view}_strip.png"))
        try:
            import imageio.v2 as imageio

            imageio.mimsave(
                os.path.join(a.out, f"env{i}_{a.view}.mp4"), frames[i], fps=int(1 / raw.step_dt)
            )
        except Exception as exc:  # noqa: BLE001
            print("video write failed:", exc)
    if not hasattr(t, "obj"):
        summ = {
            i: {
                "catches_before_first_done": next(
                    (r["catches"] for r in log[i] if r["done"]), log[i][-1]["catches"]
                ),
                "dones": sum(r["done"] for r in log[i]),
            }
            for i in envs
        }
        with open(os.path.join(a.out, "rollout.json"), "w") as fh:
            json.dump({"task": a.task, "ckpt": a.ckpt, "summary": summ, "log": log}, fh)
        print(json.dumps(summ, indent=1))
        return
    summ = {
        i: {
            "max_obj_dz_table": max(r["obj_dz_table"] for r in log[i]),
            "any_lifted": any(r["lifted"] for r in log[i]),
            "contact_frac": float(np.mean([any(r["contact"]) for r in log[i]])),
            "final_successes": log[i][-1]["successes"],
        }
        for i in envs
    }
    with open(os.path.join(a.out, "rollout.json"), "w") as fh:
        json.dump({"task": a.task, "ckpt": a.ckpt, "summary": summ, "log": log}, fh)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
