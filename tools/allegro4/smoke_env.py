"""Build a task env, step it, check shapes/NaN/contacts and render keyframes.

Usage:
  python smoke_env.py <task_id> <num_envs> <steps> <out_dir> [--policy random|zero|script]
                      [--render-envs 0,1] [--views front,side,top] [--seed 0]

Writes <out_dir>/<task>_smoke.json and PNG keyframes (step 0 and last step) for
the requested envs and views. Rendering copies world i's qpos/mocap into a CPU
MjData of the compiled scene model and uses mujoco.Renderer with fixed free
cameras (headless EGL).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import mujoco
import numpy as np
import torch

VIEWS = {
    "jside": ((0.0, 0.06, 0.62), 0.9, 180.0, -5.0),
    "jpersp": ((0.0, 0.06, 0.6), 1.0, 135.0, -20.0),
    "j3front": ((0.0, 0.0, 0.85), 1.7, 180.0, -6.0),
    "j3persp": ((0.0, 0.0, 0.8), 1.9, 145.0, -18.0),
    "j3x": (
        (-0.05, 0.0, 0.80),
        1.35,
        180.0,
        -2.0,
    ),  # three-ball close view: azimuth 180 looks along x, the hands (y = +-0.13) side by side
    # name: (lookat offset, distance, azimuth, elevation)
    "front": ((0.1, 0.0, 0.5), 1.3, 180.0, -15.0),
    "side": ((0.1, 0.0, 0.5), 1.3, 90.0, -15.0),
    "top": ((0.1, 0.0, 0.45), 1.4, 90.0, -89.0),
    "persp": ((0.15, 0.0, 0.5), 1.5, 135.0, -30.0),
}


class Snapshotter:
    def __init__(self, env, width=640, height=480):
        self.env = env
        self.model = env.sim.mj_model
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 1
        self.opt.geomgroup[3] = 0  # hide collision boxes duplicated by visual meshes

    def sync(self, i: int) -> None:
        from mjlab.viewer.model_sync import VIEWER_MODEL_FIELDS, sync_model_fields

        fields = set(self.env.sim.expanded_fields) & VIEWER_MODEL_FIELDS
        if fields:
            sync_model_fields(self.model, self.env.sim.model, fields, i)
        d = self.env.sim.data
        self.data.qpos[:] = d.qpos[i].cpu().numpy()
        self.data.qvel[:] = d.qvel[i].cpu().numpy()
        if self.model.nmocap:
            self.data.mocap_pos[:] = d.mocap_pos[i].cpu().numpy()
            self.data.mocap_quat[:] = d.mocap_quat[i].cpu().numpy()
        mujoco.mj_forward(self.model, self.data)

    def render(self, i: int, view: str, goal=None) -> np.ndarray:
        self.sync(i)
        lookat, dist, az, el = VIEWS[view]
        origin = self.env.scene.env_origins[i].cpu().numpy()
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = np.array(lookat) + origin
        cam.distance, cam.azimuth, cam.elevation = dist, az, el
        self.renderer.update_scene(self.data, camera=cam, scene_option=self.opt)
        if goal is not None:
            scn = self.renderer.scene
            pos, mat, half = goal
            if scn.ngeom < scn.maxgeom:
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(
                    g,
                    mujoco.mjtGeom.mjGEOM_BOX,
                    np.asarray(half, float),
                    pos,
                    mat.reshape(-1),
                    np.array([0.1, 0.9, 0.1, 0.35], np.float32),
                )
                scn.ngeom += 1
        return self.renderer.render()


def save_png(img: np.ndarray, path: str) -> None:
    from PIL import Image

    Image.fromarray(img).save(path)


def goal_vis(env, cmd_name, i):
    t = env.command_manager.get_term(cmd_name)
    if not hasattr(t, "goal_pos"):
        return None
    from mjlab.utils.lab_api.math import matrix_from_quat

    pos = t.goal_pos[i].cpu().numpy()
    mat = matrix_from_quat(t.goal_quat[i : i + 1])[0].cpu().numpy()
    half = (t.obj_size[i] / 2).cpu().numpy() if t.cfg.goal_orientation else np.full(3, 0.012)
    return pos, mat, half


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("num_envs", type=int)
    ap.add_argument("steps", type=int)
    ap.add_argument("out")
    ap.add_argument("--policy", default="random")
    ap.add_argument("--render-envs", default="0")
    ap.add_argument("--views", default="front,side,top")
    ap.add_argument("--render-every", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import mjlab_hand  # noqa: F401  (registers tasks)

    torch.manual_seed(args.seed)
    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    t0 = time.time()
    env = ManagerBasedRlEnv(cfg, device=args.device)
    build_s = time.time() - t0
    cmd_name = next(iter(env.command_manager.active_terms))
    obs, _ = env.reset()
    report: dict = {
        "task": args.task,
        "num_envs": args.num_envs,
        "build_s": build_s,
        "obs_dims": {k: list(v.shape) for k, v in obs.items()},
        "action_dim": int(env.action_manager.total_action_dim),
        "action_terms": {
            n: env.action_manager.get_term(n).canonical_joint_names
            for n in env.action_manager.active_terms
        },
        "obs_terms": {
            g: env.observation_manager.active_terms[g] for g in env.observation_manager.active_terms
        },
        "reward_terms": env.reward_manager.active_terms,
        "termination_terms": env.termination_manager.active_terms,
        "sensors": list(env.scene.sensors.keys()),
        "step_dt": env.step_dt,
        "physics_dt": env.physics_dt,
        "max_episode_length": int(env.max_episode_length),
    }
    snap = Snapshotter(env)
    render_envs = [int(x) for x in args.render_envs.split(",") if x != ""]
    views = [v for v in args.views.split(",") if v]

    def dump_frames(tag):
        for i in render_envs:
            for v in views:
                save_png(
                    snap.render(i, v, goal_vis(env, cmd_name, i)),
                    os.path.join(args.out, f"{args.task}_env{i}_{tag}_{v}.png"),
                )

    dump_frames("step0")
    adim = report["action_dim"]
    rew_sum = {k: 0.0 for k in env.reward_manager.active_terms}
    term_counts = {k: 0 for k in env.termination_manager.active_terms}
    nan_obs = 0
    resets = 0
    rew_total = []
    t0 = time.time()
    for s in range(args.steps):
        if args.policy == "zero":
            a = torch.zeros(args.num_envs, adim, device=args.device)
        else:
            a = torch.rand(args.num_envs, adim, device=args.device) * 2 - 1
        obs, rew, term, trunc, extras = env.step(a)
        for k in rew_sum:
            rew_sum[k] += float(
                env.reward_manager._step_reward[:, env.reward_manager.active_terms.index(k)].sum()
            )
        for k in term_counts:
            term_counts[k] += int(env.termination_manager.get_term(k).sum())
        resets += int((term | trunc).sum())
        rew_total.append(float(rew.mean()))
        for v in obs.values():
            nan_obs += int((~torch.isfinite(v)).any(dim=-1).sum())
        if args.render_every and (s + 1) % args.render_every == 0:
            dump_frames(f"step{s + 1}")
    torch.cuda.synchronize()
    run_s = time.time() - t0
    dump_frames(f"step{args.steps}")
    t = env.command_manager.get_term(cmd_name)
    if not hasattr(t, "goal_pos"):
        report.update(
            {
                "steps": args.steps,
                "run_s": run_s,
                "nan_obs_rows": nan_obs,
                "resets": resets,
                "termination_counts": term_counts,
                "reward_sum_per_term": rew_sum,
            }
        )
        with open(os.path.join(args.out, f"{args.task}_smoke.json"), "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(json.dumps(report, indent=1, default=str))
        return
    report.update(
        {
            "steps": args.steps,
            "run_s": run_s,
            "env_steps_per_s": args.steps * args.num_envs / max(run_s, 1e-9),
            "reward_sum_per_term": rew_sum,
            "reward_mean_per_step": float(np.mean(rew_total)),
            "termination_counts": term_counts,
            "resets": resets,
            "nan_obs_rows": nan_obs,
            "final_successes_mean": float(t.successes.float().mean()),
            "lifted_frac": float(t.lifted.float().mean()),
            "obj_pos_env0": t.obj.data.root_link_pos_w[0].tolist(),
            "goal_pos_env0": t.goal_pos[0].tolist(),
            "table_top_env0": float(t.table_top()[0]),
        }
    )
    with open(os.path.join(args.out, f"{args.task}_smoke.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
