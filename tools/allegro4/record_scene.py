"""Record a policy rollout as geometry + per-frame poses for offline rendering (tools/allegro4/render_blender.py).

Usage: python record_scene.py <task_id> <checkpoint.pt|zero> <out.npz>
           [--num-envs 16] [--seconds 6] [--fps 30] [--seed 2001] [--device cpu] [--warmup-s 0] [--set k=v ...]

Runs the task's play config (no added noise) with the deterministic policy in --num-envs envs and keeps the env that
runs longest without a reset (ties: higher return), so the clip shows an uninterrupted episode. Works with any run's
frozen source (only the mjlab env / runner API is used); run it with PYTHONPATH=<run>/src/src.

The .npz holds every visible geom (groups 0-2, alpha > 0, planes skipped) as a triangle mesh in its geom frame
(MuJoCo meshes as compiled, primitives tessellated here), its RGBA, entity name, and per frame its world pose relative
to the env origin; plus the goal pose of tasks with an orientation goal (drawn as a translucent ghost).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import mujoco
import numpy as np
import torch


def _uv_sphere(n_lat=12, n_lon=20):
    th = np.linspace(0, np.pi, n_lat + 1)
    ph = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    v = np.array(
        [[np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)] for t in th for p in ph]
    )
    f = []
    for i in range(n_lat):
        for j in range(n_lon):
            a, b = i * n_lon + j, i * n_lon + (j + 1) % n_lon
            c, d = a + n_lon, b + n_lon
            f += [[a, c, b], [b, c, d]]
    return v, np.array(f)


def _cylinder(n=24, caps=True):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([np.cos(ang), np.sin(ang)], 1)
    v = np.concatenate([np.c_[ring, -np.ones(n)], np.c_[ring, np.ones(n)], [[0, 0, -1], [0, 0, 1]]])
    f = []
    for j in range(n):
        k = (j + 1) % n
        f += [[j, k, n + j], [k, n + k, n + j]]
        if caps:
            f += [[2 * n, k, j], [2 * n + 1, n + j, n + k]]
    return v, np.array(f)


def primitive_mesh(gtype: int, size: np.ndarray):
    """Triangle mesh of a MuJoCo primitive in its geom frame."""
    T = mujoco.mjtGeom
    if gtype == T.mjGEOM_BOX:
        v = (
            np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
            * size[:3]
        )
        f = np.array(
            [
                [0, 1, 3],
                [0, 3, 2],
                [4, 6, 7],
                [4, 7, 5],
                [0, 4, 5],
                [0, 5, 1],
                [2, 3, 7],
                [2, 7, 6],
                [0, 2, 6],
                [0, 6, 4],
                [1, 5, 7],
                [1, 7, 3],
            ]
        )
        return v, f
    if gtype in (T.mjGEOM_SPHERE, T.mjGEOM_ELLIPSOID):
        v, f = _uv_sphere()
        return v * (size[:3] if gtype == T.mjGEOM_ELLIPSOID else size[0]), f
    if gtype == T.mjGEOM_CYLINDER:
        v, f = _cylinder()
        return v * np.array([size[0], size[0], size[1]]), f
    if gtype == T.mjGEOM_CAPSULE:
        v, f = _uv_sphere(12, 20)
        v = v * size[0]
        v[:, 2] += (
            np.sign(v[:, 2]) * size[1]
        )  # split the sphere at the equator: hemispheres + cylinder band
        return v, f
    return None, None


def export_geometry(m: mujoco.MjModel):
    keep, meshes, geoms = [], {}, []
    for g in range(m.ngeom):
        t = int(m.geom_type[g])
        if t == mujoco.mjtGeom.mjGEOM_PLANE or m.geom_group[g] > 2:
            continue
        rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
        if rgba[3] <= 0.0:
            continue
        if t == mujoco.mjtGeom.mjGEOM_MESH:
            mid = int(m.geom_dataid[g])
            key = f"mesh{mid}"
            if key not in meshes:
                va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
                meshes[key] = (m.mesh_vert[va : va + vn].copy(), m.mesh_face[fa : fa + fn].copy())
        else:
            v, f = primitive_mesh(t, m.geom_size[g])
            if v is None:
                continue
            key = f"geom{g}"
            meshes[key] = (v, f)
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
        if "collision" in name.lower() and any(
            "collision"
            not in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, k) or "collision").lower()
            and m.geom_bodyid[k] == m.geom_bodyid[g]
            and m.geom_group[k] <= 2
            for k in range(m.ngeom)
            if k != g
        ):
            continue  # the body already has a visual geom: do not draw its collision copy as well
        keep.append(g)
        geoms.append(
            {
                "geom": g,
                "mesh": key,
                "rgba": [float(x) for x in rgba],
                "name": name,
                "body": body,
                "entity": (body or name).split("/")[0],
            }
        )
    return keep, meshes, geoms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument(
        "--warmup-s", type=float, default=0.0, help="simulate this long before recording starts"
    )
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=2001)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--pick",
        default="longest",
        choices=("longest", "goals"),
        help="which env to keep: the longest run before a reset (default), or among the envs without a "
        "reset the one that reaches the most goals in the clip (shows the skill, e.g. in-hand rotation)",
    )
    ap.add_argument(
        "--set",
        nargs="*",
        default=[],
        help="command-spec overrides key=value, as in eval_three_ball.py (e.g. random_phase=true)",
    )
    a = ap.parse_args()

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    import mjlab_hand  # noqa: F401  (registers the run's tasks)

    try:
        from mjlab_hand.registration import register_tasks

        register_tasks()
    except ImportError:
        pass
    torch.manual_seed(a.seed)
    cfg = load_env_cfg(a.task, play=True)
    cfg.scene.num_envs = a.num_envs
    cfg.seed = a.seed
    spec = getattr(next(iter(cfg.commands.values())), "spec", None)
    for kv in (
        a.set
    ):  # same parsing as eval_three_ball.py, so a clip uses the evaluated start-state distribution
        k, v = kv.split("=", 1)
        old = getattr(spec, k)
        setattr(
            spec, k, v.lower() in ("1", "true", "yes") if isinstance(old, bool) else type(old)(v)
        )
    raw = ManagerBasedRlEnv(cfg, device=a.device)
    rl_cfg = load_rl_cfg(a.task)
    env = RslRlVecEnvWrapper(raw, clip_actions=rl_cfg.clip_actions)
    policy = None
    if a.ckpt != "zero":
        runner = MjlabOnPolicyRunner(env, asdict(rl_cfg), device=a.device)
        runner.load(a.ckpt, load_cfg={"actor": True}, strict=True, map_location=a.device)
        policy = runner.get_inference_policy(device=a.device)
    m = raw.sim.mj_model
    keep, meshes, geoms = export_geometry(m)
    keep_t = torch.tensor(keep, dtype=torch.long, device=raw.device)
    cmd_name = next(iter(raw.command_manager.active_terms))
    term = raw.command_manager.get_term(cmd_name)
    goal_kind = ""
    if hasattr(
        term, "goal_pos"
    ):  # orientation goal -> object ghost at the goal pose, position-only goal -> point
        # allegro4 tasks say which via cfg.goal_orientation; tasks without that flag (in-hand rotation) are orientation goals
        orient = getattr(getattr(term, "cfg", None), "goal_orientation", True)
        goal_kind = "box" if hasattr(term, "goal_quat") and orient else "point"
    has_goal = goal_kind == "box"

    dt = float(raw.step_dt)
    every = max(1, round(1.0 / (a.fps * dt)))
    n_warm = int(round(a.warmup_s / dt))
    n_rec = int(round(a.seconds / dt))
    origins = raw.scene.env_origins
    E = a.num_envs
    pos, mat, gpos, gmat, ghalf, opos, omat = [], [], [], [], [], [], []
    obj = getattr(term, "obj", None) or getattr(
        term, "object", None
    )  # manipulated object: its root pose per frame
    first_done = np.full(E, n_warm + n_rec + 1)
    ret = np.zeros(E)
    adim = raw.action_manager.total_action_dim
    obs, _ = env.reset()
    for s in range(n_warm + n_rec):
        with torch.no_grad():
            act = policy(obs) if policy is not None else torch.zeros(E, adim, device=raw.device)
        obs, rew, dones, _ = env.step(act)
        ret += rew.cpu().numpy()
        d = dones.cpu().numpy().astype(bool)
        first_done = np.where(d & (first_done > s), s, first_done)
        if s >= n_warm and (s - n_warm) % every == 0:
            xp = raw.sim.data.geom_xpos[:, keep_t] - origins[:, None, :]
            xm = raw.sim.data.geom_xmat[:, keep_t].reshape(E, len(keep), 9)
            pos.append(xp.cpu().numpy().astype(np.float32))
            mat.append(xm.cpu().numpy().astype(np.float32))
            if goal_kind:
                gpos.append((term.goal_pos - origins).cpu().numpy().astype(np.float32))
            if obj is not None:
                from mjlab.utils.lab_api.math import matrix_from_quat

                opos.append((obj.data.root_link_pos_w - origins).cpu().numpy().astype(np.float32))
                omat.append(
                    matrix_from_quat(obj.data.root_link_quat_w)
                    .reshape(E, 9)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            if has_goal:
                from mjlab.utils.lab_api.math import matrix_from_quat

                gmat.append(
                    matrix_from_quat(term.goal_quat).reshape(E, 9).cpu().numpy().astype(np.float32)
                )
                if hasattr(term, "obj_size"):
                    ghalf.append((term.obj_size / 2).cpu().numpy().astype(np.float32))
    # The env with the longest run before its first reset counts; resets during warm-up still count against it.
    goals = np.zeros(E, dtype=int)
    if gmat:  # goals reached in the clip = goal re-samples (orientation steps > 1 deg)
        G = np.stack(gmat).reshape(len(gmat), E, 3, 3)
        cos = (np.einsum("teij,teij->te", G[1:], G[:-1]) - 1) / 2
        goals = (np.degrees(np.arccos(np.clip(cos, -1, 1))) > 1.0).sum(0)
    if a.pick == "goals" and gmat:
        clean_envs = first_done >= n_warm + n_rec
        best = int(np.lexsort((-ret, -np.where(clean_envs, goals, -1)))[0])
    else:
        best = int(np.lexsort((-ret, -first_done))[0])
    clean = bool(first_done[best] >= n_warm + n_rec)
    print(
        json.dumps(
            {
                "task": a.task,
                "ckpt": a.ckpt,
                "env": best,
                "goals_in_clip": int(goals[best]),
                "goals_in_clip_all_envs": sorted(goals.tolist(), reverse=True),
                "no_reset_in_clip": clean,
                "first_reset_s": float(first_done[best] * dt),
                "return": float(ret[best]),
                "envs_without_reset": int((first_done >= n_warm + n_rec).sum()),
                "frames": len(pos),
                "fps": 1.0 / (every * dt),
            }
        )
    )
    keys = sorted(meshes)
    np.savez_compressed(
        a.out,
        meta=json.dumps(
            {
                "task": a.task,
                "ckpt": a.ckpt,
                "env": best,
                "fps": 1.0 / (every * dt),
                "dt": dt,
                "no_reset_in_clip": clean,
                "geoms": geoms,
                "mesh_keys": keys,
                "has_goal": bool(has_goal),
                "goal_kind": goal_kind,
            }
        ),
        pos=np.stack(pos)[:, best],
        mat=np.stack(mat)[:, best],
        goal_pos=np.stack(gpos)[:, best] if goal_kind else np.zeros((0, 3)),
        goal_mat=np.stack(gmat)[:, best] if has_goal else np.zeros((0, 9)),
        goal_half=np.stack(ghalf)[:, best] if ghalf else np.zeros((0, 3)),
        obj_pos=np.stack(opos)[:, best] if opos else np.zeros((0, 3)),
        obj_mat=np.stack(omat)[:, best] if omat else np.zeros((0, 9)),
        **{f"v_{k}": meshes[k][0].astype(np.float32) for k in keys},
        **{f"f_{k}": meshes[k][1].astype(np.int32) for k in keys},
    )


if __name__ == "__main__":
    main()
