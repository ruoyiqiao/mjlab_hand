"""Geometric layout probe on the real compiled scene (CPU MuJoCo kinematics).

For a task id, samples hand configurations (base joints uniform within the
workspace bounds incl. extremes, fingers uniform within joint limits) and reports:

* world direction of each base joint axis at the anchor and the palm normal;
* fingertip / hand-geom reach extents (world, env 0);
* throw: min distance between any hand collision geom and any bucket geom over
  samples with the bucket at each corner of its sampling box (must stay > 0);
* dual: min hand-hand geom distance / penetration over random and adversarial
  (both hands pushed toward the center) samples, and whether both hands can
  reach the handoff center.

Writes <out>/<task>_root_layout.json. Sampling evidence only, not a proof over
the whole action space.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os

import mujoco
import numpy as np

BASE = ("base_x", "base_y", "base_z", "base_roll", "base_pitch", "base_yaw")


def ent_joint_qadr(model, prefix):
    out = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name and name.startswith(prefix + "/"):
            out[name.split("/", 1)[1]] = (model.jnt_qposadr[j], model.jnt_range[j].copy())
    return out


def ent_geoms(model, prefix, collision_only=True):
    ids = []
    for g in range(model.ngeom):
        b = model.geom_bodyid[g]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if bname.startswith(prefix + "/"):
            if collision_only and not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            ids.append(g)
    return ids


def body_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import mjlab_hand  # noqa: F401

    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg, device=args.device)
    env.reset()
    model = env.sim.mj_model
    data = mujoco.MjData(model)
    d = env.sim.data
    data.qpos[:] = d.qpos[0].cpu().numpy()
    if model.nmocap:
        data.mocap_pos[:] = d.mocap_pos[0].cpu().numpy()
        data.mocap_quat[:] = d.mocap_quat[0].cpu().numpy()
    mujoco.mj_kinematics(model, data)
    q0 = data.qpos.copy()

    hands = [n for n in ("hand", "hand_left", "hand_right") if n in env.scene.entities]
    actions = {n: env.action_manager.get_term(n) for n in env.action_manager.active_terms}
    ws = {a.cfg.entity_name: a.cfg.workspace for a in actions.values()}
    report = {"task": args.task, "hands": {}, "samples": args.samples}

    def set_cfg(hand, base_vals, finger_rand=True):
        j = ent_joint_qadr(model, hand)
        for k, nme in enumerate(BASE):
            data.qpos[j[nme][0]] = base_vals[k]
        if finger_rand:
            for nme, (adr, rng_) in j.items():
                if nme in BASE:
                    continue
                data.qpos[adr] = rng.uniform(rng_[0], rng_[1])

    def tips(hand):
        return np.stack(
            [data.xpos[body_id(model, f"{hand}/{f}_tip_head")] for f in ("ff", "mf", "rf", "th")]
        )

    for hand in hands:
        w = ws[hand]
        data.qpos[:] = q0
        mujoco.mj_kinematics(model, data)
        palm = body_id(model, f"{hand}/palm")
        p0 = data.xpos[palm].copy()
        R0 = data.xmat[palm].reshape(3, 3).copy()
        j = ent_joint_qadr(model, hand)
        axes = {}
        for nme in BASE[:3]:
            data.qpos[:] = q0
            data.qpos[j[nme][0]] += 0.01
            mujoco.mj_kinematics(model, data)
            axes[nme] = ((data.xpos[palm] - p0) / 0.01).round(4).tolist()
        for nme in BASE[3:]:
            data.qpos[:] = q0
            data.qpos[j[nme][0]] += 0.01
            mujoco.mj_kinematics(model, data)
            dR = data.xmat[palm].reshape(3, 3) @ R0.T
            w_vec = (
                np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / 2 / 0.01
            )
            axes[nme] = w_vec.round(4).tolist()
        lo = np.array([w.joint_bounds[n][0] for n in BASE])
        hi = np.array([w.joint_bounds[n][1] for n in BASE])
        all_tips = []
        geom_pts = []
        hg = ent_geoms(model, hand)
        corners = np.array(list(itertools.product(*zip(lo, hi))))
        for s in range(args.samples):
            base = corners[s % len(corners)] if s < len(corners) else rng.uniform(lo, hi)
            data.qpos[:] = q0
            set_cfg(hand, base)
            mujoco.mj_kinematics(model, data)
            all_tips.append(tips(hand))
            geom_pts.append(data.geom_xpos[hg].copy())
        all_tips = np.concatenate(all_tips)
        geom_pts = np.concatenate(geom_pts)
        report["hands"][hand] = {
            "workspace": w.to_dict(),
            "palm_pos_home": p0.round(4).tolist(),
            "palm_normal_world(body +x)": R0[:, 0].round(4).tolist(),
            "finger_dir_world(body +z)": R0[:, 2].round(4).tolist(),
            "base_axes_world": axes,
            "tip_min": all_tips.min(0).round(4).tolist(),
            "tip_max": all_tips.max(0).round(4).tolist(),
            "geom_center_min": geom_pts.min(0).round(4).tolist(),
            "geom_center_max": geom_pts.max(0).round(4).tolist(),
        }

    t = env.command_manager.get_term(next(iter(env.command_manager.active_terms)))
    if "bucket" in env.scene.entities:
        bg = ent_geoms(model, "bucket")
        hg = ent_geoms(model, "hand")
        bmid = model.body_mocapid[
            body_id(model, "bucket/bucket") if body_id(model, "bucket/bucket") >= 0 else 0
        ]
        lo_b, hi_b = np.array(t.cfg.bucket_pos_min), np.array(t.cfg.bucket_pos_max)
        w = ws["hand"]
        lo = np.array([w.joint_bounds[n][0] for n in BASE])
        hi = np.array([w.joint_bounds[n][1] for n in BASE])
        mocap_body = None
        for b in range(model.nbody):
            if model.body_mocapid[b] >= 0:
                nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                if nm.startswith("bucket"):
                    mocap_body = model.body_mocapid[b]
        assert mocap_body is not None
        min_d = np.inf
        worst = None
        bucket_corners = list(itertools.product(*zip(lo_b, hi_b)))
        n_s = max(200, args.samples // len(bucket_corners))
        for bc in bucket_corners:
            data.mocap_pos[mocap_body] = np.array(bc)
            for s in range(n_s):
                data.qpos[:] = q0
                set_cfg(
                    "hand", rng.uniform(lo, hi) if s % 2 else np.where(rng.random(6) > 0.5, hi, lo)
                )
                mujoco.mj_kinematics(model, data)
                mujoco.mj_collision(model, data)  # updates geom poses for distance
                for g1 in hg:
                    for g2 in bg:
                        dist = mujoco.mj_geomDistance(model, data, g1, g2, 0.5, None)
                        if dist < min_d:
                            min_d, worst = (
                                dist,
                                (bc, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1)),
                            )
        report["hand_bucket_min_distance_m"] = float(min_d)
        report["hand_bucket_worst"] = str(worst)
        del bmid

    if "hand_left" in env.scene.entities and "hand_right" in env.scene.entities:
        gl, gr = ent_geoms(model, "hand_left"), ent_geoms(model, "hand_right")
        wl, wr = ws["hand_left"], ws["hand_right"]
        lol = np.array([wl.joint_bounds[n][0] for n in BASE])
        hil = np.array([wl.joint_bounds[n][1] for n in BASE])
        lor = np.array([wr.joint_bounds[n][0] for n in BASE])
        hir = np.array([wr.joint_bounds[n][1] for n in BASE])
        dists = []
        pen = 0
        for s in range(args.samples):
            data.qpos[:] = q0
            set_cfg("hand_left", rng.uniform(lol, hil))
            set_cfg("hand_right", rng.uniform(lor, hir))
            mujoco.mj_kinematics(model, data)
            md = min(mujoco.mj_geomDistance(model, data, a, b, 0.5, None) for a in gl for b in gr)
            dists.append(md)
            pen += md < 0
        # Adversarial: both hands pushed to the max toward the center (base_z hi),
        # other base joints random, random fingers.
        adv, pen_a = [], 0
        for s in range(args.samples):
            data.qpos[:] = q0
            bl = rng.uniform(lol, hil)
            bl[2] = hil[2]
            br = rng.uniform(lor, hir)
            br[2] = hir[2]
            set_cfg("hand_left", bl)
            set_cfg("hand_right", br)
            mujoco.mj_kinematics(model, data)
            md = min(mujoco.mj_geomDistance(model, data, a, b, 0.5, None) for a in gl for b in gr)
            adv.append(md)
            pen_a += md < 0
        report["hand_hand_adversarial_center"] = {
            "min_distance_m": float(np.min(adv)),
            "frac_penetrating": float(pen_a / args.samples),
        }
        report["hand_hand_random"] = {
            "min_distance_m": float(np.min(dists)),
            "p01_distance_m": float(np.percentile(dists, 1)),
            "frac_penetrating": float(pen / args.samples),
        }
    with open(os.path.join(args.out, f"{args.task}_root_layout.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(json.dumps(report, indent=1, default=str))
    del torch


if __name__ == "__main__":
    main()
