"""DualHand: how far can each hand's collision geometry reach toward the other hand?

Exact per-geom extent along world x (box: c + sum_k |R[0,k]| h_k, capsule: c + |R[0,2]| h + r,
sphere: c + r) over sampled hand configurations: base_z (the joint pointing toward the table
centre) fixed at 0, base_x/base_y/rotations at box corners and random, fingers at joint limits
and random. The base_z slide moves the whole hand exactly along world +-x (verified below), so
reach(base_z) = reach(0) + base_z. A separating plane x = 0 with margin m on each side holds for
every configuration iff base_z_max <= -m - reach_left(0) (left) and likewise for the right hand.

Usage: python dual_separation.py <out.json> [--task DualHand-Reorientation-Allegro]
           [--samples 20000] [--margin 0.02] [--device cuda:0]
"""

from __future__ import annotations

import argparse
import itertools
import json

import mujoco
import numpy as np

BASE = ("base_x", "base_y", "base_z", "base_roll", "base_pitch", "base_yaw")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--task", default="DualHand-Reorientation-Allegro")
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import mjlab_hand  # noqa: F401

    cfg = load_env_cfg(a.task)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg, device=a.device)
    env.reset()
    m = env.sim.mj_model
    d = mujoco.MjData(m)
    sd = env.sim.data
    d.qpos[:] = sd.qpos[0].cpu().numpy()
    if m.nmocap:
        d.mocap_pos[:] = sd.mocap_pos[0].cpu().numpy()
        d.mocap_quat[:] = sd.mocap_quat[0].cpu().numpy()
    q0 = d.qpos.copy()
    ws = {
        env.action_manager.get_term(n).cfg.entity_name: env.action_manager.get_term(n).cfg.workspace
        for n in env.action_manager.active_terms
    }
    res = {"task": a.task, "margin_m": a.margin, "samples": a.samples, "hands": {}}
    for hand, sign in (("hand_left", +1.0), ("hand_right", -1.0)):
        joints = {}
        for j in range(m.njnt):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if nm.startswith(hand + "/"):
                joints[nm.split("/", 1)[1]] = (m.jnt_qposadr[j], m.jnt_range[j].copy())
        geoms = [
            g
            for g in range(m.ngeom)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "").startswith(
                hand + "/"
            )
            and not (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0)
        ]
        types = sorted({int(m.geom_type[g]) for g in geoms})
        assert set(types) <= {2, 3, 6}, f"unsupported collision geom types {types}"

        def reach():
            mujoco.mj_kinematics(m, d)
            best, who = -np.inf, None
            for g in geoms:
                c = d.geom_xpos[g]
                R = d.geom_xmat[g].reshape(3, 3)
                s = m.geom_size[g]
                t = m.geom_type[g]
                cx = sign * c[0]
                rx = sign * R[0]
                if t == 6:
                    e = cx + abs(rx[0]) * s[0] + abs(rx[1]) * s[1] + abs(rx[2]) * s[2]
                elif t == 3:
                    e = cx + abs(rx[2]) * s[1] + s[0]
                else:
                    e = cx + s[0]
                if e > best:
                    best, who = e, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
            return best, who

        w = ws[hand]
        lo = np.array([w.joint_bounds[n][0] for n in BASE])
        hi = np.array([w.joint_bounds[n][1] for n in BASE])
        # verify: base_z moves the hand along sign * world x by exactly the joint displacement
        d.qpos[:] = q0
        r0, _ = reach()
        d.qpos[joints["base_z"][0]] += 0.1
        r1, _ = reach()
        lin = r1 - r0
        fingers = [n for n in joints if n not in BASE]
        corners = list(itertools.product(*[(lo[k], hi[k]) for k in (0, 1, 3, 4, 5)]))
        best, worst = -np.inf, None
        for s in range(a.samples):
            d.qpos[:] = q0
            b = (
                rng.uniform(lo, hi)
                if s >= len(corners) * 4
                else np.array([*corners[s % len(corners)][:2], 0.0, *corners[s % len(corners)][2:]])
            )
            b[2] = 0.0
            for k, n in enumerate(BASE):
                d.qpos[joints[n][0]] = b[k]
            mode = s % 3
            for n in fingers:
                adr, rg = joints[n]
                d.qpos[adr] = (
                    rg[1] if mode == 0 else (rg[0] if mode == 1 else rng.uniform(rg[0], rg[1]))
                )
            e, who = reach()
            if e > best:
                best, worst = e, {"geom": who, "base": b.round(3).tolist(), "finger_mode": mode}
        bz_max = -a.margin - best
        res["hands"][hand] = {
            "reach_toward_center_at_base_z0_m": round(float(best), 4),
            "worst": worst,
            "base_z_linearity_check": round(float(lin), 4),
            "current_base_z_bounds": list(w.joint_bounds["base_z"]),
            "anchor_x": w.anchor_pos[0],
            "base_z_max_for_margin": round(float(bz_max), 4),
        }
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
