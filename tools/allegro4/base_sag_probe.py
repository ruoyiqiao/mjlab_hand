"""Per-hand root dynamics and fingertip contact parameters of the single-hand tasks (CPU).

Usage: python base_sag_probe.py [--tasks Regrasp-Allegro,Regrasp-LEAP,...] [--steps 150]

For each task: effective inertia of each base joint at home (diagonal of the joint-space mass matrix incl. armature, the
quantity behind ``TRACK_INERTIA``), total hand mass, base kp/kv, the steady-state sag of the palm under zero action (the
target is held, gravity pulls the hand down; sag = palm drop after --steps control steps), and friction / condim of the
fingertip collision geoms. Use: compare every hand with Allegro (the reference the task layouts were tuned on).
"""

from __future__ import annotations

import argparse
import re

import mujoco
import numpy as np
import torch

BASE = ("base_x", "base_y", "base_z", "base_roll", "base_pitch", "base_yaw")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tasks",
        default="Regrasp-Allegro,Regrasp-LEAP,Regrasp-Wuji,Regrasp-Sharpa,Regrasp-XHand,Regrasp-Shadow",
    )
    ap.add_argument("--steps", type=int, default=150)
    a = ap.parse_args()
    from mjlab_hand.registration import register_tasks

    register_tasks()
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    for task in a.tasks.split(","):
        cfg = load_env_cfg(task, play=True)
        cfg.scene.num_envs = 2
        env = ManagerBasedRlEnv(cfg, device="cpu")
        m = env.sim.mj_model
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        env.reset()
        d.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
        mujoco.mj_forward(m, d)
        M = np.zeros((m.nv, m.nv))
        try:
            mujoco.mj_fullM(m, d, M)  # mujoco >= 3.5: (model, data, dst)
        except TypeError:
            mujoco.mj_fullM(m, M, d.qM)  # older: (model, dst, qM)
        inert, gains = {}, {}
        for b in BASE:
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"hand/{b}")
            v = m.jnt_dofadr[j]
            inert[b] = float(M[v, v])
            for u in range(m.nu):
                if (
                    m.actuator_trnid[u, 0] == j
                    and m.actuator_trntype[u] == mujoco.mjtTrn.mjTRN_JOINT
                ):
                    gains[b] = (float(m.actuator_gainprm[u, 0]), float(-m.actuator_biasprm[u, 2]))
        hand_bodies = [
            i
            for i in range(m.nbody)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith("hand/")
        ]
        mass = float(sum(m.body_mass[i] for i in hand_bodies))
        gravcomp = float(max(m.body_gravcomp[i] for i in hand_bodies)) if hand_bodies else 0.0
        t = next(iter(env.command_manager._terms.values()))
        act = env.action_manager.get_term("hand")
        z0 = t.palm_pose(0)[0][:, 2].clone()
        zero = torch.zeros(env.num_envs, act.action_dim)
        if (
            act.cfg.finger_mode == "absolute"
        ):  # hold the current finger targets, not the range centre
            zero[:, 6 : 6 + act._nf] = ((act._finger_target - act._f_offset) / act._f_scale).clamp(
                -1, 1
            )
        for _ in range(a.steps):
            env.step(zero)
        sag = float((z0 - t.palm_pose(0)[0][:, 2]).mean())
        pat = t.profiles[0].tip_geoms
        tips = []
        for g in range(m.ngeom):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if n.startswith("hand/") and re.fullmatch(pat, n.split("/", 1)[1]):
                tips.append(
                    (
                        n.split("/", 1)[1],
                        round(float(m.geom_friction[g, 0]), 3),
                        int(m.geom_condim[g]),
                        int(m.geom_priority[g]),
                    )
                )
        print(
            f"== {task}: hand mass {mass:.3f} kg  gravcomp {gravcomp}  palm sag after {a.steps} steps: {sag * 100:.1f} cm"
        )
        print("   inertia   ", {k: round(v, 4) for k, v in inert.items()})
        print("   base kp/kv", gains)
        print("   tips (name, friction, condim, priority)", tips)
        env.close()


if __name__ == "__main__":
    main()
