"""Kinematic thumb-opposition reach per hand (CPU, no physics): can the thumb tip meet the other fingertips?

Usage: python thumb_reach.py [--hands allegro,leap,...] [--samples 20000]

Samples finger joint configurations uniformly inside the joint limits (base fixed at the anchor) and reports, per non-thumb
tip, the minimum and the 1st-percentile distance between the thumb tip collision geom and that tip's collision geom
(mj_geomDistance, surface to surface), plus the fraction of samples where they are within 1 cm. A hand whose thumb cannot
get near the other tips cannot form the "thumb + >= 1 fingertip" grasp the throw task requires.
"""

from __future__ import annotations

import argparse
import re

import mujoco
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", default="allegro,leap,wuji,sharpa,xhand,shadow")
    ap.add_argument("--samples", type=int, default=20000)
    a = ap.parse_args()
    from mjlab_hand.tasks_common.env_builder import make_hand_entity_cfg
    from mjlab_hand.tasks_common.hand_profile import get_profile
    from mjlab_hand.tasks_common.layouts import single_hand_workspace

    rng = np.random.default_rng(0)
    for hand in a.hands.split(","):
        prof = get_profile(hand)
        m = (
            make_hand_entity_cfg(prof.spec, single_hand_workspace(hand=hand), profile=prof)
            .spec_fn()
            .compile()
        )
        d = mujoco.MjData(m)
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
        tips = [g for g, n in enumerate(names) if re.fullmatch(prof.tip_geoms, n)]
        thumb = [g for g in tips if prof.thumb_geom in names[g]]
        assert len(thumb) == 1, (hand, [names[g] for g in tips])
        th, others = thumb[0], [g for g in tips if g != thumb[0]]
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in prof.obs_joints]
        lo = np.array([m.jnt_range[j][0] for j in jids])
        hi = np.array([m.jnt_range[j][1] for j in jids])
        adr = [m.jnt_qposadr[j] for j in jids]
        dist = np.zeros((a.samples, len(others)))
        fromto = np.zeros(6)
        for s in range(a.samples):
            d.qpos[:] = m.qpos0
            d.qpos[adr] = rng.uniform(lo, hi)
            mujoco.mj_kinematics(m, d)
            for k, g in enumerate(others):
                dist[s, k] = mujoco.mj_geomDistance(m, d, th, g, 0.2, fromto)
        print(f"== {hand} ({prof.key}) thumb {names[th]}")
        for k, g in enumerate(others):
            col = dist[:, k]
            print(
                f"   {names[g]:38s} min {col.min() * 100:6.2f} cm  p1 {np.percentile(col, 1) * 100:6.2f} cm  "
                f"within 1 cm {np.mean(col < 0.01) * 100:5.2f} %"
            )


if __name__ == "__main__":
    main()
