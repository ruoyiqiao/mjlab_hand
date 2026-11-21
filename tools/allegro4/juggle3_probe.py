"""Juggle3 (two-hand three-ball cascade) parameters of a non-Allegro hand (CPU), written to hand_params/juggle3.json.

Usage: python juggle3_probe.py [--hands leap,wuji,sharpa,xhand,shadow] [--write] [--render DIR]

Only what the cascade's designed prior needs from the hand is measured; the schedule, skills, ball and rewards are
Allegro's (juggling/cascade.py). Per side (each hand's left / right model; a hand with one model uses it for both):

* finger poses by the fingertip-role rule of juggle_cup_probe.cup_pose (a joint that lifts the tips toward the palm is a
  flexion joint, the rest stay at 0): ``cup`` = the Juggle1/2 cup amplitude, ``open`` = 0.1 rad (Allegro's release pose
  is 0.0-0.1), ``grip`` = the amplitude in GRIP_AMPS with the largest drop basin for the Juggle3 ball (Allegro chose its
  grip because it keeps the ball centred: 1.4 cm vs 3-4 cm in the cup).
* the ball rest point in the grip (rel. palm origin), verified to stay within 10 mm for 2 s.
* hand_y: Allegro's 0.131 m for every hand. The static clearance of both hands at their release points (lat = 0.075 m
  inward, mj_geomDistance over collision geoms) is recorded only: it is already -23 mm for Allegro, which trains and
  passes with it, so it is not a criterion (hand-hand contact is a termination-free sensor as for Allegro).
* a side whose model has no stable grip uses the other side's model for both hands (chirality is not needed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import juggle_cup_probe as jcp  # noqa: E402

PARAMS = (
    Path(__file__).resolve().parents[2] / "src/mjlab_hand/tasks_common/hand_params/juggle3.json"
)
GRIP_AMPS = (0.5, 0.8, 1.1, 1.4)
OPEN_AMP = 0.1
HAND_Y = 0.131  # Allegro / paper: every hand keeps it (Allegro's own static clearance at release is -23 mm)
BALL = {"ball_radius": 0.03175, "ball_mass": 0.130, "ball_solref": (0.02, 1.5)}  # CascadeSpec


def side_params(hand, side, cup_amp):
    from mjlab_hand.tasks_common.hand_profile import get_profile

    prof = get_profile(hand, side)
    ov = dict(BALL, side=side)
    env = jcp.build(hand, overrides=ov)
    m = env.sim.mj_model
    cup, kinds = jcp.cup_pose(m, prof, cup_amp)
    poses = {"cup": cup, "open": jcp.cup_pose(m, prof, OPEN_AMP)[0]}
    grips = {a: jcp.cup_pose(m, prof, a)[0] for a in GRIP_AMPS}
    env.close()
    cands = []
    for a, pose in grips.items():
        rest, basin, ntot = jcp.ball_well(hand, pose, overrides=ov)
        print(
            f"   {hand}/{prof.side} grip amp {a}: rest {None if rest is None else rest.round(4)} basin {basin}/{ntot}"
        )
        if rest is not None and rest[2] > -0.02 and basin >= 5:
            cands.append((basin, a, rest, ntot))
    # Largest basin first; the first one whose rest point also holds the ball when it is placed there is taken.
    for basin, a, start, ntot in sorted(cands, key=lambda c: -c[0]):
        env = jcp.build(hand, cup_pose=grips[a], cup_rel=start, overrides=ov)
        path, v = jcp.settle(env, 200)
        env.close()
        rest = path[-1]
        env = jcp.build(hand, cup_pose=grips[a], cup_rel=rest, overrides=ov)
        path2, v2 = jcp.settle(env, 200)
        env.close()
        drift = float(np.max(np.linalg.norm(path2 - rest, axis=1)))
        ok = v < 0.02 and v2 < 0.02 and drift < 0.01
        print(
            f"== {hand}/{prof.side}: grip amp {a} rest {rest.round(4)} basin {basin}/{ntot} drift {drift * 1000:.1f} mm "
            f"v {v2:.3f} -> {'OK' if ok else 'FAIL'}"
        )
        if ok:
            break
    else:
        return None
    poses["grip"] = grips[a]
    return {
        "model_side": prof.side,
        "poses": poses,
        "kinds": kinds,
        "grip_amp": a,
        "cup_amp": cup_amp,
        "cup_rel_palm": [float(x) for x in rest],
        "basin": [basin, ntot],
        "verify_drift_m": drift,
    }


def hand_clearance(hand, hand_y):
    """Min distance between the two hands' collision geoms with both hands moved to their release points."""
    from mjlab.envs import ManagerBasedRlEnv

    from mjlab_hand.juggling.cascade import CascadeSpec, make_cascade_env_cfg

    cfg = make_cascade_env_cfg(CascadeSpec(hand=hand, hand_y=hand_y), play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg, device="cpu")
    t = env.command_manager.get_term("juggler")
    m = env.sim.mj_model
    d = mujoco.MjData(m)
    d.qpos[:] = env.sim.data.qpos[0].numpy()
    lat = t.sp.lat
    for i, h in enumerate(("hand_r", "hand_l")):
        dq = t.slide_jinv[i].numpy() @ (-t.outward[i].numpy() * lat)
        for k, jn in enumerate(("base_x", "base_y", "base_z")):
            d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{h}/{jn}")]] += (
                dq[k]
            )
    mujoco.mj_forward(m, d)
    geoms = {
        h: [
            g
            for g in range(m.ngeom)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(h + "/")
            and m.geom_contype[g] | m.geom_conaffinity[g]
        ]
        for h in ("hand_r", "hand_l")
    }
    dmin = np.inf
    for a in geoms["hand_r"]:
        for b in geoms["hand_l"]:
            dmin = min(dmin, mujoco.mj_geomDistance(m, d, a, b, 0.2, None))
    palms = [
        d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{h}/{pb}")].copy()
        for h, pb in zip(("hand_r", "hand_l"), t.palm_body_names)
    ]
    env.close()
    return float(dmin), palms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", default="leap,wuji,sharpa,xhand,shadow")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--stage", default="all", choices=("all", "poses", "spacing"))
    ap.add_argument(
        "--out",
        default="",
        help="write the poses of these hands to this JSON instead (parallel runs)",
    )
    ap.add_argument(
        "--merge",
        nargs="*",
        default=[],
        help="merge per-hand JSON files into juggle3.json and exit",
    )
    a = ap.parse_args()
    if a.merge:
        out = json.loads(PARAMS.read_text()) if PARAMS.exists() else {}
        for p in a.merge:
            out.update(json.loads(Path(p).read_text()))
        PARAMS.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
        print("merged", sorted(out), "->", PARAMS)
        return
    from mjlab_hand.juggling.juggle_env_cfg import hand_juggle_params
    from mjlab_hand.registration import register_tasks
    from mjlab_hand.tasks_common.hand_profile import get_profile

    register_tasks()
    out = json.loads(PARAMS.read_text()) if PARAMS.exists() else {}
    if a.stage in ("all", "spacing"):
        print(
            f"== allegro (reference): clearance at release {hand_clearance('allegro', 0.131)[0] * 1000:.1f} mm"
        )
    for hand in a.hands.split(","):
        if a.stage in ("all", "poses"):
            cup_amp = float(hand_juggle_params(hand)["amp"])
            rec = {"poses": {}, "cup_rel_palm": {}, "sides": {}}
            done = {}
            for side in ("right", "left"):
                ms = get_profile(hand, side).side
                if ms not in done:
                    done[ms] = side_params(hand, side, cup_amp)
            if None in done.values() and any(done.values()):
                # Fallback (plan, Stage B): a side whose model has no stable grip uses the other side's model;
                # the cascade does not need chirality.
                good = next(v for v in done.values() if v is not None)
                print(
                    f"!! {hand}: {[k for k, v in done.items() if v is None]} has no stable grip; "
                    f"both hands use the {good['model_side']} model"
                )
                done = {k: good for k in done}
            for side in ("right", "left"):
                r = done[get_profile(hand, side).side]
                if r is None:
                    print(f"!! {hand}/{side}: no stable grip, not written")
                    rec = None
                    break
                rec["poses"][side] = r["poses"]
                rec["cup_rel_palm"][side] = r["cup_rel_palm"]
                rec["sides"][side] = {
                    k: r[k]
                    for k in ("model_side", "grip_amp", "cup_amp", "basin", "verify_drift_m")
                }
            if rec is None:
                continue
            rec["hand_y"] = out.get(hand, {}).get("hand_y", 0.131)
            out[hand] = rec
            if a.out:
                Path(a.out).write_text(json.dumps({hand: rec}, indent=1, sort_keys=True) + "\n")
            elif a.write:  # the spacing stage builds the env from the file
                PARAMS.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
        if a.stage in ("all", "spacing") and hand in out:
            d0, palms = hand_clearance(hand, HAND_Y)
            print(
                f"== {hand}: clearance at release (hand_y {HAND_Y}) {d0 * 1000:.1f} mm; palms {np.round(palms, 3).tolist()}"
            )
            out[hand]["hand_y"] = HAND_Y
            out[hand]["release_clearance_m"] = d0
    if a.write:
        PARAMS.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
        print("wrote", PARAMS)


if __name__ == "__main__":
    main()
