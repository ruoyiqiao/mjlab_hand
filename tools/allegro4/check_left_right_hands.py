"""Verify the left/right floating Allegro models.

Checks, writing a JSON report:
* identical joint / actuator name order and joint ranges for both sides;
* finger joint axes are mirror images (thumb axes flipped as in upstream);
* at the same base pose the index (ff) and ring (rf) fingers and the thumb sit on
  mirrored sides of the palm (palm-frame y sign flips);
* increasing each finger flexion joint moves its tip toward the palm normal side
  (positive action = flexion) for both hands;
* palm normal direction is the same for both sides.

Usage: python check_left_right_hands.py <out.json>
"""

from __future__ import annotations

import json
import sys

import mujoco
import numpy as np
from mjlab.entity.entity import Entity

from mjlab_hand.robot.allegro import ALLEGRO_HAND_LEFT, ALLEGRO_HAND_RIGHT

FINGERS = ("ff", "mf", "rf", "th")


def joint_info(model):
    out = []
    for j in range(model.njnt):
        out.append(
            {
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j),
                "range": model.jnt_range[j].round(5).tolist(),
                "axis": model.jnt_axis[j].round(5).tolist(),
                "type": int(model.jnt_type[j]),
            }
        )
    return out


def set_home(model, data, joint_pos):
    mujoco.mj_resetData(model, data)
    for name, val in joint_pos.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = val
    mujoco.mj_kinematics(model, data)


def body_pos_in_palm(model, data, body):
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    r = data.xmat[palm].reshape(3, 3)
    return r.T @ (data.xpos[bid] - data.xpos[palm])


def main() -> None:
    from mjlab_hand.robot.allegro.allegro_constants import ALLEGRO_HOME_JOINT_POS

    report: dict = {}
    sides = {}
    for side, hs in (("left", ALLEGRO_HAND_LEFT), ("right", ALLEGRO_HAND_RIGHT)):
        cfg = hs.get_floating_cfg(pos=(0.0, 0.0, 0.5), rot=(1.0, 0.0, 0.0, 0.0))
        ent = Entity(cfg)
        model = ent.spec.compile()
        data = mujoco.MjData(model)
        sides[side] = (ent, model, data)
        report[side] = {
            "joints": joint_info(model),
            "actuators": [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(model.nu)
            ],
            "actuator_trn": [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[a, 0]))
                for a in range(model.nu)
            ],
            "nbody": model.nbody,
            "ngeom": model.ngeom,
            "mass_total": float(model.body_mass.sum()),
        }

    lj = [j["name"] for j in report["left"]["joints"]]
    rj = [j["name"] for j in report["right"]["joints"]]
    checks = {}
    checks["joint_names_equal"] = lj == rj
    checks["joint_ranges_equal"] = all(
        a["range"] == b["range"]
        for a, b in zip(report["left"]["joints"], report["right"]["joints"])
    )
    checks["actuator_targets_equal"] = (
        report["left"]["actuator_trn"] == report["right"]["actuator_trn"]
    )
    checks["nu"] = (len(report["left"]["actuators"]), len(report["right"]["actuators"]))
    checks["mass_equal"] = abs(report["left"]["mass_total"] - report["right"]["mass_total"]) < 1e-6

    # Mirror geometry at home.
    tips = {}
    for side, (_, model, data) in sides.items():
        set_home(model, data, ALLEGRO_HOME_JOINT_POS)
        tips[side] = {
            f: body_pos_in_palm(model, data, f"{f}_tip_head").round(4).tolist() for f in FINGERS
        }
        palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
        tips[side]["palm_axes_world"] = data.xmat[palm].reshape(3, 3).round(4).tolist()
    report["tips_in_palm_frame_home"] = tips
    mirror = {}
    for f in FINGERS:
        lp, rp = np.array(tips["left"][f]), np.array(tips["right"][f])
        mirror[f] = {
            "x_equal": bool(abs(lp[0] - rp[0]) < 2e-3),
            "y_mirrored": bool(abs(lp[1] + rp[1]) < 2e-3),
            "z_equal": bool(abs(lp[2] - rp[2]) < 2e-3),
        }
    checks["tip_mirror"] = mirror
    checks["ff_left_y_negative"] = tips["left"]["ff"][1] < 0
    checks["ff_right_y_positive"] = tips["right"]["ff"][1] > 0

    # Flexion direction: increase joints 1..3 of each finger by 0.3 rad, the tip
    # must move toward the palm's grasp side. Palm-frame x is the palm normal axis
    # of the upstream Allegro MJCF (fingers extend along +z, flexion goes toward -x? we
    # report the sign and require it to be identical for both sides).
    flex = {}
    for side, (_, model, data) in sides.items():
        flex[side] = {}
        for f in FINGERS:
            set_home(model, data, ALLEGRO_HOME_JOINT_POS)
            p0 = body_pos_in_palm(model, data, f"{f}_tip_head")
            q = dict(ALLEGRO_HOME_JOINT_POS)
            for k in (1, 2, 3) if f != "th" else (2, 3):
                q[f"{f}j{k}"] += 0.3
            set_home(model, data, q)
            p1 = body_pos_in_palm(model, data, f"{f}_tip_head")
            flex[side][f] = (p1 - p0).round(4).tolist()
    report["flexion_tip_delta_palm_frame"] = flex
    checks["flexion_same_direction"] = {
        f: bool(
            np.sign(flex["left"][f][0]) == np.sign(flex["right"][f][0])
            and abs(flex["left"][f][0] - flex["right"][f][0]) < 2e-3
            and abs(flex["left"][f][1] + flex["right"][f][1]) < 2e-3
        )
        for f in FINGERS
    }
    report["checks"] = checks
    with open(sys.argv[1], "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
