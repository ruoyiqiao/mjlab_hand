"""Hand frame probe (CPU): palm/finger/normal directions, base-joint axes and anchor quaternions of a floating hand.

Usage: python hand_frame_probe.py [--hands allegro,leap,...] [--out frames.json]

For every (hand, side) with a floating model it compiles the model, evaluates the home pose and the pose with
every limited finger joint at mid range, and reports, in the attachment frame (the compiled world frame at q = 0,
which is what the anchor quaternion rotates; the root body's own XML quat is part of the model):
  finger_dir  unit vector root -> mean of the long-finger tips with fingers extended
  normal      palm normal pointing to the side the fingers curl to (mid-range pose displacement)
  base_axes   each base joint axis in the hand frame (fwd = finger_dir, lat = up x fwd, up = -normal),
              with the nearest signed hand axis and the angular error
  anchor_down_px / _down_nx / _up_px   quaternions (wxyz) placing the palm facing down / up with the fingers toward +x / -x
  thumb_side  sign of the thumb tip along lat (+1: thumb on +lat)
These feed hand_profile.py; they are measured, not assumed. Validate by rendering keyframes.
"""

from __future__ import annotations

import argparse
import json

import mujoco
import numpy as np

BASE = ("base_x", "base_y", "base_z", "base_roll", "base_pitch", "base_yaw")
THUMB_TIP = {"allegro": 3, "leap": 3, "wuji": 0, "sharpa": 0, "xhand": 4, "shadow": 4}


def _unit(v):
    return v / np.linalg.norm(v)


def _quat_from_mat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


def hands():
    from mjlab_hand.robot.allegro import ALLEGRO_HAND_LEFT, ALLEGRO_HAND_RIGHT
    from mjlab_hand.robot.leap import LEAP_HAND, LEAP_HAND_LEFT
    from mjlab_hand.robot.shadow_hand import SHADOW_HAND_LEFT, SHADOW_HAND_RIGHT
    from mjlab_hand.robot.sharpa_wave import get_sharpa_hand_spec
    from mjlab_hand.robot.wuji import WUJI_HAND
    from mjlab_hand.robot.xhand import XHAND_HAND

    return [
        ("allegro", "left", ALLEGRO_HAND_LEFT),
        ("allegro", "right", ALLEGRO_HAND_RIGHT),
        ("leap", "right", LEAP_HAND),
        ("leap", "left", LEAP_HAND_LEFT),
        ("wuji", "right", WUJI_HAND),
        ("sharpa", "left", get_sharpa_hand_spec("left")),
        ("sharpa", "right", get_sharpa_hand_spec("right")),
        ("xhand", "left", XHAND_HAND),
        ("shadow", "left", SHADOW_HAND_LEFT),
        ("shadow", "right", SHADOW_HAND_RIGHT),
    ]


def probe(name, side, spec):
    cfg = spec.get_floating_cfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0))
    m = cfg.spec_fn().compile()
    d = mujoco.MjData(m)
    root = int(m.body_parentid[1] == 0 and 1 or 1)  # first body under the world
    for b in range(1, m.nbody):
        if m.body_parentid[b] == 0:
            root = b
            break
    jid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(m.njnt)}
    base_ids = [jid[n] for n in BASE]
    # wrist joints (shadow WRJ1/2) are held at 0 by the tasks, so they stay out of the probed finger poses
    fingers = [
        j
        for j in range(m.njnt)
        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) not in BASE
        and "WRJ" not in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "")
    ]

    def pose(kind):
        mujoco.mj_resetData(m, d)
        for j in fingers:
            a = m.jnt_qposadr[j]
            lo, hi = m.jnt_range[j]
            if m.jnt_limited[j] and kind == "mid":
                d.qpos[a] = 0.5 * (lo + hi)
            elif m.jnt_limited[j] and kind == "ext":
                d.qpos[a] = float(np.clip(0.0, lo, hi))
        mujoco.mj_forward(m, d)

    def body(n):
        for b in range(m.nbody):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
            if nm == n:
                return b
        raise KeyError(n)

    tips = [body(n) for n in spec.keypoint_bodies]
    th = THUMB_TIP[name]
    long_idx = [i for i in range(len(tips)) if i != th]
    pose("home")
    # Anchor frame = the frame the XML bodies are specified in (the mocap attachment frame), i.e. the compiled
    # world frame at q = 0; the root body's own XML quat is part of the model, not of the anchor.
    Rr, pr = np.eye(3), d.xpos[root].copy()

    def to_root(p):
        return Rr.T @ (p - pr)

    pose("ext")
    ext = np.array([to_root(d.xpos[t]) for t in tips])
    pose("mid")
    mid = np.array([to_root(d.xpos[t]) for t in tips])
    f = _unit(ext[long_idx].mean(0) - to_root(d.xpos[body(spec.palm_body)]))
    disp = mid[long_idx].mean(0) - ext[long_idx].mean(0)
    n = _unit(disp - (disp @ f) * f)
    up_d = -n  # palm facing down: normal -> world -z
    lat = _unit(np.cross(up_d, f))  # world +y
    M = np.stack([f, lat, up_d], axis=1)  # columns in root frame
    R_down_px = M.T  # root -> world: f->+x, lat->+y, -n->+z
    Rz = np.diag([-1.0, -1.0, 1.0])
    R_down_nx = Rz @ R_down_px
    up_u = n  # palm facing up
    lat_u = _unit(np.cross(up_u, f))
    R_up_px = np.stack([f, lat_u, up_u], axis=1).T
    pose("home")
    axes = {}
    for nme, j in zip(BASE, base_ids):
        a = Rr.T @ (d.xmat[m.jnt_bodyid[j]].reshape(3, 3) @ m.jnt_axis[j])
        comp = np.array([a @ f, a @ lat, a @ up_d])
        k = int(np.argmax(np.abs(comp)))
        axes[nme] = {
            "hand_frame": comp.round(3).tolist(),
            "axis": "fwd lat up".split()[k],
            "sign": int(np.sign(comp[k])),
            "err_deg": float(np.degrees(np.arccos(min(1.0, abs(comp[k]))))),
        }
    # Canonical base axes (the Allegro convention): base_x/roll -> -up, base_y/pitch -> lat, base_z/yaw -> fwd, hinge pivot at
    # the palm-body origin. axis_body = joint-body frame direction realising it; pos_body = pivot in the joint-body frame.
    want = {
        "base_x": -up_d,
        "base_y": lat,
        "base_z": f,
        "base_roll": -up_d,
        "base_pitch": lat,
        "base_yaw": f,
    }
    palm_o = d.xpos[body(spec.palm_body)].copy()
    fix = {}
    for nme, j in zip(BASE, base_ids):
        Rb, ob = d.xmat[m.jnt_bodyid[j]].reshape(3, 3), d.xpos[m.jnt_bodyid[j]]
        cur_axis = m.jnt_axis[j].copy()
        fix[nme] = {
            "axis_body": (Rb.T @ want[nme]).round(5).tolist(),
            "axis_xml": cur_axis.round(5).tolist(),
            "pos_body": (Rb.T @ (palm_o - ob)).round(5).tolist()
            if nme.startswith("base_r") or nme.endswith(("pitch", "yaw"))
            else [0.0, 0.0, 0.0],
            "pos_xml": m.jnt_pos[j].round(5).tolist(),
            "body": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.jnt_bodyid[j])),
        }
    thumb_lat = float(to_root(d.xpos[tips[th]]) @ lat - (to_root(d.xpos[tips[long_idx[0]]]) @ lat))
    out = {
        "root_body": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, root),
        "nbody": int(m.nbody),
        "njnt": int(m.njnt),
        "finger_dir_rootframe": f.round(4).tolist(),
        "normal_rootframe": n.round(4).tolist(),
        "f_n_orthogonal_err": float(abs(f @ n)),
        "tip_dist_from_palm_ext": [
            float(np.linalg.norm(t - to_root(d.xpos[body(spec.palm_body)]))) for t in ext
        ],
        "base_axes": axes,
        "base_joint_fix": fix,
        "palm_origin_attach": palm_o.round(5).tolist(),
        "anchor_down_px": _quat_from_mat(R_down_px).round(4).tolist(),
        "anchor_down_nx": _quat_from_mat(R_down_nx).round(4).tolist(),
        "anchor_up_px": _quat_from_mat(R_up_px).round(4).tolist(),
        "thumb_side_lat_down": float(np.sign(thumb_lat)),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    want = set(x for x in a.hands.split(",") if x)
    res = {}
    for name, side, spec in hands():
        if want and name not in want:
            continue
        try:
            res[f"{name}_{side}"] = probe(name, side, spec)
            r = res[f"{name}_{side}"]
            print(f"== {name}_{side} root={r['root_body']} f.n err={r['f_n_orthogonal_err']:.3f}")
            print(
                "   down_px",
                r["anchor_down_px"],
                "up_px",
                r["anchor_up_px"],
                "thumb lat sign",
                r["thumb_side_lat_down"],
            )
            for k, v in r["base_axes"].items():
                print(
                    f"   {k:10s} -> {v['axis']:3s} {v['sign']:+d} err {v['err_deg']:.1f} deg {v['hand_frame']}"
                )
        except Exception as e:  # report per hand, keep probing the others
            res[f"{name}_{side}"] = {"error": repr(e)}
            print(f"== {name}_{side} FAILED: {e!r}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
