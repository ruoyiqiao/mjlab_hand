"""Fingertip-support start state for PlateSpin (CPU), written to hand_params/plate.json.

Usage: python plate_probe.py --hands allegro,sharpa [--write]

Per hand, in the PlateSpin play env (palm-up anchor, zero action = hold the start pose):

1. finger chains: the finger joints on the path to each fingertip body; each joint is a flexion joint (lifts the tip
   when the palm faces up) or a lateral one (juggle_cup_probe.cup_pose rule, lateral joints stay at 0);
2. candidates ("pillar" poses): first flexion joint of each finger = a, of the thumb = a_t, the other flexion joints
   = b, over a grid; the plate is dropped 4 mm above the highest fingertip, centred over the fingertips that are
   within 25 mm of the highest one, and settles for 1.5 s; stage 2 repeats this for the 12 best grid poses with each
   finger's first flexion joint re-chosen so that all fingertips reach the same height (more fingers carry the plate);
3. a candidate is valid if only last phalanges touch the plate, at least three fingers carry it, the plate's centre of
   mass is inside their contact polygon (margin = inradius), the plate tilt is < MAX_TILT_DEG and it is at rest;
   the valid candidate with the largest margin (then the smallest tilt) is taken;
4. verification: the task's own reset with the written start state (fingers at their targets, plate at the settled
   pose + 1 mm), 3 s of zero action: no termination, no non-fingertip contact, plate drift < MAX_DRIFT. Candidates are
   tried in order of margin; the first that passes is written (a hand with none keeps its previous entry).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import juggle_cup_probe as jcp  # noqa: E402

PARAMS = Path(__file__).resolve().parents[2] / "src/mjlab_hand/tasks_common/hand_params/plate.json"
A_FINGER = (0.5, 0.7, 0.9, 1.1, 1.3, 1.5)
A_THUMB = (0.3, 0.6, 0.9, 1.2, 1.5)
B_REST = (0.0, 0.25, 0.45)
# joints held at 0 in every candidate: flexing Shadow's little-finger metacarpal (LFJ5, the first
# flexion joint of its chain) lifts the metacarpal itself into the plate
HELD = {"shadow": ("lh_LFJ5",)}
SETTLE_S, VERIFY_S = 1.5, 3.0
# the start state need not be level (the task ends an episode only beyond 60 deg plate tilt)
MAX_TILT_DEG, MAX_DRIFT = 20.0, 0.010


def build(task, n):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.envs import mdp as envs_mdp
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = n
    cfg.episode_length_s = 100.0
    cfg.terminations = {"time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True)}
    return ManagerBasedRlEnv(cfg, device="cpu")


def chains(m, tip_names, finger_joints):
    """Per fingertip body: its finger joints, root to tip."""
    out = []
    jset = set(finger_joints)
    for t in tip_names:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"hand/{t}")
        js = []
        while b > 0:
            for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
                n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j).removeprefix("hand/")
                if n in jset:
                    js.append(n)
            b = m.body_parentid[b]
        out.append(js[::-1])
    return out


def fingertip_tops(m, d, tip_sets):
    """Per finger: highest point (z) and its xy over the collision geoms of its last phalanx (AABB corners)."""
    tops = []
    for bodies in tip_sets:
        best = (-np.inf, None)
        for g in range(m.ngeom):
            if m.geom_bodyid[g] not in bodies or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
            R = d.geom_xmat[g].reshape(3, 3)
            for s in itertools.product((-1, 1), repeat=3):
                p = d.geom_xpos[g] + R @ (c + np.array(s) * h)
                if p[2] > best[0]:
                    best = (p[2], p[:2].copy())
        tops.append(best)
    return tops


def inradius(points, c):
    """Signed distance of c to the boundary of the convex hull of 2-D points (>0 inside); -inf if < 3 points."""
    if len(points) < 3:
        return -np.inf
    pts = np.unique(np.round(points, 4), axis=0)
    if len(pts) < 3:
        return -np.inf
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(ps):
        h = []
        for p in ps:
            while len(h) >= 2 and np.cross(h[-1] - h[-2], p - h[-2]) <= 0:
                h.pop()
            h.append(p)
        return h

    hull = half(pts)[:-1] + half(pts[::-1])[:-1]
    if len(hull) < 3:
        return -np.inf
    dmin = np.inf
    for a, b in zip(hull, hull[1:] + hull[:1]):
        e = b - a
        dmin = min(dmin, np.cross(e, c - a) / (np.linalg.norm(e) + 1e-12))
    return dmin


def plate_state(env, m, e, d):
    """Copy env e into d (free-joint origin removed) and mj_forward."""
    d.qpos[:] = env.sim.data.qpos[e].numpy()
    d.qvel[:] = env.sim.data.qvel[e].numpy()
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "plate/plate_joint")
    a = m.jnt_qposadr[j]
    d.qpos[a : a + 3] -= env.scene.env_origins[e].numpy()
    mujoco.mj_forward(m, d)
    return a


def contacts(m, d, plate_body, tip_of_body):
    """(hand bodies touching the plate, fingers carrying it, contact points xy of the fingertips)."""
    bodies, fingers, pts = set(), set(), []
    for c in range(d.ncon):
        con = d.contact[c]
        b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
        if plate_body not in (b1, b2):
            continue
        other = b2 if b1 == plate_body else b1
        bodies.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, other).removeprefix("hand/"))
        if other in tip_of_body:
            fingers.add(tip_of_body[other])
            pts.append(con.pos[:2].copy())
    return bodies, fingers, np.array(pts).reshape(-1, 2)


def setup(hand):
    """Hand facts shared by both stages: finger chains, flexion kinds, last-phalanx bodies per finger."""
    from mjlab_hand.palm_tasks import fingertip_bodies
    from mjlab_hand.tasks_common.hand_profile import get_profile
    from mjlab_hand.tasks_common.hand_tasks import HAND_SUFFIX

    prof = get_profile(hand)
    if hand == "allegro":
        tips, thumb, fj = ("ff_tip", "mf_tip", "rf_tip", "th_tip"), 3, list(prof.finger_joints)
    else:
        tips, thumb = tuple(prof.tip_bodies), prof.thumb_tip
        fj = [*prof.finger_joints, *prof.coupled_joints]
    c = SimpleNamespace(
        hand=hand, task=f"PlateSpin-{HAND_SUFFIX[hand]}", prof=prof, tips=tips, thumb=thumb, fj=fj
    )
    env = build(c.task, 1)
    m = env.sim.mj_model
    # cup_pose needs tip bodies of this model (Allegro's profile names the tip heads of its floating variant)
    shim = SimpleNamespace(
        finger_joints=prof.finger_joints,
        coupled_joints=prof.coupled_joints,
        tendon_targets=prof.tendon_targets,
        tip_bodies=tips,
    )
    _, c.kinds = jcp.cup_pose(m, shim, 0.5)
    c.chains = chains(m, tips, fj)
    c.allowed = set(fingertip_bodies(hand))
    c.tip_sets, c.tip_of_body = [], {}
    for k, t in enumerate(tips):  # each finger's last-phalanx bodies (tip body + allowed ancestors)
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"hand/{t}")
        s = {b}
        while (
            b > 0
            and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b).removeprefix("hand/") in c.allowed
        ):
            s.add(b)
            b = m.body_parentid[b]
        c.tip_sets.append(s)
        for x in s:
            c.tip_of_body[x] = k
    sp = env.command_manager.get_term("spin").sp
    c.under = sp.plate_thickness / 2 + sp.plate_foot_height
    c.m = m
    env.close()
    return c


def pillar(c, a_f, a_t, b):
    """First flexion joint of each finger = a_f (thumb a_t), the other flexion joints = b, lateral / HELD joints = 0."""
    pose = {}
    for k, js in enumerate(c.chains):
        first = True
        for j in js:
            kd = c.kinds.get(j, "lateral")
            if kd == "lateral" or j in HELD.get(c.hand, ()):
                pose[j] = 0.0
                continue
            amp = (a_t if k == c.thumb else a_f) if first else b
            first = False
            _, _, lo, hi = jcp.joint_info(c.m, j)
            pose[j] = float(np.clip(amp if kd == "flex+" else -amp, lo, hi))
    for j in c.fj:
        pose.setdefault(j, 0.0)
    return pose


def fk_tops(c, pose):
    d = mujoco.MjData(c.m)
    for j, v in pose.items():
        d.qpos[jcp.joint_info(c.m, j)[1]] = v
    for j, v in dict(c.prof.fixed_joints).items():
        d.qpos[jcp.joint_info(c.m, j)[1]] = v
    mujoco.mj_kinematics(c.m, d)
    return fingertip_tops(c.m, d, c.tip_sets)


def equalize(c, pose):
    """Every finger's first flexion joint re-chosen (FK scan) so its fingertip top reaches the highest one's height."""
    pose = dict(pose)
    target = max(z for z, _ in fk_tops(c, pose))
    for k, js in enumerate(c.chains):
        flex = [
            j
            for j in js
            if c.kinds.get(j, "lateral") != "lateral" and j not in HELD.get(c.hand, ())
        ]
        if not flex:
            continue
        j = flex[0]
        _, _, lo, hi = jcp.joint_info(c.m, j)
        best = (np.inf, pose[j])
        for v in np.linspace(max(lo, -2.0), min(hi, 2.0), 81):
            z = fk_tops(c, {**pose, j: float(v)})[k][0]
            best = min(best, (abs(z - target) + 0.002 * abs(v - pose[j]), float(v)))
        pose[j] = best[1]
    return pose


def settle_eval(c, poses, labels):
    """Drop the plate on each candidate pose (one env each), settle, and score the support."""
    n = len(poses)
    env = build(c.task, n)
    m = env.sim.mj_model
    d = mujoco.MjData(m)
    ent, term, plate = env.scene["hand"], env.action_manager.get_term("hand"), env.scene["plate"]
    spin = env.command_manager.get_term("spin")
    plate_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "plate/plate")
    pdof = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "plate/plate_joint")]
    env.reset()
    names = list(ent.joint_names)
    q = ent.data.joint_pos.clone()
    for e, pose in enumerate(poses):
        for j, v in pose.items():
            q[e, names.index(j)] = v
    for j, v in dict(c.prof.fixed_joints).items():
        q[:, names.index(j)] = v
    ent.write_joint_state_to_sim(q, torch.zeros_like(q))
    term.reset(None)
    tendon = {}
    for e in range(n):  # Shadow J0 tendons: lengths at the coupled joints' pose
        if c.prof.tendon_targets:
            dd = mujoco.MjData(m)
            dd.qpos[:] = env.sim.data.qpos[e].numpy()
            mujoco.mj_forward(m, dd)
            tendon[e] = {
                t: float(
                    dd.ten_length[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TENDON, f"hand/{t}")]
                )
                for t in c.prof.tendon_targets
            }
    pose0 = torch.zeros(n, 7)
    pose0[:, 3] = 1.0
    for e in range(n):
        plate_state(env, m, e, d)
        tops = fingertip_tops(m, d, c.tip_sets)
        zmax = max(z for z, _ in tops)
        xy = np.mean([p for z, p in tops if z > zmax - 0.025], axis=0)
        pose0[e, :3] = (
            torch.tensor([xy[0], xy[1], zmax + c.under + 0.004]) + env.scene.env_origins[e]
        )
    plate.write_root_link_pose_to_sim(pose0)
    plate.write_root_link_velocity_to_sim(torch.zeros(n, 6))
    zero = torch.zeros(n, env.action_manager.total_action_dim)
    for _ in range(int(SETTLE_S / env.step_dt)):
        env.step(zero)
    rows = []
    for e in range(n):
        a = plate_state(env, m, e, d)
        bodies, fingers, pts = contacts(m, d, plate_body, c.tip_of_body)
        com = d.xipos[plate_body][:2]
        tilt = float(np.degrees(np.arccos(np.clip(d.xmat[plate_body][8], -1, 1))))
        speed = float(np.linalg.norm(d.qvel[pdof : pdof + 3]))
        bad = sorted(bodies - c.allowed)
        drop = float(pose0[e, 2] - env.scene.env_origins[e, 2]) - d.qpos[a + 2]
        margin = inradius(pts, com) if not bad else -np.inf
        ok = (
            not bad
            and len(fingers) >= 3
            and margin > 0.005
            and tilt < MAX_TILT_DEG
            and speed < 0.02
            and drop < 0.03
        )
        palm_pos = spin.hand.data.body_link_pos_w[e, spin.palm_id] - env.scene.env_origins[e]
        rows.append(
            dict(
                label=labels[e],
                ok=ok,
                margin=float(margin),
                tilt=tilt,
                fingers=len(fingers),
                bad=bad,
                joints=poses[e],
                pose={**poses[e], **tendon.get(e, {})},
                plate_rel_palm=[float(x) for x in (d.qpos[a : a + 3] - palm_pos.numpy())],
                plate_quat=[float(x) for x in d.qpos[a + 3 : a + 7]],
            )
        )
        print(
            f"   {c.hand} {labels[e]}: fingers {len(fingers)} margin {margin * 1000:.1f} mm tilt {tilt:.1f} "
            f"speed {speed:.3f} bad {bad[:3]} -> {'OK' if ok else '-'}",
            flush=True,
        )
    env.close()
    return rows


def probe(hand, keep=6):
    """Valid start states of the hand, best first (at most ``keep``)."""
    c = setup(hand)
    grid = list(itertools.product(A_FINGER, A_THUMB, B_REST))
    rows = settle_eval(
        c, [pillar(c, *g) for g in grid], [f"a {g[0]} a_t {g[1]} b {g[2]}" for g in grid]
    )
    # stage 2: the most promising grid poses with every fingertip at the same height
    seeds = sorted(
        [r for r in rows if not r["bad"] and r["fingers"] >= 2], key=lambda r: -r["margin"]
    )[:12]
    if seeds:
        rows += settle_eval(
            c, [equalize(c, r["joints"]) for r in seeds], [r["label"] + " equal" for r in seeds]
        )
    good = sorted(
        (r for r in rows if r["ok"]),
        key=lambda r: (-round(r["margin"], 3) + 0.0005 * r["tilt"], -r["fingers"]),
    )
    if not good:
        print(f"!! {hand}: no fingertip-only support found")
    return [
        {
            "finger_pose": r["pose"],
            "plate_rel_palm": r["plate_rel_palm"],
            "plate_quat": r["plate_quat"],
            "candidate": r["label"],
            "support": {"fingers": r["fingers"], "margin_m": r["margin"], "tilt_deg": r["tilt"]},
        }
        for r in good[:keep]
    ]


def verify(hand):
    """The task's own reset (written start state), zero action: terminations, non-fingertip contact, drift."""
    from mjlab.envs import ManagerBasedRlEnv

    from mjlab_hand.palm_tasks import PalmTaskSpec, make_palm_task_env_cfg

    # built here, not from the registry: the registered cfg was made before plate.json was written
    cfg = make_palm_task_env_cfg(PalmTaskSpec(task="plate", hand=hand), play=True)
    cfg.scene.num_envs = 4
    env = ManagerBasedRlEnv(cfg, device="cpu")
    env.reset()
    t = env.command_manager.get_term("spin")
    p0 = (t.obj_pos()[:, 0] - t.rest_point()).clone()
    zero = torch.zeros(4, env.action_manager.total_action_dim)
    ends, nontip, drift, tips = 0, 0, 0.0, []
    for _ in range(int(VERIFY_S / env.step_dt)):
        _, _, term, _, _ = env.step(zero)
        ends += int(term.sum())
        nontip += int(t.nontip_contact().sum())
        drift = max(
            drift, float(torch.linalg.norm(t.obj_pos()[:, 0] - t.rest_point() - p0, dim=-1).max())
        )
        tips.append(float(t.tip_contacts().float().sum(-1).mean()))
    env.close()
    ok = ends == 0 and nontip == 0 and drift < MAX_DRIFT
    print(
        f"== verify {hand}: terminations {ends} non-fingertip contact steps {nontip} drift {drift * 1000:.1f} mm "
        f"fingertip bodies in contact {np.mean(tips):.1f} -> {'OK' if ok else 'FAIL'}"
    )
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", default="allegro,sharpa")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    from mjlab_hand.registration import register_tasks

    register_tasks()
    for hand in a.hands.split(","):
        cands = probe(hand)
        if not a.write:
            continue
        old = json.loads(PARAMS.read_text()).get(hand) if PARAMS.exists() else None
        # candidates in order of support margin; the first that also passes the task-reset check is kept
        for rec in cands:
            print(
                f"== {hand}: trying {rec['candidate']} (margin {rec['support']['margin_m'] * 1000:.1f} mm)"
            )
            write(hand, rec)
            if verify(hand):
                print("wrote", PARAMS)
                break
        else:
            write(
                hand, old
            )  # nothing verified: keep what was there (None = the hand has no start state)
            print(
                f"!! {hand}: no candidate passed the reset check; plate.json unchanged for this hand"
            )


def write(hand, rec):
    out = json.loads(PARAMS.read_text()) if PARAMS.exists() else {}
    if rec is None:
        out.pop(hand, None)
    else:
        out[hand] = rec
    PARAMS.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
