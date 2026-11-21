"""Juggling cup pose and cup point of a palm-up floating hand (CPU), written to hand_params/juggle.json.

Usage: python juggle_cup_probe.py [--hands leap,wuji,sharpa,xhand,shadow] [--amp 0.5] [--render DIR] [--write]
Amplitude rule: the smallest of {0.5, 0.8} rad whose well has a basin >= 5 drop points and passes the hold check (2026-11-17: 0.5 for wuji, sharpa, xhand, shadow; 0.8 for leap, basin 1 at 0.5).

Allegro's Juggle1/2 hold the fingers in a fixed "cup" (juggle_env_cfg.CUP_POSE: flexion 0.35 / 0.7 / 0.6 rad, thumb
opposed) and place ball A on the palm_collision box face. Other hands have no such box, so both are measured in the
real Juggle1 env of the hand (palm up, canonical base, zero base action = hold):
  cup pose   per actuated finger joint (and per Shadow tendon via its coupled joints), the central-difference
             displacement of the fingertips: a joint whose tips rise (+z, toward the palm-up side) is a flexion joint
             and goes to +-amp rad in the rising direction (clipped to its range); a joint that mainly moves the tips
             sideways (|dz| < 0.3 |d|) is an abduction / rotation joint and stays at its in-range zero.
  cup point  ball A is released 6 cm above the centroid of the palm-body geoms and simulated for 2 s; its rest
             position minus the palm origin is ``cup_rel_palm``. The env is rebuilt with that point and ball A must
             stay within 5 mm over 2 s (verification). For Allegro both are also computed and printed next to the
             hand-designed values for comparison (Allegro's configs are not changed).
Keyframes (front / side) are rendered to --render for visual inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

PARAMS = Path(__file__).resolve().parents[2] / "src/mjlab_hand/tasks_common/hand_params/juggle.json"


def build(hand, cup_pose=None, cup_rel=None, num_envs=1, overrides=None):
    """One palm-up hand with ball A (Juggle1 play config); ``overrides`` sets JuggleSpec fields (ball, side)."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.envs import mdp as envs_mdp
    from mjlab.managers.termination_manager import TerminationTermCfg

    from mjlab_hand.juggling.config.allegro import juggle_rl_spec
    from mjlab_hand.juggling.juggle_env_cfg import CUP_POSE, make_juggle_env_cfg

    js = juggle_rl_spec(1)
    js.hand, js.hold_limit = hand, False
    for k, v in (overrides or {}).items():
        setattr(js, k, v)
    if hand == "allegro":
        js.cup_pose = cup_pose or dict(CUP_POSE)
    else:
        from mjlab_hand.tasks_common.hand_profile import get_profile

        prof = get_profile(hand, js.side)
        js.cup_pose = cup_pose or {
            j: 0.0 for j in (*prof.finger_joints, *prof.coupled_joints, *prof.tendon_targets)
        }
        js.cup_rel_palm = tuple(cup_rel) if cup_rel is not None else (0.0, 0.0, 0.1)
    cfg = make_juggle_env_cfg(js, play=True)
    cfg.terminations = {"time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True)}
    cfg.scene.num_envs = num_envs
    return ManagerBasedRlEnv(cfg, device="cpu")


def joint_info(m, name):
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"hand/{name}")
    assert j >= 0, name
    lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-np.inf, np.inf)
    return j, m.jnt_qposadr[j], float(lo), float(hi)


def tips_xyz(m, d, prof):
    """Per tip body: mean of its geom centres (some hands put the tip body origin on the distal joint axis)."""
    out = []
    for b in prof.tip_bodies:
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"hand/{b}")
        g = [k for k in range(m.ngeom) if m.geom_bodyid[k] == i]
        out.append(np.mean([d.geom_xpos[k] for k in g], axis=0) if g else d.xpos[i])
    return np.stack(out)


def cup_pose(m, prof, amp):
    """Cup pose from the fingertip displacement of each joint (see module doc)."""
    d = mujoco.MjData(m)
    q0 = m.qpos0.copy()
    all_j = list(prof.finger_joints) + list(prof.coupled_joints)
    for n in all_j:
        _, a, lo, hi = joint_info(m, n)
        q0[a] = np.clip(0.0, lo, hi)
    pose, kind = {}, {}
    for n in all_j:
        _, a, lo, hi = joint_info(m, n)
        h = 0.02
        qa, qb = q0.copy(), q0.copy()
        qa[a], qb[a] = np.clip(q0[a] - h, lo, hi), np.clip(q0[a] + h, lo, hi)
        d.qpos[:] = qa
        mujoco.mj_kinematics(m, d)
        ta = tips_xyz(m, d, prof)
        d.qpos[:] = qb
        mujoco.mj_kinematics(m, d)
        tb = tips_xyz(m, d, prof)
        dq = qb[a] - qa[a]
        dt = (tb - ta) / max(dq, 1e-9)
        dz, dn = dt[:, 2].sum(), np.linalg.norm(dt)
        if dn < 1e-6 or abs(dz) < 0.3 * dn:
            pose[n], kind[n] = float(q0[a]), "lateral"
        else:
            pose[n], kind[n] = (
                float(np.clip(np.sign(dz) * amp, lo, hi)),
                "flex+" if dz > 0 else "flex-",
            )
    for t in prof.tendon_targets:  # Shadow J0 tendon: length at the coupled joints' cup values
        tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TENDON, f"hand/{t}")
        dd = mujoco.MjData(m)
        dd.qpos[:] = q0
        for n in prof.coupled_joints:
            dd.qpos[joint_info(m, n)[1]] = pose[n]
        mujoco.mj_forward(m, dd)
        pose[t] = float(dd.ten_length[tid])
        kind[t] = "tendon"
    return pose, kind


def settle(env, steps):
    """Zero base action for `steps` control steps; returns ball A path rel. palm and final speed."""
    t = env.command_manager.get_term("juggle")
    env.reset()
    act = torch.zeros(1, env.action_manager.total_action_dim)
    rel = []
    for _ in range(steps):
        env.step(act)
        rel.append((t.ball_pos()[0, 0] - t.palm_pos()[0]).numpy().copy())
    return np.array(rel), float(torch.linalg.norm(t.ball_vel()[0, 0]))


def ball_well(hand, pose, grid=0.02, steps=200, overrides=None):
    """Drop ball A from a grid of points above the hand, one env per point, in the real env (true actuation, base held
    at home by zero action); return the rest point (rel. palm origin) with the largest basin: the number of drop points
    whose ball comes to rest within 1 cm of it, above the palm, at speed < 2 cm/s."""
    xs = np.arange(-0.12, 0.16 + 1e-9, grid)
    ys = np.arange(-0.12, 0.12 + 1e-9, grid)
    pts = np.array([(x, y) for x in xs for y in ys])
    env = build(hand, cup_pose=pose, num_envs=len(pts), overrides=overrides)
    r = float((overrides or {}).get("ball_radius", 0.025))
    t = env.command_manager.get_term("juggle")
    env.reset()
    act = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
    for _ in range(30):  # let the hand settle with the balls parked
        env.step(act)
    m = env.sim.mj_model
    hand_g = [
        k
        for k in range(m.ngeom)
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, k) or "").startswith("hand/")
        and m.geom_contype[k] | m.geom_conaffinity[k]
    ]
    top = (
        float(env.sim.data.geom_xpos[0, hand_g, 2].max()) + r + 0.01
    )  # 0.035 for the Juggle1/2 ball
    palm = t.palm_pos()
    q = torch.zeros(env.num_envs, 4)
    q[:, 0] = 1
    pa = palm.clone()
    pa[:, :2] += torch.tensor(pts, dtype=torch.float32)
    pa[:, 2] = top
    pb = palm.clone()
    pb[:, 0] += 3.0
    pb[:, 2] = 0.03
    ids = torch.arange(env.num_envs)
    t.balls[0].write_root_link_pose_to_sim(torch.cat([pa, q], -1), env_ids=ids)
    t.balls[0].write_root_link_velocity_to_sim(torch.zeros(env.num_envs, 6), env_ids=ids)
    t.balls[1].write_root_link_pose_to_sim(torch.cat([pb, q], -1), env_ids=ids)
    for _ in range(steps):
        env.step(act)
    P = (t.ball_pos()[:, 0] - t.palm_pos()).numpy()
    V = torch.linalg.norm(t.ball_vel()[:, 0], dim=-1).numpy()
    env.close()
    keep = (V < 0.02) & (P[:, 2] > -0.03) & (np.linalg.norm(P[:, :2], axis=1) < 0.25)
    R = P[keep]
    if len(R) == 0:
        return None, 0, len(pts)
    basin = [(np.linalg.norm(R - r, axis=1) < 0.01).sum() for r in R]
    k = int(np.argmax(basin))
    return R[np.linalg.norm(R - R[k], axis=1) < 0.01].mean(0), int(basin[k]), len(pts)


def palm_centroid_rel(m, d, prof):
    pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"hand/{prof.palm_body}")
    g = [
        i
        for i in range(m.ngeom)
        if m.geom_bodyid[i] == pid and m.geom_contype[i] | m.geom_conaffinity[i]
    ]
    g = g or [i for i in range(m.ngeom) if m.geom_bodyid[i] == pid]
    return np.mean([d.geom_xpos[i] for i in g], axis=0) - d.xpos[pid]


def render(env, path):
    m = env.sim.mj_model
    d = mujoco.MjData(m)
    d.qpos[:] = env.sim.data.qpos[0].numpy()
    mujoco.mj_forward(m, d)
    t = env.command_manager.get_term("juggle")
    look = t.palm_pos()[0].numpy()
    r = mujoco.Renderer(m, 360, 480)
    imgs = []
    for az, el in ((180, -20), (90, -20), (135, -60)):
        cam = mujoco.MjvCamera()
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = look, 0.45, az, el
        r.update_scene(d, cam)
        imgs.append(r.render())
    import imageio.v2 as imageio

    imageio.imwrite(path, np.concatenate(imgs, axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", default="allegro,leap,wuji,sharpa,xhand,shadow")
    ap.add_argument("--amp", type=float, default=0.5)
    ap.add_argument("--render", default="")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    from mjlab_hand.registration import register_tasks
    from mjlab_hand.tasks_common.hand_profile import get_profile

    register_tasks()
    out = json.loads(PARAMS.read_text()) if PARAMS.exists() else {}
    for hand in a.hands.split(","):
        prof = get_profile(hand)
        env = build(hand)
        m = env.sim.mj_model
        pose, kind = cup_pose(m, prof, a.amp)
        env.close()
        if hand == "allegro":
            from mjlab_hand.juggling.juggle_env_cfg import CUP_POSE

            print(
                "== allegro probe pose vs CUP_POSE:",
                {k: (round(v, 2), CUP_POSE[k], kind[k]) for k, v in pose.items()},
            )
        start, basin, ntot = ball_well(hand, pose)
        print(f"   {hand}: well {None if start is None else start.round(4)} basin {basin}/{ntot}")
        if start is None:
            continue
        if hand == "allegro":
            env = build(hand, cup_pose=pose)
            t = env.command_manager.get_term("juggle")
            print(
                "   allegro analytic cup_rel_palm",
                (t.cup_local - t.palm_home_local).numpy().round(4),
            )
            env.close()
        env = build(hand, cup_pose=pose, cup_rel=start)
        path, v = settle(env, 200)
        rest = path[-1]
        moved = np.linalg.norm(path[-1][:2] - start[:2])
        env.close()
        env = build(hand, cup_pose=pose, cup_rel=rest)
        path2, v2 = settle(env, 200)
        drift = float(np.max(np.linalg.norm(path2 - rest, axis=1)))
        ok = v < 0.02 and v2 < 0.02 and drift < 0.01 and rest[2] > -0.02 and basin >= 5
        if a.render:
            Path(a.render).mkdir(parents=True, exist_ok=True)
            render(env, f"{a.render}/{hand}_cup.png")
        env.close()
        print(
            f"== {hand}: start {start.round(4)} rest {rest.round(4)} (xy moved {moved * 100:.1f} cm, v {v:.3f}) "
            f"verify drift {drift * 1000:.1f} mm v {v2:.3f} -> {'OK' if ok else 'FAIL'}"
        )
        print("   pose", {k: (round(val, 3), kind[k]) for k, val in pose.items()})
        if hand != "allegro" and ok:
            out[hand] = {
                "cup_pose": pose,
                "cup_rel_palm": [float(x) for x in rest],
                "kinds": kind,
                "amp": a.amp,
                "verify_drift_m": drift,
                "basin": [basin, ntot],
            }
    if a.write:
        PARAMS.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
        print("wrote", PARAMS)


if __name__ == "__main__":
    main()
