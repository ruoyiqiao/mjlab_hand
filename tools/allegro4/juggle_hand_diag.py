"""Why does a hand juggle worse? Per-hand physical quantities of the Juggle2 / Juggle2 -Full envs (CPU).

Usage: python juggle_hand_diag.py [--hands allegro,leap,wuji,sharpa,xhand,shadow]

Per hand, in the play config (ball parked in the cup / grip, zero action = hold):
* root: compiled kp / kv / force limit of the vertical slide, effective translational mass (mj_crb, as cascade.py),
  total hand mass;
* palm tracking (Juggle2): a throw-like vertical move (+0.5 action on the world-z slide for 5 control steps = 0.1 m in
  50 ms), max |target - palm| and the lag at which the palm reaches half of the move;
* finger response (Juggle2 -Full): +1 finger action for 10 control steps; control steps until the fingers have moved
  25 % / 50 % of what the targets moved (the targets move from the first step) - measures actuator delay + PD speed;
* cup / grip geometry: ball rest height above the palm origin, the highest fingertip above the ball centre and the
  horizontal distance of the nearest fingertip that is above the ball centre (an overhang blocks a vertical throw).
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np
import torch


def build(task, num_envs=1):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.envs import mdp as envs_mdp
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(task, play=True)
    cfg.terminations = {"time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True)}
    cfg.scene.num_envs = num_envs
    return ManagerBasedRlEnv(cfg, device="cpu")


def root_params(env):
    m = env.sim.mj_model
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "hand/base_x")
    act = [
        a
        for a in range(m.nu)
        if int(m.actuator_trnid[a, 0]) == jid
        and int(m.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT)
    ][0]
    d = mujoco.MjData(m)
    d.qpos[:] = env.sim.data.qpos[0].numpy()
    mujoco.mj_forward(m, d)
    e = np.zeros(m.nv)
    e[m.jnt_dofadr[jid]] = 1.0
    r = np.zeros(m.nv)
    mujoco.mj_mulM(m, d, r, e)
    hand_bodies = [
        b
        for b in range(m.nbody)
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("hand/")
    ]
    return {
        "kp": float(m.actuator_gainprm[act, 0]),
        "kv": -float(m.actuator_biasprm[act, 2]),
        "fmax": float(m.actuator_forcerange[act, 1]),
        "m_eff": float(r[m.jnt_dofadr[jid]]),
        "mass": float(sum(m.body_mass[b] for b in hand_bodies)),
    }


def palm_tracking(env):
    t = env.command_manager.get_term("juggle")
    term = env.action_manager.get_term("hand")
    env.reset()
    dim = env.action_manager.total_action_dim
    z0 = float(t.palm_pos()[0, 2])
    tgt0 = float(term.base_target[0, 0])
    err, path, tgts = [], [], []
    for k in range(30):
        a = torch.zeros(1, dim)
        if k < 5:
            a[0, 0] = 0.5
        env.step(a)
        dz_t = float(term.base_target[0, 0]) - tgt0
        dz = float(t.palm_pos()[0, 2]) - z0
        tgts.append(dz_t)
        path.append(dz)
        err.append(abs(dz_t - dz))
    half = 0.5 * tgts[-1]
    k_t = next(i for i, v in enumerate(tgts) if v >= half)
    k_p = next((i for i, v in enumerate(path) if v >= half), None)
    dt = float(env.step_dt)
    return {
        "move_m": tgts[-1],
        "max_err_m": max(err),
        "half_lag_ms": None if k_p is None else (k_p - k_t) * dt * 1000,
        "overshoot_m": max(path) - tgts[-1],
    }


def finger_response(env):
    term = env.action_manager.get_term("hand")
    ent = env.scene["hand"]
    env.reset()
    dim = env.action_manager.total_action_dim
    ids = term.joint_ids_canonical[6:]
    q0 = ent.data.joint_pos[0, ids].clone()
    tq0 = term._finger_target[0].clone()
    mv, tm = [], []
    for _ in range(20):
        a = torch.zeros(1, dim)
        a[0, 6:] = 1.0 if len(mv) < 10 else 0.0
        env.step(a)
        mv.append(float((ent.data.joint_pos[0, ids] - q0).abs().mean()))
        tm.append(float((term._finger_target[0] - tq0).abs().mean()))
    dt = float(env.step_dt)
    out = {}
    for frac in (0.25, 0.5):
        k_t = next(i for i, v in enumerate(tm) if v >= frac * tm[9])
        k_p = next((i for i, v in enumerate(mv) if v >= frac * tm[9]), None)
        out[f"lag{int(frac * 100)}_ms"] = None if k_p is None else (k_p - k_t) * dt * 1000
    out["moved_frac_at_10"] = mv[9] / max(tm[9], 1e-9)
    return out


def geometry(env, prof):
    t = env.command_manager.get_term("juggle")
    env.reset()
    dim = env.action_manager.total_action_dim
    for _ in range(50):
        env.step(torch.zeros(1, dim))
    m = env.sim.mj_model
    d = mujoco.MjData(m)
    d.qpos[:] = env.sim.data.qpos[0].numpy()
    mujoco.mj_forward(m, d)
    ball = t.ball_pos()[0, 0].numpy()
    palm = t.palm_pos()[0].numpy()
    r = float(t.cfg.ball_radius)
    tips = []
    for b in prof.tip_bodies if prof else ("ff_tip", "mf_tip", "rf_tip", "th_tip"):
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"hand/{b}")
        g = [k for k in range(m.ngeom) if m.geom_bodyid[k] == i]
        tips.append(np.mean([d.geom_xpos[k] for k in g], axis=0) if g else d.xpos[i])
    tips = np.array(tips)
    above = tips[tips[:, 2] > ball[2]]
    hd = np.linalg.norm(above[:, :2] - ball[:2], axis=1) if len(above) else np.array([np.inf])
    return {
        "ball_above_palm_m": float(ball[2] - palm[2]),
        "top_tip_above_ball_m": float(tips[:, 2].max() - ball[2]),
        "overhang_min_xy_m": float(hd.min()),
        "overhang_tips": int(len(above)),
        "ball_r": r,
        "ball_rel_palm_xy": (ball[:2] - palm[:2]).round(3).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", default="allegro,leap,wuji,sharpa,xhand,shadow")
    a = ap.parse_args()
    from mjlab_hand.registration import register_tasks
    from mjlab_hand.tasks_common.hand_profile import get_profile
    from mjlab_hand.tasks_common.hand_tasks import HAND_SUFFIX

    register_tasks()
    for hand in a.hands.split(","):
        prof = None if hand == "allegro" else get_profile(hand)
        for variant, task in (
            ("J2", f"Juggle2-OneHand-{HAND_SUFFIX[hand]}"),
            ("J2-Full", f"Juggle2-OneHand-{HAND_SUFFIX[hand]}-Full"),
        ):
            env = build(task)
            row = {"root": root_params(env), "geom": geometry(env, prof)}
            row["palm"] = palm_tracking(env)
            if variant == "J2-Full":
                row["fingers"] = finger_response(env)
            env.close()
            print(
                f"== {hand} {variant}",
                {
                    k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                    for k, v in row.items()
                },
                flush=True,
            )


if __name__ == "__main__":
    main()
