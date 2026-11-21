"""CPU probe of one palm-up Allegro hand with the paper ball (63.5 mm, 130 g): catch -> dwell -> release.

Uses the compiled model of the cascade task (one env). The right hand runs the designed prior
open loop (quintic dwell from the cushioned catch state to the release state, then the vacant segment);
the ball arrives 1 cm above the cup with the nominal incoming velocity. Reports, per finger strategy:
ball offset from the cup point while held, contact timeline, separation time and velocity vs the plan,
and the landing error at the scheduled catch time (other hand's catch point).

Usage: python cup_probe.py [--task Juggle3-Allegro-Cascade-Prior] [--out probe.json]
"""

from __future__ import annotations

import argparse
import json

import mujoco
import numpy as np


def quintic(p0, v0, a0, p1, v1, a1, T):
    A = np.array(
        [[T**3, T**4, T**5], [3 * T**2, 4 * T**3, 5 * T**4], [6 * T, 12 * T**2, 20 * T**3]]
    )
    c = np.zeros((3, 6))
    for k in range(3):
        b = np.array(
            [
                p1[k] - (p0[k] + v0[k] * T + a0[k] * T * T / 2),
                v1[k] - (v0[k] + a0[k] * T),
                a1[k] - a0[k],
            ]
        )
        c[k] = [p0[k], v0[k], a0[k] / 2, *np.linalg.solve(A, b)]
    return c


def qeval(c, s):
    p = sum(c[:, i] * s**i for i in range(6))
    v = sum(i * c[:, i] * s ** (i - 1) for i in range(1, 6))
    a = sum(i * (i - 1) * c[:, i] * s ** (i - 2) for i in range(2, 6))
    return p, v, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Juggle3-Allegro-Cascade-Prior")
    ap.add_argument("--out", default="/tmp/cup_probe.json")
    ap.add_argument("--kappas", type=float, nargs="+", default=[0.2, 0.35, 0.5])
    a = ap.parse_args()
    from mjlab_hand.registration import register_tasks

    register_tasks()
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    from mjlab_hand.juggling.cascade import G, get_juggler
    from mjlab_hand.juggling.juggle_env_cfg import CUP_POSE
    from mjlab_hand.tasks_common.actions import ALLEGRO_FINGER_JOINTS

    cfg = load_env_cfg(a.task, play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg, device="cpu")
    t = get_juggler(env)
    sp = t.sp
    m = env.sim.mj_model
    h = 0
    cup_home = t.cup_home[h].numpy()
    p_rel = t.p_rel[h].numpy()
    p_catch = t.p_catch_nom[h].numpy()
    u = t.p_catch_nom[1].numpy()
    v_in = t.incoming_velocity(h).numpy()
    v_rel = t.throw_velocity(t.p_catch_nom[1].view(1, 3).clone(), h)[0].numpy()
    jinv = t.slide_jinv[h].numpy()
    J = t.slide_j[h].numpy()
    mass = float(t.mass[h])
    damp = t.slide_damping
    kp, kv = sp.slide_gains[0], sp.slide_gains[1]
    T = sp.beat_s
    acc_rel = np.array([0.0, 0.0, -G])

    def jadr(name):
        return m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]

    def dadr(name):
        return m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]

    act = {}
    for i in range(m.nu):
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0])
        act[jn] = i
    slides = [f"hand_r/{j}" for j in ("base_x", "base_y", "base_z")]
    fingers = [f"hand_r/{j}" for j in ALLEGRO_FINGER_JOINTS]
    ball = (
        "ball_0/ball_joint"
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_0/ball_joint") >= 0
        else None
    )
    if ball is None:
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
        ball = [nm for nm in names if nm and nm.startswith("ball_0")][0]
    bq, bd = jadr(ball), dadr(ball)
    palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand_r/palm")
    ballb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball_0/ball")
    cup_off = t.cup_offset[h].numpy()
    hand_geoms = set(
        i
        for i in range(m.ngeom)
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i]) or "").startswith(
            "hand_r/"
        )
    )
    ball_geom = [i for i in range(m.ngeom) if m.geom_bodyid[i] == ballb][0]

    def pose(kind):
        p = dict(CUP_POSE)
        if kind == "shallow":
            for f in ("ff", "mf", "rf"):
                p[f + "j1"], p[f + "j2"], p[f + "j3"] = 0.2, 0.35, 0.3
        if kind == "open":
            for f in ("ff", "mf", "rf"):
                p[f + "j1"], p[f + "j2"], p[f + "j3"] = 0.0, 0.1, 0.1
        if kind == "grip":
            for f in ("ff", "mf", "rf"):
                p[f + "j1"], p[f + "j2"], p[f + "j3"] = 0.6, 1.0, 0.8
            p["thj0"], p["thj1"], p["thj2"], p["thj3"] = 1.3, 0.5, 0.8, 0.8
        if kind == "fip":  # paper FIP: index and pinky (here ff and rf) open, thumb closed
            for f in ("ff", "rf"):
                p[f + "j1"], p[f + "j2"], p[f + "j3"] = 0.0, 0.1, 0.1
        return np.array([p[j] for j in ALLEGRO_FINGER_JOINTS])

    strategies = {
        "cup": ("cup", "cup", None),
        "cup_open@-0.06": ("cup", "open", 0.06),
        "cup_open@-0.03": ("cup", "open", 0.03),
        "grip_open@-0.06": ("grip", "open", 0.06),
        "grip_open@-0.03": ("grip", "open", 0.03),
        "grip_fip@-0.03": ("grip", "fip", 0.03),
        "shallow_open@-0.03": ("shallow", "open", 0.03),
    }
    res = {}
    grid = [(k, nm, st) for k in a.kappas for nm, st in strategies.items()]
    for kap, name0, (hold, rel, lead) in grid:
        name = f"k{kap}_{name0}"
        c_dw = quintic(p_catch, kap * v_in, np.zeros(3), p_rel, v_rel, acc_rel, T)
        c_vac = quintic(p_rel, v_rel, acc_rel, p_catch, kap * v_in, np.zeros(3), T)
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        for k, nm in enumerate(slides):
            d.qpos[jadr(nm)] = (jinv @ (p_catch - cup_home))[k]
            d.qvel[dadr(nm)] = (jinv @ (kap * v_in))[k]
        for k, nm in enumerate(fingers):
            d.qpos[jadr(nm)] = pose(hold)[k]
        # other balls far away
        for b in ("ball_1", "ball_2"):
            nm = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
            jb = [x for x in nm if x and x.startswith(b)][0]
            d.qpos[jadr(jb) : jadr(jb) + 3] = [3.0, 3.0 + (b == "ball_2"), 0.05]
        d.qpos[bq : bq + 3] = p_catch + np.array([0, 0, 0.01])
        d.qpos[bq + 3 : bq + 7] = [1, 0, 0, 0]
        d.qvel[bd : bd + 3] = v_in
        mujoco.mj_forward(m, d)
        steps = int(round(2 * T / m.opt.timestep))
        log = []
        last_contact = -1.0
        for i in range(steps):
            tt = (i + 1) * m.opt.timestep
            if tt <= T:
                pr, vr, ar = qeval(c_dw, tt)
            else:
                pr, vr, ar = qeval(c_vac, tt - T)
            q_ref = jinv @ (pr - cup_home)
            qd_ref = jinv @ vr
            f_j = J.T @ (mass * (ar + np.array([0, 0, G]))) + damp * qd_ref
            ctrl = q_ref + (kv * qd_ref + f_j) / kp
            for k, nm in enumerate(slides):
                d.ctrl[act[nm]] = ctrl[k]
            if lead is not None and T - lead <= tt < T + 0.15:
                fp = pose(rel)
            elif tt < 0.06 and hold == "grip":  # close gradually after the catch
                w = tt / 0.06
                fp = (1 - w) * pose("cup") + w * pose("grip")
            elif tt >= T + 0.15:  # vacant: back to the catching cup
                fp = pose("cup")
            else:
                fp = pose(hold)
            for k, nm in enumerate(fingers):
                d.ctrl[act[nm]] = fp[k]
            mujoco.mj_step(m, d)
            contact = any(
                (
                    (c.geom1 == ball_geom and c.geom2 in hand_geoms)
                    or (c.geom2 == ball_geom and c.geom1 in hand_geoms)
                )
                for c in d.contact[: d.ncon]
            )
            if contact:
                last_contact = tt
            cup = d.xpos[palm] + cup_off
            log.append((tt, contact, *(d.xpos[ballb] - cup), *d.qvel[bd : bd + 3], *(cup - pr)))
        L = np.array(log, dtype=float)
        held = L[(L[:, 0] > 0.03) & (L[:, 0] < T - 0.05)]
        sep = last_contact
        k_after = int(np.searchsorted(L[:, 0], sep + 0.02)) if sep > 0 else 0
        out = {"separation_t": round(float(sep), 3), "planned_release_t": T}
        if 0 < k_after < len(L):
            tt = L[k_after, 0]
            # landing prediction from the state at tt (recompute from log: ball pos = cup + rel; use velocity)
            vb = L[k_after, 5:8]
            # ball world position at tt
            pr_tt = qeval(c_vac, tt - T)[0] if tt > T else qeval(c_dw, tt)[0]
            pbw = pr_tt + L[k_after, 8:11] + L[k_after, 2:5]
            dt_land = T + sp.flight_s - tt
            land = pbw + vb * dt_land + 0.5 * np.array([0, 0, -G]) * dt_land**2
            out.update(
                {
                    "ball_vel_after_sep": [round(x, 3) for x in vb],
                    "planned_v_rel": [round(x, 3) for x in v_rel],
                    "landing_err_xyz": [round(x, 3) for x in (land - u)],
                    "landing_err": round(float(np.linalg.norm(land - u)), 3),
                }
            )
        out.update(
            {
                "held_offset_mean_xyz": [round(x, 4) for x in held[:, 2:5].mean(0)]
                if len(held)
                else None,
                "held_offset_max_norm": round(float(np.linalg.norm(held[:, 2:5], axis=1).max()), 4)
                if len(held)
                else None,
                "contact_frac_dwell": round(float(L[(L[:, 0] < T), 1].mean()), 3),
                "track_err_max": round(float(np.linalg.norm(L[:, 8:11], axis=1).max()), 4),
                "ball_min_rel_z_end": round(float(L[-1, 4]), 3),
            }
        )
        out = json.loads(json.dumps(out, default=float))
        res[name] = out
        print("PROBE", name, json.dumps(out))
    with open(a.out, "w") as fh:
        json.dump({"kappas": a.kappas, "results": res}, fh, indent=1)


if __name__ == "__main__":
    main()
