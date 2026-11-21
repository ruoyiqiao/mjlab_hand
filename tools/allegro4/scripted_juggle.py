"""J1.0 scripted feasibility for Allegro toss juggling (task action space only).

Vertical palm-cup profile per half cycle H = V + D (vacant V, dwell D), flight
F = V + H (Shannon, 1 hand, 2 balls), release speed v0 = g F / 2:
  seg1 (hand empty, tau in [0, V)):  z = v0 tau - (v0 / V) tau^2   (decel 2 v0 / V > g -> release)
  seg2 (dwell,     u in [0, D)):     z = -v0 u + (v0 / D) u^2      (catch at -v0, cushion, throw at +v0)
Lateral: during seg1 the hand moves to the other column; target = predicted
landing point of the incoming ball at cup height (closed-loop catching), cubic
blend with zero end velocity. Start: ball A in the palm, ball B dropped from rest
so it reaches the cup height of column B when the first catch is scheduled
(paper: ball launcher); initial stroke from rest over ``t0``.
Mode ``one``: single ball tossed in place (period F + D, F = 2 V + D).

Base slide targets are commanded through the bounded action (a = d_target /
(v_max dt)), fingers hold a cup pose. Writes per-config results JSON.

Usage: python scripted_juggle.py <out.json> [--mode two|one] [--device cuda:0]
         [--dwell 0.15,0.2,0.25] [--half 0.3] [--reps 8]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math

import torch

G = 9.81


def hermite(p0, m0, p1, m1, T, u):
    s = u / T
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    return h00 * p0 + h10 * T * m0 + h01 * p1 + h11 * T * m1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--mode", default="two")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dwell", default="0.15,0.2,0.25")
    ap.add_argument("--half", default="0.3")
    ap.add_argument("--kp", default="2000")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--t0", type=float, default=0.25)
    ap.add_argument("--lead", default="0.0")
    ap.add_argument(
        "--lat-delay", default="0.0", help="fraction of V before the empty-hand lateral move starts"
    )
    ap.add_argument("--cup", default="deep", help="deep | shallow finger cup pose")
    ap.add_argument("--col", type=float, default=0.12)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--video-env", type=int, default=-1)
    ap.add_argument("--strip", default="")
    ap.add_argument("--trace", default="")
    ap.add_argument("--strip-seconds", type=float, default=2.0)
    ap.add_argument(
        "--hand",
        default="allegro",
        help="non-Allegro: measured fixed cup (hand_params/juggle.json)",
    )
    a = ap.parse_args()

    from mjlab.envs import ManagerBasedRlEnv

    import mjlab_hand  # noqa: F401
    from mjlab_hand.juggling.juggle_env_cfg import CUP_POSE, JuggleSpec, make_juggle_env_cfg
    from mjlab_hand.tasks_common.actions import ALLEGRO_FINGER_JOINTS

    grid = list(
        itertools.product(
            [float(x) for x in a.half.split(",")],
            [float(x) for x in a.dwell.split(",")],
            [float(x) for x in a.kp.split(",")],
            [float(x) for x in a.lead.split(",")],
            [float(x) for x in a.lat_delay.split(",")],
        )
    )
    results = []
    for H, D, kp, lead, ld in grid:
        if D >= H:
            continue
        V = H - D
        F = V + H if a.mode == "two" else 2 * V + D
        v0 = G * F / 2
        apex = v0**2 / (2 * G)
        vz_b = 0.0
        if a.mode == "two":
            # Reference-state start: ball B on the steady parabola so it reaches column B's
            # cup height at speed v0 exactly at the first scheduled catch t0 + V, with a
            # normal opening stroke t0. tau = time before its apex at t = 0.
            t0 = a.t0
            tau = (t0 + V) - F / 2
            drop_h = apex - 0.5 * G * tau**2
            vz_b = G * tau
        else:
            t0 = a.t0
            drop_h = 0.3
        js = JuggleSpec(
            num_balls=2 if a.mode == "two" else 1,
            half_cycle_s=H,
            column_offset=(0.0, a.col, 0.0),
            lin_vel_max=8.0,
            second_ball_drop_height=drop_h,
            second_ball_init_vz=vz_b,
            base_gains=(kp, 2.0 * math.sqrt(kp * 0.75) * 1.2, 400.0),
            episode_s=a.seconds + 0.5,
            hand=a.hand,
            finger_mode="absolute" if a.hand == "allegro" else "fixed",
        )
        cfg = make_juggle_env_cfg(js, play=True)
        # No terminations: drops are detected here (an auto-reset would clear the tracker).
        cfg.terminations = {}
        n = a.reps
        cfg.scene.num_envs = n
        env = ManagerBasedRlEnv(cfg, device=a.device)
        env.reset()
        dev = env.device
        t = env.command_manager.get_term("juggle")
        act = env.action_manager.get_term("hand")
        rate, dt = act._base_rate[:3], env.step_dt
        t.update()
        palm_id = t.palm_id
        R0 = t.hand.data.body_link_quat_w[:, palm_id]
        from mjlab.utils.lab_api.math import matrix_from_quat

        R = matrix_from_quat(R0)  # (n,3,3); slide axes = palm frame axes at rest
        if (
            a.hand != "allegro"
        ):  # canonical base axes are not the palm-body axes: read the slide axes in world
            import mujoco

            m_, d_ = env.sim.mj_model, mujoco.MjData(env.sim.mj_model)
            d_.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
            mujoco.mj_kinematics(m_, d_)
            cols = [
                d_.xaxis[mujoco.mj_name2id(m_, mujoco.mjtObj.mjOBJ_JOINT, f"hand/{b}")]
                for b in ("base_x", "base_y", "base_z")
            ]
            R = torch.tensor(cols, dtype=torch.float32, device=dev).T.expand(n, 3, 3).clone()
        origin = env.scene.env_origins
        cup0 = origin + t.col_local[0]
        colB = origin + t.col_local[1]
        q0 = act.base_target[:, :3].clone()
        pose = dict(CUP_POSE)
        if a.cup == "shallow":
            for f in ("ff", "mf", "rf"):
                pose[f"{f}j1"], pose[f"{f}j2"], pose[f"{f}j3"] = 0.15, 0.35, 0.3
        if a.hand == "allegro":
            fing = torch.tensor(
                [
                    (pose[j] - float(act._f_offset[i])) / float(act._f_scale[i])
                    for i, j in enumerate(ALLEGRO_FINGER_JOINTS)
                ],
                device=dev,
            ).clamp(-1, 1)
        else:
            fing = torch.zeros(0, device=dev)

        def to_action(cup_des):
            d = cup_des - cup0
            q_des = q0 + torch.einsum("nij,ni->nj", R, d)  # R^T d
            ab = torch.clamp((q_des - act.base_target[:, :3]) / (rate * dt), -1.0, 1.0)
            return torch.cat([ab, torch.zeros(n, 3, device=dev), fing.expand(n, -1)], -1)

        def safe_target(k, col_xy):
            """Column xy, nudged toward the predicted landing only while ball k is free
            and falling (never while it is in or next to the cup), clamped to +-4 cm."""
            pl = predicted_landing(k)
            ok = (
                torch.isfinite(pl).all(-1)
                & ~t.contact[:, k]
                & (t.ball_vel()[:, k, 2] < 0)
                & (t.free_t[:, k] > 0.05)
            )
            d = torch.clamp(torch.where(torch.isfinite(pl), pl, col_xy) - col_xy, -0.04, 0.04)
            return torch.where(ok.unsqueeze(-1), col_xy + d, col_xy)

        def predicted_landing(k):
            p = t.ball_pos()[:, k]
            v = t.ball_vel()[:, k]
            z_c = cup0[:, 2]
            dz = p[:, 2] - z_c
            disc = (v[:, 2] ** 2 + 2 * G * dz).clamp(min=0)
            tt = (v[:, 2] + torch.sqrt(disc)) / G
            return p[:, :2] + v[:, :2] * tt.unsqueeze(-1)

        steps = int(a.seconds / dt)
        juggle_until = torch.full((n,), float("nan"), device=dev)
        alive = torch.ones(n, dtype=torch.bool, device=dev)
        lateral_from = cup0[:, :2].clone()
        lateral_to = cup0[:, :2].clone()
        cur_col = 0
        frames = []
        for s in range(steps):
            time = s * dt + lead
            if time < t0:  # initial stroke from rest
                u = time
                bb = v0 / t0**2
                z = -bb * t0 * u**2 + bb * u**3
                xy = cup0[:, :2]
            else:
                tau = time - t0
                k = int(tau // H)
                ph = tau - k * H
                if a.mode == "two":
                    thrower = k % 2  # column of the k-th throw
                    catch_col = 1 - thrower
                else:
                    catch_col = 0
                Vv = V if a.mode == "two" else F
                if ph < Vv and a.mode == "two":
                    z = v0 * ph - (v0 / V) * ph**2
                    if ph < dt * 0.5 or k != cur_col:
                        lateral_from = (cup0 if thrower == 0 else colB)[:, :2].clone()
                        cur_col = k
                    col_xy = (cup0 if catch_col == 0 else colB)[:, :2]
                    # the ball landing in catch_col is the one thrown from it one half cycle earlier
                    incoming = (k + 1) % 2 if t.num_active > 1 else 0
                    tgt = safe_target(incoming, col_xy)
                    sgm = min(max((ph / V - ld) / max(1.0 - ld, 1e-6), 0.0), 1.0)
                    w = 3 * sgm**2 - 2 * sgm**3
                    xy = (1 - w) * lateral_from + w * tgt
                    lateral_to = tgt
                elif a.mode == "one":
                    per = F + D
                    ph1 = tau % per
                    if ph1 < F:
                        # Hermite: (0, v0) -> (-0.04, 0) at F/2 -> (0, -v0) at F; stays under the ball.
                        if ph1 < F / 2:
                            z = hermite(0.0, v0, -0.04, 0.0, F / 2, ph1)
                        else:
                            z = hermite(-0.04, 0.0, 0.0, -v0, F / 2, ph1 - F / 2)
                        xy = safe_target(0, cup0[:, :2])
                    else:
                        u = ph1 - F
                        z = -v0 * u + (v0 / D) * u**2
                        xy = lateral_to
                    lateral_to = xy
                else:
                    u = ph - V
                    z = -v0 * u + (v0 / D) * u**2
                    xy = lateral_to
            cup_des = torch.cat([xy, (cup0[:, 2] + z).unsqueeze(-1)], -1)
            env.step(to_action(cup_des))
            t.update()
            if a.trace and time - lead < 1.2:
                tr = getattr(a, "_tr", [])
                tr.append(
                    {
                        "t": round(time, 3),
                        "z_cmd": round(float(z), 4),
                        "palm": [round(x, 4) for x in (t.palm_pos()[0] - origin[0]).tolist()],
                        "ball": [round(x, 4) for x in (t.ball_pos()[0, 0] - origin[0]).tolist()],
                        "ballB": [round(x, 4) for x in (t.ball_pos()[0, -1] - origin[0]).tolist()],
                        "cmd_xy": [round(x, 4) for x in (xy[0] - origin[0, :2]).tolist()],
                        "contactB": bool(t.contact[0, -1]),
                        "vel": [round(x, 3) for x in t.ball_vel()[0, 0].tolist()],
                        "contact": bool(t.contact[0, 0]),
                        "qbase": [round(x, 4) for x in act.base_target[0].tolist()],
                    }
                )
                a._tr = tr
            newly = alive & t.dropped
            juggle_until = torch.where(
                newly, torch.full_like(juggle_until, time - lead), juggle_until
            )
            alive &= ~t.dropped
            if a.strip and time - lead < a.strip_seconds and s % 3 == 0:
                if not frames:
                    import os as _os
                    import sys as _sys

                    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
                    from smoke_env import Snapshotter

                    snap = Snapshotter(env, width=320, height=320)
                frames.append(snap.render(0, "jside"))
        dur = torch.where(
            torch.isnan(juggle_until), torch.full_like(juggle_until, a.seconds), juggle_until
        )
        res = {
            "col": a.col,
            "lat_delay": ld,
            "cup": a.cup,
            "lead": lead,
            "half": H,
            "dwell": D,
            "vacant": V,
            "flight": F,
            "v0": round(v0, 3),
            "kp": kp,
            "apex_m": round(v0**2 / (2 * G), 3),
            "num_envs": n,
            "juggle_s_mean": float(dur.mean()),
            "juggle_s": [round(float(x), 2) for x in dur],
            "full_duration_frac": float((dur >= a.seconds - 1e-6).float().mean()),
            "catches_mean": float(t.catches.float().mean()),
            "throws_mean": float(t.throws.float().mean()),
            "catches": t.catches.tolist(),
            "base_clip_mean": float(act.clip_fraction.mean()),
        }
        # catches before first drop per env
        res["first_drop_s"] = [
            None if math.isnan(float(x)) else round(float(x), 2) for x in juggle_until
        ]
        print(json.dumps(res))
        if a.trace:
            with open(a.trace.replace(".json", f"_H{H}_D{D}.json"), "w") as fh:
                json.dump(getattr(a, "_tr", []), fh)
            a._tr = []
        if a.strip and frames:
            from PIL import Image

            ims = [Image.fromarray(f) for f in frames]
            cols = 12
            rows = (len(ims) + cols - 1) // cols
            sheet = Image.new("RGB", (cols * 320, rows * 320))
            for k, im in enumerate(ims):
                sheet.paste(im, ((k % cols) * 320, (k // cols) * 320))
            sheet.save(
                a.strip.replace(".png", f"_H{H}_D{D}_kp{int(kp)}_L{lead}_ld{ld}_{a.cup}.png")
            )
        results.append(res)
        env.close()
    with open(a.out, "w") as fh:
        json.dump({"mode": a.mode, "results": results}, fh, indent=1)


if __name__ == "__main__":
    main()
