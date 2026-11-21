"""Scripted throw feasibility for Grasp-Throw-Allegro (open loop, task action space).

grasp (as scripted_grasp best setting) -> lift -> wind back -> swing toward +x with
an upward component at the max base speed -> open fingers at a per-env release
step. Records the object's trajectory after release: max horizontal travel,
landing point (first table/ground/bucket contact or z < bucket rim), whether the
tracker flagged a valid flight / in-bucket / success, and the distance from the
landing point to the bucket center. Used to check that the frozen bucket range is
physically reachable by a throw and not by placing.

Usage: python scripted_throw.py <out.json> [--device cpu] [--num-envs 48]
"""

from __future__ import annotations

import argparse
import json

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num-envs", type=int, default=48)
    a = ap.parse_args()

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import mjlab_hand  # noqa: F401

    cfg = load_env_cfg("Grasp-Throw-Allegro", play=True)
    n = a.num_envs
    cfg.scene.num_envs = n
    env = ManagerBasedRlEnv(cfg, device=a.device)
    env.reset()
    dev = env.device
    t = env.command_manager.get_term("throw_goal")
    act = env.action_manager.get_term("hand")
    rate, lo, hi = act._base_rate, act._base_lo, act._base_hi
    dt = env.step_dt
    palm0, _ = t.palm_pose(0)
    palm0 = palm0.clone()
    q0 = act.base_target.clone()
    obj0 = t.obj.data.root_link_pos_w.clone()
    top = t.table_top()

    def goal(dz, dx, dy):
        g = q0.clone()
        g[:, 0] += -dz
        g[:, 1] += dy
        g[:, 2] += dx
        return torch.clamp(g, lo, hi)

    def fingers(close):
        f = torch.full((n, 16), -0.6 + 1.4 * close, device=dev)
        for k in range(4):
            f[:, 4 * k] = 0.0
        f[:, 12] = 1.0
        return f

    dx = obj0[:, 0] - palm0[:, 0] - 0.02
    dy = obj0[:, 1] - palm0[:, 1]
    z_above = (top + 0.11) - palm0[:, 2]
    z_grasp = (top + 0.10) - palm0[:, 2]
    # Per-env swing: upward fraction and release step after swing start.
    up = torch.tensor([0.0, 0.3, 0.6, 0.9], device=dev).repeat_interleave(n // 4 if n >= 4 else 1)[
        :n
    ]
    rel = torch.tensor([4, 6, 8, 10, 12, 14], device=dev).repeat(n // 6 + 1)[:n]

    def drive(g, close, steps):
        for _ in range(steps):
            ab = torch.clamp((g - act.base_target) / (rate * dt), -1, 1)
            ab[:, 3:] = 0.0
            env.step(torch.cat([ab, fingers(close)], -1))

    drive(goal(z_above, dx, dy), 0.0, 60)
    drive(goal(z_grasp, dx, dy), 0.0, 50)
    for s in range(40):
        drive(goal(z_grasp, dx, dy), min(1.0, (s + 1) / 24), 1)
    drive(goal(z_grasp + 0.20, dx - 0.15, dy), 1.0, 60)  # lift and wind back
    lifted = t.lifted.clone()
    # Swing: +x at full speed (base_z), up at `up` fraction (base_x negative = up).
    release_pos = torch.zeros(n, 3, device=dev)
    released = torch.zeros(n, dtype=torch.bool, device=dev)
    max_dx = torch.zeros(n, device=dev)
    landed_pos = torch.full((n, 3), float("nan"), device=dev)
    any_valid = torch.zeros(n, dtype=torch.bool, device=dev)
    any_in = torch.zeros(n, dtype=torch.bool, device=dev)
    s0 = t.successes.clone()
    for s in range(120):
        ab = torch.zeros(n, 6, device=dev)
        if s < 20:
            ab[:, 2] = 1.0
            ab[:, 0] = -up
        rel_now = s >= rel
        f = fingers(1.0)
        f[rel_now] = fingers(0.0)[rel_now]
        env.step(torch.cat([ab, f], -1))
        p = t.obj.data.root_link_pos_w
        newly = rel_now & ~released
        release_pos[newly] = p[newly]
        released |= rel_now
        max_dx = torch.maximum(max_dx, p[:, 0] - release_pos[:, 0])
        land = released & torch.isnan(landed_pos[:, 0]) & (p[:, 2] < t.bucket_pos[:, 2] + 0.198)
        landed_pos[land] = p[land]
        any_valid |= t._valid_flight
        any_in |= t.in_bucket
    rows = []
    for i in range(n):
        lp = landed_pos[i].tolist()
        bc = t.bucket_pos[i].tolist()
        rows.append(
            {
                "up": float(up[i]),
                "release_step": int(rel[i]),
                "lifted_before_swing": bool(lifted[i]),
                "release_xz": [
                    round(float(release_pos[i, 0]), 3),
                    round(float(release_pos[i, 2]), 3),
                ],
                "max_dx_after_release": round(float(max_dx[i]), 3),
                "landing_xyz": [round(x, 3) for x in lp],
                "bucket_xyz": [round(x, 3) for x in bc],
                "valid_flight": bool(any_valid[i]),
                "in_bucket": bool(any_in[i]),
                "success": int(t.successes[i] - s0[i]),
            }
        )
    summary = {
        "lifted_before_swing": sum(r["lifted_before_swing"] for r in rows),
        "valid_flight": sum(r["valid_flight"] for r in rows),
        "in_bucket": sum(r["in_bucket"] for r in rows),
        "success": sum(r["success"] > 0 for r in rows),
        "landing_x_max": max(
            (r["landing_xyz"][0] for r in rows if r["landing_xyz"][0] == r["landing_xyz"][0]),
            default=None,
        ),
        "bucket_x_range": [
            cfg.commands["throw_goal"].bucket_pos_min[0],
            cfg.commands["throw_goal"].bucket_pos_max[0],
        ],
        "num_envs": n,
    }
    with open(a.out, "w") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=1)
    print(json.dumps(summary, indent=1))
    best = sorted(rows, key=lambda r: -r["max_dx_after_release"])[:8]
    print(json.dumps(best, indent=1))


if __name__ == "__main__":
    main()
