"""Scripted feasibility check: can the arm-free Allegro grasp and lift the cube?

Open-loop script in the task's own action space (no state writes after reset):
approach above the cube -> descend -> close fingers -> lift -> hold. Each env uses
a different (forward offset, height) pair; reports per-env max lift, final lift,
fingertip contacts and whether the cube stayed lifted. Feasibility evidence only;
a scripted success is not a policy result.

Usage: python scripted_grasp.py <task_id> <out.json> [--device cpu] [--reps 2]
"""

from __future__ import annotations

import argparse
import itertools
import json

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import mjlab_hand  # noqa: F401

    fwd = (0.02, 0.05, 0.08, 0.11)
    hgt = (-0.01, 0.02, 0.05)
    grid = list(itertools.product(fwd, hgt)) * a.reps
    n = len(grid)
    cfg = load_env_cfg(a.task, play=True)
    cfg.scene.num_envs = n
    env = ManagerBasedRlEnv(cfg, device=a.device)
    env.reset()
    dev = env.device
    cmd = next(iter(env.command_manager.active_terms))
    t = env.command_manager.get_term(cmd)
    act = env.action_manager.get_term("hand")
    dt = env.step_dt
    rate = act._base_rate
    lo, hi = act._base_lo, act._base_hi

    palm0, _ = t.palm_pose(0)
    palm0 = palm0.clone()
    q0 = act.base_target.clone()  # base joint targets at anchor (+noise)
    obj0 = t.obj.data.root_link_pos_w.clone()
    top = t.table_top()
    f = torch.tensor([g[0] for g in grid], device=dev)
    h = torch.tensor([g[1] for g in grid], device=dev)

    # World -> base joint mapping at the anchor (layout probe): base_x=-z, base_y=+y, base_z=+x.
    def base_goal(dz_world, dx_world, dy_world):
        g = q0.clone()
        g[:, 0] += -dz_world
        g[:, 1] += dy_world
        g[:, 2] += dx_world
        return torch.clamp(g, lo, hi)

    dx = obj0[:, 0] - palm0[:, 0] - f
    dy = obj0[:, 1] - palm0[:, 1]
    z_above = (top + 0.05 + 0.06) - palm0[:, 2]
    z_grasp = (top + 0.05 + h) - palm0[:, 2]
    z_lift = z_grasp + 0.20

    def finger_action(close: float) -> torch.Tensor:
        fa = torch.full((n, 16), -0.6 + 1.4 * close, device=dev)
        for k in range(4):  # joint 0 of each finger (spread) at neutral
            fa[:, 4 * k] = 0.0
        fa[:, 12] = 1.0  # thumb opposition
        return fa

    phases = [
        ("approach", 60, z_above, 0.0),
        ("descend", 50, z_grasp, 0.0),
        ("close", 40, z_grasp, 1.0),
        ("lift", 60, z_lift, 1.0),
        ("hold", 50, z_lift, 1.0),
    ]
    max_lift = torch.zeros(n, device=dev)
    tip_contacts_close = torch.zeros(n, device=dev)
    log = []
    for name, steps, zt, close in phases:
        goal = base_goal(zt, dx, dy)
        for s in range(steps):
            c = close if name != "close" else close * min(1.0, (s + 1) / (steps * 0.6))
            ab = torch.clamp((goal - act.base_target) / (rate * dt), -1.0, 1.0)
            ab[:, 3:] = 0.0
            env.step(torch.cat([ab, finger_action(c)], dim=-1))
            z = t.obj.data.root_link_pos_w[:, 2] - obj0[:, 2]
            max_lift = torch.maximum(max_lift, z)
        found = env.scene["hand_tip_contact"].data.found.reshape(n, -1) > 0
        if name == "close":
            tip_contacts_close = found.sum(-1).float()
        log.append(
            {
                "phase": name,
                "obj_dz_mean": float((t.obj.data.root_link_pos_w[:, 2] - obj0[:, 2]).mean()),
                "tips_in_contact_mean": float(found.sum(-1).float().mean()),
            }
        )
    final = t.obj.data.root_link_pos_w[:, 2] - obj0[:, 2]
    rows = []
    for i, (ff, hh) in enumerate(grid):
        rows.append(
            {
                "fwd": ff,
                "h": hh,
                "max_lift": float(max_lift[i]),
                "final_lift": float(final[i]),
                "tips_at_close": int(tip_contacts_close[i]),
                "held": bool(final[i] > 0.15),
            }
        )
    rep = {
        "task": a.task,
        "num_envs": n,
        "phases": log,
        "rows": rows,
        "held_fraction": sum(r["held"] for r in rows) / n,
        "best": sorted(rows, key=lambda r: -r["final_lift"])[:5],
    }
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps({k: rep[k] for k in ("held_fraction", "phases", "best")}, indent=1))


if __name__ == "__main__":
    main()
