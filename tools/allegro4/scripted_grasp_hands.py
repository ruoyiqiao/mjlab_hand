"""Hand-agnostic scripted grasp feasibility (CPU): can this floating hand grasp and lift the task object at all?

Usage: python scripted_grasp_hands.py <task_id> <out.json> [--device cpu] [--reps 2] [--close 0.65]

Open-loop script in the task's own action space (no state writes after reset): approach above the object -> descend ->
close the fingers -> lift 0.20 m -> hold. Works for every hand because (1) the base joints are canonical
(hand_profile.py: base_x = world -z, base_y = +y, base_z = +x for the palm-down anchor) and (2) the finger targets are
joint-range fractions: abduction-like joints (range roughly symmetric about 0) stay at 0, every other finger joint goes
from 15% (open) to ``--close`` of its range; tendon targets likewise. Each env tries a different (palm setback, height)
pair. Reports the fraction of envs still holding the object 0.15 m above its start after the hold phase, tip contacts
at the end of closing, and the best rows. Feasibility evidence only; not a policy result.
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
    ap.add_argument("--close", type=float, default=0.65)
    ap.add_argument(
        "--single-object",
        action="store_true",
        help="use the 5 cm cube in every env (no cuboid variants)",
    )
    a = ap.parse_args()

    from mjlab_hand.registration import register_tasks

    register_tasks()
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    setback = (0.02, 0.05, 0.08, 0.11, 0.14, 0.17)
    hgt = (-0.01, 0.02, 0.05, 0.08)
    grid = list(itertools.product(setback, hgt)) * a.reps
    n = len(grid)
    cfg = load_env_cfg(a.task, play=True)
    cfg.scene.num_envs = n
    if a.single_object:
        from mjlab_hand.tasks_common.layouts import CUBE
        from mjlab_hand.tasks_common.objects import make_cube_entity_cfg

        cfg.scene.entities["object"] = make_cube_entity_cfg(CUBE)
        cmd_cfg = next(iter(cfg.commands.values()))
        cmd_cfg.object_sizes = None
        cmd_cfg.object_size = CUBE.size
    env = ManagerBasedRlEnv(cfg, device=a.device)
    env.reset()
    dev = env.device
    t = next(iter(env.command_manager._terms.values()))
    act = env.action_manager.get_term("hand")
    dt = env.step_dt
    rate, lo, hi = act._base_rate, act._base_lo, act._base_hi
    flo, fhi = act._finger_lo[0], act._finger_hi[0]
    span = fhi - flo
    abduct = (flo + fhi).abs() < 0.25 * span  # symmetric about 0 -> spread joint, keep at 0

    def finger_action(close: float) -> torch.Tensor:
        frac = 0.15 + (a.close - 0.15) * close
        q = torch.where(abduct, torch.zeros_like(flo), flo + frac * span)
        fa = ((q - act._f_offset) / act._f_scale).clamp(-1, 1)
        parts = [fa.expand(n, -1)]
        if act._nt:
            parts.append(torch.full((n, act._nt), 2.0 * frac - 1.0, device=dev))
        return torch.cat(parts, -1)

    palm0, _ = t.palm_pose(0)
    palm0 = palm0.clone()
    q0 = act.base_target.clone()
    obj0 = t.obj.data.root_link_pos_w.clone()
    top = t.table_top()
    f = torch.tensor([g[0] for g in grid], device=dev)
    h = torch.tensor([g[1] for g in grid], device=dev)

    def base_goal(dz_world, dx_world, dy_world):
        g = q0.clone()
        g[:, 0] += -dz_world
        g[:, 1] += dy_world
        g[:, 2] += dx_world
        return torch.clamp(g, lo, hi)

    dx = obj0[:, 0] - palm0[:, 0] - f
    dy = obj0[:, 1] - palm0[:, 1]
    z_above = (top + 0.05 + 0.08) - palm0[:, 2]
    z_grasp = (top + 0.05 + h) - palm0[:, 2]
    z_lift = z_grasp + 0.20
    phases = [
        ("approach", 60, z_above, 0.0),
        ("descend", 50, z_grasp, 0.0),
        ("close", 50, z_grasp, 1.0),
        ("lift", 60, z_lift, 1.0),
        ("hold", 50, z_lift, 1.0),
    ]
    max_lift = torch.zeros(n, device=dev)
    tips_close = torch.zeros(n, device=dev)
    sensor = env.scene["hand_tip_contact"]
    log = []
    for name, steps, zt, close in phases:
        goal = base_goal(zt, dx, dy)
        for s in range(steps):
            c = close if name != "close" else close * min(1.0, (s + 1) / (steps * 0.6))
            ab = torch.clamp((goal - act.base_target) / (rate * dt), -1.0, 1.0)
            ab[:, 3:] = 0.0
            env.step(torch.cat([ab, finger_action(c)], dim=-1))
            max_lift = torch.maximum(max_lift, t.obj.data.root_link_pos_w[:, 2] - obj0[:, 2])
        found = sensor.data.found.reshape(n, -1) > 0
        if name == "close":
            tips_close = found.sum(-1).float()
        log.append(
            {
                "phase": name,
                "obj_dz_mean": round(
                    float((t.obj.data.root_link_pos_w[:, 2] - obj0[:, 2]).mean()), 4
                ),
                "tips_in_contact_mean": round(float(found.sum(-1).float().mean()), 3),
            }
        )
    final = t.obj.data.root_link_pos_w[:, 2] - obj0[:, 2]
    rows = [
        {
            "setback": ff,
            "h": hh,
            "max_lift": round(float(max_lift[i]), 3),
            "final_lift": round(float(final[i]), 3),
            "tips_at_close": int(tips_close[i]),
            "held": bool(final[i] > 0.15),
        }
        for i, (ff, hh) in enumerate(grid)
    ]
    by_setback = {}
    for r in rows:
        by_setback.setdefault(r["setback"], []).append(r["held"])
    rep = {
        "task": a.task,
        "num_envs": n,
        "close": a.close,
        "single_object": a.single_object,
        "phases": log,
        "held_fraction": round(sum(r["held"] for r in rows) / n, 3),
        "held_by_setback": {k: round(sum(v) / len(v), 2) for k, v in by_setback.items()},
        "best": sorted(rows, key=lambda r: -r["final_lift"])[:4],
    }
    json.dump({**rep, "rows": rows}, open(a.out, "w"), indent=1)
    print("GRASP", json.dumps(rep))


if __name__ == "__main__":
    main()
