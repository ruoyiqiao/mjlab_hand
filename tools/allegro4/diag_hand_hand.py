"""Diagnose DualHand hand-hand collisions of a checkpoint (play config, first episode per env).

Per step and env: min distance between any body origin of the left hand and any body origin of
the right hand (closest body pair), the hand-hand contact flag, object contact per hand, and the
object's largest dimension. Reports, for the collision steps, which body pairs are closest, how
the object was held, the object size, and the time in the current goal; plus the distribution of
the min body distance during collision-free two-hand handoffs (to choose a proximity margin).

Usage: python diag_hand_hand.py <task> <ckpt> <out.json> [--num-envs 512] [--seed 1001]
"""

from __future__ import annotations

import argparse
import collections
import json
from dataclasses import asdict

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    import mjlab_hand  # noqa: F401

    torch.manual_seed(a.seed)
    cfg = load_env_cfg(a.task, play=True)
    cfg.scene.num_envs = a.num_envs
    cfg.seed = a.seed
    raw = ManagerBasedRlEnv(cfg, device=a.device)
    rl_cfg = load_rl_cfg(a.task)
    env = RslRlVecEnvWrapper(raw, clip_actions=rl_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(env, asdict(rl_cfg), device=a.device)
    runner.load(a.ckpt, load_cfg={"actor": True}, strict=True, map_location=a.device)
    policy = runner.get_inference_policy(device=a.device)
    t = raw.command_manager.get_term("dual_goal")
    hl, hr = raw.scene["hand_left"], raw.scene["hand_right"]
    nl, nr = hl.body_names, hr.body_names
    n = a.num_envs
    active = torch.ones(n, dtype=torch.bool, device=a.device)
    coll = []
    handoff_d = []
    free_d = []
    # Collision steps terminate and reset in the same env.step, so capture them inside the
    # tracker update (post-physics, before resets).
    orig = t._update_dual
    snap = {}

    def hooked():
        orig()
        hh = t.hand_hand & active
        if hh.any():
            d = torch.cdist(hl.data.body_link_pos_w, hr.data.body_link_pos_w)
            dmin, idx = d.reshape(n, -1).min(-1)
            size = t.obj_size.max(-1).values
            for i in hh.nonzero().flatten().tolist():
                if i in snap:
                    continue
                bi, bj = divmod(int(idx[i]), len(nr))
                snap[i] = {
                    "env": i,
                    "step": cur[0],
                    "dmin": round(float(dmin[i]), 4),
                    "pair": f"{nl[bi]}|{nr[bj]}",
                    "contact_l": bool(t.hand_contact[i, 0]),
                    "contact_r": bool(t.hand_contact[i, 1]),
                    "obj_size_max": round(float(size[i]), 3),
                    "goal_time": round(float(t.goal_time[i]), 2),
                    "successes": int(t.successes[i]),
                    "lifted": bool(t.lifted[i]),
                    "goal_side": int(t.goal_side[i]),
                    "start_side": int(t.goal_start_side[i]),
                }

    t._update_dual = hooked
    cur = [0]
    obs, _ = env.reset()
    for step in range(int(raw.max_episode_length) + 2):
        if not active.any():
            break
        cur[0] = step
        with torch.no_grad():
            act = policy(obs)
        obs, _, dones, _ = env.step(act)
        d = torch.cdist(hl.data.body_link_pos_w, hr.data.body_link_pos_w)
        dmin = d.reshape(n, -1).min(-1).values
        both = t.hand_contact.all(-1)
        m = active & ~dones.bool()
        handoff_d += dmin[m & both].tolist()
        free_d += dmin[m & ~both].tolist()
        active &= ~dones.bool()
    coll = list(snap.values())

    def q(x):
        return {p: round(float(np.percentile(x, p)), 4) for p in (0.1, 1, 5, 25, 50)} if x else {}

    res = {
        "task": a.task,
        "ckpt": a.ckpt,
        "seed": a.seed,
        "num_envs": n,
        "collision_episodes": len({c["env"] for c in coll}),
        "collision_pairs": collections.Counter(c["pair"] for c in coll).most_common(10),
        "collision_dmin": q([c["dmin"] for c in coll]),
        "collision_both_holding_frac": float(
            np.mean([c["contact_l"] and c["contact_r"] for c in coll])
        )
        if coll
        else None,
        "collision_neither_holding_frac": float(
            np.mean([not (c["contact_l"] or c["contact_r"]) for c in coll])
        )
        if coll
        else None,
        "collision_obj_size_max": q([c["obj_size_max"] for c in coll]),
        "collision_goal_time": q([c["goal_time"] for c in coll]),
        "handoff_steps": len(handoff_d),
        "handoff_dmin_pct": q(handoff_d),
        "free_dmin_pct": q(free_d),
        "collisions": coll[:200],
    }
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "collisions"}, indent=1))


if __name__ == "__main__":
    main()
