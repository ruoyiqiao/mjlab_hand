"""Record a full state trace of the three-ball cascade prior (zero action) for a few envs -> npz.

Per control step: schedule step, cup pos/vel (2,3) x2, reference cup pos/vel (2,3) x2, ball pos/vel (3,3) x2,
contacts (2,3), finger joint positions of both hands (2,16), caught, fail code.
Usage: python trace_three_ball.py <task> <out.npz> [--steps 400] [--num-envs 4] [--set key=value ...]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    from mjlab_hand.registration import register_tasks

    register_tasks()
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    try:
        from mjlab_hand.juggling.cascade import get_juggler
    except (
        ImportError
    ):  # sources frozen with runs launched before the rename (rai_cascade / "rai" command)
        from mjlab_hand.juggling.rai_cascade import get_rai as get_juggler
    from mjlab_hand.tasks_common.actions import ALLEGRO_FINGER_JOINTS

    cfg = load_env_cfg(a.task, play=True)
    CMD = next(iter(cfg.commands))
    cfg.scene.num_envs = a.num_envs
    sp = next(iter(cfg.commands.values())).spec
    for kv in a.set:
        k, v = kv.split("=", 1)
        old = getattr(sp, k)
        setattr(
            sp, k, type(old)(v) if not isinstance(old, bool) else v.lower() in ("1", "true", "yes")
        )
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    env.reset()
    t = get_juggler(env, CMD)
    fids = [[h.joint_names.index(j) for j in ALLEGRO_FINGER_JOINTS] for h in t.hands]
    zero = torch.zeros(a.num_envs, env.action_manager.total_action_dim, device=env.device)
    rec = {
        k: []
        for k in (
            "step",
            "cup_p",
            "cup_v",
            "ref_p",
            "ref_v",
            "ball_p",
            "ball_v",
            "contact",
            "fingers",
            "caught",
            "fail",
            "u_cmd",
            "catch_cmd",
        )
    }
    for _ in range(a.steps):
        env.step(zero)
        rp, rv, _ = t.ref_at(t.step_idx)
        rec["step"].append(t.step_idx.cpu().numpy().copy())
        rec["cup_p"].append(t.cup_pos().cpu().numpy())
        rec["cup_v"].append(t.cup_vel().cpu().numpy())
        rec["ref_p"].append(rp.cpu().numpy())
        rec["ref_v"].append(rv.cpu().numpy())
        rec["ball_p"].append(t.ball_pos().cpu().numpy())
        rec["ball_v"].append(t.ball_vel().cpu().numpy())
        rec["contact"].append(t.contact.cpu().numpy().copy())
        rec["fingers"].append(
            np.stack([h.data.joint_pos[:, f].cpu().numpy() for h, f in zip(t.hands, fids)], 1)
        )
        rec["caught"].append(t.caught.cpu().numpy().copy())
        rec["fail"].append(t.fail_code.cpu().numpy().copy())
        rec["u_cmd"].append(t.u_cmd.cpu().numpy().copy())
        rec["catch_cmd"].append(t.catch_cmd.cpu().numpy().copy())
    np.savez(
        a.out,
        **{k: np.stack(v, 0) for k, v in rec.items()},
        p_rel=t.p_rel.cpu().numpy(),
        p_catch_nom=t.p_catch_nom.cpu().numpy(),
        cup_home=t.cup_home.cpu().numpy(),
        dt=t.dt,
    )
    print("saved", a.out)
    env.close()


if __name__ == "__main__":
    main()
