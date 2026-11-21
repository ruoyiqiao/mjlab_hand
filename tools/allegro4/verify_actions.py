"""Verify the action space of an Allegro DexPBT-style task end to end.

For every action term (hand) it checks, against the compiled model and physics:

1. layout: action index k -> joint name -> MuJoCo joint id -> the single actuator
   driving that joint (type, gain, ctrlrange, forcerange);
2. finger mapping: a=-1 / 0 / +1 map to joint lower / mid / upper (hard MJCF
   range) for all 16 finger joints, and the soft limits actually used for clamps;
3. finger response: hold a=+1 then a=-1 per finger group from home pose; report
   reached joint position vs target after 0.5 s (tracking with 0.7 Nm effort);
4. base response: a one-hot +1 on each base dimension for 0.4 s from the anchor;
   report the palm translation (world) / rotation axis-angle and compare with
   v_max * t, plus the world direction of each base axis;
5. bounds: saturating base actions for 3 s keep targets and measured joints inside
   the workspace (+ violation tolerance);
6. isolation (dual): actions on one hand leave the other hand still.

Usage: python verify_actions.py <task_id> <out.json> [--device cuda:0]
"""

from __future__ import annotations

import argparse
import json

import mujoco
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    import os

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.lab_api.math import axis_angle_from_quat, quat_conjugate, quat_mul

    import mjlab_hand  # noqa: F401

    cfg = load_env_cfg(a.task, play=True)
    # Isolated hands: put the object far away so contacts do not disturb tracking.
    cfg.scene.num_envs = 4
    # The object is parked out of reach below the table; disable terminations so
    # that does not auto-reset the hands (only time_out with a long episode).
    from mjlab.envs import mdp as envs_mdp
    from mjlab.managers.termination_manager import TerminationTermCfg

    cfg.terminations = {"time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True)}
    cfg.episode_length_s = 1.0e6
    cfg.curriculum = {}
    env = ManagerBasedRlEnv(cfg, device=a.device)
    env.reset()
    model = env.sim.mj_model
    dev = env.device
    terms = [env.action_manager.get_term(n) for n in env.action_manager.active_terms]
    total = env.action_manager.total_action_dim
    offsets = np.cumsum([0] + [t.action_dim for t in terms])
    report: dict = {"task": a.task, "total_action_dim": total, "step_dt": env.step_dt, "terms": {}}
    obj = env.scene["object"]

    def park_object():
        far = obj.data.root_link_pos_w.clone()
        far[:, 2] = -5.0
        obj.write_root_link_pose_to_sim(torch.cat([far, obj.data.root_link_quat_w], -1))
        obj.write_root_link_velocity_to_sim(torch.zeros(env.num_envs, 6, device=dev))

    def run(action_fn, steps):
        for s in range(steps):
            park_object()
            env.step(action_fn(s))

    def home():
        env.reset()
        park_object()
        env.sim.forward()

    for ti, term in enumerate(terms):
        ent = term._entity
        name = term.cfg.entity_name
        names = term.canonical_joint_names
        jids_local = term.joint_ids_canonical.tolist()
        rows = []
        for k, (jn, jl) in enumerate(zip(names, jids_local)):
            gj = int(ent.indexing.joint_ids[jl])
            acts = [
                i
                for i in range(model.nu)
                if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
                and int(model.actuator_trnid[i, 0]) == gj
            ]
            assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, gj).endswith("/" + jn)
            act = acts[0] if len(acts) == 1 else None
            rows.append(
                {
                    "k": int(offsets[ti] + k),
                    "joint": jn,
                    "mj_joint_id": gj,
                    "n_actuators": len(acts),
                    "actuator": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act)
                    if act is not None
                    else None,
                    "kp": float(model.actuator_gainprm[act, 0]) if act is not None else None,
                    "kv_bias": float(-model.actuator_biasprm[act, 2]) if act is not None else None,
                    "forcerange": model.actuator_forcerange[act].round(3).tolist()
                    if act is not None
                    else None,
                    "jnt_range": model.jnt_range[gj].round(4).tolist(),
                }
            )
        layout_ok = all(r["n_actuators"] == 1 for r in rows)

        # Finger mapping (pure function of the action term).
        nb = 6
        fmap = []
        soft = ent.data.joint_pos_limits[0, term.joint_ids_canonical].cpu().numpy()
        for i in range(16):
            r = rows[nb + i]
            lo, hi = r["jnt_range"]
            tgt = {}
            for av in (-1.0, 0.0, 1.0):
                tgt[av] = float(av * term._f_scale[i] + term._f_offset[i])
            fmap.append(
                {
                    "joint": r["joint"],
                    "a=-1": round(tgt[-1.0], 4),
                    "a=0": round(tgt[0.0], 4),
                    "a=+1": round(tgt[1.0], 4),
                    "hard_range": [lo, hi],
                    "clamp_range": soft[nb + i].round(4).tolist(),
                    "maps_to_hard_range": abs(tgt[-1.0] - lo) < 2e-3 and abs(tgt[1.0] - hi) < 2e-3,
                }
            )

        # Finger tracking.
        track = []
        for sign in (+1.0, -1.0):
            home()
            act = torch.zeros(env.num_envs, total, device=dev)
            act[:, offsets[ti] + nb : offsets[ti] + nb + 16] = sign
            run(lambda s: act, 25)
            q = ent.data.joint_pos[0, term.joint_ids_canonical[nb:]].cpu().numpy()
            tgt = (sign * term._f_scale + term._f_offset).cpu().numpy()
            clamp = soft[nb:]
            tgt = np.clip(tgt, clamp[:, 0], clamp[:, 1])
            track.append(
                {
                    "a": sign,
                    "max_abs_err_rad": float(np.abs(q - tgt).max()),
                    "mean_abs_err_rad": float(np.abs(q - tgt).mean()),
                    "worst_joint": names[nb + int(np.abs(q - tgt).argmax())],
                }
            )

        # Base response.
        base = []
        pid = ent.find_bodies("palm")[0][0]
        T = 20  # 0.4 s
        for d in range(6):
            home()
            p0 = ent.data.body_link_pos_w[0, pid].clone()
            q0 = ent.data.body_link_quat_w[0, pid].clone()
            act = torch.zeros(env.num_envs, total, device=dev)
            act[:, offsets[ti] + d] = 1.0
            run(lambda s: act, T)
            hold = torch.zeros(env.num_envs, total, device=dev)
            hold[:, :] = 0.0
            # keep fingers where they are: absolute mode -> a=0 mid targets for all
            run(lambda s: hold, 10)  # settle at the final target
            dp = (ent.data.body_link_pos_w[0, pid] - p0).cpu().numpy()
            dq = quat_mul(
                ent.data.body_link_quat_w[0, pid].unsqueeze(0), quat_conjugate(q0.unsqueeze(0))
            )
            aa = axis_angle_from_quat(dq)[0].cpu().numpy()
            vmax = float(term._base_rate[d])
            expect = vmax * T * env.step_dt
            mag = float(np.linalg.norm(dp) if d < 3 else np.linalg.norm(aa))
            vec = dp if d < 3 else aa
            base.append(
                {
                    "dim": names[d],
                    "expected": round(expect, 4),
                    "measured": round(mag, 4),
                    "ratio": round(mag / expect, 3),
                    "world_dir": (vec / max(np.linalg.norm(vec), 1e-9)).round(3).tolist(),
                    "target": float(term.base_target[0, d]),
                }
            )

        # Saturation / bounds.
        home()
        viol = 0
        maxq = np.zeros(6)
        for sign in (+1.0, -1.0):
            act = torch.zeros(env.num_envs, total, device=dev)
            act[:, offsets[ti] : offsets[ti] + 6] = sign
            for s in range(150):
                park_object()
                env.step(act)
                viol += int(term.bound_violation.any())
            qb = ent.data.joint_pos[0, term.joint_ids_canonical[:6]].cpu().numpy()
            maxq = np.maximum(maxq, np.abs(qb))
        ws = term.cfg.workspace
        bnd = [max(abs(ws.joint_bounds[n][0]), abs(ws.joint_bounds[n][1])) for n in names[:6]]
        report["terms"][name] = {
            "layout": rows,
            "layout_ok": layout_ok,
            "finger_mapping": fmap,
            "finger_mapping_ok": all(f["maps_to_hard_range"] for f in fmap),
            "finger_tracking": track,
            "base_response": base,
            "saturation": {
                "violation_steps": viol,
                "max_abs_base_q": maxq.round(4).tolist(),
                "bounds_abs": bnd,
                "within": bool(
                    np.all(
                        maxq
                        <= np.array(bnd)
                        + np.array([ws.violation_tol] * 3 + [ws.violation_tol_rot] * 3)
                    )
                ),
            },
        }

    # Isolation for multi-hand.
    if len(terms) > 1:
        home()
        e1 = terms[1]._entity
        q1 = e1.data.joint_pos[0].clone()
        act = torch.zeros(env.num_envs, total, device=dev)
        act[:, offsets[0] : offsets[1]] = 1.0
        act[:, offsets[1] :] = 0.0
        run(lambda s: act, 20)
        # right hand got a=0 (fingers -> mid targets, base -> hold): only fingers may move.
        d = (e1.data.joint_pos[0] - q1)[terms[1].joint_ids_canonical[:6]].abs().max().item()
        report["isolation_other_base_max_dq"] = d
    with open(a.out, "w") as fh:
        json.dump(report, fh, indent=1)
    s = {
        n: {
            "layout_ok": v["layout_ok"],
            "finger_mapping_ok": v["finger_mapping_ok"],
            "finger_tracking": v["finger_tracking"],
            "base": [(b["dim"], b["ratio"], b["world_dir"]) for b in v["base_response"]],
            "saturation": v["saturation"],
        }
        for n, v in report["terms"].items()
    }
    s["isolation"] = report.get("isolation_other_base_max_dq")
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
