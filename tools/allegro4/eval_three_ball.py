"""Evaluate the three-ball cascade (designed prior, memory-based learner, or an RL residual policy).

Each env is one independent trial (its own learner memory, never cleared between its runs). Every env
runs until it has completed --runs episodes ("runs separated by resets", paper Fig. 2A). Reports per run
index: mean / std of consecutive caught throws, fraction reaching 5 cycles (30 caught throws) and the
10 s survival; failure causes; landing errors (observed landing - desired) per throwing hand.

Usage: python eval_three_ball.py <task> <out.json> [--num-envs 256] [--runs 10] [--seed 2001]
         [--checkpoint model.pt] [--trace 0]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import numpy as np
import torch

FAIL = {0: "time_out/none", 1: "missed_catch", 2: "dropped", 3: "wrong_hand", 4: "not_released"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2001)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--trace", type=int, default=0, help="print env 0 state for this many steps")
    ap.add_argument("--max-steps", type=int, default=200000)
    ap.add_argument(
        "--set",
        nargs="*",
        default=[],
        help="CascadeSpec overrides key=value (command/action fields only)",
    )
    ap.add_argument("--episode-s", type=float, default=0.0, help="override episode length (s)")
    ap.add_argument(
        "--ball-pos-noise",
        type=float,
        default=0.0,
        help="std (m) per axis, balls in flight at reset",
    )
    ap.add_argument(
        "--ball-vel-noise",
        type=float,
        default=0.0,
        help="std (m/s) per axis, balls in flight at reset",
    )
    a = ap.parse_args()

    from mjlab_hand.registration import register_tasks

    register_tasks()
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    from mjlab_hand.juggling.cascade import get_juggler

    cfg = load_env_cfg(a.task, play=True)
    CMD = next(iter(cfg.commands))
    RESET_KEY = next(
        k for k in cfg.events if k.startswith("reset_") and k != "reset_scene_to_default"
    )
    cfg.scene.num_envs = a.num_envs
    cfg.seed = a.seed
    sp = next(iter(cfg.commands.values())).spec
    for kv in a.set:
        k, v = kv.split("=", 1)
        old = getattr(sp, k)
        setattr(
            sp, k, type(old)(v) if not isinstance(old, bool) else v.lower() in ("1", "true", "yes")
        )
    if a.episode_s > 0:
        cfg.episode_length_s = a.episode_s
    if a.ball_pos_noise > 0 or a.ball_vel_noise > 0:
        base_reset = cfg.events[RESET_KEY].func

        def noisy_reset(env_, env_ids, **kw):
            base_reset(env_, env_ids, **kw)
            env_.sim.forward()  # derived poses must reflect the reset write
            tt = get_juggler(env_, CMD)
            m = len(env_ids)
            o = env_.scene.env_origins[env_ids]
            for b, ball in enumerate(tt.balls):
                # contact flags are stale right after a reset: balls > 6 cm from both cups are in flight
                cp = tt.cup_pos()[env_ids]
                bp = ball.data.root_link_pos_w[env_ids] - o
                far = ((bp.unsqueeze(1) - cp).norm(dim=-1) > 0.06).all(-1)
                fl = far.float().unsqueeze(-1)
                pose = ball.data.root_link_pose_w[env_ids].clone()
                pose[:, :3] += fl * a.ball_pos_noise * torch.randn(m, 3, device=env_.device)
                vel = ball.data.root_link_vel_w[env_ids].clone()
                vel[:, :3] += fl * a.ball_vel_noise * torch.randn(m, 3, device=env_.device)
                ball.write_root_link_pose_to_sim(pose, env_ids=env_ids)
                ball.write_root_link_velocity_to_sim(vel, env_ids=env_ids)

        cfg.events[RESET_KEY].func = noisy_reset
    env = ManagerBasedRlEnv(cfg, device=a.device)
    t = get_juggler(env, CMD)
    t.record = True
    pol = None
    if a.checkpoint:
        from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

        rl = load_rl_cfg(a.task)
        wenv = RslRlVecEnvWrapper(env, clip_actions=rl.clip_actions)
        r = MjlabOnPolicyRunner(wenv, asdict(rl), device=a.device)
        r.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location=a.device)
        pol = r.get_inference_policy(device=a.device)
        obs, _ = wenv.reset()
    else:
        env.reset()
    n = a.num_envs
    dev = env.device
    runs = [[] for _ in range(n)]  # per env: list of (caught, throws, fail_code, seconds)
    done_envs = torch.zeros(n, dtype=torch.bool, device=dev)
    zero = torch.zeros(n, env.action_manager.total_action_dim, device=dev)
    step = 0
    while not bool(done_envs.all()) and step < a.max_steps:
        caught_prev = t.caught.clone()
        throws_prev = t.throws.clone()
        code_prev = t.fail_code.clone()
        eps_prev = t.ep_steps.clone()
        pat_prev = t.pat.clone() if hasattr(t, "pattern_names") else None
        if pol is not None:
            with torch.no_grad():
                act = pol(obs)
            obs, _, dn, _ = wenv.step(act)
            term = dn.bool()
        else:
            _, _, te, tr, _ = env.step(zero)
            term = te | tr
        if a.trace and step < a.trace:
            cp = t.cup_pos()[0]
            bp = t.ball_pos()[0]
            rr = (bp - cp[0]).tolist()
            rl = (bp - cp[1]).tolist()
            print(
                "TR",
                step,
                int(t.step_idx[0]),
                "cupR",
                [round(x, 3) for x in cp[0].tolist()],
                "cupL",
                [round(x, 3) for x in cp[1].tolist()],
                "b-R",
                [[round(x, 3) for x in r] for r in rr],
                "b-L",
                [[round(x, 3) for x in r] for r in rl],
                "ct",
                t.contact[0].int().tolist(),
                "caught",
                float(t.caught[0]),
                "fail",
                int(t.fail_code[0]),
            )
        # Terminations at this step come from flags set in the previous command update, so the
        # finished episode's values are the pre-step ones (the reset overwrote the current ones).
        for i in term.nonzero().flatten().tolist():
            if done_envs[i]:
                continue
            runs[i].append(
                (
                    float(caught_prev[i]),
                    float(throws_prev[i]),
                    int(code_prev[i]),
                    float(eps_prev[i] + 1) * env.step_dt,
                    int(pat_prev[i]) if pat_prev is not None else 0,
                )
            )
            if len(runs[i]) >= a.runs:
                done_envs[i] = True
        step += 1
    R = a.runs
    mat = np.full((n, R), np.nan)
    sec = np.full((n, R), np.nan)
    codes = {}
    for i in range(n):
        for k, (c, _thr, code, s, _pt) in enumerate(runs[i][:R]):
            mat[i, k] = c
            sec[i, k] = s
            codes[FAIL.get(code, str(code))] = codes.get(FAIL.get(code, str(code)), 0) + 1
    per_run = []
    for k in range(R):
        col = mat[:, k][~np.isnan(mat[:, k])]
        sc = sec[:, k][~np.isnan(sec[:, k])]
        if len(col) == 0:
            continue
        per_run.append(
            {
                "run": k + 1,
                "n": int(len(col)),
                "caught_mean": round(float(col.mean()), 2),
                "caught_std": round(float(col.std()), 2),
                "caught_median": float(np.median(col)),
                "five_cycles_frac(>=30)": round(float((col >= 30).mean()), 4),
                "survive_10s_frac": round(float((sc >= 10.0 - 1e-6).mean()), 4),
                "full_episode_frac": round(
                    float((sc >= env.cfg.episode_length_s - 2 * env.step_dt).mean()), 4
                ),
            }
        )
    per_pattern = {}
    if hasattr(t, "pattern_names"):
        for pi, nm in enumerate(t.pattern_names):
            rows = [r for i in range(n) for r in runs[i][:R] if r[4] == pi]
            if not rows:
                continue
            cs = np.array([r[0] for r in rows])
            ss = np.array([r[3] for r in rows])
            cc = {}
            for r in rows:
                cc[FAIL.get(r[2], str(r[2]))] = cc.get(FAIL.get(r[2], str(r[2])), 0) + 1
            per_pattern[nm] = {
                "n": len(rows),
                "caught_mean": round(float(cs.mean()), 2),
                "caught_median": float(np.median(cs)),
                "five_cycles_frac(>=30)": round(float((cs >= 30).mean()), 4),
                "full_episode_frac": round(
                    float((ss >= env.cfg.episode_length_s - 2 * env.step_dt).mean()), 4
                ),
                "fail_causes": cc,
            }
    land = torch.cat(t.log_land, 0).numpy() if t.log_land else np.zeros((0, 8))
    np.save(
        a.out.replace(".json", "_land.npy"), land
    )  # columns: env, hand, y - y_d (3), u - y_d (3)
    land_stats = {}
    for h, nm in ((0, "right_throws"), (1, "left_throws")):
        L = land[land[:, 1] == h]
        if len(L):
            err = L[:, 2:5]
            land_stats[nm] = {
                "n": int(len(L)),
                "mean_offset_xyz": [round(float(x), 4) for x in err.mean(0)],
                "abs_err_p50_p90": [
                    round(float(np.percentile(np.linalg.norm(err, axis=1), q)), 4) for q in (50, 90)
                ],
                "first100_abs_err_p50": round(
                    float(np.median(np.linalg.norm(err[:100], axis=1))), 4
                ),
                "last100_abs_err_p50": round(
                    float(np.median(np.linalg.norm(err[-100:], axis=1))), 4
                ),
            }
    out = {
        "task": a.task,
        "set": a.set,
        "episode_s": env.cfg.episode_length_s,
        "ball_pos_noise": a.ball_pos_noise,
        "ball_vel_noise": a.ball_vel_noise,
        "checkpoint": a.checkpoint,
        "num_envs": n,
        "runs": R,
        "seed": a.seed,
        "learner": bool(t.use_learner),
        "per_run": per_run,
        "per_pattern": per_pattern,
        "fail_causes": codes,
        "landing": land_stats,
        "steps": step,
    }
    print(json.dumps(out, indent=1))
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    env.close()


if __name__ == "__main__":
    main()
