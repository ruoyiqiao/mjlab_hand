"""Evaluator for the four Allegro DexPBT-style tasks.

Matches envs that contain a :class:`DexPBTTaskCommand` (unique command names
``throw_goal`` / ``reorient_goal`` / ``regrasp_goal`` / ``dual_goal``); it is
registered before the legacy Grasp/Rotation evaluators.

Protocol (unbiased, finite): every env starts a fresh episode at eval start and
only that **first complete episode** is counted (``num_steps`` defaults to the
episode cap + 2, so every env finishes it). Each goal inside the episode is one
attempt: it is a success when the tracker counts it, otherwise it is a failure
(drop / per-goal timeout / throw failure) or, if the episode cap ended it, a
censored attempt that is **counted as a failure** and reported separately.
Rates come with Wilson 95% intervals and explicit denominators.
"""

from __future__ import annotations

import json
import math

import torch

from mjlab_hand.eval.base import TaskEvaluator, register_evaluator
from mjlab_hand.tasks_common.tracker import DexPBTTaskCommand


def _wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def _rate(k: int, n: int) -> dict:
    return {"k": int(k), "n": int(n), "rate": (k / n) if n else float("nan"), "ci95": _wilson(k, n)}


def _find_tracker(env) -> DexPBTTaskCommand | None:
    cm = env.unwrapped.command_manager
    for name in cm.active_terms:
        t = cm.get_term(name)
        if isinstance(t, DexPBTTaskCommand):
            return t
    return None


class DexPBTTaskEvaluator(TaskEvaluator):
    name = "dexpbt"

    @classmethod
    def matches(cls, env) -> bool:
        return _find_tracker(env) is not None

    @classmethod
    def default_steps(cls, env) -> int:
        return int(env.unwrapped.max_episode_length) + 2

    def __init__(self, env, args, device: torch.device) -> None:
        self.env = env
        self.u = env.unwrapped
        self.t = _find_tracker(env)
        assert self.t is not None
        self.task = self.t.cfg.task
        self.device = device
        n = self.u.num_envs
        self.n = n
        self.dt = float(self.u.step_dt)
        self.active = torch.ones(n, dtype=torch.bool, device=device)  # first episode running
        self.goal_records: list[dict] = []
        self.episode_records: list[dict] = []
        self.actions = [
            self.u.action_manager.get_term(a) for a in self.u.action_manager.active_terms
        ]
        self.clip_sum = torch.zeros(n, device=device)
        self.clip_lo_sum = torch.zeros(len(self.actions), 6, device=device)
        self.clip_hi_sum = torch.zeros(len(self.actions), 6, device=device)
        self.viol_steps = torch.zeros(n, device=device)
        self.steps = torch.zeros(n, device=device)
        # Per-attempt stage maxima (throw).
        self.att = {
            k: torch.zeros(n, dtype=torch.bool, device=device)
            for k in ("grasped", "lifted", "released", "valid_flight", "in_bucket")
        }
        self.min_bucket_dist = torch.full((n,), float("inf"), device=device)
        self.hh_steps = torch.zeros(n, device=device)
        self._snap: dict = {}
        self.term_names = list(self.u.termination_manager.active_terms)

    # ------------------------------------------------------------------
    def bootstrap(self, env_ids: torch.Tensor) -> None:
        self.active[env_ids] = True

    def snapshot(self) -> None:
        t = self.t
        t.update()
        self._snap = {
            "successes": t.successes.clone(),
            "goal_time": t.goal_time.clone(),
            "goal_side": t.goal_side.clone(),
            "goal_start_side": t.goal_start_side.clone(),
            "handoffs": t.handoffs.clone(),
            "direct_handoffs": t.direct_handoffs.clone(),
            "indirect_handoffs": t.indirect_handoffs.clone(),
            "pos_err": t.pos_err.clone(),
            "ang_err": t.ang_err.clone(),
            "lifted": t.lifted.clone(),
        }

    def _record_attempt(self, i: int, outcome: str, s: dict, goal_time: float) -> None:
        rec = {
            "env": i,
            "outcome": outcome,
            "goal_time_s": goal_time,
            "pos_err": float(s["pos_err"][i]),
            "ang_err": float(s["ang_err"][i]),
        }
        if self.task == "dual_reorient":
            rec["start_side"] = int(s["goal_start_side"][i])
            rec["goal_side"] = int(s["goal_side"][i])
        if self.task == "throw":
            for k, v in self.att.items():
                rec[k] = bool(v[i])
            rec["min_bucket_center_dist_xy"] = float(self.min_bucket_dist[i])
        self.goal_records.append(rec)

    def _reset_attempt(self, ids: torch.Tensor) -> None:
        for v in self.att.values():
            v[ids] = False
        self.min_bucket_dist[ids] = float("inf")

    def on_step(self, obs, rewards, dones, extras) -> None:
        t, s = self.t, self._snap
        done = dones.bool().to(self.device)
        act = self.active
        # Throw stage accumulation from the pre-step snapshot is too early; use the
        # tracker's post-step flags for non-reset envs and the snapshot otherwise.
        if self.task == "throw":
            post = {
                "grasped": t.grasped,
                "lifted": t.lifted,
                "released": t.released,
                "valid_flight": t._valid_flight,
                "in_bucket": t.in_bucket,
            }
            for k in self.att:
                self.att[k] |= post[k] & act & ~done
            d = torch.linalg.norm((t.obj.data.root_link_pos_w - t.bucket_pos)[:, :2], dim=-1)
            self.min_bucket_dist = torch.where(
                act & ~done & t.released,
                torch.minimum(self.min_bucket_dist, d),
                self.min_bucket_dist,
            )
        for ai, a in enumerate(self.actions):
            self.clip_lo_sum[ai] += (a.clip_lo & act.unsqueeze(-1)).float().sum(0)
            self.clip_hi_sum[ai] += (a.clip_hi & act.unsqueeze(-1)).float().sum(0)
            self.clip_sum += a.clip_fraction * act
            self.viol_steps += a.bound_violation.float() * act
        self.steps += act.float()
        if self.task == "dual_reorient":
            self.hh_steps += t.hand_hand.float() * act

        # Successes during this step.
        post_succ = torch.where(done, t.prev_episode_successes.long(), t.successes)
        new = (post_succ - s["successes"]).clamp(min=0) * act
        for i in new.nonzero().flatten().tolist():
            for _ in range(int(new[i])):
                self._record_attempt(i, "success", s, float(s["goal_time"][i]) + self.dt)
        if new.any():
            self._reset_attempt(new.nonzero().flatten())

        # Episode ends.
        ended = (done & act).nonzero().flatten().tolist()
        if ended:
            terms = {k: self.u.termination_manager.get_term(k) for k in self.term_names}
            for i in ended:
                reasons = [k for k, v in terms.items() if bool(v[i])]
                if "max_successes" in reasons:
                    outcome = None
                elif reasons == ["time_out"]:
                    outcome = "censored_episode_cap"
                elif "goal_timeout" in reasons:
                    outcome = "goal_timeout"
                elif any(r in reasons for r in ("object_dropped", "throw_failed", "hand_far")):
                    outcome = "failure_" + "+".join(r for r in reasons if r != "time_out")
                else:
                    outcome = "failure_" + "+".join(reasons)
                if outcome is not None:
                    self._record_attempt(i, outcome, s, float(s["goal_time"][i]) + self.dt)
                rec = {
                    "env": i,
                    "successes": int(post_succ[i]),
                    "reasons": reasons,
                    "steps": int(self.steps[i]),
                }
                if self.task == "dual_reorient":
                    rec["handoffs"] = int(s["handoffs"][i])
                    rec["direct_handoffs"] = int(s["direct_handoffs"][i])
                    rec["indirect_handoffs"] = int(s["indirect_handoffs"][i])
                    rec["hand_hand_steps"] = int(self.hh_steps[i])
                self.episode_records.append(rec)
            self.active[torch.tensor(ended, device=self.device)] = False

    def on_reset(self, done_mask: torch.Tensor) -> None:
        pass

    # ------------------------------------------------------------------
    def finalize(self) -> dict:
        g, e = self.goal_records, self.episode_records
        n_ep = len(e)
        succ = [r for r in g if r["outcome"] == "success"]
        cens = [r for r in g if r["outcome"] == "censored_episode_cap"]
        m: dict = {
            "task": self.task,
            "protocol": "first complete episode per env; censored goals counted as failures",
            "num_envs": self.n,
            "episodes_completed": n_ep,
            "episodes_incomplete": int(self.active.sum()),
            "goal_attempts": len(g),
            "goal_success": _rate(len(succ), len(g)),
            "goal_censored": len(cens),
            "goal_outcomes": {},
            "successes_per_episode_mean": (sum(r["successes"] for r in e) / n_ep)
            if n_ep
            else float("nan"),
            "episodes_with_ge1": _rate(sum(r["successes"] >= 1 for r in e), n_ep),
            "episodes_with_ge3": _rate(sum(r["successes"] >= 3 for r in e), n_ep),
            "drop_episodes": _rate(sum("object_dropped" in r["reasons"] for r in e), n_ep),
            "mean_time_to_success_s": (sum(r["goal_time_s"] for r in succ) / len(succ))
            if succ
            else float("nan"),
            "base_clip_fraction_mean": float(
                self.clip_sum.sum() / max(self.steps.sum(), 1) / max(len(self.actions), 1)
            ),
            "root_bound_violation_step_frac": float(
                self.viol_steps.sum() / max(self.steps.sum(), 1) / max(len(self.actions), 1)
            ),
        }
        tot = max(float(self.steps.sum()), 1.0)
        dims = ["base_x", "base_y", "base_z", "base_roll", "base_pitch", "base_yaw"]
        m["base_clip_per_dim"] = {
            a.cfg.entity_name: {
                d: {
                    "lo": round(float(self.clip_lo_sum[ai, k]) / tot, 4),
                    "hi": round(float(self.clip_hi_sum[ai, k]) / tot, 4),
                }
                for k, d in enumerate(dims)
            }
            for ai, a in enumerate(self.actions)
        }
        for r in g:
            m["goal_outcomes"][r["outcome"]] = m["goal_outcomes"].get(r["outcome"], 0) + 1
        fails = [r for r in g if r["outcome"] != "success"]
        if fails:
            m["failed_goal_pos_err_mean"] = sum(r["pos_err"] for r in fails) / len(fails)
            m["failed_goal_ang_err_mean"] = sum(r["ang_err"] for r in fails) / len(fails)
        if self.task == "throw":
            for k in ("grasped", "lifted", "released", "valid_flight"):
                m[f"attempt_{k}"] = _rate(
                    sum(r.get(k, False) or r["outcome"] == "success" for r in g), len(g)
                )
            m["landing_success"] = m["goal_success"]
            d = [
                r["min_bucket_center_dist_xy"]
                for r in g
                if math.isfinite(r["min_bucket_center_dist_xy"])
            ]
            m["released_min_bucket_center_dist_xy_mean"] = (sum(d) / len(d)) if d else float("nan")
        if self.task == "dual_reorient":

            def sub(f):
                rs = [r for r in g if f(r)]
                return _rate(sum(r["outcome"] == "success" for r in rs), len(rs))

            m["goal_success_left_to_right"] = sub(
                lambda r: r["start_side"] == 0 and r["goal_side"] == 1
            )
            m["goal_success_right_to_left"] = sub(
                lambda r: r["start_side"] == 1 and r["goal_side"] == 0
            )
            m["goal_success_same_side_left"] = sub(
                lambda r: r["start_side"] == 0 and r["goal_side"] == 0
            )
            m["goal_success_same_side_right"] = sub(
                lambda r: r["start_side"] == 1 and r["goal_side"] == 1
            )
            m["handoffs_per_episode_mean"] = (
                (sum(r["handoffs"] for r in e) / n_ep) if n_ep else float("nan")
            )
            m["direct_handoffs_per_episode_mean"] = (
                (sum(r["direct_handoffs"] for r in e) / n_ep) if n_ep else float("nan")
            )
            m["table_handovers_per_episode_mean"] = (
                (sum(r["indirect_handoffs"] for r in e) / n_ep) if n_ep else float("nan")
            )
            m["episodes_with_table_handover"] = _rate(
                sum(r["indirect_handoffs"] > 0 for r in e), n_ep
            )
            m["episodes_with_hand_hand_contact"] = _rate(
                sum(r["hand_hand_steps"] > 0 for r in e), n_ep
            )
            m["hand_hand_contact_step_frac"] = float(self.hh_steps.sum() / max(self.steps.sum(), 1))
        m["goal_records_head"] = g[:20]
        self._all = {"goals": g, "episodes": e}
        return m

    def report(self, metrics: dict) -> None:
        print(json.dumps({k: v for k, v in metrics.items() if k != "goal_records_head"}, indent=1))


register_evaluator(DexPBTTaskEvaluator)
