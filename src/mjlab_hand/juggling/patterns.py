"""Juggling for all five patterns (cascade, tennis, half-shower, shower, box) with one hand type.

Generalizes ``cascade.py`` (same hands, ball, root tracking, finger schedule, closed-loop catch, RL residual)
from the hard-coded 3-ball cascade to any siteswap pattern of ``schedule.py``:

* The pattern's throw events are unrolled on the control grid (10 ms). Per hand they become *skills*:
  catch of ball b at step c, dwell (the hand holds b), throw at step r to a landing point after a flight of F steps
  (``p_land`` = scheduled catch point of the receiving hand; release point includes the pattern's ``rise``),
  followed by the vacant segment to the next catch. Release velocity: v = (p_land - p_rel) / F + (0, 0, g F / 2),
  hand acceleration at release = (0, 0, -g) (paper Table S2).
* Catch: closed loop, the target is the ballistic prediction of the incoming ball at the scheduled catch step from its
  current state, replanned every control step until ``replan_cutoff_s`` before the catch (paper Sec. 2A); the hand
  arrives with ``kappa`` times the ball velocity (cushioned catch, deviation from the paper's zero velocity).
* The schedule is never delayed (wall clock), a skill that cannot be completed in time is missed.
* Policy: RL residual on the designed reference (22 dims per hand: 6 root + 16 finger), zero action = designed prior.
  One policy can be trained over all patterns (pattern one-hot in the observation) or per pattern.

Timing: cascade / tennis / half-shower use the paper's 0.35 s beat; shower 0.22 s, box 0.25 s (as in
``schedule.py``); lateral offsets in/out = -/+ ``lat`` around each hand's home (|y| = 0.131 m, paper Table S1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg

from mjlab_hand.juggling import schedule as S
from mjlab_hand.juggling.cascade import (
    CascadeCommand,
    CascadeCommandCfg,
    CascadeSpec,
    G,
    ResidualAction,
    ResidualActionCfg,
    ballistic,
    get_juggler,
    make_cascade_env_cfg,
    quintic_coeffs,
    quintic_eval,
)

BIG = 10**9
PATTERN_ORDER = ("cascade", "tennis", "half_shower", "shower", "box")


def pattern_def(name: str) -> S.Pattern:
    """Pattern with the paper's 0.35 s beat (cascade, tennis, half-shower) or a searched beat (shower, box)."""
    return {
        "cascade": lambda: S._cascade(0.35, 0.35),
        "tennis": lambda: S._tennis(0.35, 0.35),
        # paper Table S4: every half-shower step is 0.35 s (a 0.49 s dwell lets the hand dip faster than g: the ball leaves the cup)
        "half_shower": lambda: S._half_shower(0.35, 0.35, 0.35),
        # shower (siteswap 51), searched over (beat, right dwell, left dwell, rise): fast pass >= 0.14 s, apex <= 1.6 m, right-hand
        # lift <= 5 g, left hand >= 0.20 s to release and re-catch. Beat 0.28, dwells 0.12 / 0.28, rise 0.20: passes 0.16 s / 1.12 s.
        "shower": lambda: S._shower(0.28, 0.12, 0.28, 0.20),
        "box": lambda: S._box(0.25, 0.25),
    }[name]()


@dataclass
class PatternSpec(CascadeSpec):
    patterns: tuple[str, ...] = PATTERN_ORDER
    horizon_s: float = 34.0
    random_phase: bool = True


def _build_pattern_tables(
    p: S.Pattern, horizon_s: float, dt: float, cup_home: np.ndarray, outward: np.ndarray, lat: dict
):
    period = int(round(p.period_beats * p.beat_s / dt))
    lead = 3 * period
    u = S.unroll(p, horizon_s + lead * dt, lateral=lat)
    rs = np.round(u.t / dt).astype(np.int64) - lead
    Fs = np.round(u.flight / dt).astype(np.int64)
    Fsec = Fs * dt
    rel = cup_home[u.hand] + outward[u.hand] * u.release_lat[:, None]
    rel[:, 2] += u.extra["rise"]
    land = cup_home[u.to_hand] + outward[u.to_hand] * u.catch_lat[:, None]
    v = (land - rel) / Fsec[:, None]
    v[:, 2] = (land[:, 2] - rel[:, 2]) / Fsec + 0.5 * G * Fsec
    vin = v.copy()
    vin[:, 2] = v[:, 2] - G * Fsec
    hand_ev: list[list] = [[], []]
    balls = []
    for b in range(3):
        idx = np.where(u.ball == b)[0]
        idx = idx[np.argsort(rs[idx])]
        balls.append([(rs[i], u.hand[i], u.to_hand[i], Fs[i], rel[i], land[i], v[i]) for i in idx])
        for i, j in zip(idx[:-1], idx[1:]):
            assert u.hand[j] == u.to_hand[i], (p.name, b)
            hand_ev[u.hand[j]].append(
                (
                    rs[i] + Fs[i],
                    rs[j],
                    b,
                    u.to_hand[j],
                    Fs[j],
                    rel[j],
                    land[j],
                    land[i],
                    vin[i],
                    v[j],
                )
            )
    for h in (0, 1):
        hand_ev[h].sort(key=lambda e: e[1])
        for a, c in zip(hand_ev[h][:-1], hand_ev[h][1:]):
            assert c[0] >= a[1], f"{p.name} hand {h}: dwell overlap {a[:2]} {c[:2]}"
    return hand_ev, balls, period


@dataclass(kw_only=True)
class PatternCommandCfg(CascadeCommandCfg):
    spec: PatternSpec = field(default_factory=PatternSpec)

    def build(self, env):
        return PatternCommand(self, env)


def _g(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather along dim 1: t (N,K[,3]), idx (N,) -> (N[,3])."""
    if t.dim() == 2:
        return t.gather(1, idx.view(-1, 1)).squeeze(1)
    return t.gather(1, idx.view(-1, 1, 1).expand(-1, 1, t.shape[-1])).squeeze(1)


class PatternCommand(CascadeCommand):
    cfg: PatternCommandCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        sp, dev, dt = self.sp, self.device, self.dt
        self.pattern_names = tuple(sp.patterns)
        P = len(self.pattern_names)
        ch = self.cup_home.cpu().numpy().astype(np.float64)
        ow = self.outward.cpu().numpy().astype(np.float64)
        lat = {"in": -sp.lat, "out": sp.lat, "mid": 0.0}
        built = [
            _build_pattern_tables(pattern_def(nm), sp.horizon_s, dt, ch, ow, lat)
            for nm in self.pattern_names
        ]
        K = max(len(b[0][h]) for b in built for h in (0, 1))
        Kb = max(len(b[1][i]) for b in built for i in range(3))
        self.K, self.Kb = K, Kb
        L = lambda a, dtype=torch.long: torch.tensor(np.array(a), dtype=dtype, device=dev)  # noqa: E731
        ev_i = np.full((P, 2, K, 5), BIG, dtype=np.int64)  # c, r, ball, to, F
        ev_i[..., 2:] = 0
        ev_f = np.zeros((P, 2, K, 5, 3))  # prel, pland, pcatch, vin, vrel
        bt_i = np.full((P, 3, Kb, 4), BIG, dtype=np.int64)  # r, hand, to, F
        bt_i[..., 1:] = 0
        bt_f = np.zeros((P, 3, Kb, 3, 3))  # prel, pland, vrel
        periods = []
        for pi, (hev, balls, per) in enumerate(built):
            periods.append(per)
            for h in (0, 1):
                for k, e in enumerate(hev[h]):
                    ev_i[pi, h, k] = [e[0], e[1], e[2], e[3], e[4]]
                    ev_f[pi, h, k] = np.stack([e[5], e[6], e[7], e[8], e[9]])
            for b in range(3):
                for k, e in enumerate(balls[b]):
                    bt_i[pi, b, k] = [e[0], e[1], e[2], e[3]]
                    bt_f[pi, b, k] = np.stack([e[4], e[5], e[6]])
        self.ev_c, self.ev_r, self.ev_ball, self.ev_to, self.ev_F = (
            L(ev_i[..., i]) for i in range(5)
        )
        ff = lambda i: L(ev_f[..., i, :], torch.float32)  # noqa: E731
        self.ev_prel, self.ev_pland, self.ev_pcatch, self.ev_vin, self.ev_vrel = (
            ff(i) for i in range(5)
        )
        self.bt_r, self.bt_hand, self.bt_to, self.bt_F = (L(bt_i[..., i]) for i in range(4))
        self.bt_prel, self.bt_pland, self.bt_vrel = (
            L(bt_f[..., i, :], torch.float32) for i in range(3)
        )
        self.period_steps = L(periods)
        self.pat = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        self.pend_target = torch.zeros(self.num_envs, 3, 3, device=dev)
        self.pend_F = torch.zeros(self.num_envs, 3, dtype=torch.long, device=dev)
        self._bp = self._bv = None

    # ------------------------------------------------------------------ queries
    @property
    def command(self) -> torch.Tensor:
        return torch.nn.functional.one_hot(self.pat, len(self.pattern_names)).float()

    def cursor(self, h: int, now: torch.Tensor | None = None):
        """k = index of the first event of hand h whose release is after `now` (the current or next dwell)."""
        now = self.step_idx if now is None else now
        r_all = self.ev_r[self.pat, h]
        k = (r_all <= now.view(-1, 1)).sum(-1)
        return k, r_all

    def cur_event(self, h: int):
        now = self.step_idx
        k, r_all = self.cursor(h, now)
        kc = k.clamp(max=self.K - 1)
        c_k = _g(self.ev_c[self.pat, h], kc)
        r_k = _g(r_all, kc)
        prev_r = _g(r_all, (k - 1).clamp(min=0))
        dwell = now >= c_k
        tin = torch.where(dwell, now - c_k, now - prev_r).float() * self.dt
        return {
            "k": k,
            "kc": kc,
            "c": c_k,
            "r": r_k,
            "dwell": dwell,
            "tin": tin,
            "prev_r": prev_r,
            "ball": _g(self.ev_ball[self.pat, h], kc),
        }

    # ------------------------------------------------------------------ planning
    def _cur_ref(self, ids, h, now):
        return quintic_eval(
            self.coef[ids, h], (now[ids] - self.seg_start[ids, h]).float() * self.dt
        )

    def plan_throw_ev(self, ids, h, now, k):
        if ids.numel() == 0:
            return
        pat = self.pat[ids]
        kk = k[ids].clamp(max=self.K - 1)
        r_k = self.ev_r[pat, h, kk]
        p0, v0, a0 = self._cur_ref(ids, h, now)
        a1 = torch.zeros_like(v0)
        a1[:, 2] = -G
        T = ((r_k - now[ids]).float() * self.dt).clamp(min=self.dt)
        self.coef[ids, h] = quintic_coeffs(
            p0, v0, a0, self.ev_prel[pat, h, kk], self.ev_vrel[pat, h, kk], a1, T
        )
        self.seg_start[ids, h] = now[ids]
        self.u_cmd[ids, h] = self.ev_pland[pat, h, kk]
        self.x_state[ids, h] = torch.cat([self.catch_cmd[ids, h], self.ev_pland[pat, h, kk]], -1)

    def plan_catch_ev(self, ids, h, now, k):
        if ids.numel() == 0:
            return
        sp = self.sp
        pat = self.pat[ids]
        kk = k[ids].clamp(max=self.K - 1)
        c_k = self.ev_c[pat, h, kk]
        b = self.ev_ball[pat, h, kk]
        bp, bv = self._bp[ids, b], self._bv[ids, b]
        free = ~self.contact[ids][torch.arange(ids.numel(), device=self.device), :, b].any(-1)
        dtt = (c_k - now[ids]).float() * self.dt
        pc, vc = ballistic(bp, bv, dtt)
        pc = torch.where(free.unsqueeze(-1), pc, self.ev_pcatch[pat, h, kk])
        vc = torch.where(free.unsqueeze(-1), vc, self.ev_vin[pat, h, kk])
        box = torch.tensor(sp.catch_box, device=self.device)
        pc = torch.max(torch.min(pc, self.cup_home[h] + box), self.cup_home[h] - box)
        p0, v0, a0 = self._cur_ref(ids, h, now)
        self.coef[ids, h] = quintic_coeffs(
            p0, v0, a0, pc, sp.kappa * vc, torch.zeros_like(vc), dtt.clamp(min=self.dt)
        )
        self.seg_start[ids, h] = now[ids]
        self.catch_cmd[ids, h] = pc

    def init_plans_p(self, ids, t0):
        """Nominal (prior) segment of both hands at step t0 (steady state of the pattern)."""
        sp, dt = self.sp, self.dt
        m = ids.numel()
        pat = self.pat[ids]
        t = t0[ids]
        zero = torch.zeros(m, 3, device=self.device)
        a_rel = zero.clone()
        a_rel[:, 2] = -G
        for h in (0, 1):
            r_all = self.ev_r[pat, h]
            k = (r_all <= t.view(-1, 1)).sum(-1)
            kc, kp = k.clamp(max=self.K - 1), (k - 1).clamp(min=0)
            c_k, r_k, r_j = _g(self.ev_c[pat, h], kc), _g(r_all, kc), _g(r_all, kp)
            prel_k, pcatch_k, vin_k = (
                _g(x[pat, h], kc) for x in (self.ev_prel, self.ev_pcatch, self.ev_vin)
            )
            prel_j, vrel_j = _g(self.ev_prel[pat, h], kp), _g(self.ev_vrel[pat, h], kp)
            vrel_k = _g(self.ev_vrel[pat, h], kc)
            dwell = t >= c_k
            T_dw = ((r_k - c_k).float() * dt).clamp(min=dt)
            T_vac = ((c_k - r_j).float() * dt).clamp(min=dt)
            c_dw = quintic_coeffs(pcatch_k, sp.kappa * vin_k, zero, prel_k, vrel_k, a_rel, T_dw)
            c_vac = quintic_coeffs(prel_j, vrel_j, a_rel, pcatch_k, sp.kappa * vin_k, zero, T_vac)
            self.coef[ids, h] = torch.where(dwell.view(-1, 1, 1), c_dw, c_vac)
            self.seg_start[ids, h] = torch.where(dwell, c_k, r_j)
            self.u_cmd[ids, h] = _g(self.ev_pland[pat, h], kc)
            self.catch_cmd[ids, h] = pcatch_k

    def nominal_ball_state_p(self, ids, t0):
        dt = self.dt
        m = ids.numel()
        pat = self.pat[ids]
        t = t0[ids]
        rp, rv, _ = quintic_eval(
            self.coef[ids], (t.view(-1, 1) - self.seg_start[ids]).float() * dt
        )  # (m,2,3)
        pos = torch.zeros(m, 3, 3, device=self.device)
        vel = torch.zeros(m, 3, 3, device=self.device)
        for b in range(3):
            r_all = self.bt_r[pat, b]
            cnt = (r_all <= t.view(-1, 1)).sum(-1)
            last = (cnt - 1).clamp(min=0)
            r_l, to_l, F_l = (
                _g(r_all, last),
                _g(self.bt_to[pat, b], last),
                _g(self.bt_F[pat, b], last),
            )
            rel_l, v_l = _g(self.bt_prel[pat, b], last), _g(self.bt_vrel[pat, b], last)
            in_flight = (t - r_l) < F_l
            pf, vf = ballistic(rel_l, v_l, (t - r_l).float() * dt)
            ph = rp.gather(1, to_l.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
            vh = rv.gather(1, to_l.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
            pos[:, b] = torch.where(in_flight.unsqueeze(-1), pf, ph)
            vel[:, b] = torch.where(in_flight.unsqueeze(-1), vf, vh)
        return pos, vel

    def valid_start(self, pat: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """A start step is valid when no hand released a ball in the last start_steps steps."""
        ok = torch.ones_like(cand, dtype=torch.bool)
        for h in (0, 1):
            r_all = self.ev_r[pat, h]
            k = (r_all <= cand.view(-1, 1)).sum(-1)
            prev_r = _g(r_all, (k - 1).clamp(min=0))
            ok &= (k == 0) | ((cand - prev_r) >= self.sp.start_steps)
        return ok

    # ------------------------------------------------------------------ per-step update (after physics)
    def update(self):
        sp, n, dt = self.sp, self.num_envs, self.dt
        env = self._env
        live = ~self.just_reset
        self.just_reset[:] = False
        self.catch_now.zero_()
        self.land_err_now.zero_()
        self.throw_now.zero_()
        for hi, h in enumerate(self.cfg.hand_names):
            for bi, b in enumerate(self.cfg.ball_names):
                s = env.scene[f"{h}_{b}_contact"]
                self.contact[:, hi, bi] = (s.data.found.reshape(n, -1) > 0).any(-1)
        self.step_idx = torch.where(live, self.step_idx + 1, self.step_idx)
        self.ep_steps = torch.where(live, self.ep_steps + 1, self.ep_steps)
        self.last_ct = torch.where(self.contact, self.step_idx.view(-1, 1, 1), self.last_ct)
        now = self.step_idx
        p, v = self.ball_pos(), self.ball_vel()
        self._bp, self._bv = p, v
        # 1) landing measurement of pending throws
        free_b = ~self.contact.any(1)
        since = now.view(-1, 1) - self.pend_rel_step
        meas = self.pending_y & live.view(-1, 1) & free_b & (since >= 2)
        # a ball touched by the receiving hand before it was ever measured: take the measurement at the first contact
        meas = meas | (
            self.pending_y
            & live.view(-1, 1)
            & ~free_b
            & ~self.contact.gather(1, self.pend_hand.unsqueeze(1)).squeeze(1)
            & (since >= 2)
        )
        # not released: still touching the THROWING hand 8 steps after the release (a short pass, e.g. the shower's
        # 12-step "1", can already be touched by the receiving hand after 8 steps)
        thrower_ct = self.contact.gather(1, self.pend_hand.unsqueeze(1)).squeeze(1)
        late = self.pending_y & live.view(-1, 1) & thrower_ct & (since >= 8)
        if meas.any():
            e, b = meas.nonzero(as_tuple=True)
            t_land = self.pend_rel_step[e, b] + self.pend_F[e, b]
            y, _ = ballistic(p[e, b], v[e, b], (t_land - now[e]).float() * dt)
            tgt = self.pend_target[e, b]
            err = (y - tgt).norm(dim=-1)
            self.land_err_now.index_put_((e,), err, accumulate=True)
            self.metrics["land_err"][e] = err
            if self.record:
                h = self.pend_hand[e, b]
                self.log_land.append(
                    torch.cat(
                        [
                            e.float().unsqueeze(-1),
                            h.float().unsqueeze(-1),
                            y - tgt,
                            torch.zeros_like(y),
                        ],
                        -1,
                    ).cpu()
                )
            self.pending_y[e, b] = False
        if late.any():
            e, b = late.nonzero(as_tuple=True)
            self.failed[e] = True
            self.fail_code[e] = torch.where(
                self.fail_code[e] == 0, torch.full_like(self.fail_code[e], 4), self.fail_code[e]
            )
            self.pending_y[e, b] = False
        # 2) catch checks, check_steps after each scheduled catch
        cup = self.cup_pos()
        for h in (0, 1):
            m_ = (self.ev_c[self.pat, h] == (now - self.check_steps).view(-1, 1)) & live.view(-1, 1)
            ids = m_.any(-1).nonzero(as_tuple=True)[0]
            if ids.numel() == 0:
                continue
            idx = m_.float().argmax(-1)[ids]
            b = self.ev_ball[self.pat[ids], h, idx]
            near = (p[ids, b] - cup[ids, h]).norm(dim=-1) < 0.07
            ok = (now[ids] - self.last_ct[ids, h, b] <= 3) & near
            wrong = (now[ids] - self.last_ct[ids, 1 - h, b] <= 3) & ~ok
            self.caught[ids] += ok.float()
            self.catch_now[ids] += ok.float()
            bad = ~ok
            self.failed[ids] |= bad
            code = torch.where(wrong, torch.full_like(b, 3), torch.full_like(b, 1))
            self.fail_code[ids] = torch.where(
                bad & (self.fail_code[ids] == 0), code, self.fail_code[ids]
            )
        # 3) drops
        dropped = (p[..., 2] < self.cup_home[0, 2] - sp.drop_below).any(-1) & live
        self.fail_code = torch.where(
            dropped & (self.fail_code == 0), torch.full_like(self.fail_code, 2), self.fail_code
        )
        self.failed |= dropped
        # 4) planning
        for h in (0, 1):
            k, r_all = self.cursor(h, now)
            kc = k.clamp(max=self.K - 1)
            c_k = _g(self.ev_c[self.pat, h], kc)
            r_prev = _g(r_all, (k - 1).clamp(min=0))
            rel_now = live & (k > 0) & (r_prev == now)
            has_next = c_k < BIG // 2
            ids = rel_now.nonzero(as_tuple=True)[0]
            if ids.numel():
                j = k[ids] - 1
                pat = self.pat[ids]
                b = self.ev_ball[pat, h, j]
                self.pending_y[ids, b] = True
                self.pend_hand[ids, b] = h
                self.pend_rel_step[ids, b] = now[ids]
                self.pend_target[ids, b] = self.ev_pland[pat, h, j]
                self.pend_F[ids, b] = self.ev_F[pat, h, j]
                self.throws[ids] += 1.0
                self.throw_now[ids] = True
                go = has_next[ids] & ((c_k[ids] - now[ids]) >= 2)
                self.plan_catch_ev(ids[go], h, now, k)
            ids = (live & (c_k == now)).nonzero(as_tuple=True)[0]
            self.plan_throw_ev(ids, h, now, k)
            vac = live & ~rel_now & (now < c_k) & has_next & ((c_k - now) >= self.cut_steps)
            self.plan_catch_ev(vac.nonzero(as_tuple=True)[0], h, now, k)


# ---------------------------------------------------------------------- reset
def reset_pattern(env, env_ids, command_name: str = "juggler"):
    t: PatternCommand = get_juggler(env, command_name)
    sp = t.sp
    n = len(env_ids)
    if n == 0:
        return
    dev = env.device
    P = len(t.pattern_names)
    pat = torch.randint(0, P, (n,), device=dev)
    t.pat[env_ids] = pat
    if sp.random_phase:
        start = (torch.rand(n, device=dev) * t.period_steps[pat]).long()
        for _ in range(12):
            bad = ~t.valid_start(pat, start)
            if not bad.any():
                break
            start = torch.where(
                bad, (torch.rand(n, device=dev) * t.period_steps[pat]).long(), start
            )
    else:
        start = torch.full((n,), -1, dtype=torch.long, device=dev)
        for d_ in range(400):
            cand = torch.full((n,), sp.start_steps + d_, dtype=torch.long, device=dev)
            pick = (start < 0) & t.valid_start(pat, cand)
            start = torch.where(pick, cand, start)
        assert bool((start >= 0).all())
    t.step_idx[env_ids] = start
    t.ep_steps[env_ids] = 0
    t.caught[env_ids] = 0.0
    t.throws[env_ids] = 0.0
    t.failed[env_ids] = False
    t.fail_code[env_ids] = 0
    t.pending_y[env_ids] = False
    t.last_ct[env_ids] = -1000
    t.just_reset[env_ids] = True
    t.init_plans_p(env_ids, t.step_idx)
    rp, rv, _ = t.ref_at(t.step_idx)
    for hi, h in enumerate(t.hands):
        dq = torch.einsum("ij,nj->ni", t.slide_jinv[hi], rp[env_ids, hi] - t.cup_home[hi])
        dv = torch.einsum("ij,nj->ni", t.slide_jinv[hi], rv[env_ids, hi])
        h.write_joint_state_to_sim(dq, dv, joint_ids=t.slide_ids[hi], env_ids=env_ids)
    bp, bv = t.nominal_ball_state_p(env_ids, t.step_idx)
    o = env.scene.env_origins[env_ids]
    q = torch.zeros(n, 4, device=dev)
    q[:, 0] = 1.0
    for b, ball in enumerate(t.balls):
        vel = torch.zeros(n, 6, device=dev)
        vel[:, :3] = bv[:, b]
        ball.write_root_link_pose_to_sim(torch.cat([bp[:, b] + o, q], -1), env_ids=env_ids)
        ball.write_root_link_velocity_to_sim(vel, env_ids=env_ids)


# ---------------------------------------------------------------------- action (finger schedule per event)
@dataclass(kw_only=True)
class PatternActionCfg(ResidualActionCfg):
    def build(self, env):
        return PatternAction(self, env)


class PatternAction(ResidualAction):
    def _prior_fingers(self, t, h, frac):
        sp = t.sp
        if (
            self._poses is None
        ):  # the hand's finger poses (Allegro: finger_pose; others: measured, juggle3.json)
            self._poses = {k: self._pose(k) for k in {"cup", sp.hold_pose, sp.release_pose}}
        hold, rel = self._poses[sp.hold_pose], self._poses[sp.release_pose]
        ev = t.cur_event(h)
        tin = ev["tin"] + frac * t.dt
        to_rel = (ev["r"] - t.step_idx).float() * t.dt - frac * t.dt
        n = tin.shape[0]
        in_dwell = torch.where(
            (to_rel <= sp.release_lead_s).unsqueeze(-1), rel.expand(n, -1), hold.expand(n, -1)
        )
        in_vac = torch.where(
            (tin < sp.release_hold_s).unsqueeze(-1), rel.expand(n, -1), hold.expand(n, -1)
        )
        return torch.where(ev["dwell"].unsqueeze(-1), in_dwell, in_vac)


# ---------------------------------------------------------------------- observations
def obs_phase_pattern(env, command_name="juggler"):
    t = get_juggler(env, command_name)
    now = t.step_idx
    out = [t.command]
    for h in (0, 1):
        ev = t.cur_event(h)
        out += [
            ev["dwell"].float().unsqueeze(-1),
            ((ev["r"] - now).float() * t.dt).clamp(0.0, 1.5).unsqueeze(-1) / 1.5,
            ((ev["c"] - now).float() * t.dt).clamp(-1.0, 1.5).unsqueeze(-1) / 1.5,
            torch.nn.functional.one_hot(ev["ball"], 3).float(),
            t.catch_cmd[:, h] - t.cup_home[h],
            t.u_cmd[:, h] - t.cup_home[h],
        ]
    return torch.cat(out, -1)


# ---------------------------------------------------------------------- env config
def make_pattern_env_cfg(sp: PatternSpec, play: bool = False):
    cfg = make_cascade_env_cfg(sp, play=play)
    spec = cfg.commands["juggler"].spec  # deep copy of sp (play: random_phase off), a PatternSpec
    spec.horizon_s = max(sp.horizon_s, (30.0 if play else sp.episode_s) + 4.0)
    cascade_cmd = cfg.commands["juggler"]
    # keep the hand-specific fields of the cascade config (palm body, measured cup point, finger targets / poses)
    cfg.commands = {
        "juggler": PatternCommandCfg(
            spec=spec, palm_body=cascade_cmd.palm_body, cup_rel_palm=cascade_cmd.cup_rel_palm
        )
    }
    cfg.actions = {
        h: PatternActionCfg(
            entity_name=h,
            hand_index=a.hand_index,
            finger_names=a.finger_names,
            tendon_names=a.tendon_names,
            held_joint_pos=a.held_joint_pos,
            poses=a.poses,
        )
        for h, a in cfg.actions.items()
    }
    for grp in cfg.observations.values():
        grp.terms["phase"] = ObservationTermCfg(func=obs_phase_pattern)
    cfg.events["reset_cascade"] = EventTermCfg(func=reset_pattern, mode="reset")
    return cfg
