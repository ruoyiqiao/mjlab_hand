"""Juggling task state, rewards, terminations, observations and resets.

:class:`JuggleCommand` tracks per ball (from hand-ball contact sensors and ball
state, once per env step on post-physics data):

* ``held``: in hand contact continuously for >= ``hold_debounce_s``;
* throw event: held -> no hand contact for ``free_debounce_s`` with upward speed
  >= ``throw_min_vz``;
* catch event: a ball airborne for >= ``min_flight_s`` becomes held again;
* ``juggle_time``: time with both balls above the drop height (paper reward).

Pattern clock: throws are scheduled every ``half_cycle_s`` (alternating balls /
columns), exposed as sin/cos phase in observations (closed-loop policy only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


@dataclass(kw_only=True)
class JuggleCommandCfg(CommandTermCfg):
    hand_name: str = "hand"
    ball_names: tuple[str, ...] = ("ball_a", "ball_b")
    ball_radius: float = 0.025
    column_offset: tuple[float, float, float] = (0.0, 0.12, 0.0)
    """World offset (m) of column B from column A (A = palm cup at home)."""
    drop_height_below_cup: float = 0.15
    """Drop line = cup height - this (paper: balls >= 0.6 m above the floor)."""
    max_horizontal_dist: float = 0.4
    hold_debounce_s: float = 0.03
    free_debounce_s: float = 0.02
    throw_min_vz: float = 0.5
    min_flight_s: float = 0.15
    half_cycle_s: float = 0.30
    # Start protocol (paper: ball 1 in the hand, ball 2 released from above).
    second_ball_drop_height: float = 0.30
    """Height (m) of ball B above column B's cup point at reset (zero velocity)."""
    second_ball_init_vz: float = 0.0
    """Initial upward speed of ball B (m/s): reference-state start on the steady
    parabola (deviation from the paper's launcher, which releases at rest)."""
    second_ball_present: bool = True
    ref_tau_range: tuple[float, float] | None = None
    """Reference-state init: if set, ball B starts on a parabola that lands on column B
    after tau ~ U(range) seconds with downward speed ref_land_speed (overrides the
    drop height / init vz)."""
    ref_land_speed: float = 1.96
    max_hold_s: float = 0.5
    """A ball in continuous hand contact longer than this = not juggling (failure).
    Deviation from the paper (open-loop policy could not exploit holding)."""
    initial_hold_grace_s: float = 0.6
    """Hold limit for ball A at the episode start (it starts in the hand)."""
    max_both_held_s: float = 0.15
    target_apex: float = 0.5
    min_throw_apex: float = 0.25
    """A throw counts only if the predicted apex at release is >= this above the cup."""
    min_flight_rise: float = 0.20
    """A catch counts only if the ball rose >= this above the cup during the flight."""
    airborne_prob: float = 0.0
    """Training-only reference-state init: with this probability both balls start on
    the steady two-column pattern (ball A thrown from column A u s ago, ball B thrown
    from column B u + T s ago, u ~ U(0, V)), exact projectile states."""
    throw_speed: float = 3.19
    vacant_s: float = 0.25
    max_cycle_s: float = 1.2
    entry_deadline_s: float | None = None
    alive_recent_throw_s: float | None = None
    max_throw_apex: float | None = None
    cap_actual_apex: bool = False
    """Training-only: judge the cap on the ball's actual flight height, not only the predicted apex at release
    (loophole: a gentle valid release, then the rising hand re-hits the ball in flight). An overshooting flight
    pays too_high once, stops the alive reward and its catch does not count."""
    """Training-only: releases whose predicted apex exceeds this are not valid throws
    (stall timer keeps running, a later catch does not count). None = no upper limit (eval)."""
    """Training-only: the alive reward is paid only while every active ball was validly
    thrown (or started thrown) within this time (steady pattern: every 2T = 0.8 s)."""
    """Training-only: from the paper start, ball A must be validly thrown within this
    time, else the episode stalls (None = off; evaluation uses the 1.2 s stall only)."""
    """Stall: an active ball without a valid throw for this long ends the episode."""
    """Target apex height (m) above the cup for throw-quality shaping."""
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    debug_vis: bool = False
    palm_body: str = "palm"
    cup_rel_palm: tuple[float, float, float] | None = None
    """Ball-centre rest point in the cup minus the palm-body origin (world frame, home pose), measured by
    tools/allegro4/juggle_cup_probe.py (hand_params/juggle.json). None = Allegro: palm_collision box face."""

    def build(self, env: ManagerBasedRlEnv) -> JuggleCommand:
        return JuggleCommand(self, env)


class JuggleCommand(CommandTerm):
    cfg: JuggleCommandCfg

    def __init__(self, cfg: JuggleCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        self.hand = env.scene[cfg.hand_name]
        self.balls = [env.scene[b] for b in cfg.ball_names]
        self.nb = len(self.balls)
        pid, _ = self.hand.find_bodies(cfg.palm_body)
        self.palm_id = pid[0]
        self.cup_local, self.palm_home_local = self._compute_cup_point(env)  # env-local (3,)
        self.cup_offset = self.cup_local - self.palm_home_local
        self.col_local = torch.stack(
            [self.cup_local, self.cup_local + torch.tensor(cfg.column_offset, device=dev)]
        )
        z = (n, self.nb)
        self.contact = torch.zeros(z, dtype=torch.bool, device=dev)
        self.contact_t = torch.zeros(z, device=dev)
        self.free_t = torch.zeros(z, device=dev)
        self.held = torch.zeros(z, dtype=torch.bool, device=dev)
        self.flying = torch.zeros(z, dtype=torch.bool, device=dev)
        self.flight_t = torch.zeros(z, device=dev)
        self.apex = torch.zeros(z, device=dev)
        self.throws = torch.zeros(n, dtype=torch.long, device=dev)
        self.catches = torch.zeros(n, dtype=torch.long, device=dev)
        self.throw_now = torch.zeros(z, dtype=torch.bool, device=dev)
        self.release_now = torch.zeros(z, dtype=torch.bool, device=dev)
        self.overshot = torch.zeros(z, dtype=torch.bool, device=dev)
        self.overshoot_now = torch.zeros(z, dtype=torch.bool, device=dev)
        self.released = torch.zeros(z, dtype=torch.bool, device=dev)
        self.catch_now = torch.zeros(z, dtype=torch.bool, device=dev)
        self.juggling = torch.zeros(n, dtype=torch.bool, device=dev)
        self.alive = torch.zeros(n, dtype=torch.bool, device=dev)
        self.juggle_time = torch.zeros(n, device=dev)
        self.ep_time = torch.zeros(n, device=dev)
        self.dropped = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ball_ball = torch.zeros(n, dtype=torch.bool, device=dev)
        self.both_held_t = torch.zeros(n, device=dev)
        self.since_throw = torch.zeros(z, device=dev)
        self.thrown = torch.zeros(z, dtype=torch.bool, device=dev)
        self.stalled = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ever_thrown = torch.zeros(z, dtype=torch.bool, device=dev)
        self.lost_hold = torch.zeros(z, dtype=torch.bool, device=dev)
        self.held_too_long = torch.zeros(n, dtype=torch.bool, device=dev)
        self.prev_juggle_time = torch.zeros(n, device=dev)
        self.prev_catches = torch.zeros(n, device=dev)
        self._last_step = -1
        for k in ("catches", "throws", "juggle_time", "max_apex", "dropped"):
            self.metrics[k] = torch.zeros(n, device=dev)

    def _compute_cup_point(self, env) -> torch.Tensor:
        """Ball-on-palm point (env-local) at the hand's home pose, CPU kinematics."""
        model = env.sim.mj_model
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        f = lambda x: torch.tensor(x, dtype=torch.float32, device=self.device)  # noqa: E731
        if self.cfg.cup_rel_palm is not None:
            pid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"{self.cfg.hand_name}/{self.cfg.palm_body}"
            )
            assert pid >= 0, self.cfg.palm_body
            palm = data.xpos[pid].copy()
            return f(palm + np.asarray(self.cfg.cup_rel_palm)), f(palm)
        gname = f"{self.cfg.hand_name}/palm_collision"
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        assert gid >= 0, gname
        c = data.geom_xpos[gid].copy()
        R = data.geom_xmat[gid].reshape(3, 3)
        half = model.geom_size[gid]
        normal = R[:, 0]  # palm grasp side = body +x (verified: fingers flex to +x)
        if normal[2] < 0.5:
            raise ValueError(f"palm must face up for juggling, normal={normal}")
        cup = c + normal * (half[0] + self.cfg.ball_radius + 0.001)
        pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{self.cfg.hand_name}/palm")
        palm = data.xpos[pid].copy()
        return f(cup), f(palm)

    @property
    def command(self) -> torch.Tensor:
        return self.col_local.reshape(-1).expand(self.num_envs, -1)

    @property
    def num_active(self) -> int:
        return self.nb if (self.nb == 1 or self.cfg.second_ball_present) else 1

    @property
    def step_dt(self) -> float:
        return float(self._env.step_dt)

    def ball_pos(self) -> torch.Tensor:
        return torch.stack([b.data.root_link_pos_w for b in self.balls], dim=1)

    def ball_vel(self) -> torch.Tensor:
        return torch.stack([b.data.root_link_lin_vel_w for b in self.balls], dim=1)

    def palm_pos(self) -> torch.Tensor:
        return self.hand.data.body_link_pos_w[:, self.palm_id]

    def cup_pos(self) -> torch.Tensor:
        """Current cup point (world), assuming small palm rotation."""
        return self.palm_pos() + self.cup_offset

    def predicted_landing(self) -> torch.Tensor:
        """(N, nb, 2) xy where each ball crosses the cup height descending."""
        p, v = self.ball_pos(), self.ball_vel()
        zc = self.cup_pos()[:, 2].unsqueeze(-1)
        disc = (v[..., 2] ** 2 + 2 * 9.81 * (p[..., 2] - zc)).clamp(min=0)
        tt = (v[..., 2] + torch.sqrt(disc)) / 9.81
        return p[..., :2] + v[..., :2] * tt.unsqueeze(-1)

    def drop_line(self) -> torch.Tensor:
        return (
            self._env.scene.env_origins[:, 2] + self.cup_local[2] - self.cfg.drop_height_below_cup
        )

    def phase(self) -> torch.Tensor:
        return (self.ep_time / (2 * self.cfg.half_cycle_s)) % 1.0

    # ------------------------------------------------------------------
    def update(self) -> None:
        step = int(self._env.common_step_counter)
        if step == self._last_step:
            return
        self._last_step = step
        cfg, dt = self.cfg, self.step_dt
        for k, b in enumerate(cfg.ball_names):
            s = self._env.scene[f"{cfg.hand_name}_{b}_contact"]
            self.contact[:, k] = (s.data.found.reshape(self.num_envs, -1) > 0).any(-1)
        p, v = self.ball_pos(), self.ball_vel()
        self.contact_t = torch.where(
            self.contact, self.contact_t + dt, torch.zeros_like(self.contact_t)
        )
        self.free_t = torch.where(self.contact, torch.zeros_like(self.free_t), self.free_t + dt)
        held_prev = self.held.clone()
        # Throw: was held, now free for the debounce, moving up, and the ballistic apex
        # is high enough (no micro-hops inside the cup).
        cup_z = self.cup_pos()[:, 2].unsqueeze(-1)
        pred_apex = p[..., 2] + v[..., 2].clamp(min=0) ** 2 / (2 * 9.81) - cup_z
        # Any upward release (valid or not); used by the too-high penalty so that releases made
        # invalid by max_throw_apex are still penalised (a10 bug: they were not).
        self.release_now = (
            held_prev
            & (self.free_t >= cfg.free_debounce_s)
            & (v[..., 2] >= cfg.throw_min_vz)
            & ~self.flying
            & ~self.released
        )
        self.released = (self.released | self.release_now) & ~(
            self.contact_t >= cfg.hold_debounce_s
        )
        self.throw_now = (
            held_prev
            & (self.free_t >= cfg.free_debounce_s)
            & (v[..., 2] >= cfg.throw_min_vz)
            & (pred_apex >= cfg.min_throw_apex)
            & ~self.flying
        )
        if cfg.max_throw_apex is not None:
            self.throw_now &= pred_apex <= cfg.max_throw_apex
        new_held = self.contact_t >= cfg.hold_debounce_s
        # Catch: after a valid throw, the ball really rose, and is held again.
        rose = (self.apex - cup_z) >= cfg.min_flight_rise
        caught = self.flying & (self.flight_t >= cfg.min_flight_s) & rose & new_held
        # Only balls thrown by the hand count (the initial in-flight ball B is free).
        self.catch_now = caught & self.thrown
        if cfg.cap_actual_apex:
            self.catch_now &= ~self.overshot
        self.thrown = (self.thrown | self.throw_now) & ~caught
        self.flying = (self.flying | self.throw_now) & ~new_held
        self.flight_t = torch.where(
            self.flying, self.flight_t + dt, torch.zeros_like(self.flight_t)
        )
        self.apex = torch.where(
            self.flying, torch.maximum(self.apex, p[..., 2]), torch.zeros_like(self.apex)
        )
        if cfg.cap_actual_apex and cfg.max_throw_apex is not None:
            over = self.flying & ((p[..., 2] - cup_z) > cfg.max_throw_apex)
            self.overshoot_now = over & ~self.overshot
            self.overshot = (self.overshot | over) & ~new_held
        self.held = new_held | (held_prev & ~(self.free_t >= cfg.free_debounce_s))
        # Held -> free without a valid throw (fell off / dropped away / batted).
        self.lost_hold = held_prev & ~self.held & ~self.throw_now
        self.ever_thrown |= self.throw_now
        self.since_throw = torch.where(
            self.throw_now, torch.zeros_like(self.since_throw), self.since_throw + dt
        )
        self.throws += self.throw_now.sum(-1)
        self.catches += self.catch_now.sum(-1)
        line = self.drop_line()
        na = self.num_active
        pa = p[:, :na]
        above = (pa[..., 2] >= line.unsqueeze(-1)).all(-1)
        horiz = torch.linalg.norm((pa - self.palm_pos().unsqueeze(1))[..., :2], dim=-1)
        self.dropped = ~above | (horiz > cfg.max_horizontal_dist).any(-1)
        na_contact_t = self.contact_t[:, :na]
        limit = torch.where(
            self.ep_time < cfg.initial_hold_grace_s,
            torch.full_like(self.ep_time, cfg.initial_hold_grace_s),
            torch.full_like(self.ep_time, cfg.max_hold_s),
        )
        both = self.contact[:, :na].all(-1) if na > 1 else torch.zeros_like(self.dropped)
        self.both_held_t = torch.where(
            both, self.both_held_t + dt, torch.zeros_like(self.both_held_t)
        )
        self.held_too_long = (na_contact_t > limit.unsqueeze(-1)).any(-1) | (
            self.both_held_t > cfg.max_both_held_s
        )
        self.stalled = (self.since_throw[:, :na] > cfg.max_cycle_s).any(-1)
        if cfg.entry_deadline_s is not None:
            self.stalled |= ~self.ever_thrown[:, 0] & (self.ep_time > cfg.entry_deadline_s)
        self.juggling = above & ~self.held_too_long & ~self.stalled
        self.alive = self.juggling
        if cfg.alive_recent_throw_s is not None:
            self.alive = self.alive & (self.since_throw[:, :na] <= cfg.alive_recent_throw_s).all(-1)
        if cfg.cap_actual_apex:
            self.alive = self.alive & ~self.overshot[:, :na].any(-1)
        self.juggle_time += dt * self.juggling.float()
        self.ep_time += dt
        if self.num_active > 1:
            s = self._env.scene["ball_ball_contact"]
            self.ball_ball = (s.data.found.reshape(self.num_envs, -1) > 0).any(-1)

    # ------------------------------------------------------------------
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        pass

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        for t in (
            self.contact_t,
            self.free_t,
            self.flight_t,
            self.apex,
            self.juggle_time,
            self.ep_time,
        ):
            t[env_ids] = 0.0
        for t in (
            self.contact,
            self.held,
            self.flying,
            self.throw_now,
            self.catch_now,
            self.dropped,
            self.ball_ball,
            self.release_now,
            self.released,
            self.overshot,
            self.overshoot_now,
        ):
            t[env_ids] = False
        self.throws[env_ids] = 0
        self.catches[env_ids] = 0
        self.both_held_t[env_ids] = 0.0
        self.since_throw[env_ids] = 0.0
        self.thrown[env_ids] = False
        self.ever_thrown[env_ids] = False
        self.lost_hold[env_ids] = False
        self.stalled[env_ids] = False
        self.held_too_long[env_ids] = False
        # Ball A starts resting in the palm: count it as held from the start
        # (unless the reference-state init put it in the air, see reset_balls).
        air = getattr(self, "_airborne_init", None)
        a_air = (
            air[env_ids]
            if air is not None
            else torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        )
        self.held[env_ids, 0] = ~a_air
        if self.num_active > 1:
            idx = env_ids[a_air]
            self.flying[idx] = True
            self.thrown[idx] = True
            self.ever_thrown[idx] = True
            self.flight_t[idx] = self.cfg.min_flight_s
            cup_z = self._env.scene.env_origins[idx, 2] + self.cup_local[2]
            self.apex[idx] = (cup_z + self.cfg.target_apex).unsqueeze(-1).expand(-1, self.nb)
        # Ball B starts in the air (paper launcher / reference state): its first catch counts.
        if self.num_active > 1:
            self.flying[env_ids, 1] = True
            self.flight_t[env_ids, 1] = self.cfg.min_flight_s
        return super().reset(env_ids)

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    def _update_metrics(self) -> None:
        m = self.metrics
        m["catches"] = self.catches.float()
        m["throws"] = self.throws.float()
        m["juggle_time"] = self.juggle_time.clone()
        m["max_apex"] = self.apex.amax(-1) - self._env.scene.env_origins[:, 2] - self.cup_local[2]
        m["dropped"] = self.dropped.float()

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        for b in visualizer.get_env_indices(self.num_envs):
            o = self._env.scene.env_origins[b]
            for c in range(2):
                visualizer.add_sphere(
                    center=(self.col_local[c] + o).cpu().numpy(),
                    radius=0.008,
                    color=(0.2, 0.9, 0.2, 0.5),
                    label=f"col{c}_{b}",
                )


def get_juggle(env, command_name: str = "juggle") -> JuggleCommand:
    t = env.command_manager.get_term(command_name)
    t.update()
    return t


# ----------------------------------------------------------------------------
# Rewards
# ----------------------------------------------------------------------------
def juggling_alive(env, command_name: str = "juggle"):
    """Paper reward: 1 while every ball is above the drop line."""
    return get_juggle(env, command_name).alive.float()


def catch_event(env, command_name: str = "juggle"):
    return get_juggle(env, command_name).catch_now.float().sum(-1)


def throw_event(env, command_name: str = "juggle"):
    return get_juggle(env, command_name).throw_now.float().sum(-1)


def lost_hold(env, command_name: str = "juggle"):
    """Count of active balls that went held -> free without a valid throw."""
    t = get_juggle(env, command_name)
    return t.lost_hold[:, : t.num_active].float().sum(-1)


def ball_ball_contact(env, command_name: str = "juggle"):
    return get_juggle(env, command_name).ball_ball.float()


def joint_vel_l1(env, asset_cfg: SceneEntityCfg):
    a = env.scene[asset_cfg.name]
    return a.data.joint_vel[:, asset_cfg.joint_ids].abs().sum(-1)


def throw_quality(
    env, command_name: str = "juggle", apex_sigma: float = 0.1, xy_sigma: float = 0.03
):
    """At each throw event: exp(-|apex - target| / s) * exp(-|landing - other column| / s).
    Throws from column k should land on column 1 - k (two columns, one hand)."""
    t = get_juggle(env, command_name)
    if not t.throw_now.any():
        return torch.zeros(env.num_envs, device=env.device)
    p, v = t.ball_pos(), t.ball_vel()
    apex = p[..., 2] + v[..., 2].clamp(min=0) ** 2 / (2 * 9.81) - t.cup_pos()[:, 2].unsqueeze(-1)
    land = t.predicted_landing()
    cols = t.col_local[:, :2].unsqueeze(0) + env.scene.env_origins[:, :2].unsqueeze(1)  # (N,2,2)
    # thrown from the column nearest to the cup now -> target the other column
    cup_xy = t.cup_pos()[:, :2]
    near = torch.linalg.norm(cols - cup_xy.unsqueeze(1), dim=-1).argmin(-1)  # (N,)
    target = (
        cols[torch.arange(env.num_envs, device=env.device), 1 - near]
        if t.num_active > 1
        else cols[:, 0]
    )
    q = torch.exp(-(apex - t.cfg.target_apex).abs() / apex_sigma) * torch.exp(
        -torch.linalg.norm(land - target.unsqueeze(1), dim=-1) / xy_sigma
    )
    return (q * t.throw_now.float()).sum(-1)


def throw_too_high(env, command_name: str = "juggle", factor: float = 1.6):
    """Throws whose apex exceeds factor * target_apex (flight too long for the rhythm)."""
    t = get_juggle(env, command_name)
    if t.cfg.cap_actual_apex:
        return t.overshoot_now[:, : t.num_active].float().sum(-1)
    ev = t.release_now | t.throw_now
    if not ev.any():
        return torch.zeros(env.num_envs, device=env.device)
    p, v = t.ball_pos(), t.ball_vel()
    apex = p[..., 2] + v[..., 2].clamp(min=0) ** 2 / (2 * 9.81) - t.cup_pos()[:, 2].unsqueeze(-1)
    return ((apex > factor * t.cfg.target_apex) & ev).float().sum(-1)


def catch_alignment(env, command_name: str = "juggle", sigma: float = 0.03):
    """Per step: for each airborne descending ball, exp(-(|landing_xy - cup_xy| / s)^2)."""
    t = get_juggle(env, command_name)
    d = torch.linalg.norm(t.predicted_landing() - t.cup_pos()[:, :2].unsqueeze(1), dim=-1)
    falling = t.flying & (t.ball_vel()[..., 2] < 0)
    return (torch.exp(-((d / sigma) ** 2)) * falling.float())[:, : t.num_active].sum(-1)


# ----------------------------------------------------------------------------
# Terminations
# ----------------------------------------------------------------------------
def ball_dropped(env, command_name: str = "juggle"):
    return get_juggle(env, command_name).dropped


def failure(env, command_name: str = "juggle"):
    """1 on the step the episode fails (drop / stall / held too long)."""
    t = get_juggle(env, command_name)
    return (t.dropped | t.stalled | t.held_too_long).float()


def stall_event(env, command_name: str = "juggle"):
    """1 on the step the episode stalls (no valid throw of some ball for max_cycle_s)."""
    return get_juggle(env, command_name).stalled.float()


def held_too_long(env, command_name: str = "juggle"):
    return get_juggle(env, command_name).held_too_long


def stalled(env, command_name: str = "juggle"):
    return get_juggle(env, command_name).stalled


# ----------------------------------------------------------------------------
# Observations
# ----------------------------------------------------------------------------
def palm_state(env, command_name: str = "juggle"):
    t = get_juggle(env, command_name)
    h = t.hand
    pos = t.palm_pos() - env.scene.env_origins
    return torch.cat(
        [
            pos,
            h.data.body_link_quat_w[:, t.palm_id],
            h.data.body_link_lin_vel_w[:, t.palm_id],
            h.data.body_link_ang_vel_w[:, t.palm_id],
        ],
        -1,
    )


def balls_rel_palm(env, command_name: str = "juggle"):
    """Per ball: position rel. palm (3), linear velocity (3), contact flag, flight time."""
    t = get_juggle(env, command_name)
    rel = t.ball_pos() - t.palm_pos().unsqueeze(1)
    feats = torch.cat(
        [
            rel,
            t.ball_vel(),
            t.contact.float().unsqueeze(-1),
            t.flight_t.unsqueeze(-1),
            t.contact_t.unsqueeze(-1),
        ],
        -1,
    )
    return feats.reshape(env.num_envs, -1)


def columns_rel_palm(env, command_name: str = "juggle"):
    t = get_juggle(env, command_name)
    cols = t.col_local.unsqueeze(0) + env.scene.env_origins.unsqueeze(1)
    return (cols - t.palm_pos().unsqueeze(1)).reshape(env.num_envs, -1)


def pattern_clock(env, command_name: str = "juggle"):
    ph = get_juggle(env, command_name).phase() * 2 * np.pi
    return torch.stack([torch.sin(ph), torch.cos(ph)], -1)


# ----------------------------------------------------------------------------
# Reset event
# ----------------------------------------------------------------------------
def reset_balls(
    env,
    env_ids: torch.Tensor,
    command_name: str = "juggle",
    ball_b_xy_noise: float = 0.0,
    ball_b_height_noise: float = 0.0,
):
    """Paper start: ball A resting on the palm cup (column A); ball B released at
    rest ``second_ball_drop_height`` above column B (paper: ball launcher)."""
    t: JuggleCommand = env.command_manager.get_term(command_name)
    n = len(env_ids)
    if n == 0:
        return
    if not hasattr(t, "_airborne_init"):
        t._airborne_init = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    air = (torch.rand(n, device=env.device) < t.cfg.airborne_prob) & torch.tensor(
        t.num_active > 1, device=env.device
    )
    t._airborne_init[env_ids] = air
    o = env.scene.env_origins[env_ids]
    q = torch.zeros(n, 4, device=env.device)
    q[:, 0] = 1.0
    zero = torch.zeros(n, 6, device=env.device)
    pa = o + t.col_local[0]
    t.balls[0].write_root_link_pose_to_sim(torch.cat([pa, q], -1), env_ids=env_ids)
    t.balls[0].write_root_link_velocity_to_sim(zero, env_ids=env_ids)
    if t.nb > 1:
        pb = o + t.col_local[1]
        pb = pb.clone()
        vz_b = torch.full((n,), t.cfg.second_ball_init_vz, device=env.device)
        if t.cfg.ref_tau_range is not None:
            lo, hi = t.cfg.ref_tau_range
            tau = lo + torch.rand(n, device=env.device) * (hi - lo)
            v = t.cfg.ref_land_speed
            g = 9.81
            pb[:, 2] += v * tau - 0.5 * g * tau**2
            vz_b = g * tau - v
        else:
            pb[:, 2] += t.cfg.second_ball_drop_height
        if ball_b_xy_noise:
            pb[:, :2] += (torch.rand(n, 2, device=env.device) * 2 - 1) * ball_b_xy_noise
        if ball_b_height_noise:
            pb[:, 2] += (torch.rand(n, device=env.device) * 2 - 1) * ball_b_height_noise
        if not t.cfg.second_ball_present:
            # Parked on the floor out of reach (curriculum stage with one ball).
            pb = o + torch.tensor([0.6, 0.6, t.cfg.ball_radius + 0.001], device=env.device)
        vb = zero.clone()
        if t.cfg.second_ball_present:
            vb[:, 2] = vz_b
        if air.any():
            g, v0, T, V = 9.81, t.cfg.throw_speed, t.cfg.half_cycle_s, t.cfg.vacant_s
            u = torch.rand(n, device=env.device) * V
            za = v0 * u - 0.5 * g * u**2
            ub = u + T
            zb = v0 * ub - 0.5 * g * ub**2
            pa2 = pa.clone()
            pa2[:, 2] += za
            pb2 = (o + t.col_local[1]).clone()
            pb2[:, 2] += zb
            va = zero.clone()
            va[:, 2] = v0 - g * u
            vb2 = zero.clone()
            vb2[:, 2] = v0 - g * ub
            m = air.unsqueeze(-1)
            t.balls[0].write_root_link_pose_to_sim(
                torch.cat([torch.where(m, pa2, pa), q], -1), env_ids=env_ids
            )
            t.balls[0].write_root_link_velocity_to_sim(torch.where(m, va, zero), env_ids=env_ids)
            pb = torch.where(m, pb2, pb)
            vb = torch.where(m, vb2, vb)
        t.balls[1].write_root_link_pose_to_sim(torch.cat([pb, q], -1), env_ids=env_ids)
        t.balls[1].write_root_link_velocity_to_sim(vb, env_ids=env_ids)
