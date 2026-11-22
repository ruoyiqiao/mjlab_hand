"""Per-env DexPBT task state as an mjlab command term.

One :class:`DexPBTTaskCommand` per task owns the goal and every piece of task
state that DexPBT keeps on the env (``allegro_kuka_base.py``):

* ``closest_fingertip_dist`` / ``closest_keypoint_max_dist`` best-distance
  histories with the ``-1`` sentinel re-initialisation (B:764, 819, 1044, 1077);
* ``lifted_object`` (sticky, B:788-812) using ``z_lift = 0.05 + z - z_init``;
* ``near_goal`` (DexPBT: ``kp_max <= tol * keypointScale``) – here an explicit
  position (+ angle) tolerance scaled by the tolerance curriculum;
* a **consecutive** hold timer (s) for success (deliberate deviation: DexPBT's
  ``near_goal_steps`` is cumulative; the cumulative count is kept as a metric);
* ``successes`` counter (monotonic within an episode) and a per-goal timer
  (DexPBT resets ``progress_buf`` on success, so its timeout is per goal);
* for throw: grasp / lift / release / flight / landing stage flags.

Timing in mjlab 1.6.0 (``ManagerBasedRlEnv.step``): physics -> terminations ->
rewards -> resets -> ``sim.forward()`` -> ``command_manager.compute`` ->
observations. :meth:`update` is called lazily (once per env step, keyed on
``env.common_step_counter``) by terminations/rewards, i.e. on post-physics state.
On-success goal changes run in :meth:`_update_command` (after rewards, like
DexPBT's next-step ``reset_target_pose``) and refresh kinematics with
``sim.forward()`` when they teleport the object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat

from mjlab_hand.grasp.mdp.table_utils import get_table_reference_pos_w, get_table_top_height
from mjlab_hand.tasks_common.hand_profile import profile_by_key
from mjlab_hand.tasks_common.keypoints import (
    keypoint_max_dist,
    keypoints_world,
    quat_angle_error,
    random_quat,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

TIP_BODIES = ("ff_tip_head", "mf_tip_head", "rf_tip_head", "th_tip_head")


@dataclass(kw_only=True)
class DexPBTTaskCommandCfg(CommandTermCfg):
    task: Literal["reorient", "regrasp", "throw", "dual_reorient"]
    hand_names: tuple[str, ...] = ("hand",)
    hand_profiles: tuple[str, ...] = ("allegro_left",)
    """``hand_profile`` key of each hand entity (same order as ``hand_names``): tip/palm bodies, thumb column."""
    object_name: str = "object"
    table_name: str = "table"
    bucket_name: str | None = None

    # Object / keypoints.
    object_size: tuple[float, float, float] = (0.05, 0.05, 0.05)
    object_sizes: tuple[tuple[float, float, float], ...] | None = None
    """Per-variant object sizes (m), ordered like the object VariantEntityCfg."""
    object_base_size: float = 0.05
    keypoint_pattern: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),)
    keypoint_scale: float = 1.5
    object_rest_height: float = 0.025
    """Kept for reference; lift reference uses lift_ref_height_above_table."""

    # DexPBT lifting (B:788-812): z_lift = lift_offset + z - z_init.
    lift_offset: float = 0.05
    lift_ref_offset: float = (
        0.075  # DexPBT constant, not used: z_init comes from lift_ref_height_above_table
    )
    lift_ref_height_above_table: float = 0.10
    """z_init = table top + this, for every object (DexPBT object_init_state z
    0.63 = table top 0.53 + 0.10, independent of the object's size)."""
    lift_bonus_threshold: float = 0.15
    lift_clearance_threshold: float = 0.05
    lift_shaping_mode: Literal["dexpbt", "clearance"] = "clearance"
    """dexpbt: clip(0.05 + z - z_init, 0, 0.5) (2.5 cm dead zone for the 5 cm cube,
    object-size dependent). clearance: clip(lowest point height above table, 0, 0.5)
    (deviation: pays from the first millimetre, same for every cuboid)."""
    max_spawn_height: float = 0.11
    """Resample spawn orientations so the object stands <= this tall (m): keeps long
    sticks from spawning upright into the hand (deviation; DexPBT drops from 0.1 m)."""

    # Goal sampling (env-local box), orientation goal on/off.
    goal_pos_min: tuple[float, float, float] = (-0.1, -0.1, 0.6)
    goal_pos_max: tuple[float, float, float] = (0.1, 0.1, 0.8)
    goal_orientation: bool = False
    dual_side_offset: float = 0.0
    """dual_reorient: goal x = center x +/- this, side chosen at random."""

    # Success.
    pos_tolerance: float = 0.03
    ang_tolerance: float = 0.2
    tolerance_scale_init: float = 2.5
    """Curriculum scale on tolerances (1.0 = final). DexPBT: 0.075 -> 0.01."""
    hold_time_s: float = 0.5
    ang_tol_exponent: float = 1.0
    """Angle tolerance = ang_tolerance * tol_scale ** exponent. With 2.955 the start (scale
    2.5) allows ~3 rad (orientation effectively free, like DexPBT's loose initial keypoint
    tolerance) and reaches ang_tolerance exactly at scale 1.0 (final, evaluation)."""

    # Object resets (regrasp / throw / episode resets).
    object_reset_xy: tuple[float, float] = (0.05, 0.05)
    object_reset_center_xy: tuple[float, float] = (0.0, 0.0)
    dual_object_side_offset: float = 0.0
    """dual_reorient: object start x = +/- this (random side)."""
    forbid_table_handover: bool = False
    """dual_reorient: if True the ``table_handover`` termination is armed: a handover through the
    table (object resting on the table with nobody holding it, then picked up by the OTHER hand)
    ends the episode. Regrasping on the table with the same hand is allowed."""

    # Throw-specific.
    release_debounce_s: float = 0.06
    min_flight_time_s: float = 0.12
    min_flight_dist: float = 0.15
    bucket_pos_min: tuple[float, float, float] = (0.6, -0.2, 0.25)
    bucket_pos_max: tuple[float, float, float] = (0.8, 0.2, 0.35)
    bucket_floor_top: float = 0.010
    bucket_floor_inner_radius: float = 0.070
    bucket_rim_inner_radius: float = 0.103
    bucket_height: float = 0.198
    bucket_margin: float = 0.005
    settle_speed: float = 0.3
    goal_height_above_bucket: float = 0.05
    grasp_min_tips: int = 2
    hold_requires_grasp: bool = False
    """The near-goal hold (success timer) also requires thumb + >= 1 fingertip on the object, not just any hand contact. XHand regrasp
    with the lift gate still swatted (lifted 0.97 of steps, thumb opposition 0.19, 0 successes): ``lifted`` is sticky, so a brief grasp
    unlocks the rest of the goal. Training and evaluation both use it, so it is a stricter success definition (never looser)."""
    lift_requires_grasp: bool = False
    """Throw on the non-Allegro hands (2026-11-17): the object only becomes ``lifted`` (lift bonus, then keypoint reward)
    and earns lift shaping while a hand holds it with the thumb + >= 1 fingertip. Without it LEAP learned to swat the cube
    into the air (99.8 % of lifted steps without hand contact), which collected the lift/keypoint reward but can never be
    a success. Success rules are unchanged; Allegro keeps False."""

    debug_vis: bool = True
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    def build(self, env: ManagerBasedRlEnv) -> DexPBTTaskCommand:
        return DexPBTTaskCommand(self, env)


class DexPBTTaskCommand(CommandTerm):
    cfg: DexPBTTaskCommandCfg

    def __init__(self, cfg: DexPBTTaskCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        self.obj = env.scene[cfg.object_name]
        self.table = env.scene[cfg.table_name]
        self.hands = [env.scene[h] for h in cfg.hand_names]
        assert len(cfg.hand_profiles) == len(cfg.hand_names), (cfg.hand_names, cfg.hand_profiles)
        self.profiles = [profile_by_key(k) for k in cfg.hand_profiles]
        self.tip_ids = []
        self.palm_ids = []
        for h, prof in zip(self.hands, self.profiles):
            ids, names = h.find_bodies(list(prof.tip_bodies), preserve_order=True)
            assert list(names) == list(prof.tip_bodies), names
            self.tip_ids.append(ids)
            pid, _ = h.find_bodies(prof.palm_body)
            self.palm_ids.append(pid[0])
        self.tips_per_hand = self.profiles[0].n_tips
        assert all(p.n_tips == self.tips_per_hand for p in self.profiles), (
            "hands in one task must share a tip count"
        )
        self.num_tips = self.tips_per_hand * len(self.hands)
        self.bucket = env.scene[cfg.bucket_name] if cfg.bucket_name else None

        # Per-env object size (variant objects) -> per-env keypoint offsets.
        if cfg.object_sizes is not None:
            w2v = env.sim.world_to_variant.get(cfg.object_name)
            assert w2v is not None, "object_sizes given but the object entity has no variants"
            table = torch.tensor(cfg.object_sizes, dtype=torch.float32, device=dev)
            self.variant_idx = w2v.to(dev).long()
            self.obj_size = table[self.variant_idx]
        else:
            self.variant_idx = torch.zeros(n, dtype=torch.long, device=dev)
            self.obj_size = torch.tensor(cfg.object_size, device=dev).expand(n, 3).clone()
        kp = torch.tensor(cfg.keypoint_pattern, dtype=torch.float32, device=dev)
        self.kp_offsets = kp.unsqueeze(0) * self.obj_size.unsqueeze(1) * cfg.keypoint_scale / 2.0
        self.goal_pos = torch.zeros(n, 3, device=dev)
        self.goal_quat = torch.zeros(n, 4, device=dev)
        self.goal_quat[:, 0] = 1.0
        self.goal_side = torch.zeros(
            n, dtype=torch.long, device=dev
        )  # dual: 0=left(-x),1=right(+x)
        self.bucket_pos = torch.zeros(n, 3, device=dev)

        self.best_tip = -torch.ones(n, self.num_tips, device=dev)
        self.best_kp = -torch.ones(n, device=dev)
        self.lifted = torch.zeros(n, dtype=torch.bool, device=dev)
        self.z_init = torch.zeros(n, device=dev)
        self.hold_time = torch.zeros(n, device=dev)
        self.near_cum = torch.zeros(n, dtype=torch.long, device=dev)
        self.successes = torch.zeros(n, dtype=torch.long, device=dev)
        self.goal_time = torch.zeros(n, device=dev)
        self.goal_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.tol_scale = float(cfg.tolerance_scale_init)
        self.prev_episode_successes = torch.zeros(n, device=dev)

        # Throw stages.
        self.grasped = torch.zeros(n, dtype=torch.bool, device=dev)
        self.released = torch.zeros(n, dtype=torch.bool, device=dev)
        self.no_contact_time = torch.zeros(n, device=dev)
        self.release_pos = torch.zeros(n, 3, device=dev)
        self.flight_time = torch.zeros(n, device=dev)
        self.flight_invalid = torch.zeros(n, dtype=torch.bool, device=dev)
        self.in_bucket = torch.zeros(n, dtype=torch.bool, device=dev)
        self.throw_failed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._valid_flight = torch.zeros(n, dtype=torch.bool, device=dev)

        # Per-step outputs.
        self.tip_delta = torch.zeros(n, device=dev)
        self.tip_delta_gated = torch.zeros(n, device=dev)
        self.lift_rew = torch.zeros(n, device=dev)
        self.lift_bonus = torch.zeros(n, device=dev)
        self.kp_delta = torch.zeros(n, device=dev)
        self.kp_dist = torch.zeros(n, device=dev)
        self.pos_err = torch.zeros(n, device=dev)
        self.ang_err = torch.zeros(n, device=dev)
        self.near_goal = torch.zeros(n, dtype=torch.bool, device=dev)
        self.success_now = torch.zeros(n, dtype=torch.bool, device=dev)
        self.tip_dist = torch.zeros(n, self.num_tips, device=dev)
        self.hand_contact = torch.zeros(n, len(self.hands), dtype=torch.bool, device=dev)
        self.tip_found = torch.zeros(
            n, len(self.hands), self.tips_per_hand, dtype=torch.bool, device=dev
        )
        self.grasp_contact = torch.zeros(n, dtype=torch.bool, device=dev)
        self.grasp_rew = torch.zeros(n, device=dev)
        self.clearance = torch.zeros(n, device=dev)
        self._thumb_cols: list[int] | None = None  # resolved lazily from sensor names
        self.teleported = torch.zeros(n, dtype=torch.bool, device=dev)
        # Dual-hand bookkeeping.
        self.owner = -torch.ones(n, dtype=torch.long, device=dev)  # -1 none/both, 0 left, 1 right
        self.last_owner = -torch.ones(n, dtype=torch.long, device=dev)
        self.handoffs = torch.zeros(n, dtype=torch.long, device=dev)
        self.table_touch = torch.zeros(n, dtype=torch.bool, device=dev)
        self.indirect_handover = torch.zeros(n, dtype=torch.bool, device=dev)
        self.direct_handoffs = torch.zeros(n, dtype=torch.long, device=dev)
        self.hand_rewarded = torch.zeros(n, dtype=torch.bool, device=dev)
        self.direct_bonus = torch.zeros(n, device=dev)
        self.indirect_handoffs = torch.zeros(n, dtype=torch.long, device=dev)
        self.goal_start_side = torch.zeros(n, dtype=torch.long, device=dev)
        self.hand_hand = torch.zeros(n, dtype=torch.bool, device=dev)
        self.hand_hand_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self._need_side = torch.zeros(n, dtype=torch.bool, device=dev)
        self._last_step = -1
        self._pending_forward = False

        for k in (
            "successes",
            "lifted",
            "pos_err",
            "ang_err",
            "near_goal",
            "goal_time",
            "near_cum",
            "grasped",
            "released",
            "valid_flight",
            "in_bucket",
            "tol_scale",
        ):
            self.metrics[k] = torch.zeros(n, device=dev)

    # ------------------------------------------------------------------
    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.goal_pos - self._env.scene.env_origins, self.goal_quat], dim=-1)

    @property
    def step_dt(self) -> float:
        return float(self._env.step_dt)

    @property
    def hold_steps(self) -> int:
        return max(1, int(round(self.cfg.hold_time_s / self.step_dt)))

    def object_pose(self):
        return self.obj.data.root_link_pos_w, self.obj.data.root_link_quat_w

    def tips_w(self) -> torch.Tensor:
        return torch.cat(
            [h.data.body_link_pos_w[:, ids] for h, ids in zip(self.hands, self.tip_ids)], dim=1
        )

    def palm_pose(self, i: int):
        h = self.hands[i]
        return h.data.body_link_pos_w[:, self.palm_ids[i]], h.data.body_link_quat_w[
            :, self.palm_ids[i]
        ]

    def object_keypoints(self) -> torch.Tensor:
        p, q = self.object_pose()
        return keypoints_world(p, q, self.kp_offsets)

    def goal_keypoints(self) -> torch.Tensor:
        return keypoints_world(self.goal_pos, self.goal_quat, self.kp_offsets)

    def table_top(self) -> torch.Tensor:
        return get_table_top_height(self.table)

    def _sensor_found(self, name: str) -> torch.Tensor:
        return self._env.scene[name].data.found.reshape(self.num_envs, -1) > 0

    # ------------------------------------------------------------------
    def update(self) -> None:
        """Compute this step's task state once (post-physics)."""
        step = int(self._env.common_step_counter)
        if step == self._last_step:
            return
        self._last_step = step
        cfg, dt = self.cfg, self.step_dt
        obj_pos, obj_quat = self.object_pose()

        # Contacts.
        for i, h in enumerate(cfg.hand_names):
            self.hand_contact[:, i] = self._sensor_found(f"{h}_object_contact").any(-1)
            self.tip_found[:, i] = self._sensor_found(f"{h}_tip_contact")
        if self._thumb_cols is None:
            self._thumb_cols = []
            for h, prof in zip(cfg.hand_names, self.profiles):
                names = list(self._env.scene[f"{h}_tip_contact"].primary_names)
                cols = [i for i, nme in enumerate(names) if prof.thumb_geom in nme]
                assert len(names) == self.tips_per_hand and len(cols) == 1, (prof.key, names)
                self._thumb_cols.append(cols[0])
        thumb = torch.stack([self.tip_found[:, i, c] for i, c in enumerate(self._thumb_cols)], -1)
        n_other = self.tip_found.sum(-1) - thumb.long()
        # Opposition grasp evidence: in some hand, thumb tip + >= 1 other fingertip.
        self.grasp_contact = (thumb & (n_other >= 1)).any(-1)
        # Lowest point of the (box) object above the table top.
        rz = matrix_from_quat(obj_quat)[:, 2, :]
        self.clearance = obj_pos[:, 2] - (rz.abs() * self.obj_size / 2).sum(-1) - self.table_top()

        # Fingertip progress (B:762-778): clip(best - curr, 0, 10) summed; gated later.
        tips = self.tips_w()
        self.tip_dist = torch.linalg.norm(tips - obj_pos.unsqueeze(1), dim=-1)
        self.best_tip = torch.where(self.best_tip < 0, self.tip_dist, self.best_tip)
        self.tip_delta = torch.clamp(self.best_tip - self.tip_dist, 0.0, 10.0).sum(-1)
        self.best_tip = torch.minimum(self.best_tip, self.tip_dist)
        lifted_prev = self.lifted.clone()
        self.tip_delta_gated = self.tip_delta * (~lifted_prev).float()

        # Lifting (B:788-812).
        z_lift = cfg.lift_offset + obj_pos[:, 2] - self.z_init
        shaped = z_lift if cfg.lift_shaping_mode == "dexpbt" else self.clearance
        self.lift_rew = torch.clamp(shaped, 0.0, 0.5) * (~lifted_prev).float()
        if cfg.lift_requires_grasp:
            self.lift_rew = self.lift_rew * self.grasp_contact.float()
        self.grasp_rew = (self.grasp_contact & ~lifted_prev).float()
        if cfg.lift_shaping_mode == "dexpbt":
            lift_now = z_lift > cfg.lift_bonus_threshold
        else:
            # a5+: lifted once the object's lowest point clears the table by
            # lift_clearance_threshold (size-independent; DexPBT's centre threshold made
            # goals below table+0.20 unreachable without first over-lifting).
            lift_now = self.clearance > cfg.lift_clearance_threshold
        if cfg.lift_requires_grasp:
            lift_now = lift_now & self.grasp_contact
        self.lifted = lift_now | lifted_prev
        self.lift_bonus = (self.lifted & ~lifted_prev).float()

        # Keypoint progress (B:816-826), gated by lifted (post-update).
        kp_o = self.object_keypoints()
        kp_g = self.goal_keypoints()
        self.kp_dist = keypoint_max_dist(kp_o, kp_g)
        self.best_kp = torch.where(self.best_kp < 0, self.kp_dist, self.best_kp)
        self.kp_delta = torch.clamp(self.best_kp - self.kp_dist, 0.0, 100.0) * self.lifted.float()
        self.best_kp = torch.minimum(self.best_kp, self.kp_dist)

        self.pos_err = torch.linalg.norm(obj_pos - self.goal_pos, dim=-1)
        self.ang_err = quat_angle_error(obj_quat, self.goal_quat)
        pos_tol = cfg.pos_tolerance * self.tol_scale
        near = self.pos_err <= pos_tol
        if cfg.goal_orientation:
            near = near & (self.ang_err <= cfg.ang_tolerance * self.tol_scale**cfg.ang_tol_exponent)

        if cfg.task == "dual_reorient":
            self._update_dual()
        if cfg.task == "throw":
            near = self._update_throw(obj_pos, near, dt)
        else:
            # The object must be lifted and held (no free-fall coincidences).
            near = near & self.lifted & self.hand_contact.any(-1)
            if cfg.hold_requires_grasp:
                near = near & self.grasp_contact
        self.near_goal = near
        self.near_cum += near.long()
        self.hold_time = torch.where(near, self.hold_time + dt, torch.zeros_like(self.hold_time))
        self.success_now = self.hold_time >= cfg.hold_time_s - 1e-6
        self.goal_time += dt
        self.goal_steps += 1
        self.teleported[:] = False

    def _update_dual(self) -> None:
        side = self._object_side()
        self.goal_start_side = torch.where(self._need_side, side, self.goal_start_side)
        self._need_side[:] = False
        c = self.hand_contact  # (N, 2) left, right
        single = c.sum(-1) == 1
        owner = torch.where(single, c[:, 1].long(), -torch.ones_like(self.owner))
        self.owner = owner
        changed = single & (self.last_owner >= 0) & (owner != self.last_owner)
        self.handoffs += changed.long()
        # Handover through the table: the object rested on the table while no single hand held it
        # since the previous owner; a direct handover (both hands touching, or a flight between the
        # hands) leaves table_touch False.
        on_table = self._sensor_found("object_table_contact").any(-1)
        self.table_touch = self.table_touch | (on_table & ~single)
        self.indirect_handover = changed & self.table_touch
        self.indirect_handoffs += self.indirect_handover.long()
        self.direct_handoffs += (changed & ~self.table_touch).long()
        # One-time bonus per goal: a direct handover that gives the object to the hand on the goal's
        # side (cannot be farmed by passing back and forth).
        to_goal = changed & ~self.table_touch & (owner == self.goal_side) & ~self.hand_rewarded
        self.direct_bonus = to_goal.float()
        self.hand_rewarded |= to_goal
        self.table_touch = torch.where(single, torch.zeros_like(self.table_touch), self.table_touch)
        self.last_owner = torch.where(single, owner, self.last_owner)
        s = self._env.scene["hand_hand_contact"].data
        hh = s.found.reshape(self.num_envs, -1) > 0
        if s.dist_history is not None:
            hh = hh | (s.dist_history.reshape(self.num_envs, -1) < 0).any(-1, keepdim=True)
        self.hand_hand = hh.any(-1)
        self.hand_hand_steps += self.hand_hand.long()

    def _object_side(self) -> torch.Tensor:
        """0 if the object is on the left half (x < table center), else 1."""
        x = self.obj.data.root_link_pos_w[:, 0] - get_table_reference_pos_w(self.table)[:, 0]
        return (x > 0).long()

    def _update_throw(self, obj_pos, near_pos, dt):
        cfg = self.cfg
        sensor = self._env.scene[f"{cfg.hand_names[0]}_tip_contact"]
        tip_found = sensor.data.found.reshape(self.num_envs, -1) > 0  # (N, tips) model order
        if not hasattr(self, "_thumb_col"):
            names = list(sensor.primary_names)
            cols = [i for i, nme in enumerate(names) if self.profiles[0].thumb_geom in nme]
            assert len(cols) == 1 and len(names) == self.tips_per_hand, (
                self.profiles[0].key,
                names,
            )
            self._thumb_col = cols[0]
        any_hand = self.hand_contact.any(-1)
        on_table = self._sensor_found("object_table_contact").any(-1)
        n_tips = tip_found.sum(-1)
        # Grasp evidence: >= grasp_min_tips tips incl. thumb on the object, off the table.
        grasp_now = (n_tips >= cfg.grasp_min_tips) & tip_found[:, self._thumb_col] & ~on_table
        self.grasped |= grasp_now
        # Release: grasped & lifted, then no hand contact for release_debounce_s.
        self.no_contact_time = torch.where(
            any_hand, torch.zeros_like(self.no_contact_time), self.no_contact_time + dt
        )
        newly_released = (
            ~self.released
            & self.grasped
            & self.lifted
            & (self.no_contact_time >= cfg.release_debounce_s)
        )
        self.release_pos = torch.where(newly_released.unsqueeze(-1), obj_pos, self.release_pos)
        self.released |= newly_released
        # Re-grabbing after a release cancels it (a throw must be the final release).
        regrab = self.released & any_hand
        self.released &= ~regrab
        self.flight_time = torch.where(
            self.released, self.flight_time, torch.zeros_like(self.flight_time)
        )
        self.flight_invalid &= self.released

        # Bucket containment: object center inside the flared inner cone (bucket
        # frame = world-aligned, origin at the bucket bottom center).
        rel = obj_pos - self.bucket_pos
        m = cfg.bucket_margin
        zf, zh = cfg.bucket_floor_top, cfg.bucket_height
        frac = ((rel[:, 2] - zf) / (zh - zf)).clamp(0.0, 1.0)
        r_in = cfg.bucket_floor_inner_radius + frac * (
            cfg.bucket_rim_inner_radius - cfg.bucket_floor_inner_radius
        )
        inside = (
            (torch.linalg.norm(rel[:, :2], dim=-1) <= r_in - m)
            & (rel[:, 2] >= zf)
            & (rel[:, 2] <= zh)
        )
        self.in_bucket = inside
        in_air = self.released & ~inside & ~on_table
        self.flight_time = self.flight_time + dt * in_air.float()
        # Touching the table after release = failed throw.
        self.throw_failed = self.released & on_table
        flight_dist = torch.linalg.norm((obj_pos - self.release_pos)[:, :2], dim=-1)
        valid_flight = (
            self.released
            & (self.flight_time >= cfg.min_flight_time_s)
            & (flight_dist >= cfg.min_flight_dist)
        )
        self._valid_flight = valid_flight
        speed = torch.linalg.norm(self.obj.data.root_link_lin_vel_w, dim=-1)
        landed = inside & valid_flight & ~any_hand & (speed <= cfg.settle_speed)
        return landed

    # ------------------------------------------------------------------
    def _sample_goal(self, env_ids: torch.Tensor) -> None:
        cfg, n = self.cfg, len(env_ids)
        origins = self._env.scene.env_origins[env_ids]
        if cfg.task == "throw":
            lo = torch.tensor(cfg.bucket_pos_min, device=self.device)
            hi = torch.tensor(cfg.bucket_pos_max, device=self.device)
            bpos = lo + torch.rand(n, 3, device=self.device) * (hi - lo) + origins
            self.bucket_pos[env_ids] = bpos
            self.goal_pos[env_ids] = bpos + torch.tensor(
                [0.0, 0.0, cfg.goal_height_above_bucket], device=self.device
            )
            self.goal_quat[env_ids] = torch.tensor([1.0, 0, 0, 0], device=self.device)
            assert self.bucket is not None
            pose = torch.cat([bpos, self.goal_quat[env_ids]], dim=-1)
            self.bucket.write_mocap_pose_to_sim(pose, env_ids=env_ids)
            return
        lo = torch.tensor(cfg.goal_pos_min, device=self.device)
        hi = torch.tensor(cfg.goal_pos_max, device=self.device)
        pos = lo + torch.rand(n, 3, device=self.device) * (hi - lo)
        if cfg.task == "dual_reorient":
            side = (torch.rand(n, device=self.device) > 0.5).long()
            self.goal_side[env_ids] = side
            pos[:, 0] += torch.where(side == 1, 1.0, -1.0) * cfg.dual_side_offset
            # Side the object is on when this goal starts; read at the next
            # post-physics update (derived poses are stale right after writes).
            self._need_side[env_ids] = True
        self.goal_pos[env_ids] = pos + origins
        if cfg.goal_orientation:
            self.goal_quat[env_ids] = random_quat(n, self.device)
        else:
            self.goal_quat[env_ids] = torch.tensor([1.0, 0, 0, 0], device=self.device)

    def _reset_object(self, env_ids: torch.Tensor) -> None:
        place_object_on_table(self._env, env_ids, self.cfg, self.obj_size / 2.0)
        top = self.table_top()[env_ids]
        self.z_init[env_ids] = top + self.cfg.lift_ref_height_above_table
        self.lifted[env_ids] = False
        self.best_tip[env_ids] = -1.0
        self.teleported[env_ids] = True
        self.grasped[env_ids] = False
        self.released[env_ids] = False
        self.no_contact_time[env_ids] = 0.0
        self.flight_time[env_ids] = 0.0
        self.throw_failed[env_ids] = False

    def _clear_goal_state(self, env_ids: torch.Tensor) -> None:
        self.best_kp[env_ids] = -1.0
        self.hand_rewarded[env_ids] = False
        self.direct_bonus[env_ids] = 0.0
        self.hold_time[env_ids] = 0.0
        self.near_cum[env_ids] = 0
        self.goal_time[env_ids] = 0.0
        self.goal_steps[env_ids] = 0
        self.success_now[env_ids] = False
        self.near_goal[env_ids] = False

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Called on episode reset (via CommandTerm.reset)."""
        self._sample_goal(env_ids)
        self._clear_goal_state(env_ids)

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        # A success detected on the terminal step still counts for that episode.
        self.successes[env_ids] += self.success_now[env_ids].long()
        self.prev_episode_successes[env_ids] = self.successes[env_ids].float()
        self.success_now[env_ids] = False
        # Episode-level state; the object itself is placed by the reset event.
        self.successes[env_ids] = 0
        self.lifted[env_ids] = False
        self.best_tip[env_ids] = -1.0
        top = self.table_top()[env_ids]
        self.z_init[env_ids] = top + self.cfg.lift_ref_height_above_table
        self.grasped[env_ids] = False
        self.released[env_ids] = False
        self.no_contact_time[env_ids] = 0.0
        self.flight_time[env_ids] = 0.0
        self.throw_failed[env_ids] = False
        self.in_bucket[env_ids] = False
        self.owner[env_ids] = -1
        self.last_owner[env_ids] = -1
        self.handoffs[env_ids] = 0
        self.table_touch[env_ids] = False
        self.indirect_handover[env_ids] = False
        self.direct_handoffs[env_ids] = 0
        self.indirect_handoffs[env_ids] = 0
        self.hand_hand[env_ids] = False
        self.hand_hand_steps[env_ids] = 0
        # Cached per-step quantities (kp_dist, tip_dist, errors, ...) were computed before this
        # reset; if the env is being reset because its physics went NaN (nan termination), they
        # are NaN and observations of the reset env would read them. Clear non-finite values.
        for v in vars(self).values():
            if (
                isinstance(v, torch.Tensor)
                and v.is_floating_point()
                and v.dim() >= 1
                and v.shape[0] == self.num_envs
            ):
                sub = v[env_ids]
                if not torch.isfinite(sub).all():
                    v[env_ids] = torch.nan_to_num(sub, nan=0.0, posinf=0.0, neginf=0.0)
        extras = super().reset(env_ids)
        self._pending_forward = False
        return extras

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is not None:
            return
        ids = self.success_now.nonzero().flatten()
        if len(ids) > 0:
            self.successes[ids] += 1
            if self.cfg.task in ("regrasp", "throw"):
                # DexPBT regrasping/throw _reset_target: object back on the table,
                # lifted flag cleared. The hand keeps its state.
                self._reset_object(ids)
                self._pending_forward = True
            self._sample_goal(ids)
            if self.cfg.task == "throw":
                self._pending_forward = True
            self._clear_goal_state(ids)
        if self._pending_forward:
            self._pending_forward = False
            self._env.sim.forward()

    def _update_metrics(self) -> None:
        m = self.metrics
        m["successes"] = self.successes.float()
        m["lifted"] = self.lifted.float()
        m["pos_err"] = self.pos_err
        m["ang_err"] = self.ang_err
        m["near_goal"] = self.near_goal.float()
        m["goal_time"] = self.goal_time
        m["near_cum"] = self.near_cum.float()
        m["grasped"] = self.grasped.float()
        m["released"] = self.released.float()
        m["valid_flight"] = getattr(self, "_valid_flight", self.released).float()
        m["in_bucket"] = self.in_bucket.float()
        m["tol_scale"] = torch.full_like(m["tol_scale"], self.tol_scale)

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        for b in visualizer.get_env_indices(self.num_envs):
            pos = self.goal_pos[b].cpu().numpy()
            if self.cfg.goal_orientation:
                mat = matrix_from_quat(self.goal_quat[b : b + 1])[0].cpu().numpy()
                h = (self.obj_size[b] / 2).tolist()
                visualizer.add_box(
                    center=pos,
                    size=tuple(h),
                    mat=mat,
                    color=(0.2, 0.9, 0.2, 0.35),
                    label=f"goal_{b}",
                )
                # Axis tick to show orientation.
                tip = pos + mat[:, 0] * 0.05
                visualizer.add_arrow(start=pos, end=tip, color=(0.9, 0.1, 0.1, 0.8))
            else:
                visualizer.add_sphere(
                    center=pos, radius=0.015, color=(0.2, 0.9, 0.2, 0.5), label=f"goal_{b}"
                )


def place_object_on_table(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    cfg: DexPBTTaskCommandCfg,
    half_size: torch.Tensor,
) -> None:
    """DexPBT ``reset_object_pose`` analogue on a real table.

    Position: table center + ``object_reset_center_xy`` + U(-r, r) (DexPBT: +-0.1 x/y);
    for ``dual_reorient`` the x center is mirrored to a random side.
    Orientation: uniform random (DexPBT ``get_random_quat``).
    Height: the rotated box's lowest point 1 mm above the table top, computed
    analytically per env from that env's cuboid half extents (the template-model
    vertex cache of ``reset_geometry.object_lowest_offset`` is not valid for
    per-world variants), using the table's mocap pose (valid on cold resets, no
    FK dependence). Only writes state.
    """
    if len(env_ids) == 0:
        return
    n, dev = len(env_ids), env.device
    obj = env.scene[cfg.object_name]
    table = env.scene[cfg.table_name]
    center = get_table_reference_pos_w(table)[env_ids].clone()
    cx, cy = cfg.object_reset_center_xy
    rx, ry = cfg.object_reset_xy
    x = cx + (torch.rand(n, device=dev) * 2 - 1) * rx
    if cfg.task == "dual_reorient" and cfg.dual_object_side_offset > 0:
        side = torch.where(torch.rand(n, device=dev) > 0.5, 1.0, -1.0)
        x = x + side * cfg.dual_object_side_offset
    y = cy + (torch.rand(n, device=dev) * 2 - 1) * ry
    # Uniform random orientation (DexPBT get_random_quat), rejection-sampled so the
    # object's vertical extent stays <= max_spawn_height; fallback: lowest candidate.
    k = 16
    cand = random_quat(n * k, dev).reshape(n, k, 4)
    rz = matrix_from_quat(cand.reshape(-1, 4))[:, 2, :].reshape(n, k, 3)
    ext = 2.0 * (rz.abs() * half_size[env_ids].unsqueeze(1)).sum(-1)
    ok = ext <= cfg.max_spawn_height
    first_ok = torch.where(ok.any(-1), ok.float().argmax(-1), ext.argmin(-1))
    quat = cand[torch.arange(n, device=dev), first_ok]
    pos = center
    pos[:, 0] += x
    pos[:, 1] += y
    # Lowest point offset of a box: -sum_j |R[2, j]| * h_j.
    rz = matrix_from_quat(quat)[:, 2, :]
    lowest = -(rz.abs() * half_size[env_ids]).sum(-1)
    pos[:, 2] = get_table_top_height(table)[env_ids] + 0.001 - lowest
    obj.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    obj.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)


def reset_object_event(env: ManagerBasedRlEnv, env_ids: torch.Tensor, command_name: str) -> None:
    """Reset event: place the object with the task command's distribution."""
    term = env.command_manager.get_term(command_name)
    place_object_on_table(env, env_ids, term.cfg, term.obj_size / 2.0)


def get_tracker(env: ManagerBasedRlEnv, command_name: str) -> DexPBTTaskCommand:
    term = env.command_manager.get_term(command_name)
    assert isinstance(term, DexPBTTaskCommand)
    term.update()
    return term
