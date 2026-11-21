"""Bounded floating-hand action for arm-free Allegro tasks.

Layout of the (canonical) action vector, independent of actuator ordering::

    [base_x, base_y, base_z, base_roll, base_pitch, base_yaw,
     ffj0..ffj3, mfj0..mfj3, rfj0..rfj3, thj0..thj3]

Mapping (DexPBT ``allegro_kuka_base.py`` ``pre_physics_step`` analogue):

* base: ``target += a * v_max * step_dt`` (DexPBT arm: ``prev + dofSpeedScale*dt*a``),
  then clamped to the workspace bounds **on the applied target**;
* fingers (``finger_mode="absolute"``, DexPBT default): ``target = scale*a + offset``
  mapped onto the joint range (DexPBT ``scale(a, lower, upper)``), with optional EMA
  (DexPBT ``actionsMovingAverage``; 1.0 = no smoothing);
* fingers (``finger_mode="delta"``, mjlab_hand grasp style): integrated velocity.

All indices are resolved by joint name and asserted. ``clip_fraction`` and
``bound_violation`` are exposed for logging/evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import mujoco
import torch
from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values

from mjlab_hand.tasks_common.workspace import BASE_JOINTS, HandRootWorkspace

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

ALLEGRO_FINGER_JOINTS: tuple[str, ...] = tuple(
    f"{f}j{i}" for f in ("ff", "mf", "rf", "th") for i in range(4)
)


@dataclass(kw_only=True)
class BoundedFloatingHandActionCfg(BaseActionCfg):
    workspace: HandRootWorkspace
    finger_joint_names: tuple[str, ...] = ALLEGRO_FINGER_JOINTS
    finger_tendon_names: tuple[str, ...] = ()
    """Tendon-length targets appended after the finger joints in the action (Shadow's coupled J2+J1 tendons). Absolute map:
    target = centre + a * half-range of the tendon actuator's control range (EMA as for the finger joints)."""
    held_joint_pos: dict[str, float] = field(default_factory=dict)
    """Actuated joints driven to a fixed target every step and excluded from the action (Shadow wrist)."""
    finger_mode: Literal["absolute", "delta", "fixed"] = "absolute"
    fixed_finger_pos: dict[str, float] = field(default_factory=dict)
    """finger_mode="fixed": fingers held at these joint targets; action = 6 base dims."""
    finger_scale: dict[str, float] = field(default_factory=dict)
    finger_offset: dict[str, float] = field(default_factory=dict)
    finger_vel_max: float = 2.5
    """rad/s, only for ``finger_mode="delta"``."""
    finger_ema: float = 1.0
    base_ema: float = 1.0
    base_mode: Literal["delta", "absolute"] = "delta"
    """"delta": target += a*v_max*dt (DexPBT arm). "absolute": mjlab tracking/G1 style,
    target = default_pos + a*scale, scale = reach to the farther workspace bound
    (no integration, no rate limit; the PD gains set the speed)."""
    base_slew: bool = False
    """base_mode="absolute" only: limit the target change per step to workspace lin/ang_vel_max * dt
    (measured on dualtrack_fartrack_s42 it 3700: unlimited absolute targets + saturated policy = 4.6 m/s
    object flings, joint speeds up to 28 rad/s)."""
    actuator_names: tuple[str, ...] | list[str] = (".*",)

    def __post_init__(self):
        self.transmission_type = TransmissionType.JOINT
        for a in (self.finger_ema, self.base_ema):
            if not 0.0 < a <= 1.0:
                raise ValueError(f"EMA alpha must be in (0, 1], got {a}")
        if self.finger_mode == "fixed" and not self.fixed_finger_pos:
            raise ValueError("fixed finger mode requires fixed_finger_pos")
        if self.finger_mode == "absolute" and bool(self.finger_scale) != bool(self.finger_offset):
            raise ValueError(
                "finger_scale and finger_offset must be given together (or both empty = joint range map)"
            )

    def build(self, env: ManagerBasedRlEnv) -> BoundedFloatingHandAction:
        return BoundedFloatingHandAction(self, env)


class BoundedFloatingHandAction(BaseAction):
    cfg: BoundedFloatingHandActionCfg

    def __init__(self, cfg: BoundedFloatingHandActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        names = list(self._target_names)
        canonical = list(BASE_JOINTS) + list(cfg.finger_joint_names)
        missing = [n for n in canonical if n not in names]
        extra = [n for n in names if n not in canonical and n not in cfg.held_joint_pos]
        if missing or extra:
            raise ValueError(
                f"{cfg.entity_name}: action joints mismatch, missing={missing} extra={extra}"
            )
        # perm[k] = position (within target ids) of canonical joint k.
        perm = [names.index(n) for n in canonical]
        self._perm = torch.tensor(perm, device=self.device, dtype=torch.long)
        self._joint_ids_canonical = self._target_ids[self._perm]
        self._canonical_names = canonical
        self._nb = len(BASE_JOINTS)
        self._nf = len(cfg.finger_joint_names)
        self._nt = len(cfg.finger_tendon_names)
        held = list(cfg.held_joint_pos)
        self._held_ids = (
            self._target_ids[
                torch.tensor([names.index(n) for n in held], device=self.device, dtype=torch.long)
            ]
            if held
            else None
        )
        self._held_pos = (
            torch.tensor([cfg.held_joint_pos[n] for n in held], device=self.device).expand(
                self.num_envs, -1
            )
            if held
            else None
        )
        if self._nt:
            tids, tnames = self._entity.find_tendons(
                list(cfg.finger_tendon_names), preserve_order=True
            )
            assert list(tnames) == list(cfg.finger_tendon_names), tnames
            self._tendon_ids = torch.tensor(tids, device=self.device, dtype=torch.long)
            lo_hi = self._tendon_ctrl_ranges(env, cfg)
            self._t_lo = (
                torch.tensor([r[0] for r in lo_hi], device=self.device)
                .expand(self.num_envs, -1)
                .clone()
            )
            self._t_hi = (
                torch.tensor([r[1] for r in lo_hi], device=self.device)
                .expand(self.num_envs, -1)
                .clone()
            )
        self._step_dt = float(env.step_dt)

        ws = cfg.workspace
        lo = torch.tensor([ws.joint_bounds[n][0] for n in BASE_JOINTS], device=self.device)
        hi = torch.tensor([ws.joint_bounds[n][1] for n in BASE_JOINTS], device=self.device)
        limits = self._entity.data.joint_pos_limits[:, self._joint_ids_canonical]  # (E, J, 2)
        self._base_lo = torch.maximum(lo.expand(self.num_envs, -1), limits[:, : self._nb, 0])
        self._base_hi = torch.minimum(hi.expand(self.num_envs, -1), limits[:, : self._nb, 1])
        self._finger_lo = limits[:, self._nb :, 0].clone()
        self._finger_hi = limits[:, self._nb :, 1].clone()
        self._base_rate = torch.tensor(
            [ws.lin_vel_max] * 3 + [ws.ang_vel_max] * 3, device=self.device
        )
        self._viol_tol = torch.tensor(
            [ws.violation_tol] * 3 + [ws.violation_tol_rot] * 3, device=self.device
        )
        self._ws_lo = lo
        self._ws_hi = hi
        # mjlab JointPositionAction (use_default_offset=True): target = default_pos + scale * a, so a = 0
        # means "hold home". Scale = reach to the farther bound (the workspace is asymmetric, e.g.
        # base_z in (-0.20, +0.115)); the clamp to the bounds then saturates on the short side.
        dflt = self._entity.data.default_joint_pos[:, self._joint_ids_canonical[: self._nb]]
        self._abs_centre = dflt.clone()
        self._abs_scale = torch.maximum(self._base_hi - dflt, dflt - self._base_lo)

        if cfg.finger_mode == "absolute":
            fnames = list(cfg.finger_joint_names)
            if cfg.finger_scale:
                self._f_scale = torch.ones(self._nf, device=self.device)
                self._f_offset = torch.zeros(self._nf, device=self.device)
                idx, _, vals = resolve_matching_names_values(cfg.finger_scale, fnames)
                self._f_scale[idx] = torch.tensor(vals, device=self.device)
                idx, _, vals = resolve_matching_names_values(cfg.finger_offset, fnames)
                self._f_offset[idx] = torch.tensor(vals, device=self.device)
            else:
                # target = centre + a * half-range of the joint limits (what the hand-specific dicts encode)
                self._f_scale = 0.5 * (self._finger_hi - self._finger_lo)
                self._f_offset = 0.5 * (self._finger_hi + self._finger_lo)

        n = self.num_envs
        self._fixed_tendons = None  # tendon start / hold lengths; None = mid-range
        if cfg.finger_mode == "fixed":
            fixed = torch.tensor(
                [cfg.fixed_finger_pos[j] for j in cfg.finger_joint_names], device=self.device
            )
            self._fixed_fingers = torch.clamp(
                fixed.expand(n, -1), self._finger_lo, self._finger_hi
            ).clone()
            if self._nt:  # tendon targets (Shadow J0 = J1 + J2) are held at fixed lengths too
                ft = torch.tensor(
                    [cfg.fixed_finger_pos[t] for t in cfg.finger_tendon_names], device=self.device
                )
                self._fixed_tendons = torch.minimum(
                    torch.maximum(ft.expand(n, -1), self._t_lo), self._t_hi
                ).clone()
            self._action_dim = self._nb
        else:
            self._action_dim = self._nb + self._nf + self._nt
            if self._nt and all(t in cfg.fixed_finger_pos for t in cfg.finger_tendon_names):
                # delta mode with a start pose (e.g. a measured grip): tendons start there instead of mid-range
                ft = torch.tensor(
                    [cfg.fixed_finger_pos[t] for t in cfg.finger_tendon_names], device=self.device
                )
                self._fixed_tendons = torch.minimum(
                    torch.maximum(ft.expand(n, -1), self._t_lo), self._t_hi
                ).clone()
        self._raw_actions = torch.zeros(n, self._action_dim, device=self.device)
        self._base_target = torch.zeros(n, self._nb, device=self.device)
        self._finger_target = torch.zeros(n, self._nf, device=self.device)
        self._applied = torch.zeros(n, self._nb + self._nf, device=self.device)
        self._tendon_target = torch.zeros(n, self._nt, device=self.device)
        self._tendon_applied = torch.zeros(n, self._nt, device=self.device)
        self.clip_fraction = torch.zeros(n, device=self.device)
        self.clip_lo = torch.zeros(n, self._nb, dtype=torch.bool, device=self.device)
        self.clip_hi = torch.zeros(n, self._nb, dtype=torch.bool, device=self.device)
        self.bound_violation = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._init_targets(slice(None))

    # ------------------------------------------------------------------
    @property
    def canonical_joint_names(self) -> list[str]:
        return list(self._canonical_names)

    @property
    def joint_ids_canonical(self) -> torch.Tensor:
        return self._joint_ids_canonical

    @property
    def base_target(self) -> torch.Tensor:
        return self._applied[:, : self._nb]

    @staticmethod
    def _tendon_ctrl_ranges(
        env: ManagerBasedRlEnv, cfg: BoundedFloatingHandActionCfg
    ) -> list[tuple[float, float]]:
        """Control range of the position actuator driving each tendon target (from the compiled model)."""
        m = env.sim.mj_model
        rng = [
            m.actuator_ctrlrange[actuator_of(m, cfg.entity_name, t, "tendon")]
            for t in cfg.finger_tendon_names
        ]
        return [(float(lo), float(hi)) for lo, hi in rng]

    def _current_pos(self, env_ids) -> torch.Tensor:
        return self._entity.data.joint_pos[env_ids][:, self._joint_ids_canonical]

    def _init_targets(self, env_ids) -> None:
        q = self._current_pos(env_ids)
        base = torch.clamp(q[:, : self._nb], self._base_lo[env_ids], self._base_hi[env_ids])
        self._base_target[env_ids] = base
        self._finger_target[env_ids] = q[:, self._nb :]
        self._applied[env_ids] = torch.cat([base, q[:, self._nb :]], dim=-1)
        if self._nt:
            mid = (
                self._fixed_tendons[env_ids]
                if self._fixed_tendons is not None
                else 0.5 * (self._t_lo[env_ids] + self._t_hi[env_ids])
            )
            self._tendon_target[env_ids] = mid
            self._tendon_applied[env_ids] = mid
        self.clip_fraction[env_ids] = 0.0
        self.bound_violation[env_ids] = False

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        a = torch.clamp(actions, -1.0, 1.0)
        # Raw actions arrive in the canonical layout.
        a_base = a[:, : self._nb]
        a_f = a[:, self._nb : self._nb + self._nf] if self.cfg.finger_mode != "fixed" else None
        a_t = a[:, self._nb + self._nf :] if self._nt else None

        if self.cfg.base_mode == "absolute":
            unclamped = self._abs_centre + a_base * self._abs_scale
            if self.cfg.base_slew:
                step = self._base_rate * self._step_dt
                unclamped = torch.minimum(
                    torch.maximum(unclamped, self._base_target - step), self._base_target + step
                )
        else:
            unclamped = self._base_target + a_base * self._base_rate * self._step_dt
        self._base_target = torch.clamp(unclamped, self._base_lo, self._base_hi)
        clipped = (unclamped - self._base_target).abs() > 1e-9
        self.clip_fraction = clipped.float().mean(dim=-1)
        self.clip_lo = clipped & (unclamped < self._base_target)
        self.clip_hi = clipped & (unclamped > self._base_target)

        if self.cfg.finger_mode == "fixed":
            ft = self._fixed_fingers
        elif self.cfg.finger_mode == "absolute":
            ft = a_f * self._f_scale + self._f_offset
        else:
            ft = self._finger_target + a_f * self.cfg.finger_vel_max * self._step_dt
        self._finger_target = torch.clamp(ft, self._finger_lo, self._finger_hi)

        ab, af = self.cfg.base_ema, self.cfg.finger_ema
        self._applied[:, : self._nb] = (
            ab * self._base_target + (1 - ab) * self._applied[:, : self._nb]
        )
        self._applied[:, self._nb :] = (
            af * self._finger_target + (1 - af) * self._applied[:, self._nb :]
        )
        if self._nt:
            centre, half = 0.5 * (self._t_hi + self._t_lo), 0.5 * (self._t_hi - self._t_lo)
            if self.cfg.finger_mode == "fixed":
                tt = self._fixed_tendons
            elif self.cfg.finger_mode == "delta":
                tt = self._tendon_target + a_t * self.cfg.finger_vel_max * self._step_dt
            else:
                tt = centre + a_t * half
            self._tendon_target = torch.minimum(torch.maximum(tt, self._t_lo), self._t_hi)
            self._tendon_applied = af * self._tendon_target + (1 - af) * self._tendon_applied
        self._processed_actions = (
            torch.cat([self._applied, self._tendon_applied], dim=-1)
            if self._nt
            else self._applied.clone()
        )

        # Monitor actual root joints (closed-loop tracking) against workspace bounds.
        qb = self._entity.data.joint_pos[:, self._joint_ids_canonical[: self._nb]]
        self.bound_violation = (
            (qb < self._ws_lo - self._viol_tol) | (qb > self._ws_hi + self._viol_tol)
        ).any(dim=-1)

    def apply_actions(self) -> None:
        self._entity.set_joint_position_target(self._applied, joint_ids=self._joint_ids_canonical)
        if self._held_ids is not None:
            self._entity.set_joint_position_target(self._held_pos, joint_ids=self._held_ids)
        if self._nt:
            self._entity.set_tendon_len_target(self._tendon_applied, tendon_ids=self._tendon_ids)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        self._init_targets(env_ids)


# ---------------------------------------------------------------------- utilities
def actuator_of(model: mujoco.MjModel, entity: str, target: str, kind: str = "joint") -> int:
    """Index of the single actuator driving ``<entity>/<target>`` (a joint or a tendon) in the compiled model."""
    obj, trn = (
        (mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtTrn.mjTRN_JOINT)
        if kind == "joint"
        else (mujoco.mjtObj.mjOBJ_TENDON, mujoco.mjtTrn.mjTRN_TENDON)
    )
    tid = mujoco.mj_name2id(model, obj, f"{entity}/{target}")
    assert tid >= 0, f"{kind} {entity}/{target} not found"
    acts = [
        a
        for a in range(model.nu)
        if int(model.actuator_trntype[a]) == int(trn) and int(model.actuator_trnid[a, 0]) == tid
    ]
    assert len(acts) == 1, (target, acts)
    return acts[0]
