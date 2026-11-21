"""Hand-root workspace description for arm-free floating Allegro hands.

Frames
------
* ``anchor_pos`` / ``anchor_rot`` are **env-local** (relative to
  ``env.scene.env_origins``); they become the entity ``init_state`` pose, i.e. the
  pose of the auto-generated mocap root that carries the hand. ``env_origins`` are
  added exactly once by ``reset_scene_to_default``.
* The six base joints (``base_x/y/z`` slides, ``base_roll/pitch/yaw`` hinges) are
  chained joints on the ``palm`` body, so their axes live in the palm frame at the
  anchor (anchor_rot * palm body quat). ``joint_bounds`` therefore bound the root
  pose *relative to the anchor* in that frame, which is exactly what the actuator
  target controls. The world-frame meaning of each axis is computed by
  ``describe_world_axes`` and serialized into ``root_layout.json``.
* Bounds are enforced on the final applied actuator targets (see
  :class:`mjlab_hand.tasks_common.actions.BoundedFloatingHandAction`), and actual
  joint positions are monitored for violations.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field

BASE_JOINTS: tuple[str, ...] = (
    "base_x",
    "base_y",
    "base_z",
    "base_roll",
    "base_pitch",
    "base_yaw",
)


@dataclass(frozen=True)
class HandRootWorkspace:
    """Anchor pose, bounds and limits of one floating hand root."""

    side: str
    """``"left"`` or ``"right"``."""

    entity_name: str
    """Scene entity name of the hand (e.g. ``"hand"``, ``"hand_left"``)."""

    anchor_pos: tuple[float, float, float]
    """Env-local anchor position (m)."""

    anchor_rot: tuple[float, float, float, float]
    """Env-local anchor orientation, quaternion **wxyz**."""

    joint_bounds: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "base_x": (-0.3, 0.3),
            "base_y": (-0.3, 0.3),
            "base_z": (-0.3, 0.3),
            "base_roll": (-0.6, 0.6),
            "base_pitch": (-0.6, 0.6),
            "base_yaw": (-0.6, 0.6),
        }
    )
    """Bounds on base joint targets relative to the anchor (m / rad, palm frame)."""

    lin_vel_max: float = 0.5
    """Max base translation target speed per axis (m/s)."""

    ang_vel_max: float = 2.0
    """Max base rotation target speed per axis (rad/s)."""

    reset_noise: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "base_x": (-0.02, 0.02),
            "base_y": (-0.02, 0.02),
            "base_z": (-0.02, 0.02),
            "base_roll": (-0.05, 0.05),
            "base_pitch": (-0.05, 0.05),
            "base_yaw": (-0.05, 0.05),
        }
    )
    """Uniform reset noise added to base joints at episode reset (m / rad)."""

    violation_tol: float = 0.02
    """Tolerance (m) for flagging actual base translation joints outside bounds."""

    violation_tol_rot: float = 0.05
    """Tolerance (rad) for base rotation joints; verify_actions measured up to 0.03 rad
    overshoot under saturated rotation with finger-table contact (DualHand left, pitch)."""

    base_rot_joint_range: float = 1.25
    """MJCF range (rad, symmetric) set on base_roll/pitch/yaw and their position
    actuators for these tasks only (the shared XML has +-0.7853). Must exceed the
    rotation bounds so the bound, not the joint limit, is what clamps."""

    notes: str = ""

    def __post_init__(self) -> None:
        if self.side not in ("left", "right"):
            raise ValueError(f"side must be left/right, got {self.side}")
        for name in BASE_JOINTS:
            lo, hi = self.joint_bounds[name]
            if not lo < hi:
                raise ValueError(f"empty bound for {name}: {(lo, hi)}")
            nlo, nhi = self.reset_noise.get(name, (0.0, 0.0))
            if nlo < lo or nhi > hi:
                raise ValueError(f"reset noise for {name} exceeds bounds")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
