"""Frozen layouts (anchor poses, bounds, goal/bucket ranges) for the four tasks.

All positions are env-local metres; quaternions wxyz. The single-hand anchor
puts the palm 0.20 m above the table top, palm normal -z, fingers +x; base joint
axes then map to world as base_x -> -z, base_y -> +y, base_z -> +x (verified by
tools/allegro4/probe_layout.py and saved to root_layout.json). Final values are
frozen after the geometry/reach checks, before the first tuning round.
"""

from __future__ import annotations

from mjlab_hand.tasks_common.hand_profile import get_profile
from mjlab_hand.tasks_common.objects import BucketSpec, CubeSpec, TableSpec
from mjlab_hand.tasks_common.workspace import HandRootWorkspace

PALM_DOWN_FINGERS_PX = (0.0, 1.0, 0.0, 0.0)
PALM_DOWN_FINGERS_NX = (0.0, 0.0, 1.0, 0.0)

TABLE = TableSpec(center=(0.0, 0.0, 0.40), half_size=(0.30, 0.30, 0.02))
TABLE_TOP = TABLE.top_z  # 0.42
CUBE = CubeSpec(edge=0.05, mass=0.05)  # DexPBT density 400 kg/m^3

SINGLE_ANCHOR = (-0.05, 0.0, TABLE_TOP + 0.20)

SINGLE_BOUNDS = {
    "base_x": (-0.25, 0.25),  # world -z
    "base_y": (-0.25, 0.25),  # world +y
    "base_z": (-0.25, 0.25),  # world +x
    # +-1.2 rad: with +-0.6 the policies held roll / yaw at the bound about half of the time
    "base_roll": (-1.2, 1.2),
    "base_pitch": (-1.2, 1.2),
    "base_yaw": (-1.2, 1.2),
}


def single_hand_workspace(
    lin_vel_max: float = 1.0, ang_vel_max: float = 4.0, hand: str = "allegro"
) -> HandRootWorkspace:
    """Palm-down workspace of one floating hand: palm origin 0.20 m above the table, fingers toward +x.

    ``SINGLE_BOUNDS`` are in the canonical hand frame (hand_profile.py), so every hand uses the same bounds."""
    prof = get_profile(hand)
    rot = prof.anchor_rot(down=True, fingers_px=True)
    return HandRootWorkspace(
        side=prof.side,
        entity_name="hand",
        anchor_pos=prof.anchor_pos(SINGLE_ANCHOR, rot),
        anchor_rot=rot,
        joint_bounds=dict(SINGLE_BOUNDS),
        lin_vel_max=lin_vel_max,
        ang_vel_max=ang_vel_max,
        notes=f"Arm-free floating {prof.side} {prof.hand.capitalize()}; palm-down above the table.",
    )


# Goal box for reorientation / regrasp (env-local), reachable by the bounded hand.
GOAL_MIN = (-0.12, -0.12, TABLE_TOP + 0.12)
GOAL_MAX = (0.12, 0.12, TABLE_TOP + 0.30)

# Throw: bucket outside the hand's reach (checked by probe_layout.py), beyond the
# +x table edge, floor top below the table top.
BUCKET = BucketSpec()  # original DexPBT bucket dimensions, round
BUCKET_POS_MIN = (0.50, -0.12, TABLE_TOP - 0.20)
BUCKET_POS_MAX = (0.66, 0.12, TABLE_TOP - 0.12)

# Dual hand: wide table; left hand at -x (fingers +x), right hand at +x (fingers -x).
DUAL_TABLE = TableSpec(center=(0.0, 0.0, 0.40), half_size=(0.55, 0.30, 0.02))
DUAL_HALF_SPAN = 0.30
DUAL_BOUNDS = {
    "base_x": (-0.25, 0.25),
    "base_y": (-0.20, 0.20),
    "base_z": (-0.20, 0.25),  # toward the table center for both hands
    # +-1.2 rad like the single-hand tasks (+-0.5 rad: orientation error stuck at ~2 rad).
    "base_roll": (-1.2, 1.2),
    "base_pitch": (-1.2, 1.2),
    "base_yaw": (-1.2, 1.2),
}
DUAL_SIDE_OFFSET = 0.20
DUAL_GOAL_MIN = (-0.05, -0.10, TABLE_TOP + 0.12)
DUAL_GOAL_MAX = (0.05, 0.10, TABLE_TOP + 0.28)


# Separated layout (like the DexPBT two-arms bases at x = +-1.1 m): base_z (toward the centre) is capped so every
# collision geom of each hand stays >= 2 cm from the centre plane x = 0 for all base rotations / finger poses
# (tools/allegro4/dual_separation.py: reach at base_z = 0 is -0.138 m -> base_z_max 0.118 m). Hands cannot touch.
DUAL_SEP_BASE_Z_MAX = 0.115


# Far layout: hands farther apart than Sep, original table, handover must be direct hand to hand (resting on the
# table in between ends the episode, tracker.indirect_handover). Anchors +-0.34 m; the hand reaches 0.162 m
# toward the centre at base_z = 0 (dual_separation.py), so with base_z <= 0.115 the closest hand parts stay 12.6 cm
# apart (4.6 cm in Sep): the object is passed with a short toss.
DUAL_FAR_HALF_SPAN = 0.34
DUAL_FAR_SIDE_OFFSET = 0.24
DUAL_FAR_LIN_VEL = (
    1.5  # root speed limits of the far layout (m/s, rad/s); 3 / 8 flung the object off the table
)
DUAL_FAR_ANG_VEL = 5.0


def dual_workspaces(
    separated: bool = False,
    far: bool = False,
    lin_vel: float | None = None,
    ang_vel: float | None = None,
) -> tuple[HandRootWorkspace, HandRootWorkspace]:
    span = DUAL_FAR_HALF_SPAN if far else DUAL_HALF_SPAN
    lv = lin_vel if lin_vel is not None else (DUAL_FAR_LIN_VEL if far else 1.0)
    av = ang_vel if ang_vel is not None else (DUAL_FAR_ANG_VEL if far else 4.0)
    left = HandRootWorkspace(
        side="left",
        entity_name="hand_left",
        anchor_pos=(-span, 0.0, TABLE_TOP + 0.20),
        anchor_rot=PALM_DOWN_FINGERS_PX,
        joint_bounds=_dual_bounds(separated or far),
        lin_vel_max=lv,
        ang_vel_max=av,
        notes="left hand, fingers toward +x",
    )
    right = HandRootWorkspace(
        side="right",
        entity_name="hand_right",
        anchor_pos=(span, 0.0, TABLE_TOP + 0.20),
        anchor_rot=PALM_DOWN_FINGERS_NX,
        joint_bounds=_dual_bounds(separated or far),
        lin_vel_max=lv,
        ang_vel_max=av,
        notes="right hand, fingers toward -x",
    )
    return left, right


def _dual_bounds(separated: bool) -> dict:
    b = dict(DUAL_BOUNDS)
    if separated:
        b["base_z"] = (DUAL_BOUNDS["base_z"][0], DUAL_SEP_BASE_Z_MAX)
    return b
