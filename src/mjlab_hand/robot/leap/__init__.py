"""LEAP hand package."""

from functools import partial

from mjlab_hand.robot.hand_spec import HandSpec
from mjlab_hand.robot.leap.leap_constants import (
    LEAP_ACTION_LIMIT_OFFSET,
    LEAP_ACTION_LIMIT_SCALE,
    LEAP_KEYPOINT_GEOMS,
    get_leap_floating_hand_cfg,
    get_leap_hand_cfg,
)

LEAP_HAND = HandSpec(
    name="leap",
    palm_body="palm",
    keypoint_bodies=("if_ds", "mf_ds", "rf_ds", "th_ds"),
    keypoint_geom_pattern=LEAP_KEYPOINT_GEOMS,
    num_keypoints=4,
    get_fixed_cfg=get_leap_hand_cfg,
    get_floating_cfg=get_leap_floating_hand_cfg,
    action_limit_scale=LEAP_ACTION_LIMIT_SCALE,
    action_limit_offset=LEAP_ACTION_LIMIT_OFFSET,
)

# Left hand for the task layer: the floating model is generated from menagerie's left_hand.xml
# (tools/allegro4/make_floating_xml.py). Action limits come from the joint ranges (the left thumb CMC range differs).
LEAP_HAND_LEFT = HandSpec(
    name="leap_left",
    palm_body="palm",
    keypoint_bodies=("if_ds", "mf_ds", "rf_ds", "th_ds"),
    keypoint_geom_pattern=LEAP_KEYPOINT_GEOMS,
    num_keypoints=4,
    get_fixed_cfg=get_leap_hand_cfg,
    get_floating_cfg=partial(get_leap_floating_hand_cfg, side="left"),
)
LEAP_HAND_RIGHT = LEAP_HAND

# Same right hand with menagerie's own finger position actuators (kp 3.0, kv 0.01; no SysID torque limit, no delay).
# Used by LEAP Juggle2 only: Juggle2 learns with it, the other LEAP tasks only with the SysID PD.
LEAP_HAND_MENAGERIE = HandSpec(
    name="leap_menagerie",
    palm_body="palm",
    keypoint_bodies=("if_ds", "mf_ds", "rf_ds", "th_ds"),
    keypoint_geom_pattern=LEAP_KEYPOINT_GEOMS,
    num_keypoints=4,
    get_fixed_cfg=get_leap_hand_cfg,
    get_floating_cfg=partial(get_leap_floating_hand_cfg, actuators="menagerie"),
    action_limit_scale=LEAP_ACTION_LIMIT_SCALE,
    action_limit_offset=LEAP_ACTION_LIMIT_OFFSET,
)

__all__ = ["LEAP_HAND", "LEAP_HAND_LEFT", "LEAP_HAND_MENAGERIE", "LEAP_HAND_RIGHT"]
