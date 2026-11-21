"""XHand1 left hand sourced from DexScrew."""

from mjlab_hand.robot.hand_spec import HandSpec
from mjlab_hand.robot.xhand.xhand_constants import (
    XHAND_ACTION_LIMIT_OFFSET,
    XHAND_ACTION_LIMIT_SCALE,
    XHAND_KEYPOINT_BODIES,
    XHAND_KEYPOINT_GEOMS,
    get_xhand_cfg,
    get_xhand_floating_cfg,
)

XHAND_HAND = HandSpec(
    name="xhand",
    palm_body="left_hand_link",
    keypoint_bodies=XHAND_KEYPOINT_BODIES,
    keypoint_geom_pattern=XHAND_KEYPOINT_GEOMS,
    num_keypoints=5,
    get_fixed_cfg=get_xhand_cfg,
    get_floating_cfg=get_xhand_floating_cfg,
    action_limit_scale=XHAND_ACTION_LIMIT_SCALE,
    action_limit_offset=XHAND_ACTION_LIMIT_OFFSET,
    floating_init_pos=(-0.2, 0.0, 0.8),
    floating_init_rot=(0.70710678, 0.0, -0.70710678, 0.0),
)

__all__ = ["XHAND_HAND"]
