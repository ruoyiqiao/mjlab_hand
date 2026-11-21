"""Allegro hand package."""

from mjlab_hand.robot.allegro.allegro_constants import (
    ALLEGRO_ACTION_LIMIT_OFFSET,
    ALLEGRO_ACTION_LIMIT_SCALE,
    ALLEGRO_KEYPOINT_GEOMS,
    get_allegro_floating_hand_cfg,
    get_allegro_hand_cfg,
    get_allegro_right_fixed_hand_cfg,
    get_allegro_right_floating_hand_cfg,
)
from mjlab_hand.robot.hand_spec import HandSpec

ALLEGRO_HAND = HandSpec(
    name="allegro",
    palm_body="palm",
    keypoint_bodies=("ff_tip_head", "mf_tip_head", "rf_tip_head", "th_tip_head"),
    keypoint_geom_pattern=ALLEGRO_KEYPOINT_GEOMS,
    num_keypoints=4,
    get_fixed_cfg=get_allegro_hand_cfg,
    get_floating_cfg=get_allegro_floating_hand_cfg,
    action_limit_scale=ALLEGRO_ACTION_LIMIT_SCALE,
    action_limit_offset=ALLEGRO_ACTION_LIMIT_OFFSET,
)

# Explicit side descriptors for multi-hand tasks. ALLEGRO_HAND (left) is unchanged.
ALLEGRO_HAND_LEFT = ALLEGRO_HAND

ALLEGRO_HAND_RIGHT = HandSpec(
    name="allegro_right",
    palm_body="palm",
    keypoint_bodies=("ff_tip_head", "mf_tip_head", "rf_tip_head", "th_tip_head"),
    keypoint_geom_pattern=ALLEGRO_KEYPOINT_GEOMS,
    num_keypoints=4,
    get_fixed_cfg=get_allegro_right_fixed_hand_cfg,
    get_floating_cfg=get_allegro_right_floating_hand_cfg,
    action_limit_scale=ALLEGRO_ACTION_LIMIT_SCALE,
    action_limit_offset=ALLEGRO_ACTION_LIMIT_OFFSET,
)

__all__ = ["ALLEGRO_HAND", "ALLEGRO_HAND_LEFT", "ALLEGRO_HAND_RIGHT"]
