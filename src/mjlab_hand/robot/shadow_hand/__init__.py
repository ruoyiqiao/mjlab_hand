"""Shadow hand package."""

from mjlab_hand.robot.hand_spec import HandSpec
from mjlab_hand.robot.shadow_hand.shadow_constants import (
    SHADOW_KEYPOINT_GEOMS,
    get_shadow_actions,
    get_shadow_floating_hand_cfg,
    get_shadow_hand_cfg,
)

SHADOW_HAND = HandSpec(
    name="shadow",
    palm_body="lh_palm",
    keypoint_bodies=(
        "lh_ffmiddle",
        "lh_mfmiddle",
        "lh_rfmiddle",
        "lh_lfmiddle",
        "lh_thmiddle",
    ),
    keypoint_geom_pattern=SHADOW_KEYPOINT_GEOMS,
    num_keypoints=5,
    get_fixed_cfg=get_shadow_hand_cfg,
    get_floating_cfg=get_shadow_floating_hand_cfg,
    has_tendon_actuation=True,
    extra_actions_factory=get_shadow_actions,
    floating_init_pos=(-0.4, 0.0, 0.9),
)

__all__ = ["SHADOW_HAND"]
