"""Shadow hand package."""

from functools import partial

from mjlab_hand.robot.hand_spec import HandSpec
from mjlab_hand.robot.shadow_hand.shadow_constants import (
    SHADOW_KEYPOINT_GEOMS,
    SHADOW_TIP_BODIES,
    SHADOW_TIP_GEOMS,
    for_side,
    get_shadow_actions,
    get_shadow_floating_hand_cfg,
    get_shadow_hand_cfg,
    get_shadow_task_floating_hand_cfg,
    side_prefix,
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


def get_shadow_task_spec(side: str) -> HandSpec:
    """Floating Shadow hand for the task layer: distal-phalanx tips, wrist held at 0, finger action = 14 joints + 4 tendons."""
    return HandSpec(
        name=f"shadow_{side}",
        palm_body=f"{side_prefix(side)}palm",
        keypoint_bodies=for_side(SHADOW_TIP_BODIES, side),
        keypoint_geom_pattern=for_side(SHADOW_TIP_GEOMS, side),
        num_keypoints=5,
        get_fixed_cfg=get_shadow_hand_cfg,
        get_floating_cfg=partial(get_shadow_task_floating_hand_cfg, side=side),
        has_tendon_actuation=True,
    )


SHADOW_HAND_LEFT = get_shadow_task_spec("left")
SHADOW_HAND_RIGHT = get_shadow_task_spec("right")

__all__ = ["SHADOW_HAND", "SHADOW_HAND_LEFT", "SHADOW_HAND_RIGHT", "get_shadow_task_spec"]
