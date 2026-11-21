"""Left and right official Sharpa Wave hands."""

from functools import partial

from mjlab_hand.robot.hand_spec import HandSpec
from mjlab_hand.robot.sharpa_wave.sharpa_constants import (
    HandSide,
    asset_path,
    get_sharpa_floating_hand_cfg,
    get_sharpa_hand_cfg,
)


def get_sharpa_hand_spec(side: HandSide = "left") -> HandSpec:
    asset_path(side)
    return HandSpec(
        name=f"sharpa_{side}",
        palm_body=f"{side}_hand_C_MC",
        keypoint_bodies=tuple(
            f"{side}_{finger}_DP" for finger in ("thumb", "index", "middle", "ring", "pinky")
        ),
        keypoint_geom_pattern=rf"{side}_(thumb|index|middle|ring|pinky)_elastomer",
        num_keypoints=5,
        get_fixed_cfg=partial(get_sharpa_hand_cfg, side=side),
        get_floating_cfg=partial(get_sharpa_floating_hand_cfg, side=side),
        floating_init_pos=(-0.2, 0.0, 0.8),
        floating_init_rot=(0.71, 0.0, 0.71, 0.0),
    )


SHARPA_LEFT_HAND = get_sharpa_hand_spec("left")
SHARPA_RIGHT_HAND = get_sharpa_hand_spec("right")
SHARPA_HAND = SHARPA_LEFT_HAND

__all__ = ["SHARPA_HAND", "SHARPA_LEFT_HAND", "SHARPA_RIGHT_HAND", "get_sharpa_hand_spec"]
