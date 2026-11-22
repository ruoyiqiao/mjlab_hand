"""Grasp-Throw task definition.

DexPBT reference: ``allegro_kuka_throw.py`` (+ base). Same reward as the base task
(single center keypoint, goal = bucket + 0.05 z, no tolerance curriculum, success
steps 5, no random forces, episode 300 steps = 5 s per goal in IsaacGymEnvs).

Deviations from DexPBT:
* bucket within an arm-free throwing range (outside the hand's reach);
* success needs the full chain grasp -> lift -> release -> valid flight -> settled
  inside the bucket (DexPBT counts keypoint proximity only);
* post-release table/floor contact terminates as a failed throw.
"""

from __future__ import annotations

from mjlab_hand.tasks_common.dexpbt_objects import THROW_SETS
from mjlab_hand.tasks_common.env_builder import SingleHandTaskSpec, TaskTiming
from mjlab_hand.tasks_common.keypoints import CENTER_KEYPOINT
from mjlab_hand.tasks_common.layouts import (
    BUCKET,
    BUCKET_POS_MAX,
    BUCKET_POS_MIN,
    CUBE,
    TABLE,
    single_hand_workspace,
)
from mjlab_hand.tasks_common.tracker import DexPBTTaskCommandCfg


def make_grasp_throw_spec(hand: str = "allegro") -> SingleHandTaskSpec:
    cmd = DexPBTTaskCommandCfg(
        task="throw",
        hand_names=("hand",),
        bucket_name="bucket",
        object_size=CUBE.size,
        object_rest_height=CUBE.edge / 2,
        keypoint_pattern=CENTER_KEYPOINT,
        goal_orientation=False,
        pos_tolerance=0.1125,  # DexPBT throw: 0.075 * keypointScale 1.5 (unused for success)
        tolerance_scale_init=1.0,
        hold_time_s=0.1,  # DexPBT throw successSteps 5
        bucket_pos_min=BUCKET_POS_MIN,
        bucket_pos_max=BUCKET_POS_MAX,
        bucket_floor_top=BUCKET.floor_thickness,
        bucket_floor_inner_radius=BUCKET.floor_inner_radius,
        bucket_rim_inner_radius=BUCKET.rim_inner_radius,
        bucket_height=BUCKET.height,
        goal_height_above_bucket=0.05,
        lift_requires_grasp=hand != "allegro",
    )
    return SingleHandTaskSpec(
        task="throw",
        command=cmd,
        workspace=single_hand_workspace(lin_vel_max=4.0, ang_vel_max=10.0, hand=hand),
        hand=hand,
        timing=TaskTiming(episode_s=24.0, per_goal_s=8.0, max_successes=50),
        table=TABLE,
        cube=CUBE,
        bucket=BUCKET,
        object_sets=THROW_SETS,  # env/throw.yaml: cube + small cuboids only
        use_curriculum=False,
    )
