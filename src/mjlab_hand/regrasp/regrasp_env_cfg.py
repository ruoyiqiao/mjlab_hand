"""Regrasp task definition.

DexPBT reference: ``allegro_kuka_regrasping.py`` (+ base): single center
keypoint (position goal only), target position uniform in a volume, on every
success ``_reset_target`` puts the object back on the table
(``reset_object_pose``) and clears ``lifted_object``; the hand keeps its state.
IsaacGymEnvs: successSteps 30 (cumulative), episode 300 steps per goal, tolerance
curriculum 0.075 -> 0.01.

Here: success = position error <= 0.03 m held continuously 0.5 s (tolerance x
curriculum scale 2.5 -> 1.0), object must be lifted and in hand contact.
"""

from __future__ import annotations

from mjlab_hand.tasks_common.env_builder import SingleHandTaskSpec, TaskTiming
from mjlab_hand.tasks_common.keypoints import CENTER_KEYPOINT
from mjlab_hand.tasks_common.layouts import CUBE, GOAL_MAX, GOAL_MIN, TABLE, single_hand_workspace
from mjlab_hand.tasks_common.tracker import DexPBTTaskCommandCfg


def make_regrasp_spec(hand: str = "allegro") -> SingleHandTaskSpec:
    cmd = DexPBTTaskCommandCfg(
        task="regrasp",
        hand_names=("hand",),
        object_size=CUBE.size,
        object_rest_height=CUBE.edge / 2,
        keypoint_pattern=CENTER_KEYPOINT,
        goal_pos_min=GOAL_MIN,
        goal_pos_max=GOAL_MAX,
        goal_orientation=False,
        pos_tolerance=0.03,
        tolerance_scale_init=2.5,
        hold_time_s=0.5,
        # Sharpa (2026-11-17): swats the cube off the table (episodes ~42 steps, cube ~1.8 m from the goal, lift bonus always paid, 0 grasp
        # reward). Same fix as Grasp-Throw: lifting counts only while thumb + >= 1 fingertip hold the object. LEAP / Wuji regrasp already
        # reach 95.7-96.4 % without it, so the gate is limited to the hands that need it.
        lift_requires_grasp=hand in ("sharpa", "xhand", "shadow"),
        hold_requires_grasp=hand == "xhand",
    )
    return SingleHandTaskSpec(
        task="regrasp",
        command=cmd,
        workspace=single_hand_workspace(hand=hand),
        hand=hand,
        # Shadow and XHand farmed the per-step grasp shaping (held the cube, never lifted: Shadow lifted 0.00 after 3000 it, XHand 0.01 after
        # 2000 it); without it Shadow regrasp lifts within 1000 it and passes (96.3 / 96.4 %).
        grasp_contact_weight=0.0 if hand in ("shadow", "xhand") else 0.005,
        timing=TaskTiming(episode_s=30.0, per_goal_s=8.0, max_successes=50),
        table=TABLE,
        cube=CUBE,
        use_curriculum=True,
        goal_bonus_mode="on_success",
        goal_timeout_is_failure=True,
    )
