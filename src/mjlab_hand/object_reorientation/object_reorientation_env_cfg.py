"""Object-Reorientation task definition.

DexPBT reference: ``allegro_kuka_reorientation.py`` (+ base): 4 corner keypoints,
goal = random position in a target volume + uniform random quaternion, only the
goal is resampled on success (the object keeps being manipulated), extra reset
when any fingertip is > 1.5 m from the object, tolerance curriculum
0.075 -> 0.01 (x0.9 every 3000 steps when mean successes >= 3).

Here: success = position error <= 0.03 m and angle error <= 0.20 rad held
continuously for 0.5 s (tolerances x curriculum scale 2.5 -> 1.0). The cube is
treated without symmetry (full SO(3) target; colored faces show orientation).
"""

from __future__ import annotations

from mjlab_hand.tasks_common.env_builder import SingleHandTaskSpec, TaskTiming
from mjlab_hand.tasks_common.keypoints import CORNER_KEYPOINTS
from mjlab_hand.tasks_common.layouts import CUBE, GOAL_MAX, GOAL_MIN, TABLE, single_hand_workspace
from mjlab_hand.tasks_common.tracker import DexPBTTaskCommandCfg


def make_object_reorientation_spec(hand: str = "allegro") -> SingleHandTaskSpec:
    cmd = DexPBTTaskCommandCfg(
        task="reorient",
        hand_names=("hand",),
        object_size=CUBE.size,
        object_rest_height=CUBE.edge / 2,
        keypoint_pattern=CORNER_KEYPOINTS,
        goal_pos_min=GOAL_MIN,
        goal_pos_max=GOAL_MAX,
        goal_orientation=True,
        pos_tolerance=0.03,
        ang_tolerance=0.20,
        tolerance_scale_init=2.5,
        hold_time_s=0.5,
        # Non-Allegro hands (2026-11-17): training-only angle curriculum, as in DualHand. LEAP holds the cube and brings it within
        # 11-15 cm but never learns the orientation (0.3 successes/episode after 8000 it, Allegro breaks through at it ~1800); with
        # exponent 1.0 the angle tolerance starts at 0.5 rad. 2.955 makes it ~3 rad at scale 2.5 (position-only successes first, so
        # the tolerance curriculum starts advancing) and exactly 0.20 rad at scale 1.0 (evaluation is unchanged).
        ang_tol_exponent=2.955 if hand != "allegro" else 1.0,
        # Shadow swats the cube (lifted steps: hand contact 17 %, thumb opposition 0.2 %, cube 0.7 m from the goal, 0 successes in 3000 it);
        # as for throw and Shadow/Sharpa regrasp, lifting only counts while the thumb + >= 1 fingertip hold it. Other hands hold the cube
        # without the thumb-opposed contact this gate needs (the gate made LEAP worse), so it stays off for them.
        # Sharpa / XHand reorientation (2026-11-17, s42_b1 runs): same swatting signature (0 successes, cube 0.7-0.9 m from the goal after
        # 2000-2700 it); the gate fixed Sharpa regrasp.
        lift_requires_grasp=hand in ("shadow", "sharpa", "xhand"),
        # XHand (2026-11-17): with the lift gate every run ended at 0 successes and lifted 0 (the cube is held on the table, never
        # lifted). Hold gate instead: no lift requirement, but a success needs the cube held (thumb + >= 1 fingertip) at the goal.
        hold_requires_grasp=hand == "xhand",
    )
    return SingleHandTaskSpec(
        task="reorient",
        command=cmd,
        workspace=single_hand_workspace(hand=hand),
        hand=hand,
        # Keep the per-step grasp shaping here: with the lift gate it is the only signal that leads to the grasp the gate requires
        # (Shadow reorientation with gate and without shaping: lifted 0.00 after 2200 it). Regrasp drops it for Shadow / XHand.
        grasp_contact_weight=0.005,
        curriculum_min_successes=3.0 if hand == "allegro" else 1.0,
        timing=TaskTiming(episode_s=40.0, per_goal_s=10.0, max_successes=50),
        table=TABLE,
        cube=CUBE,
        use_curriculum=True,
        hand_far_reset=True,
    )
