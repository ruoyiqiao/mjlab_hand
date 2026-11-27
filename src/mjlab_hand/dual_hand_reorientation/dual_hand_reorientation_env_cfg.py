"""DualHand-Reorientation: two floating Allegro hands (left / right MJCF), one shared object.

DexPBT two-arms semantics (``allegro_kuka_two_arms*.py``) without arms: the object starts on a random side, the goal
(random quaternion) is on a random side, so about half of the goals need a handover. Fingertip progress is summed over
both hands and not gated by lift; no velocity penalties; tolerance curriculum. Hand-hand contact is monitored.

Layouts: ``default`` (original), ``sep`` (each hand's reach stops 2 cm before the centre plane: hands cannot touch),
``far`` (anchors +-0.34 m, root 1.5 m/s / 5 rad/s, stiffer root PD; a handover through the table ends the episode).
``handoff_shaping`` / ``track`` are training-only; play and evaluation use the same rules.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from mjlab_hand.robot.allegro import ALLEGRO_HAND_LEFT, ALLEGRO_HAND_RIGHT
from mjlab_hand.tasks_common import mdp as tmdp
from mjlab_hand.tasks_common import sensors
from mjlab_hand.tasks_common.dexpbt_objects import ALL_SETS, make_dexpbt_object_cfg
from mjlab_hand.tasks_common.env_builder import (
    TaskTiming,
    common_terminations,
    dexpbt_rewards,
    hand_obs_terms,
    hand_reset_events,
    make_action_cfg,
    make_hand_entity_cfg,
    make_sim_cfg,
    task_obs_terms,
    track_root_gains,
)
from mjlab_hand.tasks_common.keypoints import CORNER_KEYPOINTS
from mjlab_hand.tasks_common.layouts import (
    CUBE,
    DUAL_FAR_SIDE_OFFSET,
    DUAL_GOAL_MAX,
    DUAL_GOAL_MIN,
    DUAL_SIDE_OFFSET,
    DUAL_TABLE,
    dual_workspaces,
)
from mjlab_hand.tasks_common.objects import make_table_entity_cfg
from mjlab_hand.tasks_common.tracker import DexPBTTaskCommandCfg, reset_object_event

FAR_SLIDE_GAINS = (
    3000.0,
    120.0,
    400.0,
)  # root PD of the far layout (kp, kv, force limit); default PD lags 0.1-0.25 m
FAR_ROT_GAINS = (300.0, 12.0, 80.0)
HANDOFF_SHAPING = (
    30.0  # x0.01: +30 once per goal for a direct handover to the goal side, -30 through the table
)
TRACK_ACTION_RATE = 1e-3  # weight of the root action-rate penalty of the tracking-style action
TRACK_INERTIA = (
    0.668,
    0.0128,
)  # effective inertia of the root slides (kg) / hinges (kg m^2) at the home pose
DUAL_TIMING = TaskTiming(episode_s=40.0, per_goal_s=10.0, max_successes=50)


def make_dual_command(side_offset: float = DUAL_SIDE_OFFSET) -> DexPBTTaskCommandCfg:
    return DexPBTTaskCommandCfg(
        task="dual_reorient",
        hand_names=("hand_left", "hand_right"),
        hand_profiles=("allegro_left", "allegro_right"),
        object_size=CUBE.size,
        object_rest_height=CUBE.edge / 2,
        keypoint_pattern=CORNER_KEYPOINTS,
        goal_pos_min=DUAL_GOAL_MIN,
        goal_pos_max=DUAL_GOAL_MAX,
        goal_orientation=True,
        dual_side_offset=side_offset,
        dual_object_side_offset=side_offset,
        object_reset_xy=(0.04, 0.08),
        pos_tolerance=0.03,
        ang_tolerance=0.20,
        tolerance_scale_init=2.5,
        ang_tol_exponent=2.955,
        hold_time_s=0.5,
    )


def make_dual_hand_env_cfg(
    play: bool = False,
    layout: str = "default",
    handoff_shaping: bool = False,
    track: bool = False,
    finger_action_rate: float = 0.0,
    finger_ema: float = 1.0,
) -> ManagerBasedRlEnvCfg:
    """layout: default | sep | far. track: absolute slew-limited root action with G1-recipe PD (far layout only).
    finger_action_rate (<= 0) / finger_ema (< 1): finger smoothing (penalty during training; the EMA also in play)."""
    assert layout in ("default", "sep", "far") and (not track or layout == "far")
    far, separated = layout == "far", layout in ("sep", "far")
    cmd = make_dual_command(DUAL_FAR_SIDE_OFFSET if far else DUAL_SIDE_OFFSET)
    cmd.forbid_table_handover = far
    obj_cfg, obj_table = make_dexpbt_object_cfg(ALL_SETS)
    cmd.object_sizes = tuple(s for _, s in obj_table)
    if play:
        cmd.tolerance_scale_init = 1.0
    cmd_name = "dual_goal"
    left_ws, right_ws = dual_workspaces(separated, far)
    slide_gains, rot_gains = (FAR_SLIDE_GAINS, FAR_ROT_GAINS) if far else (None, None)
    if track:
        slide_gains, rot_gains = track_root_gains(TRACK_INERTIA[0], TRACK_INERTIA[1], left_ws)
    hands = ["hand_left", "hand_right"]
    hold_steps = max(1, int(round(cmd.hold_time_s / (0.005 * 4))))
    obs = {}
    for i, h in enumerate(hands):
        obs.update(hand_obs_terms(h, i, cmd_name))
    obs.update(task_obs_terms(cmd_name, orientation=True, throw=False))
    observations = {
        "actor": ObservationGroupCfg(dict(obs), enable_corruption=False),
        "critic": ObservationGroupCfg(dict(obs), enable_corruption=False),
    }
    observations["actor"].nan_policy = "error"
    observations["critic"].nan_policy = "error"

    events = {
        "reset_scene_to_default": EventTermCfg(func=envs_mdp.reset_scene_to_default, mode="reset")
    }
    for h, w in zip(hands, (left_ws, right_ws)):
        events.update(hand_reset_events(h, w))
    events["reset_object"] = EventTermCfg(
        func=reset_object_event, mode="reset", params={"command_name": cmd_name}
    )

    sensor_cfgs = []
    for h in hands:
        sensor_cfgs += [
            sensors.hand_object_contact(h),
            sensors.tip_object_contact(h),
            sensors.hand_table_contact(h),
        ]
    sensor_cfgs += [sensors.object_table_contact(), sensors.hand_hand_contact(history=4)]

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=1,
            env_spacing=2.0,
            sensors=tuple(sensor_cfgs),
        ),
        observations=observations,
        actions={
            h: make_action_cfg(
                h,
                w,
                finger_mode="absolute",
                finger_ema=finger_ema,
                base_ema=1.0,
                base_mode="absolute" if track else "delta",
                base_slew=track,
            )
            for h, w in zip(hands, (left_ws, right_ws))
        },
        commands={cmd_name: cmd},
        events=events,
        rewards=dexpbt_rewards(
            cmd_name, hands, hold_steps, gate_tips_by_lift=False, velocity_penalties=False
        ),
        terminations={
            **common_terminations(cmd_name, DUAL_TIMING),
            "hand_hand_collision": TerminationTermCfg(
                func=tmdp.hand_hand_collision, params={"command_name": cmd_name}
            ),
            **(
                {
                    "table_handover": TerminationTermCfg(
                        func=tmdp.table_handover, params={"command_name": cmd_name}
                    )
                }
                if far
                else {}
            ),
        },
        curriculum={}
        if play
        else {
            "tolerance": CurriculumTermCfg(
                func=tmdp.tolerance_curriculum, params={"command_name": cmd_name}
            )
        },
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.WORLD,
            lookat=(0.0, 0.0, 0.5),
            distance=1.6,
            elevation=-25.0,
            azimuth=90.0,
        ),
        sim=make_sim_cfg(nconmax=300, njmax=1500),
        decimation=4,
        episode_length_s=DUAL_TIMING.episode_s,
        scale_rewards_by_dt=False,
        is_finite_horizon=False,
    )
    if track and not play:
        cfg.rewards["base_action_rate"] = RewardTermCfg(
            func=tmdp.base_action_rate_l2, weight=-TRACK_ACTION_RATE, params={"hands": tuple(hands)}
        )
    if finger_action_rate and not play:
        cfg.rewards["finger_action_rate"] = RewardTermCfg(
            func=tmdp.finger_action_rate_l2,
            weight=finger_action_rate,
            params={"hands": tuple(hands)},
        )
    if handoff_shaping and not play:
        cfg.rewards["direct_handoff"] = RewardTermCfg(
            func=tmdp.direct_handoff_bonus,
            weight=HANDOFF_SHAPING * 0.01,
            params={"command_name": cmd_name},
        )
        cfg.rewards["table_handover_pen"] = RewardTermCfg(
            func=tmdp.table_handover_f,
            weight=-HANDOFF_SHAPING * 0.01,
            params={"command_name": cmd_name},
        )
    cfg.scene.entities = {
        "hand_left": make_hand_entity_cfg(
            ALLEGRO_HAND_LEFT,
            left_ws,
            slide_joint_limits=separated,
            base_gains=slide_gains,
            rot_gains=rot_gains,
        ),
        "hand_right": make_hand_entity_cfg(
            ALLEGRO_HAND_RIGHT,
            right_ws,
            slide_joint_limits=separated,
            base_gains=slide_gains,
            rot_gains=rot_gains,
        ),
        "table": make_table_entity_cfg(DUAL_TABLE),
        "object": obj_cfg,
    }
    return cfg
