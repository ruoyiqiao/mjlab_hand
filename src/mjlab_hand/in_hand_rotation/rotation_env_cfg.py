from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from mjlab_hand.defaults import NUM_TRAIN_ENVS
from mjlab_hand.in_hand_rotation import mdp as rotation_mdp
from mjlab_hand.in_hand_rotation.mdp import RotationCommandCfg
from mjlab_hand.in_hand_rotation.mdp.actions import JointPositionEMAActionCfg
from mjlab_hand.mdp.contact_explorer import (
    contact_explorer_reset_buffers,
    contact_explorer_reward,
)
from mjlab_hand.mdp.contact_sensor import make_keypoint_object_contact_sensor_cfg
from mjlab_hand.object.object import make_contactdb_spec_fn
from mjlab_hand.robot.hand_spec import HandSpec


def make_rotation_env_cfg(
    spec: HandSpec,
    *,
    contactdb_object: str | None = "cylinder_medium",
    contactdb_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> ManagerBasedRlEnvCfg:
    palm_cfg = SceneEntityCfg("hand", body_names=[spec.palm_body])

    policy_terms = {
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("hand")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("hand")},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "object_pos": ObservationTermCfg(
            func=rotation_mdp.object_position_in_base,
            params={
                "object_name": "object",
                "asset_cfg": SceneEntityCfg("hand", body_names=[spec.palm_body]),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "object_quat": ObservationTermCfg(
            func=rotation_mdp.object_orientation_in_base,
            params={
                "object_name": "object",
                "asset_cfg": SceneEntityCfg("hand", body_names=[spec.palm_body]),
            },
        ),
        "object_linvel": ObservationTermCfg(
            func=rotation_mdp.object_linear_velocity,
            params={"object_name": "object", "vel_scale": 0.2},
        ),
        "object_angvel": ObservationTermCfg(
            func=rotation_mdp.object_angular_velocity,
            params={"object_name": "object", "vel_scale": 0.2},
        ),
        "goal_quat": ObservationTermCfg(
            func=rotation_mdp.goal_orientation,
            params={"command_name": "rotation"},
        ),
        "goal_rel_rot": ObservationTermCfg(
            func=rotation_mdp.goal_relative_rotation,
            params={"command_name": "rotation", "object_name": "object"},
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }

    critic_terms = {**policy_terms}

    observations = {
        "actor": ObservationGroupCfg(policy_terms, enable_corruption=True),
        "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
    }

    observations["actor"].nan_policy = "error"
    observations["critic"].nan_policy = "error"

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionEMAActionCfg(
            entity_name="hand",
            actuator_names=(".*",),
            scale=1.0,
            offset=0.0,
            use_default_offset=True,
            ema_alpha=0.8,
        ),
    }

    commands: dict[str, CommandTermCfg] = {
        "rotation": RotationCommandCfg(
            entity_name="object",
            resampling_time_range=(8.0, 12.0),
            debug_vis=True,
            success_tolerance=0.1,
            fall_dist=0.24,
            goal_pos_offset_from_hand_base=(0.0, 0.0, 0.12),
            hand_asset_cfg=SceneEntityCfg("hand", body_names=[spec.palm_body]),
        )
    }

    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("hand"),
            },
        ),
        "reset_hand_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.1, 0.1),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("hand", joint_names=(".*",)),
            },
        ),
        "keypoint_friction_slide": EventTermCfg(
            mode="startup",
            func=mdp.dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("hand", geom_names=spec.keypoint_geom_pattern),
                "operation": "abs",
                "distribution": "uniform",
                "axes": [0],
                "ranges": (0.7, 1.3),
            },
        ),
        "keypoint_friction_spin": EventTermCfg(
            mode="startup",
            func=mdp.dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("hand", geom_names=spec.keypoint_geom_pattern),
                "operation": "abs",
                "distribution": "log_uniform",
                "axes": [1],
                "ranges": (1e-4, 2e-2),
            },
        ),
        "keypoint_friction_roll": EventTermCfg(
            mode="startup",
            func=mdp.dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("hand", geom_names=spec.keypoint_geom_pattern),
                "operation": "abs",
                "distribution": "log_uniform",
                "axes": [2],
                "ranges": (1e-5, 5e-3),
            },
        ),
    }

    if contactdb_object is not None:
        events["contact_explorer_reset"] = EventTermCfg(
            func=contact_explorer_reset_buffers,
            mode="reset",
            params={
                "contactdb_object_name": contactdb_object,
                "contactdb_scale": contactdb_scale,
                "use_goal_state_feature": True,
            },
        )
    rewards = {
        "rot_rew": RewardTermCfg(
            func=rotation_mdp.rotation_alignment_reward,
            weight=0.02,
            params={
                "command_name": "rotation",
                "object_name": "object",
                "rot_eps": 0.1,
            },
        ),
        "reach_goal_bonus": RewardTermCfg(
            func=rotation_mdp.reach_goal_bonus,
            weight=0.01,
            params={
                "command_name": "rotation",
                "object_name": "object",
                "success_tolerance": 0.1,
                "bonus": 100.0,
            },
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-6),
    }

    if contactdb_object is not None:
        _keypoints_cfg = SceneEntityCfg(
            "hand", body_names=list(spec.keypoint_bodies), preserve_order=True
        )
        rewards["contact_explorer_reward"] = RewardTermCfg(
            func=contact_explorer_reward,
            weight=0.02,
            params={
                "scene_object_name": "object",
                "contactdb_object_name": contactdb_object,
                "contactdb_scale": contactdb_scale,
                "contact_sensor_name": "keypoint_contacts",
                "asset_cfg": SceneEntityCfg("hand"),
                "keypoints_cfg": _keypoints_cfg,
                "num_keypoints": spec.num_keypoints,
                "num_key_states": 32,
                "reward_scale": 1.0,
                "kernel_param": 0.03,
                "command_name": "rotation",
                "use_goal_state_feature": True,
            },
        )

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "nan_term": TerminationTermCfg(func=mdp.nan_detection, time_out=False),
        "object_dropped": TerminationTermCfg(
            func=rotation_mdp.object_dropped,
            params={
                "command_name": "rotation",
                "object_name": "object",
                "fall_dist": 0.24,
            },
        ),
    }

    _sensors = ()
    if contactdb_object is not None:
        _sensors = (make_keypoint_object_contact_sensor_cfg(spec.keypoint_geom_pattern),)

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=NUM_TRAIN_ENVS,
            env_spacing=0.75,
            sensors=_sensors,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="hand",
            body_name=spec.palm_body,
            distance=0.5,
            elevation=-30.0,
            azimuth=150.0,
        ),
        sim=SimulationCfg(
            nconmax=100,
            njmax=800,
            mujoco=MujocoCfg(
                timestep=0.010,
                iterations=20,
                ls_iterations=20,
                impratio=10,
                cone="elliptic",
            ),
        ),
        decimation=5,
        episode_length_s=20.0,
    )

    cfg.scene.entities["hand"] = spec.get_fixed_cfg()

    if contactdb_object is not None:
        cfg.scene.entities["object"] = EntityCfg(
            spec_fn=make_contactdb_spec_fn(contactdb_object, scale=contactdb_scale),
        )

    if spec.action_limit_scale is not None or spec.action_limit_offset is not None:
        joint_pos_action = cfg.actions["joint_pos"]
        if spec.action_limit_scale is not None:
            joint_pos_action.scale = spec.action_limit_scale
        if spec.action_limit_offset is not None:
            joint_pos_action.offset = spec.action_limit_offset
        joint_pos_action.use_default_offset = False

    return cfg
