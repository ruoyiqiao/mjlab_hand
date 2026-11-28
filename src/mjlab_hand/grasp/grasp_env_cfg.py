import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
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
from mjlab_hand.grasp import mdp as grasp_mdp
from mjlab_hand.mdp.contact_sensor import (
    make_hand_force_sensor_cfg,
    make_keypoint_object_contact_sensor_cfg,
)
from mjlab_hand.object.object import make_contactdb_spec_fn
from mjlab_hand.robot.hand_spec import HandSpec


def make_grasp_env_cfg(
    spec: HandSpec,
    *,
    contactdb_object: str = "cylinder_medium",
    contactdb_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> ManagerBasedRlEnvCfg:
    """Create a complete grasping task configuration for the given hand.

    The returned cfg is fully wired with the HandSpec — no secondary mutation
    helper is required.
    """
    _base_cfg = SceneEntityCfg("hand", body_names=[spec.palm_body])
    _keypoints_cfg = SceneEntityCfg(
        "hand", body_names=list(spec.keypoint_bodies), preserve_order=True
    )

    policy_terms = {
        "hand_dof_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("hand")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "hand_dof_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("hand")},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "hand_root_pos": ObservationTermCfg(
            func=grasp_mdp.hand_position,
            params={"asset_cfg": _base_cfg},
            noise=Unoise(n_min=-0.005, n_max=0.005),
        ),
        "hand_root_quat": ObservationTermCfg(
            func=grasp_mdp.hand_orientation,
            params={"asset_cfg": _base_cfg},
        ),
        "hand_root_linvel": ObservationTermCfg(
            func=grasp_mdp.hand_linear_velocity,
            params={"asset_cfg": _base_cfg, "vel_scale": 0.2},
        ),
        "hand_root_angvel": ObservationTermCfg(
            func=grasp_mdp.hand_angular_velocity,
            params={"asset_cfg": _base_cfg, "vel_scale": 0.2},
        ),
        "object_pos": ObservationTermCfg(
            func=grasp_mdp.object_position,
            params={"object_name": "object"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "object_quat": ObservationTermCfg(
            func=grasp_mdp.object_orientation,
            params={"object_name": "object"},
        ),
        "object_pos_rel_hand": ObservationTermCfg(
            func=grasp_mdp.object_position_in_base,
            params={"object_name": "object", "asset_cfg": _base_cfg},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "object_quat_rel_hand": ObservationTermCfg(
            func=grasp_mdp.object_orientation_in_base,
            params={"object_name": "object", "asset_cfg": _base_cfg},
        ),
        "object_linvel": ObservationTermCfg(
            func=grasp_mdp.object_linear_velocity,
            params={"object_name": "object", "vel_scale": 0.2},
        ),
        "object_angvel": ObservationTermCfg(
            func=grasp_mdp.object_angular_velocity,
            params={"object_name": "object", "vel_scale": 0.2},
        ),
        "object_goal_pos": ObservationTermCfg(
            func=grasp_mdp.object_goal_position,
            params={"command_name": "pose"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "object_goal_dist": ObservationTermCfg(
            func=grasp_mdp.object_goal_distance,
            params={"object_name": "object", "command_name": "pose"},
        ),
        "keypoint_pos_rel": ObservationTermCfg(
            func=grasp_mdp.keypoint_positions_in_base,
            params={
                "base_cfg": _base_cfg,
                "keypoints_cfg": _keypoints_cfg,
            },
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

    actions: dict[str, ActionTermCfg] = {}

    commands = {
        "grasp_metrics": grasp_mdp.GraspMetricsCommandCfg(
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=False,
            hand_name="hand",
            object_name="object",
            table_name="table",
            asset_cfg=SceneEntityCfg("hand", body_names=[spec.palm_body]),
            keypoints_cfg=SceneEntityCfg(
                "hand", body_names=list(spec.keypoint_bodies), preserve_order=True
            ),
            frame_scale=0.03,
            axis_radius=0.005,
            contactdb_object_name=contactdb_object,
            contactdb_scale=contactdb_scale,
        ),
        "pose": grasp_mdp.PoseCommandCfg(
            resampling_time_range=(10.0, 15.0),
            object_name="object",
            table_name="table",
            pos_x_min=-0.15,
            pos_x_max=0.15,
            pos_y_min=-0.15,
            pos_y_max=0.15,
            pos_z_min=0.1,
            pos_z_max=0.5,
            min_height_above_table=0.05,
            debug_vis=True,
        ),
    }

    events = {
        "reset_scene_to_default": EventTermCfg(
            func=mdp.reset_scene_to_default,
            mode="reset",
        ),
        "reset_hand_base": EventTermCfg(
            func=grasp_mdp.reset_hand_base,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.05, 0.05),
                    "y": (-0.05, 0.05),
                    "z": (-0.02, 0.02),
                    "roll": (-0.1, 0.1),
                    "pitch": (-0.1, 0.1),
                    "yaw": (-0.1, 0.1),
                },
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
        "reset_object": EventTermCfg(
            func=grasp_mdp.reset_object_on_table,
            mode="reset",
            params={
                "object_name": "object",
                "table_name": "table",
                "position_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                "orientation_range": {
                    "roll": (-1.57, 1.57),
                    "pitch": (-1.57, 1.57),
                    "yaw": (-3.14, 3.14),
                },
            },
        ),
        "contact_explorer_reset": EventTermCfg(
            func=grasp_mdp.contact_explorer_reset_buffers,
            mode="reset",
            params={
                "contactdb_object_name": contactdb_object,
                "contactdb_scale": contactdb_scale,
                "use_goal_state_feature": True,
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
                "ranges": (0.5, 1.0),
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

    # Reward formulation from paper:
    # r = r_smooth + r_grasp + I_grasped * r_goal
    # r_grasp = r_approach + (1 - I_grasped) * r_lift
    rewards = {
        # r_smooth: penalize L1 joint velocities
        "smoothness": RewardTermCfg(
            func=grasp_mdp.smoothness_reward,
            weight=0.0001,
            params={
                "asset_cfg": SceneEntityCfg("hand", joint_names=(".*",)),
                "joint_names": (".*",),
                "scale": 0.1,
            },
        ),
        # r_approach: per-finger progress toward object (pre-lift)
        "approach": RewardTermCfg(
            func=grasp_mdp.EpisodicApproachReward,
            weight=0.01,
            params={
                "object_name": "object",
                "asset_cfg": SceneEntityCfg("hand"),
                "keypoints_cfg": _keypoints_cfg,
                "lift_threshold": 0.05,
            },
        ),
        # r_goal: dense progress + success bonus toward goal (post-lift)
        "goal_progress": RewardTermCfg(
            func=grasp_mdp.GoalProgressReward,
            weight=0.01,
            params={
                "object_name": "object",
                "command_name": "pose",
                "success_tolerance": 0.03,
                "success_bonus": 10.0,
            },
        ),
        "contact_explorer_reward": RewardTermCfg(
            func=grasp_mdp.contact_explorer_reward,
            weight=0.01,
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
                "kernel_param": 0.05,
                "command_name": "pose",
                "use_goal_state_feature": True,
                "force_sensor_name": "hand_force",
                "force_threshold": 5e10,
            },
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "nan_term": TerminationTermCfg(func=mdp.nan_detection, time_out=False),
        "object_fallen": TerminationTermCfg(
            func=grasp_mdp.object_fallen,
            params={
                "object_name": "object",
                "table_name": "table",
                "fall_height": -0.2,
            },
        ),
    }

    metrics = {
        "object_height_beyond_table": MetricsTermCfg(
            func=grasp_mdp.object_height_beyond_table,
            params={
                "object_name": "object",
                "table_name": "table",
            },
        ),
        "hand_object_distance": MetricsTermCfg(
            func=grasp_mdp.hand_object_distance,
            params={
                "object_name": "object",
                "asset_cfg": SceneEntityCfg("hand", body_names=[spec.palm_body]),
            },
        ),
    }

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=NUM_TRAIN_ENVS,
            env_spacing=1.0,
            sensors=(
                make_keypoint_object_contact_sensor_cfg(spec.keypoint_geom_pattern),
                make_hand_force_sensor_cfg(),
            ),
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        metrics=metrics,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="hand",
            body_name=spec.palm_body,
            distance=1.0,
            elevation=-30.0,
            azimuth=150.0,
        ),
        sim=SimulationCfg(
            nconmax=200,
            njmax=1000,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=50,
                ls_iterations=50,
                impratio=10,
                cone="elliptic",
            ),
        ),
        decimation=4,
        episode_length_s=10.0,
        scale_rewards_by_dt=False,
    )

    cfg.scene.entities = {
        "hand": spec.make_floating_entity_cfg(),
        "table": make_grasp_table_entity_cfg(),
        "object": EntityCfg(
            spec_fn=make_contactdb_spec_fn(contactdb_object, scale=contactdb_scale),
        ),
    }

    return cfg


_TABLE_XML = """
<mujoco model="table">
  <worldbody>
    <body name="table" pos="0 0 0.4">
      <geom name="table_surface" type="box" size="0.3 0.3 0.02"
            rgba="0.5 0.35 0.2 1" friction="1.0 0.005 0.0001"
            solref="0.005 1" solimp="0.99 0.99 0.001"/>
    </body>
  </worldbody>
</mujoco>
"""


def get_table_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string(_TABLE_XML)


def make_grasp_table_entity_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=get_table_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )
