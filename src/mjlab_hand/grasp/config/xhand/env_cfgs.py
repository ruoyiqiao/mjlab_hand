from math import sqrt

import numpy as np
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_hand.grasp.grasp_env_cfg import make_grasp_env_cfg
from mjlab_hand.grasp.mdp.actions import FloatingHandDeltaActionCfg
from mjlab_hand.robot.xhand import XHAND_HAND
from mjlab_hand.robot.xhand.xhand_constants import PARAMETERS


def xhand_grasp_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = make_grasp_env_cfg(XHAND_HAND)
    # A downward-facing open palm can enclose the object from above. The
    # screwdriver home used by the asset is not a table-grasp initial pose.
    hand = cfg.scene.entities["hand"]
    hand.init_state.pos = (0.0, -0.10, 0.68)
    hand.init_state.rot = (sqrt(0.5), sqrt(0.5), 0.0, 0.0)
    hand.init_state.joint_pos = {
        "left_hand_index_bend_joint": 0.0,
        **{
            f"left_hand_{finger}_joint{joint}": 0.15
            for finger in ("index", "mid", "pinky", "ring")
            for joint in (1, 2)
        },
        "left_hand_thumb_bend_joint": 1.0,
        "left_hand_thumb_rota_joint1": 0.25,
        "left_hand_thumb_rota_joint2": 0.15,
    }
    # Base pose noise is already applied by reset_hand_base. Restrict the
    # following joint event so it does not overwrite that pose with +/-0.1 m.
    cfg.events["reset_hand_joints"].params["asset_cfg"] = SceneEntityCfg(
        "hand", joint_names=("left_hand_.*",)
    )
    cfg.actions = {
        "delta_control": FloatingHandDeltaActionCfg(
            entity_name="hand",
            actuator_names=(".*",),
            trans_vel_min=-0.2,
            trans_vel_max=0.2,
            rot_vel_min=-np.pi / 4,
            rot_vel_max=np.pi / 4,
            joint_vel_min=-1.0,
            joint_vel_max=1.0,
            ema_alpha=0.3,
            use_relative=True,
            clip_to_joint_limits=True,
        )
    }
    cfg.sim.mujoco.timestep = PARAMETERS["sim_dt"]
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
    return cfg
