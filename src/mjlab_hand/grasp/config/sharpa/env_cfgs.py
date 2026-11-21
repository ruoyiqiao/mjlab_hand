from math import sqrt

import numpy as np
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_hand.grasp.grasp_env_cfg import make_grasp_env_cfg
from mjlab_hand.grasp.mdp.actions import FloatingHandDeltaActionCfg
from mjlab_hand.robot.sharpa_wave import get_sharpa_hand_spec
from mjlab_hand.robot.sharpa_wave.sharpa_constants import HandSide


def sharpa_grasp_env_cfg(
    play: bool = False,
    *,
    side: HandSide = "left",
) -> ManagerBasedRlEnvCfg:
    cfg = make_grasp_env_cfg(get_sharpa_hand_spec(side))
    hand = cfg.scene.entities["hand"]
    # The official floating model places the left/right palm 0.15 m to either
    # side of its root. Compensate in the task pose, preserving the asset.
    hand.init_state.pos = (-0.14, 0.17 if side == "left" else -0.17, 0.66)
    hand.init_state.rot = (sqrt(0.5), 0.0, sqrt(0.5), 0.0)
    hand.init_state.joint_pos = {
        f"{side}_{joint}": value
        for joint, value in {
            "thumb_CMC_FE": 0.8,
            "thumb_CMC_AA": 0.0,
            "thumb_MCP_FE": 0.3,
            "thumb_MCP_AA": 0.0,
            "thumb_IP": 0.2,
            "pinky_CMC": 0.1,
            **{
                f"{finger}_{joint}": 0.0 if joint == "MCP_AA" else 0.15
                for finger in ("index", "middle", "ring", "pinky")
                for joint in ("MCP_FE", "MCP_AA", "PIP", "DIP")
            },
        }.items()
    }
    cfg.events["reset_hand_joints"].params["asset_cfg"] = SceneEntityCfg(
        "hand", joint_names=(f"{side}_.*",)
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
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
    return cfg
