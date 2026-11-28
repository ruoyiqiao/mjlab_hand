import numpy as np
from mjlab.envs import ManagerBasedRlEnvCfg

from mjlab_hand.grasp.grasp_env_cfg import make_grasp_env_cfg
from mjlab_hand.grasp.mdp.actions import FloatingHandDeltaActionCfg
from mjlab_hand.robot.leap import LEAP_HAND


def leap_grasp_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = make_grasp_env_cfg(LEAP_HAND)
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
        cfg.scene.num_envs = 1
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
    return cfg
