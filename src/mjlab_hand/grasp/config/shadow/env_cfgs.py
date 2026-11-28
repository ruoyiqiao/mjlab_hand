import numpy as np
from mjlab.envs import ManagerBasedRlEnvCfg

from mjlab_hand.grasp.grasp_env_cfg import make_grasp_env_cfg
from mjlab_hand.grasp.mdp.actions import (
    FloatingHandDeltaActionCfg,
    TendonDeltaActionCfg,
)
from mjlab_hand.robot.shadow_hand import SHADOW_HAND


def shadow_grasp_env_cfg(
    play: bool = False,
    use_tendon_control: bool = True,
) -> ManagerBasedRlEnvCfg:
    cfg = make_grasp_env_cfg(SHADOW_HAND)
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
    if use_tendon_control:
        base_actions = SHADOW_HAND.extra_actions_factory(use_tendon_control=True)
        if "tendon_pos" in base_actions:
            tendon_action = base_actions["tendon_pos"]
            cfg.actions["tendon_control"] = TendonDeltaActionCfg(
                entity_name=tendon_action.entity_name,
                actuator_names=tendon_action.actuator_names,
                scale=tendon_action.scale,
                offset=tendon_action.offset,
                tendon_vel_min=-1.0,
                tendon_vel_max=1.0,
                ema_alpha=0.3,
            )
    if play:
        cfg.scene.num_envs = 1
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
    return cfg
