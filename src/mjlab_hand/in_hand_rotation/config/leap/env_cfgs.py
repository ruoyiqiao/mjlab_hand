from mjlab.envs import ManagerBasedRlEnvCfg

from mjlab_hand.in_hand_rotation.rotation_env_cfg import make_rotation_env_cfg
from mjlab_hand.robot.leap import LEAP_HAND


def leap_rotation_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = make_rotation_env_cfg(
        LEAP_HAND,
        contactdb_object="cylinder_medium",
        contactdb_scale=(1.0, 1.0, 1.0),
    )
    cfg.commands["rotation"].goal_pos_offset_from_hand_base = (0.0, 0.0, 0.22)

    if play:
        cfg.scene.num_envs = 1
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False

    return cfg
