from mjlab.envs import ManagerBasedRlEnvCfg

from mjlab_hand.in_hand_rotation.rotation_env_cfg import make_rotation_env_cfg
from mjlab_hand.robot.allegro import ALLEGRO_HAND


def allegro_rotation_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = make_rotation_env_cfg(
        ALLEGRO_HAND,
        contactdb_object="cylinder_medium",
        contactdb_scale=(1.0, 1.0, 1.0),
    )

    if play:
        cfg.scene.num_envs = 1
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False

    return cfg
