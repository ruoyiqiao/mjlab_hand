from mjlab.envs import ManagerBasedRlEnvCfg

from mjlab_hand.in_hand_rotation.rotation_env_cfg import make_rotation_env_cfg
from mjlab_hand.robot.sharpa_wave import get_sharpa_hand_spec
from mjlab_hand.robot.sharpa_wave.sharpa_constants import HandSide

_OBJECT_SCALE = (0.8, 0.8, 0.8)


def sharpa_rotation_env_cfg(
    play: bool = False,
    *,
    side: HandSide = "left",
) -> ManagerBasedRlEnvCfg:
    cfg = make_rotation_env_cfg(
        get_sharpa_hand_spec(side),
        contactdb_scale=_OBJECT_SCALE,
    )
    cfg.commands["rotation"].goal_pos_offset_from_hand_base = (-0.05, 0.0, 0.12)
    cfg.commands["rotation"].obj_init_pos_offset_from_hand_base = (-0.05, 0.0, 0.20)

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False

    return cfg
