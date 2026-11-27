from mjlab.envs import ManagerBasedRlEnvCfg

from mjlab_hand.in_hand_rotation.mdp.actions import JointPositionEMAActionCfg
from mjlab_hand.in_hand_rotation.rotation_env_cfg import make_rotation_env_cfg
from mjlab_hand.robot.shadow_hand import SHADOW_HAND
from mjlab_hand.robot.shadow_hand.shadow_constants import TendonPositionActionCfg

_OBJECT_SCALE = (1.0, 1.0, 1.0)


def shadow_rotation_env_cfg(
    play: bool = False,
    use_tendon_control: bool = True,
) -> ManagerBasedRlEnvCfg:
    cfg = make_rotation_env_cfg(
        SHADOW_HAND,
        contactdb_scale=_OBJECT_SCALE,
    )
    cfg.commands["rotation"].goal_pos_offset_from_hand_base = (-0.10, 0.0, 0.12)
    cfg.commands["rotation"].obj_init_pos_offset_from_hand_base = (-0.10, 0.0, 0.12)

    cfg.actions = SHADOW_HAND.extra_actions_factory(use_tendon_control=use_tendon_control)

    # Smooth joint position targets with EMA while preserving Shadow affine mapping.
    joint_pos_action = cfg.actions["joint_pos"]
    cfg.actions["joint_pos"] = JointPositionEMAActionCfg(
        entity_name=joint_pos_action.entity_name,
        actuator_names=joint_pos_action.actuator_names,
        clip=joint_pos_action.clip,
        preserve_order=joint_pos_action.preserve_order,
        scale=joint_pos_action.scale,
        offset=joint_pos_action.offset,
        use_default_offset=joint_pos_action.use_default_offset,
        ema_alpha=0.8,
    )
    if "tendon_pos" in cfg.actions:
        tendon_pos_action = cfg.actions["tendon_pos"]
        cfg.actions["tendon_pos"] = TendonPositionActionCfg(
            entity_name=tendon_pos_action.entity_name,
            actuator_names=tendon_pos_action.actuator_names,
            clip=tendon_pos_action.clip,
            preserve_order=tendon_pos_action.preserve_order,
            scale=tendon_pos_action.scale,
            offset=tendon_pos_action.offset,
            use_default_offset=tendon_pos_action.use_default_offset,
            ema_alpha=0.8,
        )

    if play:
        cfg.scene.num_envs = 1
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False

    return cfg
