from math import cos, radians, sin, sqrt

from mjlab.envs import ManagerBasedRlEnvCfg

from mjlab_hand.in_hand_rotation.mdp.actions import JointPositionDeltaActionCfg
from mjlab_hand.in_hand_rotation.rotation_env_cfg import make_rotation_env_cfg
from mjlab_hand.robot.xhand import XHAND_HAND
from mjlab_hand.robot.xhand.xhand_constants import PARAMETERS

_OBJECT_SCALE = (0.8, 0.8, 0.8)
_OBJECT_OFFSET = (0.0, -0.10, 0.07)
_ROTATION_HOME_JOINT_POS = {
    "left_hand_index_bend_joint": 0.0,
    **{
        f"left_hand_{finger}_joint{joint}": 0.3
        for finger in ("index", "mid", "pinky", "ring")
        for joint in (1, 2)
    },
    "left_hand_thumb_bend_joint": 1.0,
    "left_hand_thumb_rota_joint1": 0.25,
    "left_hand_thumb_rota_joint2": 0.15,
}


def xhand_rotation_env_cfg(
    play: bool = False,
    *,
    palm_tilt_deg: float = 0.0,
) -> ManagerBasedRlEnvCfg:
    """Build the task; positive palm tilt raises the fingers relative to the wrist."""
    cfg = make_rotation_env_cfg(
        XHAND_HAND,
        contactdb_scale=_OBJECT_SCALE,
    )
    # The upstream screwdriver pose faces downward. General object rotation
    # starts with the palm facing upward and enough room for the 64 mm cylinder.
    hand = cfg.scene.entities["hand"]
    # XHand's local palm normal is -Y, and its open fingers point along -Z.
    # Pitch about world -X from the palm-up pose, raising the fingertips.
    tilt = radians(palm_tilt_deg)
    half_cos, half_sin = cos(tilt / 2), sin(tilt / 2)
    hand.init_state.rot = (
        sqrt(0.5) * (half_cos - half_sin),
        -sqrt(0.5) * (half_cos + half_sin),
        0.0,
        0.0,
    )
    # RotationCommand offsets use world axes. Rotate both together to preserve
    # the palm-local object/goal position when changing the hand's pitch.
    x, y, z = _OBJECT_OFFSET
    object_offset = (
        x,
        y * cos(tilt) + z * sin(tilt),
        -y * sin(tilt) + z * cos(tilt),
    )
    hand.init_state.joint_pos = _ROTATION_HOME_JOINT_POS.copy()
    cfg.commands["rotation"].goal_pos_offset_from_hand_base = object_offset
    cfg.commands["rotation"].obj_init_pos_offset_from_hand_base = object_offset

    # Match DexScrew's command integration: target += 0.05 * action, with
    # physical joint limits and the same nominal PD gains/effort limits.
    cfg.actions["joint_pos"] = JointPositionDeltaActionCfg(
        entity_name="hand",
        actuator_names=(".*",),
        use_default_offset=False,
        use_soft_joint_pos_limits=False,
        delta_min=-PARAMETERS["action_scale"],
        delta_max=PARAMETERS["action_scale"],
    )
    cfg.sim.mujoco.timestep = PARAMETERS["sim_dt"]
    cfg.decimation = PARAMETERS["control_decimation"]
    if play:
        cfg.scene.num_envs = 1
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False

    return cfg
