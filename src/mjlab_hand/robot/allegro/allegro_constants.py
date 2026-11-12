"""Allegro hand constants."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg, XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from mjlab_hand.utils import update_assets

##
# MJCF and assets.
##

_HERE = Path(__file__).parent

ALLEGRO_XML: Path = _HERE / "xmls" / "left_hand.xml"
ALLEGRO_FLOATING_XML: Path = _HERE / "xmls" / "left_hand_floating.xml"
assert ALLEGRO_XML.exists()
assert ALLEGRO_FLOATING_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, ALLEGRO_XML.parent / "assets", meshdir)
    return assets


def _name_unnamed_geoms(spec: mujoco.MjSpec) -> None:
    """Assign names to unnamed collision/visual geoms based on parent body."""
    counter: dict[str, int] = {}
    for geom in spec.geoms:
        if geom.name:
            continue
        body_name = geom.parent.name if geom.parent else "world"
        is_visual = geom.contype == 0 and geom.conaffinity == 0
        suffix = "visual" if is_visual else "collision"
        key = f"{body_name}_{suffix}"
        idx = counter.get(key, 0)
        counter[key] = idx + 1
        geom.name = f"{key}_{idx}" if idx > 0 else key


def _get_spec(xml_path: Path) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(xml_path))
    spec.assets = get_assets(spec.meshdir)
    _name_unnamed_geoms(spec)
    return spec


def get_spec() -> mujoco.MjSpec:
    return _get_spec(ALLEGRO_XML)


def get_floating_spec() -> mujoco.MjSpec:
    return _get_spec(ALLEGRO_FLOATING_XML)


##
# Actuator config.
#
# The PD gains below are derived from the real Allegro Hand ROS driver
# (allegro_node_pd). The raw k_p/k_d values from the driver are divided by
# the hardware torqueConversion factor (~800) to obtain the effective
# joint-level stiffness and damping in Nm/rad and Nm/(rad/s).
#
# Joint mapping from C++ (jXY) to MuJoCo names:
#   j00-j03 -> ffj0-ffj3  (first finger)
#   j10-j13 -> mfj0-mfj3  (middle finger)
#   j20-j23 -> rfj0-rfj3  (ring finger)
#   j30-j33 -> thj0-thj3  (thumb)
#
# Manufacturer max torque: 0.7 Nm.
##

EFFORT_LIMIT = 0.7

# Estimated reflected rotor inertia for Allegro Hand gear ratio 369:1.
# Using a conservative motor rotor inertia estimate (~1.5e-8 kg*m^2),
# reflected inertia = 1.5e-8 * 369^2 ≈ 0.002 kg*m^2.
ARMATURE = 0.002

# Base and medial joints of ff/mf/rf, plus thumb distal.
# k_p = 600/800 = 0.75, k_d = 15/800 = 0.01875
ALLEGRO_ACTUATOR_KP0_KD0 = BuiltinPositionActuatorCfg(
    target_names_expr=("ffj0", "ffj2", "mfj0", "mfj2", "rfj0", "rfj2", "thj3"),
    stiffness=1.5,
    damping=0.0375,
    effort_limit=EFFORT_LIMIT,
    armature=ARMATURE,
)

# Proximal joints of ff/mf/rf.
# k_p = 600/800 = 0.75, k_d = 20/800 = 0.025
ALLEGRO_ACTUATOR_KP0_KD1 = BuiltinPositionActuatorCfg(
    target_names_expr=("ffj1", "mfj1", "rfj1"),
    stiffness=1.5,
    damping=0.05,
    effort_limit=EFFORT_LIMIT,
    armature=ARMATURE,
)

# Distal joints of ff/mf/rf.
# k_p = 1000/800 = 1.25, k_d = 15/800 = 0.01875
ALLEGRO_ACTUATOR_KP1_KD0 = BuiltinPositionActuatorCfg(
    target_names_expr=("ffj3", "mfj3", "rfj3"),
    stiffness=2.5,
    damping=0.0375,
    effort_limit=EFFORT_LIMIT,
    armature=ARMATURE,
)

# Thumb base.
# k_p = 1000/800 = 1.25, k_d = 30/800 = 0.0375
ALLEGRO_ACTUATOR_THUMB_BASE = BuiltinPositionActuatorCfg(
    target_names_expr=("thj0",),
    stiffness=2.5,
    damping=0.075,
    effort_limit=EFFORT_LIMIT,
    armature=ARMATURE,
)

# Thumb proximal and medial.
# k_p = 1000/800 = 1.25, k_d = 20/800 = 0.025
ALLEGRO_ACTUATOR_THUMB_PROX_MED = BuiltinPositionActuatorCfg(
    target_names_expr=("thj1", "thj2"),
    stiffness=2.5,
    damping=0.05,
    effort_limit=EFFORT_LIMIT,
    armature=ARMATURE,
)

##
# Keyframe config.
#
# Palm facing up with fingers slightly curled inward to cradle an object.
##

ALLEGRO_HOME_POS: tuple[float, float, float] = (0.0, 0.0, 0.5)
ALLEGRO_HOME_ROT: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
ALLEGRO_HOME_JOINT_POS: dict[str, float] = {
    "ffj0": 0.0,
    "ffj1": 0.4,
    "ffj2": 0.4,
    "ffj3": 0.4,
    "mfj0": 0.0,
    "mfj1": 0.4,
    "mfj2": 0.4,
    "mfj3": 0.4,
    "rfj0": 0.0,
    "rfj1": 0.4,
    "rfj2": 0.4,
    "rfj3": 0.4,
    "thj0": 0.0,
    "thj1": 0.5,
    "thj2": 0.4,
    "thj3": 0.4,
}


def make_home_state(
    pos: tuple[float, float, float] = ALLEGRO_HOME_POS,
    rot: tuple[float, float, float, float] = ALLEGRO_HOME_ROT,
) -> EntityCfg.InitialStateCfg:
    """Return a fresh InitialStateCfg so callers never share state."""
    return EntityCfg.InitialStateCfg(
        pos=pos,
        rot=rot,
        joint_pos=dict(ALLEGRO_HOME_JOINT_POS),
        joint_vel={".*": 0.0},
    )


##
# Collision config.
#
# Enable condim=6 on fingertip capsule geoms for rich frictional contact
# with the manipulation object.
##

ALLEGRO_KEYPOINT_GEOMS = r"(ff|mf|rf|th)_tip_collision"

FULL_COLLISION = CollisionCfg(
    contype=1,
    conaffinity=1,
    geom_names_expr=(".*_collision",),
    condim={
        ALLEGRO_KEYPOINT_GEOMS: 6,
        ".*_collision": 3,
    },
    friction={
        ALLEGRO_KEYPOINT_GEOMS: (1.0, 5e-3, 5e-4),
        ".*_collision": (0.6,),
    },
    solref={
        ALLEGRO_KEYPOINT_GEOMS: (0.01, 1),
    },
    priority={
        ALLEGRO_KEYPOINT_GEOMS: 1,
        ".*": 0,
    },
)

##
# Final config.
##

ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        ALLEGRO_ACTUATOR_KP0_KD0,
        ALLEGRO_ACTUATOR_KP0_KD1,
        ALLEGRO_ACTUATOR_KP1_KD0,
        ALLEGRO_ACTUATOR_THUMB_BASE,
        ALLEGRO_ACTUATOR_THUMB_PROX_MED,
    ),
    soft_joint_pos_limit_factor=0.9,
)

ALLEGRO_BASE_XML_ACTUATOR = XmlActuatorCfg(
    target_names_expr=("base_(x|y|z|roll|pitch|yaw)",),
)

ARTICULATION_FLOATING = EntityArticulationInfoCfg(
    actuators=(
        ALLEGRO_BASE_XML_ACTUATOR,
        ALLEGRO_ACTUATOR_KP0_KD0,
        ALLEGRO_ACTUATOR_KP0_KD1,
        ALLEGRO_ACTUATOR_KP1_KD0,
        ALLEGRO_ACTUATOR_THUMB_BASE,
        ALLEGRO_ACTUATOR_THUMB_PROX_MED,
    ),
    soft_joint_pos_limit_factor=0.9,
)


def get_allegro_hand_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=make_home_state(),
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=ARTICULATION,
    )


def get_allegro_floating_hand_cfg(
    pos: tuple[float, float, float] = ALLEGRO_HOME_POS,
    rot: tuple[float, float, float, float] = ALLEGRO_HOME_ROT,
) -> EntityCfg:
    return EntityCfg(
        init_state=make_home_state(pos=pos, rot=rot),
        collisions=(FULL_COLLISION,),
        spec_fn=get_floating_spec,
        articulation=ARTICULATION_FLOATING,
    )


# ALLEGRO_ACTION_SCALE: dict[str, float] = {}
# for _a in ARTICULATION.actuators:
#   assert isinstance(_a, BuiltinPositionActuatorCfg)
#   _e = _a.effort_limit
#   _s = _a.stiffness
#   _names = _a.target_names_expr
#   assert _e is not None
#   for _n in _names:
#     ALLEGRO_ACTION_SCALE[_n] = 0.25 * _e / _s


# Affine map from normalized actions in [-1, 1] to Allegro joint limits:
# target = action * scale + offset
ALLEGRO_ACTION_LIMIT_SCALE: dict[str, float] = {
    "(rf|mf|ff)j0": 0.47,
    "(rf|mf|ff)j1": 0.903,
    "(rf|mf|ff)j2": 0.9415,
    "(rf|mf|ff)j3": 0.9225,
    "thj0": 0.5665,
    "thj1": 0.634,
    "thj2": 0.9165,
    "thj3": 0.9405,
}

ALLEGRO_ACTION_LIMIT_OFFSET: dict[str, float] = {
    "(rf|mf|ff)j0": 0.0,
    "(rf|mf|ff)j1": 0.707,
    "(rf|mf|ff)j2": 0.7675,
    "(rf|mf|ff)j3": 0.6955,
    "thj0": 0.8295,
    "thj1": 0.529,
    "thj2": 0.7275,
    "thj3": 0.7785,
}


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    hand = Entity(get_allegro_hand_cfg())
    viewer.launch(hand.spec.compile())
