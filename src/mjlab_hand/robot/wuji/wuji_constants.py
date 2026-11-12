"""Wuji hand constants and EntityCfg for mjlab."""

from __future__ import annotations

from pathlib import Path

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from mjlab_hand.utils import update_assets

##
# MJCF and assets.
##

_HERE = Path(__file__).parent

WUJI_XML: Path = _HERE / "xmls" / "right.xml"
WUJI_FLOATING_XML: Path = _HERE / "xmls" / "right_floating.xml"
assert WUJI_XML.exists(), f"Missing MJCF: {WUJI_XML}"
assert WUJI_FLOATING_XML.exists(), f"Missing MJCF: {WUJI_FLOATING_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
    """Embed mesh assets into MjSpec.assets."""
    assets: dict[str, bytes] = {}
    side = "left" if "left" in meshdir else "right"
    update_assets(assets, WUJI_XML.parent / "assets" / side, meshdir)
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


def get_spec() -> mujoco.MjSpec:
    """Create MjSpec and attach embedded assets."""
    spec = mujoco.MjSpec.from_file(str(WUJI_XML))
    _name_unnamed_geoms(spec)
    spec.assets = get_assets(spec.meshdir)
    return spec


def get_floating_spec() -> mujoco.MjSpec:
    """Create floating-base MjSpec with 6-DOF base."""
    spec = mujoco.MjSpec.from_file(str(WUJI_FLOATING_XML))
    _name_unnamed_geoms(spec)
    spec.assets = get_assets(spec.meshdir)
    return spec


##
# Collision config.
##

WUJI_KEYPOINT_GEOMS = r"right_finger.*_link4_collision_1"

FULL_COLLISION = CollisionCfg(
    contype=1,
    conaffinity=1,
    geom_names_expr=(".*_collision.*",),
    condim={
        WUJI_KEYPOINT_GEOMS: 6,
        ".*_collision.*": 3,
    },
    friction={
        WUJI_KEYPOINT_GEOMS: (1.0, 5e-3, 5e-4),
        ".*_collision.*": (0.6,),
    },
    solref={
        WUJI_KEYPOINT_GEOMS: (0.01, 1),
    },
    priority={
        WUJI_KEYPOINT_GEOMS: 1,
        ".*": 0,
    },
)


##
# Initial pose (slightly curled, non-singular).
##

WUJI_HOME_POS: tuple[float, float, float] = (0.0, 0.0, 0.5)
WUJI_HOME_ROT: tuple[float, float, float, float] = (0.87, 0.0, -0.50, 0.0)
WUJI_HOME_JOINT_POS: dict[str, float] = {
    # Thumb (finger 1)
    "right_finger1_joint1": 1.5,
    "right_finger1_joint2": 0.0,
    "right_finger1_joint3": 0.5,
    "right_finger1_joint4": 0.5,
    # Index (finger 2)
    "right_finger2_joint1": 0.3,
    "right_finger2_joint2": 0.0,
    "right_finger2_joint3": 0.4,
    "right_finger2_joint4": 0.4,
    # Middle (finger 3)
    "right_finger3_joint1": 0.3,
    "right_finger3_joint2": 0.0,
    "right_finger3_joint3": 0.4,
    "right_finger3_joint4": 0.4,
    # Ring (finger 4)
    "right_finger4_joint1": 0.3,
    "right_finger4_joint2": 0.0,
    "right_finger4_joint3": 0.4,
    "right_finger4_joint4": 0.4,
    # Pinky (finger 5)
    "right_finger5_joint1": 0.3,
    "right_finger5_joint2": 0.0,
    "right_finger5_joint3": 0.4,
    "right_finger5_joint4": 0.4,
}


def make_home_state(
    pos: tuple[float, float, float] = WUJI_HOME_POS,
    rot: tuple[float, float, float, float] = WUJI_HOME_ROT,
) -> EntityCfg.InitialStateCfg:
    """Return a fresh InitialStateCfg so callers never share state."""
    return EntityCfg.InitialStateCfg(
        pos=pos,
        rot=rot,
        joint_pos=dict(WUJI_HOME_JOINT_POS),
        joint_vel={".*": 0.0},
    )


##
# Action space limits.
#
# Affine map from normalized actions in [-1, 1] to joint position targets:
#   target = action * scale + offset
##


def _compute_action_limits(
    spec: mujoco.MjSpec,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute scale/offset from XML joint ranges."""
    scale: dict[str, float] = {}
    offset: dict[str, float] = {}
    for jnt in spec.joints:
        if jnt.type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        lo, hi = float(jnt.range[0]), float(jnt.range[1])
        scale[jnt.name] = (hi - lo) / 2.0
        offset[jnt.name] = (hi + lo) / 2.0
    return scale, offset


WUJI_ACTION_LIMIT_SCALE, WUJI_ACTION_LIMIT_OFFSET = _compute_action_limits(get_spec())


##
# Actuator config.
#
# Wuji XML already defines 20 position actuators with tuned kp/kv.
# We reference them directly via XmlActuatorCfg.
##

WUJI_XML_ACTUATORS = XmlActuatorCfg(
    target_names_expr=("right_finger.*",),
)

WUJI_BASE_XML_ACTUATORS = XmlActuatorCfg(
    target_names_expr=("base_(x|y|z|roll|pitch|yaw)",),
)

ARTICULATION = EntityArticulationInfoCfg(
    actuators=(WUJI_XML_ACTUATORS,),
    soft_joint_pos_limit_factor=0.9,
)

ARTICULATION_FLOATING = EntityArticulationInfoCfg(
    actuators=(WUJI_BASE_XML_ACTUATORS, WUJI_XML_ACTUATORS),
    soft_joint_pos_limit_factor=0.9,
)


##
# Final EntityCfg factories.
##


def get_wuji_hand_cfg() -> EntityCfg:
    """Return mjlab EntityCfg for Wuji right hand (fixed base)."""
    return EntityCfg(
        init_state=make_home_state(),
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=ARTICULATION,
    )


def get_wuji_floating_hand_cfg(
    pos: tuple[float, float, float] = WUJI_HOME_POS,
    rot: tuple[float, float, float, float] = WUJI_HOME_ROT,
) -> EntityCfg:
    """Return floating-base Wuji hand cfg with 6-DOF base."""
    return EntityCfg(
        init_state=make_home_state(pos=pos, rot=rot),
        collisions=(FULL_COLLISION,),
        spec_fn=get_floating_spec,
        articulation=ARTICULATION_FLOATING,
    )


if __name__ == "__main__":
    import mujoco.viewer
    from mjlab.entity.entity import Entity

    hand = Entity(get_wuji_hand_cfg())
    mujoco.viewer.launch(hand.spec.compile())
