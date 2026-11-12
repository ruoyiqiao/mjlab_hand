"""Sharpa hand constants."""

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

SHARPA_XML: Path = _HERE / "xmls" / "left_sharpa_ha4_v2_1.xml"
SHARPA_FLOATING_XML: Path = _HERE / "xmls" / "left_sharpa_ha4_v2_1_floating.xml"
assert SHARPA_XML.exists()
assert SHARPA_FLOATING_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, SHARPA_XML.parent / "meshes", meshdir)
    return assets


def _get_spec(xml_path: Path) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(xml_path))
    spec.assets = get_assets(spec.meshdir)
    return spec


def get_spec() -> mujoco.MjSpec:
    return _get_spec(SHARPA_XML)


def get_floating_spec() -> mujoco.MjSpec:
    return _get_spec(SHARPA_FLOATING_XML)


def _compute_action_limits(
    spec: mujoco.MjSpec,
) -> tuple[dict[str, float], dict[str, float]]:
    scale: dict[str, float] = {}
    offset: dict[str, float] = {}
    for jnt in spec.joints:
        if jnt.type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        lo, hi = float(jnt.range[0]), float(jnt.range[1])
        scale[jnt.name] = (hi - lo) / 2.0
        offset[jnt.name] = (hi + lo) / 2.0
    return scale, offset


SHARPA_ACTION_LIMIT_SCALE, SHARPA_ACTION_LIMIT_OFFSET = _compute_action_limits(get_spec())

SHARPA_KEYPOINT_GEOMS = r".*elastomer"

FULL_COLLISION = CollisionCfg(
    contype=1,
    conaffinity=1,
    geom_names_expr=(".*",),
    condim={
        SHARPA_KEYPOINT_GEOMS: 6,
        ".*": 3,
    },
    friction={
        SHARPA_KEYPOINT_GEOMS: (1.0, 5e-3, 5e-4),
        ".*": (0.6,),
    },
    solref={
        SHARPA_KEYPOINT_GEOMS: (0.01, 1),
    },
    priority={
        SHARPA_KEYPOINT_GEOMS: 1,
        ".*": 0,
    },
)

##
# Keyframe config.
#
# Slightly curled open pose for stable object cradling.
##

SHARPA_HOME_POS: tuple[float, float, float] = (0.0, 0.0, 0.5)
# rot=(0.71, 0.0, -0.71, 0.0) is the original 90deg variant; we use a 60deg tilt.
SHARPA_HOME_ROT: tuple[float, float, float, float] = (0.87, 0.0, -0.50, 0.0)
SHARPA_HOME_JOINT_POS: dict[str, float] = {
    "left_thumb_CMC_FE": 0.5,
    "left_thumb_CMC_AA": 0.0,
    "left_thumb_MCP_FE": 0.3,
    "left_thumb_MCP_AA": 0.0,
    "left_thumb_IP": 0.3,
    "left_index_MCP_FE": 0.3,
    "left_index_MCP_AA": 0.0,
    "left_index_PIP": 0.4,
    "left_index_DIP": 0.3,
    "left_middle_MCP_FE": 0.3,
    "left_middle_MCP_AA": 0.0,
    "left_middle_PIP": 0.4,
    "left_middle_DIP": 0.3,
    "left_ring_MCP_FE": 0.3,
    "left_ring_MCP_AA": 0.0,
    "left_ring_PIP": 0.4,
    "left_ring_DIP": 0.3,
    "left_pinky_CMC": 0.1,
    "left_pinky_MCP_FE": 0.3,
    "left_pinky_MCP_AA": 0.0,
    "left_pinky_PIP": 0.4,
    "left_pinky_DIP": 0.3,
}


def make_home_state(
    pos: tuple[float, float, float] = SHARPA_HOME_POS,
    rot: tuple[float, float, float, float] = SHARPA_HOME_ROT,
) -> EntityCfg.InitialStateCfg:
    """Return a fresh InitialStateCfg so callers never share state."""
    return EntityCfg.InitialStateCfg(
        pos=pos,
        rot=rot,
        joint_pos=dict(SHARPA_HOME_JOINT_POS),
        joint_vel={".*": 0.0},
    )


##
# Actuator config.
#
# Wrap existing XML position actuators.
##

SHARPA_ACTUATOR_CFG = XmlActuatorCfg(
    target_names_expr=("left_.*",),
)

ARTICULATION = EntityArticulationInfoCfg(
    actuators=(SHARPA_ACTUATOR_CFG,),
    soft_joint_pos_limit_factor=0.9,
)

SHARPA_BASE_ACTUATOR_CFG = XmlActuatorCfg(
    target_names_expr=("base_(x|y|z|roll|pitch|yaw)",),
)

ARTICULATION_FLOATING = EntityArticulationInfoCfg(
    actuators=(SHARPA_BASE_ACTUATOR_CFG, SHARPA_ACTUATOR_CFG),
    soft_joint_pos_limit_factor=0.9,
)


def get_sharpa_hand_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=make_home_state(),
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=ARTICULATION,
    )


def get_sharpa_floating_hand_cfg(
    pos: tuple[float, float, float] = SHARPA_HOME_POS,
    rot: tuple[float, float, float, float] = SHARPA_HOME_ROT,
) -> EntityCfg:
    return EntityCfg(
        init_state=make_home_state(pos=pos, rot=rot),
        collisions=(FULL_COLLISION,),
        spec_fn=get_floating_spec,
        articulation=ARTICULATION_FLOATING,
    )


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    hand = Entity(get_sharpa_hand_cfg())
    viewer.launch(hand.spec.compile())
