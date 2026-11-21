"""Official Sharpa Wave assets; preserve upstream dynamics and collision policy."""

from functools import partial
from pathlib import Path
from typing import Literal

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from mjlab_hand.robot.mjcf import load_mjcf

HandSide = Literal["left", "right"]
_ASSETS = Path(__file__).parent / "assets" / "wave_01"
BASE_AXES = ("x", "y", "z", "roll", "pitch", "yaw")
SHARPA_KEYPOINT_GEOMS = r"(left|right)_(thumb|index|middle|ring|pinky)_elastomer"
SHARPA_HOME_POS = (0.0, 0.0, 0.5)
SHARPA_HOME_ROT = (0.87, 0.0, -0.50, 0.0)

_HOME_FINGERS = {
    "thumb_CMC_FE": 0.5,
    "thumb_CMC_AA": 0.0,
    "thumb_MCP_FE": 0.3,
    "thumb_MCP_AA": 0.0,
    "thumb_IP": 0.3,
    "index_MCP_FE": 0.3,
    "index_MCP_AA": 0.0,
    "index_PIP": 0.4,
    "index_DIP": 0.3,
    "middle_MCP_FE": 0.3,
    "middle_MCP_AA": 0.0,
    "middle_PIP": 0.4,
    "middle_DIP": 0.3,
    "ring_MCP_FE": 0.3,
    "ring_MCP_AA": 0.0,
    "ring_PIP": 0.4,
    "ring_DIP": 0.3,
    "pinky_CMC": 0.1,
    "pinky_MCP_FE": 0.3,
    "pinky_MCP_AA": 0.0,
    "pinky_PIP": 0.4,
    "pinky_DIP": 0.3,
}
SHARPA_HOME_JOINT_POS = {f"left_{name}": value for name, value in _HOME_FINGERS.items()}


def asset_path(side: HandSide = "left", floating: bool = False) -> Path:
    if side not in ("left", "right"):
        raise ValueError(f"Unsupported Sharpa side: {side}")
    if floating:
        name = f"{side}_sharpa_wave_with_float_base"
        return _ASSETS / "sharpa_wave_float_base_urdf_usd" / name / f"{name}.xml"
    name = f"{side}_sharpa_wave"
    return _ASSETS / name / f"{name}.xml"


SHARPA_XML = asset_path()
SHARPA_FLOATING_XML = asset_path(floating=True)


def get_spec(side: HandSide = "left") -> mujoco.MjSpec:
    return load_mjcf(asset_path(side))


def get_floating_spec(side: HandSide = "left") -> mujoco.MjSpec:
    names = {f"{side}_{axis}_joint": f"base_{axis}" for axis in BASE_AXES}
    return load_mjcf(asset_path(side, floating=True), joint_renames=names)


def make_home_state(
    pos: tuple[float, float, float] = SHARPA_HOME_POS,
    rot: tuple[float, float, float, float] = SHARPA_HOME_ROT,
    *,
    side: HandSide = "left",
) -> EntityCfg.InitialStateCfg:
    asset_path(side)  # Validate before building names.
    return EntityCfg.InitialStateCfg(
        pos=pos,
        rot=rot,
        joint_pos={f"{side}_{name}": value for name, value in _HOME_FINGERS.items()},
        joint_vel={".*": 0.0},
    )


def _articulation(side: HandSide, floating: bool) -> EntityArticulationInfoCfg:
    actuators = [XmlActuatorCfg(target_names_expr=(f"{side}_.*",))]
    if floating:
        actuators.insert(0, XmlActuatorCfg(target_names_expr=(r"base_(x|y|z|roll|pitch|yaw)",)))
    return EntityArticulationInfoCfg(
        actuators=tuple(actuators),
        soft_joint_pos_limit_factor=0.9,
    )


def get_sharpa_hand_cfg(side: HandSide = "left") -> EntityCfg:
    return EntityCfg(
        init_state=make_home_state(side=side),
        spec_fn=partial(get_spec, side=side),
        articulation=_articulation(side, floating=False),
    )


def get_sharpa_floating_hand_cfg(
    pos: tuple[float, float, float] = SHARPA_HOME_POS,
    rot: tuple[float, float, float, float] = SHARPA_HOME_ROT,
    *,
    side: HandSide = "left",
) -> EntityCfg:
    return EntityCfg(
        init_state=make_home_state(pos, rot, side=side),
        spec_fn=partial(get_floating_spec, side=side),
        articulation=_articulation(side, floating=True),
    )
