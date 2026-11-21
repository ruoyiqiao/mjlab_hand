"""DexScrew-parameterized XHand1 left hand."""

import json
import math
from pathlib import Path

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from mjlab_hand.robot.mjcf import load_mjcf

_HERE = Path(__file__).parent
PARAMETERS = json.loads((_HERE / "dexscrew_parameters.json").read_text())
XHAND_XML = _HERE / "xmls" / "left.xml"
XHAND_FLOATING_XML = _HERE / "xmls" / "left_floating.xml"
XHAND_JOINT_NAMES = tuple(PARAMETERS["joint_order"])
XHAND_HOME_POS = (0.0, 0.0, 0.5)
# DexScrew uses a 65 degree rotation about X for the left palm.
XHAND_HOME_ROT = (math.cos(math.radians(65) / 2), math.sin(math.radians(65) / 2), 0.0, 0.0)
XHAND_KEYPOINT_BODIES = (
    "left_hand_index_rota_tip",
    "left_hand_mid_tip",
    "left_hand_pinky_tip",
    "left_hand_ring_tip",
    "left_hand_thumb_rota_tip",
)
XHAND_KEYPOINT_GEOMS = r"left_hand_(index_rota|mid|pinky|ring|thumb_rota)_tip_collision"
XHAND_ACTION_LIMIT_SCALE = dict(
    zip(
        XHAND_JOINT_NAMES,
        [
            (hi - lo) / 2
            for lo, hi in zip(PARAMETERS["lower_limits"], PARAMETERS["upper_limits"], strict=True)
        ],
        strict=True,
    )
)
XHAND_ACTION_LIMIT_OFFSET = dict(
    zip(
        XHAND_JOINT_NAMES,
        [
            (hi + lo) / 2
            for lo, hi in zip(PARAMETERS["lower_limits"], PARAMETERS["upper_limits"], strict=True)
        ],
        strict=True,
    )
)


def get_spec() -> mujoco.MjSpec:
    return load_mjcf(XHAND_XML)


def get_floating_spec() -> mujoco.MjSpec:
    return load_mjcf(XHAND_FLOATING_XML)


def make_home_state(pos=XHAND_HOME_POS, rot=XHAND_HOME_ROT) -> EntityCfg.InitialStateCfg:
    return EntityCfg.InitialStateCfg(
        pos=pos,
        rot=rot,
        joint_pos=dict(zip(XHAND_JOINT_NAMES, PARAMETERS["home_positions"], strict=True)),
        joint_vel={".*": 0.0},
    )


def _articulation(floating: bool) -> EntityArticulationInfoCfg:
    actuators = [XmlActuatorCfg(target_names_expr=XHAND_JOINT_NAMES)]
    if floating:
        actuators.insert(0, XmlActuatorCfg(target_names_expr=(r"base_(x|y|z|roll|pitch|yaw)",)))
    return EntityArticulationInfoCfg(
        actuators=tuple(actuators),
        soft_joint_pos_limit_factor=1.0,
    )


def get_xhand_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=make_home_state(),
        spec_fn=get_spec,
        articulation=_articulation(False),
    )


def get_xhand_floating_cfg(pos=XHAND_HOME_POS, rot=XHAND_HOME_ROT) -> EntityCfg:
    return EntityCfg(
        init_state=make_home_state(pos, rot),
        spec_fn=get_floating_spec,
        articulation=_articulation(True),
    )
