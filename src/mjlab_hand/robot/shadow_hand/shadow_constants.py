"""Shadow hand constants."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import torch
from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTerm
from mjlab.utils.spec_config import CollisionCfg

from mjlab_hand.utils import update_assets

##
# MJCF and assets.
##

_HERE = Path(__file__).parent

SHADOW_XML: Path = _HERE / "xmls" / "left_hand.xml"
SHADOW_FLOATING_XML: Path = _HERE / "xmls" / "left_hand_floating.xml"
SHADOW_FLOATING_XML_BY_SIDE: dict[str, Path] = {
    "left": SHADOW_FLOATING_XML,
    "right": _HERE
    / "xmls"
    / "right_hand_floating.xml",  # tools/allegro4/make_floating_xml.py from menagerie right_hand.xml
}
assert SHADOW_XML.exists()
assert all(p.exists() for p in SHADOW_FLOATING_XML_BY_SIDE.values())


def side_prefix(side: str) -> str:
    """Body/joint/tendon name prefix of the menagerie Shadow hand: ``lh_`` (left), ``rh_`` (right)."""
    return {"left": "lh_", "right": "rh_"}[side]


def for_side(names, side: str):
    """Left-hand (``lh_``) names -> the names of ``side`` (tuples/lists/dict keys/strings)."""
    pre = side_prefix(side)
    if isinstance(names, str):
        return names.replace("lh_", pre)
    if isinstance(names, dict):
        return {k.replace("lh_", pre): v for k, v in names.items()}
    return type(names)(n.replace("lh_", pre) for n in names)


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, SHADOW_XML.parent / "assets", meshdir)
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
    return _get_spec(SHADOW_XML)


def get_floating_spec(side: str = "left") -> mujoco.MjSpec:
    return _get_spec(SHADOW_FLOATING_XML_BY_SIDE[side])


def _get_actuator_ctrl_ranges(spec: mujoco.MjSpec) -> dict[str, tuple[float, float]]:
    """Extract control ranges from XML actuators."""
    ranges: dict[str, tuple[float, float]] = {}
    for actuator in spec.actuators:
        target = actuator.target
        if target:
            ranges[target] = (
                float(actuator.ctrlrange[0]),
                float(actuator.ctrlrange[1]),
            )
    return ranges


def _compute_action_limits(
    ctrl_ranges: dict[str, tuple[float, float]],
    target_names: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute scale/offset for specific targets."""
    scale: dict[str, float] = {}
    offset: dict[str, float] = {}
    for name in target_names:
        if name in ctrl_ranges:
            lo, hi = float(ctrl_ranges[name][0]), float(ctrl_ranges[name][1])
            scale[name] = (hi - lo) / 2.0
            offset[name] = (hi + lo) / 2.0
    return scale, offset


_ALL_CTRL_RANGES = _get_actuator_ctrl_ranges(get_spec())

# Joint-actuated targets
SHADOW_JOINT_TARGETS = (
    "lh_WRJ2",
    "lh_WRJ1",
    "lh_FFJ4",
    "lh_FFJ3",
    "lh_MFJ4",
    "lh_MFJ3",
    "lh_RFJ4",
    "lh_RFJ3",
    "lh_LFJ5",
    "lh_LFJ4",
    "lh_LFJ3",
    "lh_THJ5",
    "lh_THJ4",
    "lh_THJ3",
    "lh_THJ2",
    "lh_THJ1",
)

# Tendon-actuated targets (coupled joints)
SHADOW_TENDON_TARGETS = ("lh_FFJ0", "lh_MFJ0", "lh_RFJ0", "lh_LFJ0")

# Compute scale/offset for each group
SHADOW_JOINT_ACTION_LIMIT_SCALE, SHADOW_JOINT_ACTION_LIMIT_OFFSET = _compute_action_limits(
    _ALL_CTRL_RANGES, SHADOW_JOINT_TARGETS
)
SHADOW_TENDON_ACTION_LIMIT_SCALE, SHADOW_TENDON_ACTION_LIMIT_OFFSET = _compute_action_limits(
    _ALL_CTRL_RANGES, SHADOW_TENDON_TARGETS
)

SHADOW_KEYPOINT_GEOMS = r"lh_(ff|mf|rf|lf|th)middle_collision$"

FULL_COLLISION = CollisionCfg(
    contype=1,
    conaffinity=1,
    geom_names_expr=(".*_collision",),
    condim={
        SHADOW_KEYPOINT_GEOMS: 6,
        ".*_collision": 3,
    },
    friction={
        SHADOW_KEYPOINT_GEOMS: (1.0, 5e-3, 5e-4),
        ".*_collision": (0.6,),
    },
    solref={
        SHADOW_KEYPOINT_GEOMS: (0.01, 1),
    },
    priority={
        SHADOW_KEYPOINT_GEOMS: 1,
        ".*": 0,
    },
)

##
# Keyframe config.
#
# Pre-grasp pose from keyframes.xml.
##

SHADOW_HOME_POS: tuple[float, float, float] = (0.0, 0.0, 0.5)
SHADOW_HOME_ROT: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
SHADOW_HOME_JOINT_POS: dict[str, float] = {
    "lh_WRJ2": -0.03896,
    "lh_WRJ1": -0.5694,
    "lh_FFJ4": -0.3497,
    "lh_FFJ3": 0.1074,
    "lh_FFJ2": 0.1424,
    "lh_FFJ1": -0.008296,
    "lh_MFJ4": -0.05406,
    "lh_MFJ3": -0.07509,
    "lh_MFJ2": 0.4658,
    "lh_MFJ1": 0.01033,
    "lh_RFJ4": -0.1062,
    "lh_RFJ3": 0.2738,
    "lh_RFJ2": 0.001251,
    "lh_RFJ1": 0.02586,
    "lh_LFJ5": 0.1551,
    "lh_LFJ4": -0.3593,
    "lh_LFJ3": 0.1445,
    "lh_LFJ2": -0.00691,
    "lh_LFJ1": -0.001588,
    "lh_THJ5": -0.1413,
    "lh_THJ4": 0.8383,
    "lh_THJ3": 0.199,
    "lh_THJ2": 0.4945,
    "lh_THJ1": 0.1291,
}


def make_home_state(
    pos: tuple[float, float, float] = SHADOW_HOME_POS,
    rot: tuple[float, float, float, float] = SHADOW_HOME_ROT,
) -> EntityCfg.InitialStateCfg:
    """Return a fresh InitialStateCfg so callers never share state."""
    return EntityCfg.InitialStateCfg(
        pos=pos,
        rot=rot,
        joint_pos=dict(SHADOW_HOME_JOINT_POS),
        joint_vel={".*": 0.0},
    )


##
# Actuator config.
#
# Shadow Hand has 20 actuators:
# - 16 joint actuators (wrists, knuckles, proximal, thumb)
# - 4 tendon actuators (coupled middle/distal joints for fingers)
##

# Joint-actuated elements (direct joint control)
SHADOW_JOINT_ACTUATORS = XmlActuatorCfg(
    target_names_expr=SHADOW_JOINT_TARGETS,
)

# Tendon-actuated elements (coupled joints)
SHADOW_TENDON_ACTUATORS = XmlActuatorCfg(
    target_names_expr=SHADOW_TENDON_TARGETS,
    transmission_type=TransmissionType.TENDON,
)

ARTICULATION = EntityArticulationInfoCfg(
    actuators=(SHADOW_JOINT_ACTUATORS, SHADOW_TENDON_ACTUATORS),
    soft_joint_pos_limit_factor=0.9,
)

SHADOW_BASE_ACTUATORS = XmlActuatorCfg(
    target_names_expr=("base_(x|y|z|roll|pitch|yaw)",),
)

ARTICULATION_FLOATING = EntityArticulationInfoCfg(
    actuators=(SHADOW_BASE_ACTUATORS, SHADOW_JOINT_ACTUATORS, SHADOW_TENDON_ACTUATORS),
    soft_joint_pos_limit_factor=0.9,
)


def get_shadow_hand_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=make_home_state(),
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=ARTICULATION,
    )


def get_shadow_floating_hand_cfg(
    pos: tuple[float, float, float] = SHADOW_HOME_POS,
    rot: tuple[float, float, float, float] = SHADOW_HOME_ROT,
) -> EntityCfg:
    return EntityCfg(
        init_state=make_home_state(pos=pos, rot=rot),
        collisions=(FULL_COLLISION,),
        spec_fn=get_floating_spec,
        articulation=ARTICULATION_FLOATING,
    )


##
# Floating hands for the shared task layer (tasks_common): both sides, distal-phalanx fingertips.
#
# Policy-controlled finger targets: 14 independent finger/thumb joints + 4 tendon lengths (J2 + J1 of FF/MF/RF/LF, the
# coupled joints stay observable); the two wrist joints are redundant with the floating base and are held at 0.
##
SHADOW_WRIST_JOINTS = ("lh_WRJ2", "lh_WRJ1")
SHADOW_FINGER_JOINT_TARGETS = tuple(n for n in SHADOW_JOINT_TARGETS if n not in SHADOW_WRIST_JOINTS)
SHADOW_COUPLED_JOINTS = tuple(f"lh_{f}J{j}" for f in ("FF", "MF", "RF", "LF") for j in (2, 1))
SHADOW_TIP_BODIES = ("lh_ffdistal", "lh_mfdistal", "lh_rfdistal", "lh_lfdistal", "lh_thdistal")
SHADOW_TIP_GEOMS = r"lh_(ff|mf|rf|lf|th)distal_collision$"


def _task_collision(side: str) -> CollisionCfg:
    tip = for_side(SHADOW_TIP_GEOMS, side)
    return CollisionCfg(
        contype=1,
        conaffinity=1,
        geom_names_expr=(".*_collision",),
        condim={tip: 6, ".*_collision": 3},
        friction={tip: (1.0, 5e-3, 5e-4), ".*_collision": (0.6,)},
        solref={tip: (0.01, 1)},
        priority={tip: 1, ".*": 0},
    )


def get_shadow_task_floating_hand_cfg(
    pos: tuple[float, float, float] = SHADOW_HOME_POS,
    rot: tuple[float, float, float, float] = SHADOW_HOME_ROT,
    side: str = "left",
) -> EntityCfg:
    """Floating Shadow hand of ``side`` for tasks_common: wrist at 0, grippy distal fingertips."""
    home = {k: v for k, v in SHADOW_HOME_JOINT_POS.items() if k not in SHADOW_WRIST_JOINTS}
    home.update({w: 0.0 for w in SHADOW_WRIST_JOINTS})
    init = EntityCfg.InitialStateCfg(
        pos=pos, rot=rot, joint_pos=for_side(home, side), joint_vel={".*": 0.0}
    )
    joints = XmlActuatorCfg(target_names_expr=for_side(SHADOW_JOINT_TARGETS, side))
    tendons = XmlActuatorCfg(
        target_names_expr=for_side(SHADOW_TENDON_TARGETS, side),
        transmission_type=TransmissionType.TENDON,
    )
    return EntityCfg(
        init_state=init,
        collisions=(_task_collision(side),),
        spec_fn=lambda: get_floating_spec(side),
        articulation=EntityArticulationInfoCfg(
            actuators=(SHADOW_BASE_ACTUATORS, joints, tendons), soft_joint_pos_limit_factor=0.9
        ),
    )


##
# Action configurations for in-hand rotation task.
#
# These are used in the environment config to set up the action space.
# Shadow Hand supports two modes:
#   1. Standard (16 DoF): Controls only independent joints
#   2. Full (20 DoF): Controls joints + tendon-coupled joints
##


@dataclass(kw_only=True)
class TendonPositionActionCfg(JointPositionActionCfg):
    """Configuration for tendon position control.

    This action controls tendon lengths, which couple multiple joints together.
    For Shadow Hand: each tendon couples proximal and middle finger joints.
    """

    ema_alpha: float = 1.0
    """EMA smoothing factor in [0, 1].

  - 1.0: no smoothing (direct affine mapping)
  - 0.2: mild smoothing
  """

    def __post_init__(self):
        self.transmission_type = TransmissionType.TENDON
        if not 0.0 <= self.ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in [0, 1], got {self.ema_alpha}")

    def build(self, env: ManagerBasedRlEnv) -> "TendonPositionAction":
        return TendonPositionAction(self, env)


class TendonPositionAction(ActionTerm):
    """Control tendons via length targets.

    This properly handles tendon actuators by calling set_tendon_len_target
    instead of set_joint_position_target.
    """

    cfg: TendonPositionActionCfg
    _entity: "Entity"
    _target_ids: torch.Tensor
    _target_names: list[str]
    _action_dim: int
    _raw_actions: torch.Tensor
    _processed_actions: torch.Tensor
    _ema_target: torch.Tensor
    _scale: torch.Tensor | float
    _offset: torch.Tensor | float

    def __init__(self, cfg: TendonPositionActionCfg, env: ManagerBasedRlEnv):
        from mjlab.utils.lab_api.string import resolve_matching_names_values

        self.cfg = cfg
        self._env = env
        self._entity = env.scene[cfg.entity_name]

        # Find tendon targets
        target_ids, target_names = self._entity.find_tendons(
            cfg.actuator_names, preserve_order=cfg.preserve_order
        )
        self._target_ids = torch.tensor(target_ids, device=self.device, dtype=torch.long)
        self._target_names = target_names
        self._action_dim = len(target_ids)

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # Setup scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            index_list, _, value_list = resolve_matching_names_values(cfg.scale, self._target_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}")

        # Setup offset
        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            self._offset = torch.zeros_like(self._raw_actions)
            index_list, _, value_list = resolve_matching_names_values(
                cfg.offset, self._target_names
            )
            self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}")

        self._ema_target = torch.zeros_like(self._raw_actions)
        if isinstance(self._offset, torch.Tensor):
            self._ema_target[:] = self._offset
        else:
            self._ema_target[:] = float(self._offset)
        self._processed_actions[:] = self._ema_target

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def device(self) -> str:
        return self._env.device

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def target_names(self) -> list[str]:
        return self._target_names

    def process_actions(self, actions: torch.Tensor):
        """Apply scale/offset and optional EMA smoothing."""
        self._raw_actions[:] = actions
        new_target = self._raw_actions * self._scale + self._offset

        if self.cfg.ema_alpha >= 1.0:
            self._processed_actions[:] = new_target
            self._ema_target[:] = new_target
            return

        self._ema_target[:] = (
            self.cfg.ema_alpha * new_target + (1.0 - self.cfg.ema_alpha) * self._ema_target
        )
        self._processed_actions[:] = self._ema_target

    def apply_actions(self) -> None:
        """Apply processed actions as tendon length targets."""
        self._entity.set_tendon_len_target(self._processed_actions, tendon_ids=self._target_ids)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Reset raw actions and EMA state to neutral offset."""
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        if isinstance(self._offset, torch.Tensor):
            self._ema_target[env_ids] = self._offset[env_ids]
            self._processed_actions[env_ids] = self._offset[env_ids]
        else:
            self._ema_target[env_ids] = float(self._offset)
            self._processed_actions[env_ids] = float(self._offset)


# Joint position action - 16 DoF (independent joints only)
SHADOW_JOINT_ACTION_CFG = JointPositionActionCfg(
    entity_name="hand",
    actuator_names=(".*",),
    scale=SHADOW_JOINT_ACTION_LIMIT_SCALE,
    offset=SHADOW_JOINT_ACTION_LIMIT_OFFSET,
    use_default_offset=False,
)

# Tendon position action - 4 DoF (coupled proximal+middle joints)
# This enables full 20-DoF control when combined with joint action.
# Tendon coupling (1:1 ratio):
#   - lh_FFJ0: couples lh_FFJ2 (proximal) + lh_FFJ1 (middle)
#   - lh_MFJ0: couples lh_MFJ2 (proximal) + lh_MFJ1 (middle)
#   - lh_RFJ0: couples lh_RFJ2 (proximal) + lh_RFJ1 (middle)
#   - lh_LFJ0: couples lh_LFJ2 (proximal) + lh_LFJ1 (middle)
SHADOW_TENDON_ACTION_CFG = TendonPositionActionCfg(
    entity_name="hand",
    actuator_names=(".*",),
    scale=SHADOW_TENDON_ACTION_LIMIT_SCALE,
    offset=SHADOW_TENDON_ACTION_LIMIT_OFFSET,
    use_default_offset=False,
)


def get_shadow_actions(use_tendon_control: bool = False):
    """Get Shadow Hand action configuration.

    Args:
      use_tendon_control: If True, includes tendon action for full 20-DoF control.
                         If False, returns only joint action (16 DoF).

    Returns:
      Dict of action name -> ActionTermCfg
    """
    actions = {"joint_pos": SHADOW_JOINT_ACTION_CFG}
    if use_tendon_control:
        actions["tendon_pos"] = SHADOW_TENDON_ACTION_CFG
    return actions


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    hand = Entity(get_shadow_hand_cfg())
    viewer.launch(hand.spec.compile())
