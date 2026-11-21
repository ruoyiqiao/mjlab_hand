"""Juggle2-OneHand-Allegro / Juggle1-OneHand-Allegro environment configs.

One floating palm-up Allegro (genuine left MJCF, anchor rotation identity ->
palm normal +z, fingers +x; base_x -> world +z, base_y -> world -y,
base_z -> world +x), two free low-restitution balls, no table, floor plane.
Physics 0.005 s, decimation 2 -> 100 Hz control (0.3 s throw period).

Deviations from Ploeger et al. 2020 (recorded in juggling_plan.md): Allegro hand
with a finger cup instead of a WAM funnel; 5 cm balls (75 mm in the paper);
drop line relative to the cup height (paper: 0.6 m above the floor); task-local
stiffer base slides (kp 2000 / kv 90 vs 300 / 40) to track a 0.3 s cycle.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from mjlab_hand.juggling import mdp as jmdp
from mjlab_hand.robot.allegro import ALLEGRO_HAND_LEFT
from mjlab_hand.tasks_common import mdp as tmdp
from mjlab_hand.tasks_common.actions import ALLEGRO_FINGER_JOINTS
from mjlab_hand.tasks_common.env_builder import (
    hand_joint_cfg,
    make_action_cfg,
    make_hand_entity_cfg,
)
from mjlab_hand.tasks_common.hand_profile import get_profile
from mjlab_hand.tasks_common.workspace import BASE_JOINTS, HandRootWorkspace

PALM_UP = (1.0, 0.0, 0.0, 0.0)
ANCHOR = (0.0, 0.0, 0.50)

# Finger "cup" pose (rad): fingers curled up around the palm, thumb opposed.
CUP_POSE = {
    "ffj0": 0.0,
    "ffj1": 0.35,
    "ffj2": 0.7,
    "ffj3": 0.6,
    "mfj0": 0.0,
    "mfj1": 0.35,
    "mfj2": 0.7,
    "mfj3": 0.6,
    "rfj0": 0.0,
    "rfj1": 0.35,
    "rfj2": 0.7,
    "rfj3": 0.6,
    "thj0": 1.2,
    "thj1": 0.3,
    "thj2": 0.5,
    "thj3": 0.5,
}

_PARAMS = Path(__file__).parents[1] / "tasks_common" / "hand_params" / "juggle.json"


def hand_juggle_params(hand: str) -> dict:
    """Measured cup pose / cup point of a non-Allegro hand (tools/allegro4/juggle_cup_probe.py)."""
    return json.loads(_PARAMS.read_text())[hand]


@dataclass
class JuggleSpec:
    num_balls: int = 2
    ball_radius: float = 0.025
    ball_mass: float = 0.05
    ball_solref: tuple[float, float] = (0.01, 1.0)
    ball_friction: tuple[float, float, float] = (1.0, 0.005, 0.0005)
    column_offset: tuple[float, float, float] = (0.0, 0.12, 0.0)
    half_cycle_s: float = 0.30
    second_ball_drop_height: float = 0.30
    second_ball_init_vz: float = 0.0
    episode_s: float = 10.0
    base_gains: tuple[float, float, float] = (2000.0, 90.0, 300.0)
    lin_vel_max: float = 4.0
    ang_vel_max: float = 6.0
    bounds: dict = field(
        default_factory=lambda: {
            "base_x": (-0.20, 0.30),  # world +z (dwell dip ~0.12 m, post-throw rise ~0.2 m)
            "base_y": (-0.20, 0.10),  # world -y (column B is +y -> base_y negative)
            "base_z": (-0.10, 0.10),  # world +x
            "base_roll": (-0.3, 0.3),
            "base_pitch": (-0.3, 0.3),
            "base_yaw": (-0.3, 0.3),
        }
    )
    finger_mode: str = "absolute"
    ref_tau_range: tuple[float, float] | None = None
    ref_land_speed: float = 1.96
    max_hold_s: float = 0.5
    hold_limit: bool = True
    reward_failure: float = 0.0
    reward_too_high: float = 0.0
    airborne_prob: float = 0.0
    throw_speed: float = 3.19
    vacant_s: float = 0.25
    reward_throw_quality: float = 0.0
    reward_catch_align: float = 0.0
    target_apex: float = 0.5
    reward_alive: float = 0.01
    reward_catch: float = 0.0
    reward_throw: float = 0.0
    reward_ball_ball: float = 0.0
    reward_lost_hold: float = 0.0
    entry_deadline_s: float | None = None
    alive_recent_throw_s: float | None = None
    max_throw_apex: float | None = None
    cap_actual_apex: bool = False
    reward_stall: float = 0.0
    finger_vel_penalty: float = -3e-6
    base_vel_penalty: float = -3e-5
    hand: str = "allegro"
    cup_pose: dict | None = None
    """Fixed finger targets (joints + Shadow tendons); None = CUP_POSE (Allegro) / the hand's measured pose."""
    cup_rel_palm: tuple[float, float, float] | None = None
    """Cup point minus palm origin; None = Allegro's palm_collision face / the hand's measured point."""
    side: str | None = None
    """Model side for non-Allegro hands (None = the hand's default side); used by the Juggle3 probe."""
    leap_pd: str = "sysid"
    """LEAP finger actuators: "sysid" (default) or "menagerie" (menagerie XML PD; LEAP Juggle2)."""


def make_ball_spec(
    r: float, m: float, solref, friction, rgba=(0.86, 0.47, 0.42, 1.0)
) -> mujoco.MjSpec:
    inertia = 2.0 / 5.0 * m * r * r
    xml = f"""
<mujoco model="ball">
  <asset>
    <material name="ball_mat" rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}" specular="0.05" shininess="0.05" reflectance="0"/>
  </asset>
  <worldbody>
    <body name="ball">
      <freejoint name="ball_joint"/>
      <inertial pos="0 0 0" mass="{m}" diaginertia="{inertia} {inertia} {inertia}"/>
      <geom name="ball_collision" type="sphere" size="{r}" material="ball_mat"
            solref="{solref[0]} {solref[1]}" condim="6"
            friction="{friction[0]} {friction[1]} {friction[2]}" priority="1" mass="0"/>
    </body>
  </worldbody>
</mujoco>
"""
    return mujoco.MjSpec.from_string(xml)


def juggle_workspace(js: JuggleSpec) -> HandRootWorkspace:
    if js.hand == "allegro":
        side, pos, rot = "left", ANCHOR, PALM_UP
    else:  # canonical base axes (hand_profile.py): the same bounds act like Allegro's
        prof = get_profile(js.hand, js.side)
        rot = prof.anchor_rot(down=False, fingers_px=True)
        side, pos = prof.side, prof.anchor_pos(ANCHOR, rot)
    return HandRootWorkspace(
        side=side,
        entity_name="hand",
        anchor_pos=pos,
        anchor_rot=rot,
        joint_bounds=dict(js.bounds),
        lin_vel_max=js.lin_vel_max,
        ang_vel_max=js.ang_vel_max,
        reset_noise={n: (0.0, 0.0) for n in BASE_JOINTS},
        notes=f"palm-up floating {js.hand.capitalize()} for toss juggling",
    )


def _contact(name, hand, ball, root="palm"):
    return ContactSensorCfg(
        name=name,
        primary=ContactMatch(mode="subtree", pattern=root, entity=hand),
        secondary=ContactMatch(mode="subtree", pattern="ball", entity=ball),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
    )


def make_juggle_env_cfg(js: JuggleSpec, play: bool = False) -> ManagerBasedRlEnvCfg:
    js = copy.deepcopy(js)
    balls = ("ball_a", "ball_b")
    ws = juggle_workspace(js)
    prof = None if js.hand == "allegro" else get_profile(js.hand, js.side)
    if js.leap_pd != "sysid":
        assert js.hand == "leap" and js.leap_pd == "menagerie" and prof.side == "right", (
            js.hand,
            js.leap_pd,
        )
        from mjlab_hand.robot.leap import LEAP_HAND_MENAGERIE

        prof = dataclasses.replace(prof, spec=LEAP_HAND_MENAGERIE)
    hp = (
        hand_juggle_params(js.hand)
        if prof is not None and (js.cup_pose is None or js.cup_rel_palm is None)
        else {}
    )
    cup_pose = dict(js.cup_pose or hp.get("cup_pose") or CUP_POSE)
    cup_rel = js.cup_rel_palm or (tuple(hp["cup_rel_palm"]) if prof is not None else None)
    extra = {} if prof is None else {"palm_body": prof.palm_body, "cup_rel_palm": cup_rel}
    cmd = jmdp.JuggleCommandCfg(
        ball_names=balls,
        ball_radius=js.ball_radius,
        column_offset=js.column_offset,
        half_cycle_s=js.half_cycle_s,
        second_ball_drop_height=js.second_ball_drop_height,
        second_ball_init_vz=js.second_ball_init_vz,
        ref_tau_range=js.ref_tau_range,
        ref_land_speed=js.ref_land_speed,
        max_hold_s=js.max_hold_s if js.hold_limit else 1.0e9,
        initial_hold_grace_s=0.6 if js.hold_limit else 1.0e9,
        max_both_held_s=0.15 if js.hold_limit else 1.0e9,
        target_apex=js.target_apex,
        airborne_prob=0.0 if play else js.airborne_prob,
        entry_deadline_s=None if play else js.entry_deadline_s,
        alive_recent_throw_s=None if play else js.alive_recent_throw_s,
        max_throw_apex=None if play else js.max_throw_apex,
        cap_actual_apex=False if play else js.cap_actual_apex,
        throw_speed=js.throw_speed,
        vacant_s=js.vacant_s,
        second_ball_present=js.num_balls >= 2,
        **extra,
    )
    obs = {
        "joint_pos": ObservationTermCfg(
            func=tmdp.hand_joint_pos_unit, params={"asset_cfg": hand_joint_cfg("hand", prof)}
        ),
        "joint_vel": ObservationTermCfg(
            func=tmdp.hand_joint_vel, params={"asset_cfg": hand_joint_cfg("hand", prof)}
        ),
        "palm": ObservationTermCfg(func=jmdp.palm_state),
        "balls": ObservationTermCfg(func=jmdp.balls_rel_palm),
        "columns": ObservationTermCfg(func=jmdp.columns_rel_palm),
        "clock": ObservationTermCfg(func=jmdp.pattern_clock),
        "actions": ObservationTermCfg(func=envs_mdp.last_action),
    }
    observations = {
        "actor": ObservationGroupCfg(dict(obs), enable_corruption=False),
        "critic": ObservationGroupCfg(dict(obs), enable_corruption=False),
    }
    observations["actor"].nan_policy = "error"
    observations["critic"].nan_policy = "error"

    events = {
        "reset_scene_to_default": EventTermCfg(func=envs_mdp.reset_scene_to_default, mode="reset"),
        "reset_base": EventTermCfg(
            func=tmdp.reset_floating_base,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("hand"), "noise": dict(ws.reset_noise)},
        ),
        "reset_balls": EventTermCfg(
            func=jmdp.reset_balls,
            mode="reset",
            params={"ball_b_xy_noise": 0.005, "ball_b_height_noise": 0.01},
        ),
    }
    rewards = {
        "alive": RewardTermCfg(func=jmdp.juggling_alive, weight=js.reward_alive),
        "base_vel_penalty": RewardTermCfg(
            func=jmdp.joint_vel_l1,
            weight=js.base_vel_penalty,
            params={
                "asset_cfg": SceneEntityCfg(
                    "hand", joint_names=list(BASE_JOINTS), preserve_order=True
                )
            },
        ),
        "finger_vel_penalty": RewardTermCfg(
            func=jmdp.joint_vel_l1,
            weight=js.finger_vel_penalty,
            params={
                "asset_cfg": SceneEntityCfg(
                    "hand",
                    joint_names=list(prof.obs_joints if prof else ALLEGRO_FINGER_JOINTS),
                    preserve_order=True,
                )
            },
        ),
    }
    if js.reward_catch:
        rewards["catch"] = RewardTermCfg(func=jmdp.catch_event, weight=js.reward_catch)
    if js.reward_throw:
        rewards["throw"] = RewardTermCfg(func=jmdp.throw_event, weight=js.reward_throw)
    if js.reward_too_high:
        rewards["too_high"] = RewardTermCfg(func=jmdp.throw_too_high, weight=js.reward_too_high)
    if js.reward_failure:
        rewards["failure"] = RewardTermCfg(func=jmdp.failure, weight=js.reward_failure)
    if js.reward_throw_quality:
        rewards["throw_quality"] = RewardTermCfg(
            func=jmdp.throw_quality, weight=js.reward_throw_quality
        )
    if js.reward_catch_align:
        rewards["catch_align"] = RewardTermCfg(
            func=jmdp.catch_alignment, weight=js.reward_catch_align
        )
    if js.reward_stall:
        rewards["stall"] = RewardTermCfg(func=jmdp.stall_event, weight=js.reward_stall)
    if js.reward_lost_hold:
        rewards["lost_hold"] = RewardTermCfg(func=jmdp.lost_hold, weight=js.reward_lost_hold)
    if js.reward_ball_ball:
        rewards["ball_ball"] = RewardTermCfg(
            func=jmdp.ball_ball_contact, weight=js.reward_ball_ball
        )
    terms = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "dropped": TerminationTermCfg(func=jmdp.ball_dropped),
        "held_too_long": TerminationTermCfg(func=jmdp.held_too_long),
        "stalled": TerminationTermCfg(func=jmdp.stalled),
        "nan": TerminationTermCfg(func=envs_mdp.nan_detection),
    }
    root = prof.root_body if prof else "palm"
    sensors = [
        _contact("hand_ball_a_contact", "hand", "ball_a", root),
        _contact("hand_ball_b_contact", "hand", "ball_b", root),
        ContactSensorCfg(
            name="ball_ball_contact",
            primary=ContactMatch(mode="subtree", pattern="ball", entity="ball_a"),
            secondary=ContactMatch(mode="subtree", pattern="ball", entity="ball_b"),
            fields=("found",),
            reduce="netforce",
            num_slots=1,
        ),
    ]
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=1,
            env_spacing=1.0,
            sensors=tuple(sensors),
        ),
        observations=observations,
        actions={
            "hand": make_action_cfg(
                "hand",
                ws,
                finger_mode=js.finger_mode,
                finger_ema=1.0,
                base_ema=1.0,
                fixed_finger_pos=cup_pose,
                profile=prof,
            )
        },
        commands={"juggle": cmd},
        events=events,
        rewards=rewards,
        terminations=terms,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.WORLD,
            lookat=(0.0, 0.06, 0.6),
            distance=1.0,
            elevation=-10.0,
            azimuth=180.0,
        ),
        sim=SimulationCfg(
            nconmax=100,
            njmax=600,
            mujoco=MujocoCfg(
                timestep=0.005, iterations=50, ls_iterations=50, impratio=10, cone="elliptic"
            ),
        ),
        decimation=2,
        episode_length_s=js.episode_s,
        scale_rewards_by_dt=False,
        is_finite_horizon=False,
    )
    if prof is None:
        hand = make_hand_entity_cfg(ALLEGRO_HAND_LEFT, ws, base_gains=js.base_gains)
        hand.init_state.joint_pos = dict(
            cup_pose
        )  # = CUP_POSE unless the spec gives a pose (Juggle2 -Full)
    else:
        hand = make_hand_entity_cfg(prof.spec, ws, base_gains=js.base_gains, profile=prof)
        init = dict(hand.init_state.joint_pos or {})
        init.update({j: v for j, v in cup_pose.items() if j not in prof.tendon_targets})
        hand.init_state.joint_pos = init
    cfg.scene.entities = {
        "hand": hand,
        "ball_a": EntityCfg(
            spec_fn=lambda: make_ball_spec(
                js.ball_radius, js.ball_mass, js.ball_solref, js.ball_friction
            )
        ),
        "ball_b": EntityCfg(
            spec_fn=lambda: make_ball_spec(
                js.ball_radius,
                js.ball_mass,
                js.ball_solref,
                js.ball_friction,
                rgba=(0.88, 0.72, 0.36, 1.0),
            )
        ),
    }
    return cfg
