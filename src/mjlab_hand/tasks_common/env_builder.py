"""Builders for the Allegro DexPBT-style env configs.

Single-hand tasks (throw / reorientation / regrasp) share one builder; the
dual-hand task builds on the same pieces with two namespaced hands. Every call
returns fresh config objects (no shared mutable state between tasks).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import mujoco
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from mjlab_hand.defaults import NUM_TRAIN_ENVS
from mjlab_hand.robot.allegro import ALLEGRO_HAND_RIGHT
from mjlab_hand.tasks_common import mdp as tmdp
from mjlab_hand.tasks_common import sensors
from mjlab_hand.tasks_common.actions import ALLEGRO_FINGER_JOINTS, BoundedFloatingHandActionCfg
from mjlab_hand.tasks_common.dexpbt_objects import make_dexpbt_object_cfg
from mjlab_hand.tasks_common.hand_profile import HandProfile, get_profile
from mjlab_hand.tasks_common.objects import (
    BucketSpec,
    CubeSpec,
    TableSpec,
    make_bucket_entity_cfg,
    make_cube_entity_cfg,
    make_table_entity_cfg,
)
from mjlab_hand.tasks_common.tracker import DexPBTTaskCommandCfg, reset_object_event
from mjlab_hand.tasks_common.workspace import BASE_JOINTS, HandRootWorkspace

# DexPBT AllegroKuka.yaml:43-50 coefficients and rl_games reward_shaper scale.
DEXPBT_REWARD_SCALE = 0.01
GRASP_CONTACT_WEIGHT = 0.005  # per step; < lift shaping at 5 cm clearance (0.01)
DEXPBT = {
    "distanceDeltaRewScale": 50.0,
    "liftingRewScale": 20.0,
    "liftingBonus": 300.0,
    "keypointRewScale": 200.0,
    "reachGoalBonus": 1000.0,
    "kukaActionsPenaltyScale": 0.003,
    "allegroActionsPenaltyScale": 0.0003,
}

CANONICAL_JOINTS = list(BASE_JOINTS) + list(ALLEGRO_FINGER_JOINTS)


def canonical_joints(profile: HandProfile | None = None) -> list[str]:
    """Policy joint order of one hand: the six base joints, then the finger joints."""
    return list(BASE_JOINTS) + list((profile.obs_joints if profile else ALLEGRO_FINGER_JOINTS))


PALM_DOWN_FINGERS_PX = (0.0, 1.0, 0.0, 0.0)
"""Anchor rot (wxyz): palm normal -z (down), fingers +x (see root_layout.json)."""
PALM_DOWN_FINGERS_NX = (0.0, 0.0, 1.0, 0.0)
"""Anchor rot (wxyz): palm normal -z (down), fingers -x."""


@dataclass
class TaskTiming:
    episode_s: float
    per_goal_s: float
    max_successes: int = 50


@dataclass
class SingleHandTaskSpec:
    task: str
    command: DexPBTTaskCommandCfg
    workspace: HandRootWorkspace
    timing: TaskTiming
    hand: str = "allegro"
    """Hand model (``hand_profile``); the workspace must have been built for the same hand."""
    table: TableSpec = field(default_factory=TableSpec)
    cube: CubeSpec = field(default_factory=CubeSpec)
    bucket: BucketSpec | None = None
    object_sets: tuple[str, ...] | None = ("cube", "small", "big", "sticks")
    """DexPBT cuboid sets (per-world variants); None = single ``cube``."""
    use_curriculum: bool = True
    finger_mode: str = "absolute"
    finger_ema: float = 1.0
    base_ema: float = 1.0
    hand_far_reset: bool = False
    goal_timeout_is_failure: bool = False
    """Treat the per-goal timeout as a terminal failure (no value bootstrap); with bootstrap the regrasp
    policy learned to end episodes by stalling near the goal."""
    goal_bonus_mode: str = "per_step"
    """per_step: reachGoalBonus/hold_steps per near-goal step (DexPBT form).
    on_success: reachGoalBonus once per completed goal (needed when success makes the
    next state harder, e.g. regrasp: a5 farmed per-step bonus while breaking the hold)."""
    base_bound_penalty: float = 0.0
    """Weight (<= 0) of the base workspace-bound pressure penalty; 0 disables."""
    curriculum_min_successes: float = 3.0
    """Tolerance curriculum (training only): advance when the mean successes of finished episodes reach this. Successes per episode are
    geometric (episodes end at the first timed-out goal), so 3 = 75 % per goal and 1 = 50 %. The non-Allegro reorientation runs parked at
    ~1 (LEAP s42: tolerance 1.33, never trained at the final tolerance, held-out 2 %), so they use 1.0; Allegro keeps 3."""
    grasp_contact_weight: float = GRASP_CONTACT_WEIGHT
    """Per-step pre-lift opposition-grasp shaping (a deviation from DexPBT); 0 removes it."""
    finger_action_rate: float = 0.0
    """Weight (<= 0) of the finger action-rate penalty (sum |a_t - a_{t-1}|^2 over the finger dims); 0 disables."""
    finger_action_acc: float = 0.0
    """Weight (<= 0) of the finger action-acceleration penalty (second difference); 0 disables."""


def hand_joint_cfg(hand: str, profile: HandProfile | None = None) -> SceneEntityCfg:
    return SceneEntityCfg(
        hand,
        joint_names=canonical_joints(profile) if profile else CANONICAL_JOINTS,
        preserve_order=True,
    )


def make_action_cfg(
    hand: str,
    ws: HandRootWorkspace,
    finger_mode: str,
    finger_ema: float,
    base_ema: float,
    fixed_finger_pos: dict | None = None,
    base_mode: str = "delta",
    base_slew: bool = False,
    profile: HandProfile | None = None,
) -> BoundedFloatingHandActionCfg:
    profile = profile or get_profile("allegro")
    return BoundedFloatingHandActionCfg(
        fixed_finger_pos=dict(fixed_finger_pos or {}),
        entity_name=hand,
        workspace=ws,
        finger_mode=finger_mode,  # type: ignore[arg-type]
        finger_joint_names=tuple(profile.finger_joints),
        finger_tendon_names=tuple(profile.tendon_targets),
        held_joint_pos=dict(profile.fixed_joints),
        finger_scale=dict(profile.spec.action_limit_scale or {}),
        finger_offset=dict(profile.spec.action_limit_offset or {}),
        finger_ema=finger_ema,
        base_ema=base_ema,
        base_mode=base_mode,  # type: ignore[arg-type]
        base_slew=base_slew,
    )


def hand_obs_terms(
    hand: str, idx: int, cmd: str, profile: HandProfile | None = None
) -> dict[str, ObservationTermCfg]:
    p = f"{hand}_" if idx >= 0 else ""
    return {
        f"{p}joint_pos": ObservationTermCfg(
            func=tmdp.hand_joint_pos_unit, params={"asset_cfg": hand_joint_cfg(hand, profile)}
        ),
        f"{p}joint_vel": ObservationTermCfg(
            func=tmdp.hand_joint_vel, params={"asset_cfg": hand_joint_cfg(hand, profile)}
        ),
        f"{p}palm": ObservationTermCfg(
            func=tmdp.palm_state, params={"command_name": cmd, "hand_index": max(idx, 0)}
        ),
        f"{p}tips_rel_palm": ObservationTermCfg(
            func=tmdp.fingertips_rel_palm, params={"command_name": cmd, "hand_index": max(idx, 0)}
        ),
        f"{p}kp_rel_palm": ObservationTermCfg(
            func=tmdp.keypoints_rel_palm, params={"command_name": cmd, "hand_index": max(idx, 0)}
        ),
    }


def task_obs_terms(cmd: str, orientation: bool, throw: bool) -> dict[str, ObservationTermCfg]:
    terms = {
        "object": ObservationTermCfg(func=tmdp.object_state, params={"command_name": cmd}),
        "object_scale": ObservationTermCfg(func=tmdp.object_scale, params={"command_name": cmd}),
        "kp_rel_goal": ObservationTermCfg(
            func=tmdp.keypoints_rel_goal, params={"command_name": cmd}
        ),
        "goal": ObservationTermCfg(func=tmdp.goal_pose, params={"command_name": cmd}),
        "progress": ObservationTermCfg(func=tmdp.task_progress, params={"command_name": cmd}),
        "contacts": ObservationTermCfg(func=tmdp.contact_flags, params={"command_name": cmd}),
    }
    if orientation:
        terms["obj_quat_rel_goal"] = ObservationTermCfg(
            func=tmdp.object_quat_rel_goal, params={"command_name": cmd}
        )
    if throw:
        terms["throw_stage"] = ObservationTermCfg(
            func=tmdp.throw_stage, params={"command_name": cmd}
        )
    terms["actions"] = ObservationTermCfg(func=envs_mdp.last_action)
    return terms


def dexpbt_rewards(
    cmd: str,
    hands: list[str],
    hold_steps: int,
    gate_tips_by_lift: bool,
    velocity_penalties: bool,
    profile: HandProfile | None = None,
    grasp_contact_weight: float = GRASP_CONTACT_WEIGHT,
) -> dict[str, RewardTermCfg]:
    s = DEXPBT_REWARD_SCALE
    finger_names = list(profile.obs_joints if profile else ALLEGRO_FINGER_JOINTS)
    r = {
        "fingertip_progress": RewardTermCfg(
            func=tmdp.fingertip_progress,
            weight=DEXPBT["distanceDeltaRewScale"] * s,
            params={"command_name": cmd, "gate_by_lift": gate_tips_by_lift},
        ),
        "lift_shaping": RewardTermCfg(
            func=tmdp.lift_shaping,
            weight=DEXPBT["liftingRewScale"] * s,
            params={"command_name": cmd},
        ),
        "lift_bonus": RewardTermCfg(
            func=tmdp.lift_bonus, weight=DEXPBT["liftingBonus"] * s, params={"command_name": cmd}
        ),
        # Deviation from DexPBT: small pre-lift opposition-grasp shaping.
        "grasp_contact": RewardTermCfg(
            func=tmdp.grasp_contact, weight=grasp_contact_weight, params={"command_name": cmd}
        ),
        "keypoint_progress": RewardTermCfg(
            func=tmdp.keypoint_progress,
            weight=DEXPBT["keypointRewScale"] * s,
            params={"command_name": cmd},
        ),
        "goal_bonus": RewardTermCfg(
            func=tmdp.goal_bonus,
            weight=DEXPBT["reachGoalBonus"] / hold_steps * s,
            params={"command_name": cmd},
        ),
    }
    if velocity_penalties:
        for h in hands:
            r[f"{h}_base_vel_penalty"] = RewardTermCfg(
                func=tmdp.joint_vel_l1,
                weight=-DEXPBT["kukaActionsPenaltyScale"] * s,
                params={
                    "asset_cfg": SceneEntityCfg(
                        h, joint_names=list(BASE_JOINTS), preserve_order=True
                    )
                },
            )
            r[f"{h}_finger_vel_penalty"] = RewardTermCfg(
                func=tmdp.joint_vel_l1,
                weight=-DEXPBT["allegroActionsPenaltyScale"] * s,
                params={
                    "asset_cfg": SceneEntityCfg(h, joint_names=finger_names, preserve_order=True)
                },
            )
    return r


def hand_reset_events(
    hand: str, ws: HandRootWorkspace, profile: HandProfile | None = None
) -> dict[str, EventTermCfg]:
    finger_names = list(profile.obs_joints if profile else ALLEGRO_FINGER_JOINTS)
    return {
        f"{hand}_reset_base": EventTermCfg(
            func=tmdp.reset_floating_base,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg(hand), "noise": dict(ws.reset_noise)},
        ),
        f"{hand}_reset_fingers": EventTermCfg(
            func=envs_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.1, 0.1),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg(hand, joint_names=finger_names),
            },
        ),
    }


def common_terminations(
    cmd: str, timing: TaskTiming, goal_timeout_is_failure: bool = False
) -> dict[str, TerminationTermCfg]:
    return {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "goal_timeout": TerminationTermCfg(
            func=tmdp.goal_timeout,
            time_out=not goal_timeout_is_failure,
            params={"command_name": cmd, "per_goal_s": timing.per_goal_s},
        ),
        "max_successes": TerminationTermCfg(
            func=tmdp.max_successes,
            time_out=True,
            params={"command_name": cmd, "max_successes": timing.max_successes},
        ),
        "object_dropped": TerminationTermCfg(
            func=tmdp.object_dropped, params={"command_name": cmd}
        ),
        "nan": TerminationTermCfg(func=envs_mdp.nan_detection),
    }


def make_sim_cfg(nconmax: int = 200, njmax: int = 1000) -> SimulationCfg:
    return SimulationCfg(
        nconmax=nconmax,
        njmax=njmax,
        mujoco=MujocoCfg(
            timestep=0.005, iterations=50, ls_iterations=50, impratio=10, cone="elliptic"
        ),
    )


ALLEGRO_BASE_DEFAULTS = {"slide": (300.0, 40.0, 200.0), "hinge": (60.0, 8.0, 80.0)}
"""Allegro floating-base XML gains (kp, kv, force limit) and joint params, the reference for match_base_dynamics."""
ALLEGRO_BASE_JOINT = {
    "slide": {"armature": 0.02, "damping": 3.0, "frictionloss": 0.0},
    "hinge": {"armature": 0.01, "damping": 0.5, "frictionloss": 0.0},
}
_SLIDES, _HINGES = ("base_x", "base_y", "base_z"), ("base_roll", "base_pitch", "base_yaw")
_ALLEGRO_INERTIA: dict[str, float] | None = None


def base_inertia(spec) -> dict[str, float]:
    """Effective inertia (joint-space mass matrix diagonal incl. armature) of the six base joints at qpos0."""
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    import numpy as np

    M = np.zeros((m.nv, m.nv))
    try:
        mujoco.mj_fullM(m, d, M)
    except TypeError:
        mujoco.mj_fullM(m, M, d.qM)
    out = {}
    for n in _SLIDES + _HINGES:
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        assert j >= 0, n
        out[n] = float(M[m.jnt_dofadr[j], m.jnt_dofadr[j]])
    return out


def allegro_base_inertia() -> dict[str, float]:
    global _ALLEGRO_INERTIA
    if _ALLEGRO_INERTIA is None:
        _ALLEGRO_INERTIA = base_inertia(ALLEGRO_HAND_RIGHT.get_floating_cfg().spec_fn())
    return _ALLEGRO_INERTIA


def make_hand_entity_cfg(
    hand_spec,
    ws: HandRootWorkspace,
    base_gains: tuple[float, float, float] | None = None,
    slide_joint_limits: bool = False,
    rot_gains: tuple[float, float, float] | None = None,
    profile: HandProfile | None = None,
):
    """Floating hand for these tasks: anchor pose + widened base hinge range.

    Wraps the hand's spec_fn (fresh cfg per call; the shared XML and the legacy
    tasks are untouched): base_roll/pitch/yaw joint range and the matching
    position-actuator ctrlrange become +-ws.base_rot_joint_range. For hands whose own base axes are not the canonical
    ones (every hand except Allegro), ``profile.base_fix`` re-orients the six base joint axes and puts the hinge pivots at
    the palm origin, so one layout works for all hands (hand_profile.py).
    """
    cfg = hand_spec.get_floating_cfg(pos=ws.anchor_pos, rot=ws.anchor_rot)
    base_fn = cfg.spec_fn
    r = float(ws.base_rot_joint_range)
    for n in ("base_roll", "base_pitch", "base_yaw"):
        lo, hi = ws.joint_bounds[n]
        assert -r <= lo and hi <= r, (n, ws.joint_bounds[n], r)

    def spec_fn():
        spec = base_fn()
        if profile is not None and profile.base_fix:
            fixed = 0
            for j in spec.joints:
                f = profile.base_fix.get(j.name)
                if f is not None:
                    j.axis = list(f["axis_body"])
                    if j.name in ("base_roll", "base_pitch", "base_yaw"):
                        j.pos = list(f["pos_body"])
                    fixed += 1
            assert fixed == 6, f"canonical-axis fix touched {fixed} of 6 base joints"
        match = profile is not None and profile.match_base_dynamics
        if match:
            for j in spec.joints:
                if j.name in _SLIDES + _HINGES:
                    for k, v in ALLEGRO_BASE_JOINT[
                        "slide" if j.name in _SLIDES else "hinge"
                    ].items():
                        cur = getattr(j, k)
                        if hasattr(
                            cur, "__len__"
                        ):  # e.g. MjsJoint.damping is a 3-vector (linear, quadratic, ...)
                            vec = [0.0] * len(cur)
                            vec[0] = v
                            setattr(j, k, vec)
                        else:
                            setattr(j, k, v)
        names = {"base_roll", "base_pitch", "base_yaw"}
        hit = 0
        for j in spec.joints:
            if j.name in names:
                j.range = [-r, r]
                hit += 1
        for a in spec.actuators:
            if a.target in names:
                a.ctrlrange = [-r, r]
                hit += 1
        assert hit == 6, f"expected 3 joints + 3 actuators, got {hit}"
        if slide_joint_limits:
            # Hard joint limits on the base slides = workspace bounds (physics enforces the
            # reach limit even under contact pushes / PD overshoot, like an arm's kinematics).
            n_set = 0
            for j in spec.joints:
                if j.name in ("base_x", "base_y", "base_z"):
                    j.range = list(ws.joint_bounds[j.name])
                    j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
                    n_set += 1
            assert n_set == 3
        ratio = {}
        if match:
            ih, ia = base_inertia(spec), allegro_base_inertia()
            ratio = {n: ih[n] / ia[n] for n in _SLIDES + _HINGES}
        sg = (
            base_gains
            if base_gains is not None
            else (ALLEGRO_BASE_DEFAULTS["slide"] if match else None)
        )
        rg = (
            rot_gains
            if rot_gains is not None
            else (ALLEGRO_BASE_DEFAULTS["hinge"] if match else None)
        )
        for gains, names in ((sg, _SLIDES), (rg, _HINGES)):
            # (kp, kv, force limit): task-specific base gains (juggling / DualHand-Far), given for Allegro; scaled by the
            # per-joint inertia ratio when the hand's base is matched to Allegro's dynamics.
            if gains is None:
                continue
            n_set = 0
            for a_ in spec.actuators:
                if a_.target in names:
                    k_i = ratio.get(a_.target, 1.0)
                    kp, kv, fmax = (g * k_i for g in gains)
                    a_.gainprm[0] = kp
                    a_.biasprm[1] = -kp
                    a_.biasprm[2] = -kv
                    a_.forcerange = [-fmax, fmax]
                    n_set += 1
            assert n_set == 3
        return spec

    cfg.spec_fn = spec_fn
    return cfg


def make_object_entity_cfg(ts) -> object:
    """DexPBT cuboid variants (sizes recorded into the command cfg) or one cube."""
    if ts.object_sets is None:
        ts.command.object_sizes = None
        return make_cube_entity_cfg(ts.cube)
    cfg, table = make_dexpbt_object_cfg(ts.object_sets)
    ts.command.object_sizes = tuple(s for _, s in table)
    return cfg


def _goal_bonus_override(rewards: dict, cmd: str, mode: str) -> dict:
    if mode == "on_success":
        rewards["goal_bonus"] = RewardTermCfg(
            func=tmdp.success_bonus_event,
            weight=DEXPBT["reachGoalBonus"] * DEXPBT_REWARD_SCALE,
            params={"command_name": cmd},
        )
    return rewards


def make_single_hand_env_cfg(ts: SingleHandTaskSpec, play: bool = False) -> ManagerBasedRlEnvCfg:
    ts = copy.deepcopy(ts)
    if play:
        # Evaluation/play always uses the final (frozen) tolerances.
        ts.command.tolerance_scale_init = 1.0
        ts.use_curriculum = False
    cmd_name = ts.command.task + "_goal"
    hand = "hand"
    profile = get_profile(ts.hand)
    ts.command.hand_profiles = (profile.key,)
    hold_steps = max(1, int(round(ts.command.hold_time_s / (0.005 * 4))))
    throw = ts.task == "throw"
    obs = {
        **hand_obs_terms(hand, -1, cmd_name, profile),
        **task_obs_terms(cmd_name, ts.command.goal_orientation, throw),
    }
    observations = {
        "actor": ObservationGroupCfg(dict(obs), enable_corruption=False),
        "critic": ObservationGroupCfg(dict(obs), enable_corruption=False),
    }
    observations["actor"].nan_policy = "error"
    observations["critic"].nan_policy = "error"

    events = {
        "reset_scene_to_default": EventTermCfg(func=envs_mdp.reset_scene_to_default, mode="reset")
    }
    events.update(hand_reset_events(hand, ts.workspace, profile))
    events["reset_object"] = EventTermCfg(
        func=reset_object_event, mode="reset", params={"command_name": cmd_name}
    )

    terms = common_terminations(cmd_name, ts.timing, ts.goal_timeout_is_failure)
    if ts.hand_far_reset:
        terms["hand_far"] = TerminationTermCfg(
            func=tmdp.hand_far, params={"command_name": cmd_name}
        )
    if throw:
        terms["throw_failed"] = TerminationTermCfg(
            func=tmdp.throw_failed, params={"command_name": cmd_name}
        )

    sensor_cfgs = [
        sensors.hand_object_contact(hand, profile=profile),
        sensors.tip_object_contact(hand, profile=profile),
        sensors.object_table_contact(),
        sensors.hand_table_contact(hand, profile=profile),
    ]
    if throw:
        sensor_cfgs += [
            sensors.object_bucket_contact(),
            sensors.hand_bucket_contact(hand, profile=profile),
        ]

    curriculum = {}
    if ts.use_curriculum:
        cur_params = {"command_name": cmd_name}
        if ts.curriculum_min_successes != 3.0:
            cur_params["min_mean_successes"] = ts.curriculum_min_successes
        curriculum["tolerance"] = CurriculumTermCfg(
            func=tmdp.tolerance_curriculum, params=cur_params
        )

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=1 if play else NUM_TRAIN_ENVS,
            env_spacing=1.5,
            sensors=tuple(sensor_cfgs),
        ),
        observations=observations,
        actions={
            "hand": make_action_cfg(
                hand,
                ts.workspace,
                finger_mode=ts.finger_mode,
                finger_ema=ts.finger_ema,
                base_ema=ts.base_ema,
                profile=profile,
            )
        },
        commands={cmd_name: ts.command},
        events=events,
        rewards={
            **_goal_bonus_override(
                dexpbt_rewards(
                    cmd_name,
                    [hand],
                    hold_steps,
                    gate_tips_by_lift=True,
                    velocity_penalties=True,
                    profile=profile,
                    grasp_contact_weight=ts.grasp_contact_weight,
                ),
                cmd_name,
                ts.goal_bonus_mode,
            ),
            **(
                {
                    "base_bound_pressure": RewardTermCfg(
                        func=tmdp.base_bound_pressure,
                        weight=ts.base_bound_penalty,
                        params={"action_name": "hand"},
                    )
                }
                if ts.base_bound_penalty
                else {}
            ),
            **(
                {
                    "finger_action_rate": RewardTermCfg(
                        func=tmdp.finger_action_rate_l2,
                        weight=ts.finger_action_rate,
                        params={"hands": (hand,)},
                    )
                }
                if ts.finger_action_rate
                else {}
            ),
            **(
                {
                    "finger_action_acc": RewardTermCfg(
                        func=tmdp.finger_action_acc_l2,
                        weight=ts.finger_action_acc,
                        params={"hands": (hand,)},
                    )
                }
                if ts.finger_action_acc
                else {}
            ),
        },
        terminations=terms,
        curriculum=curriculum,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.WORLD,
            lookat=(0.1, 0.0, 0.5),
            distance=1.4,
            elevation=-25.0,
            azimuth=135.0,
        ),
        sim=make_sim_cfg(),
        decimation=4,
        episode_length_s=ts.timing.episode_s,
        scale_rewards_by_dt=False,
        is_finite_horizon=False,
    )
    entities = {
        hand: make_hand_entity_cfg(profile.spec, ts.workspace, profile=profile),
        "table": make_table_entity_cfg(ts.table),
        "object": make_object_entity_cfg(ts),
    }
    if throw:
        assert ts.bucket is not None
        lo, hi = ts.command.bucket_pos_min, ts.command.bucket_pos_max
        mid = tuple((a + b) / 2 for a, b in zip(lo, hi))
        entities["bucket"] = make_bucket_entity_cfg(ts.bucket, pos=mid)  # type: ignore[arg-type]
    cfg.scene.entities = entities
    if play:
        cfg.episode_length_s = ts.timing.episode_s  # finite: eval must stay finite.
    return cfg


__all__ = [
    "ALLEGRO_HAND_RIGHT",
    "DEXPBT",
    "DEXPBT_REWARD_SCALE",
    "PALM_DOWN_FINGERS_NX",
    "PALM_DOWN_FINGERS_PX",
    "SingleHandTaskSpec",
    "TaskTiming",
    "make_single_hand_env_cfg",
]


def track_root_gains(
    inertia_slide: float,
    inertia_rot: float,
    ws: HandRootWorkspace,
    natural_freq_hz: float = 10.0,
    damping_ratio: float = 2.0,
    effort_fraction: float = 0.25,
):
    """mjlab G1 recipe applied to the floating root (g1_constants.py).

    stiffness = I * (2 pi f)^2, damping = 2 * zeta * I * (2 pi f); the effort limit is set so that a
    full action (|a| = 1, i.e. target offset = scale) loads the motor with ``effort_fraction`` of its
    limit, the same invariant as ``G1_ACTION_SCALE = 0.25 * effort / stiffness``. ``I`` is the effective
    inertia of the base joint measured at the home pose (mj_crb incl. armature).
    Returns ((kp, kv, fmax) slides, (kp, kv, fmax) hinges)."""
    import math

    w = 2.0 * math.pi * natural_freq_hz
    out = []
    for inertia, names in (
        (inertia_slide, ("base_x", "base_y", "base_z")),
        (inertia_rot, ("base_roll", "base_pitch", "base_yaw")),
    ):
        reach = max(max(abs(ws.joint_bounds[n][0]), abs(ws.joint_bounds[n][1])) for n in names)
        kp = inertia * w * w
        kv = 2.0 * damping_ratio * inertia * w
        out.append((kp, kv, kp * reach / effort_fraction))
    return tuple(out)
