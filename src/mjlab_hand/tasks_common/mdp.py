"""Rewards, observations, terminations, events and curriculum for the Allegro
DexPBT-style tasks. All task state comes from :class:`DexPBTTaskCommand`.

Reward weights in the task configs are DexPBT coefficients times a global 0.01
(rl_games ``reward_shaper.scale_value``), with ``scale_rewards_by_dt=False`` so a
weight is a per-control-step reward exactly as in IsaacGymEnvs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_conjugate, quat_mul

from mjlab_hand.tasks_common.tracker import DexPBTTaskCommand, get_tracker
from mjlab_hand.tasks_common.workspace import BASE_JOINTS

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# ----------------------------------------------------------------------------
# Rewards (DexPBT compute_kuka_reward terms, B:854-898 / T2:818-847)
# ----------------------------------------------------------------------------
def fingertip_progress(env: ManagerBasedRlEnv, command_name: str, gate_by_lift: bool = True):
    """sum_tips clip(best - curr, 0, 10) * (~lifted if gated). DexPBT x50."""
    t = get_tracker(env, command_name)
    return t.tip_delta_gated if gate_by_lift else t.tip_delta


def lift_shaping(env: ManagerBasedRlEnv, command_name: str):
    """clip(z_lift, 0, 0.5) * ~lifted. DexPBT x20."""
    return get_tracker(env, command_name).lift_rew


def grasp_contact(env: ManagerBasedRlEnv, command_name: str):
    """Pre-lift opposition contact (thumb + >=1 fingertip on the object). Deviation:
    not in DexPBT; added after attempt a2 showed no lifting in 45M steps."""
    return get_tracker(env, command_name).grasp_rew


def lift_bonus(env: ManagerBasedRlEnv, command_name: str):
    """1 on the step lifted first becomes true. DexPBT x300."""
    return get_tracker(env, command_name).lift_bonus


def keypoint_progress(env: ManagerBasedRlEnv, command_name: str):
    """clip(best_kp - kp_max, 0, 100) * lifted. DexPBT x200."""
    return get_tracker(env, command_name).kp_delta


def goal_bonus(env: ManagerBasedRlEnv, command_name: str):
    """near_goal per step; weight = reachGoalBonus / hold_steps (DexPBT 1000/success_steps)."""
    return get_tracker(env, command_name).near_goal.float()


def base_bound_pressure(env: ManagerBasedRlEnv, action_name: str):
    """Fraction of base dims whose integrated target was clipped by the workspace
    bound this step (pushing against the reach limit). Deviation, see mapping."""
    return env.action_manager.get_term(action_name).clip_fraction


def success_bonus_event(env: ManagerBasedRlEnv, command_name: str):
    """1 on the step a goal is completed (hold satisfied). Lump-sum goal bonus."""
    return get_tracker(env, command_name).success_now.float()


def joint_vel_l1(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg):
    """sum |qvel| over the selected joints (DexPBT 'actions penalty' is on dof vel)."""
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids].abs().sum(dim=-1)


# ----------------------------------------------------------------------------
# Observations (DexPBT compute_observations, B:1091-1172)
# ----------------------------------------------------------------------------
def _env_local(env, pos):
    return pos - env.scene.env_origins


def hand_joint_pos_unit(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg):
    """Joint positions unscaled to [-1, 1] by their limits (DexPBT unscale)."""
    a = env.scene[asset_cfg.name]
    q = a.data.joint_pos[:, asset_cfg.joint_ids]
    lim = a.data.joint_pos_limits[:, asset_cfg.joint_ids]
    return 2.0 * (q - lim[..., 0]) / (lim[..., 1] - lim[..., 0]) - 1.0


def hand_joint_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg):
    return env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]


def palm_state(env: ManagerBasedRlEnv, command_name: str, hand_index: int = 0):
    """Palm position (env-local), quat (wxyz), lin vel, ang vel: 13."""
    t = get_tracker(env, command_name)
    h = t.hands[hand_index]
    pid = t.palm_ids[hand_index]
    pos, quat = t.palm_pose(hand_index)
    return torch.cat(
        [
            _env_local(env, pos),
            quat,
            h.data.body_link_lin_vel_w[:, pid],
            h.data.body_link_ang_vel_w[:, pid],
        ],
        dim=-1,
    )


def fingertips_rel_palm(env: ManagerBasedRlEnv, command_name: str, hand_index: int = 0):
    """Fingertip positions relative to the palm (world-aligned): 3 per tip."""
    t = get_tracker(env, command_name)
    pos, _ = t.palm_pose(hand_index)
    k = t.tips_per_hand
    tips = t.tips_w()[:, k * hand_index : k * (hand_index + 1)]
    return (tips - pos.unsqueeze(1)).reshape(env.num_envs, -1)


def object_state(env: ManagerBasedRlEnv, command_name: str):
    """Object pos (env-local), quat, lin vel, ang vel: 13."""
    t = get_tracker(env, command_name)
    p, q = t.object_pose()
    return torch.cat(
        [_env_local(env, p), q, t.obj.data.root_link_lin_vel_w, t.obj.data.root_link_ang_vel_w],
        dim=-1,
    )


def keypoints_rel_palm(env: ManagerBasedRlEnv, command_name: str, hand_index: int = 0):
    t = get_tracker(env, command_name)
    pos, _ = t.palm_pose(hand_index)
    return (t.object_keypoints() - pos.unsqueeze(1)).reshape(env.num_envs, -1)


def keypoints_rel_goal(env: ManagerBasedRlEnv, command_name: str):
    t = get_tracker(env, command_name)
    return (t.object_keypoints() - t.goal_keypoints()).reshape(env.num_envs, -1)


def goal_pose(env: ManagerBasedRlEnv, command_name: str):
    """Goal pos (env-local) and quat: 7."""
    t = get_tracker(env, command_name)
    return torch.cat([_env_local(env, t.goal_pos), t.goal_quat], dim=-1)


def object_quat_rel_goal(env: ManagerBasedRlEnv, command_name: str):
    t = get_tracker(env, command_name)
    _, q = t.object_pose()
    return quat_mul(q, quat_conjugate(t.goal_quat))


def task_progress(env: ManagerBasedRlEnv, command_name: str):
    """[closest_kp_dist, closest_tip_dists..., lifted, log(goal_steps/10+1), log(successes+1)]."""
    t = get_tracker(env, command_name)
    best_kp = torch.where(t.best_kp < 0, t.kp_dist, t.best_kp).unsqueeze(-1)
    best_tip = torch.where(t.best_tip < 0, t.tip_dist, t.best_tip)
    return torch.cat(
        [
            best_kp,
            best_tip,
            t.lifted.float().unsqueeze(-1),
            torch.log(t.goal_steps.float() / 10.0 + 1.0).unsqueeze(-1),
            torch.log(t.successes.float() + 1.0).unsqueeze(-1),
        ],
        dim=-1,
    )


def throw_stage(env: ManagerBasedRlEnv, command_name: str):
    """Throw stage flags: grasped, released, in_bucket, flight_time; plus bucket pos (env-local)."""
    t = get_tracker(env, command_name)
    return torch.cat(
        [
            t.grasped.float().unsqueeze(-1),
            t.released.float().unsqueeze(-1),
            t.in_bucket.float().unsqueeze(-1),
            t.flight_time.unsqueeze(-1),
            _env_local(env, t.bucket_pos),
        ],
        dim=-1,
    )


def object_scale(env: ManagerBasedRlEnv, command_name: str):
    """Object dimensions / objectBaseSize (DexPBT obs 'object scales', 3)."""
    t = get_tracker(env, command_name)
    return t.obj_size / t.cfg.object_base_size


def contact_flags(env: ManagerBasedRlEnv, command_name: str):
    """Per-hand object contact flags."""
    return get_tracker(env, command_name).hand_contact.float()


# ----------------------------------------------------------------------------
# Terminations
# ----------------------------------------------------------------------------
def object_dropped(env: ManagerBasedRlEnv, command_name: str, below_table: float = 0.1):
    """Object center below the table top - margin (DexPBT: z < 0.1 world).

    For throw only before release (post-release failures handled separately).
    """
    t = get_tracker(env, command_name)
    p, _ = t.object_pose()
    dropped = p[:, 2] < t.table_top() - below_table
    if t.cfg.task == "throw":
        dropped = dropped & ~t.released
    return dropped


def throw_failed(env: ManagerBasedRlEnv, command_name: str, floor_z: float = 0.05):
    """After release: object hit the table, or fell below the bucket floor / near ground."""
    t = get_tracker(env, command_name)
    p, _ = t.object_pose()
    low = p[:, 2] < torch.minimum(t.bucket_pos[:, 2] - 0.02, t.table_top() - 0.1)
    low = low | (p[:, 2] < floor_z)
    return t.released & (t.throw_failed | (low & ~t.in_bucket))


def goal_timeout(env: ManagerBasedRlEnv, command_name: str, per_goal_s: float):
    """DexPBT per-goal timeout (progress_buf is reset on success)."""
    return get_tracker(env, command_name).goal_time >= per_goal_s


def max_successes(env: ManagerBasedRlEnv, command_name: str, max_successes: int):
    return get_tracker(env, command_name).successes >= max_successes


def direct_handoff_bonus(env: ManagerBasedRlEnv, command_name: str):
    """1 on the step a direct (no table) handover gives the object to the hand on the goal side
    (once per goal)."""
    return get_tracker(env, command_name).direct_bonus


def table_handover_f(env: ManagerBasedRlEnv, command_name: str):
    """Reward-side float of the table handover (penalty weight < 0)."""
    t = get_tracker(env, command_name)
    return (t.indirect_handover & t.cfg.forbid_table_handover).float()


def table_handover(env: ManagerBasedRlEnv, command_name: str):
    """Dual hand (Far layout): the object was picked up by the other hand after resting on the table
    with nobody holding it = handover through the table = failure (only if the command arms it)."""
    t = get_tracker(env, command_name)
    return t.indirect_handover & t.cfg.forbid_table_handover


def hand_hand_collision(env: ManagerBasedRlEnv, command_name: str):
    """Dual hand: any inter-hand contact (found or penetration in any substep) = failure.

    Deviation from DexPBT two-arms (arms never meet there); acceptance requires
    zero unintended hand-hand collisions, so contact ends the episode.
    """
    return get_tracker(env, command_name).hand_hand


def hand_far(env: ManagerBasedRlEnv, command_name: str, max_dist: float = 1.5):
    """Reorientation extra reset rule: any fingertip > max_dist from the object."""
    return get_tracker(env, command_name).tip_dist.amax(dim=-1) > max_dist


# ----------------------------------------------------------------------------
# Events
# ----------------------------------------------------------------------------
def reset_floating_base(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg, noise: dict
) -> None:
    """Reset base joints to anchor (default 0) + uniform noise; zero velocity.

    Base joints are looked up by name on the named entity (no 'hand' default).
    Finger joints are reset by mdp.reset_joints_by_offset separately.
    """
    hand = env.scene[asset_cfg.name]
    n = len(env_ids)
    ids = [hand.joint_names.index(j) for j in BASE_JOINTS]
    q = hand.data.default_joint_pos[env_ids][:, ids].clone()
    for k, j in enumerate(BASE_JOINTS):
        lo, hi = noise.get(j, (0.0, 0.0))
        q[:, k] += torch.rand(n, device=env.device) * (hi - lo) + lo
    hand.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=ids, env_ids=env_ids)


# ----------------------------------------------------------------------------
# Curriculum (DexPBT allegro_kuka_utils.tolerance_curriculum)
# ----------------------------------------------------------------------------
def tolerance_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    interval_steps: int = 3000,
    min_mean_successes: float = 3.0,
    increment: float = 0.9,
):
    """Every >= interval_steps control steps, if the mean successes of recently
    finished episodes >= min_mean_successes, multiply the tolerance scale by
    ``increment`` (clamped to >= 1.0 = final tolerance)."""
    t: DexPBTTaskCommand = env.command_manager.get_term(command_name)
    step = int(env.common_step_counter)
    last = getattr(t, "_curriculum_last", 0)
    if step - last >= interval_steps:
        mean_succ = float(t.prev_episode_successes.mean().item())
        if mean_succ >= min_mean_successes and t.tol_scale > 1.0:
            t.tol_scale = max(1.0, t.tol_scale * increment)
            t._curriculum_last = step
    return torch.tensor(t.tol_scale)


def _hand_action_slices(env: ManagerBasedRlEnv, hands: tuple[str, ...]):
    am = env.action_manager
    dims = dict(zip(am.active_terms, am.action_term_dim))
    off = 0
    for name in am.active_terms:
        if name in hands:
            yield off, dims[name]
        off += dims[name]


def finger_action_rate_l2(env: ManagerBasedRlEnv, hands: tuple[str, ...], n_base: int = 6):
    """Sum over hands of |a_t - a_{t-1}|^2 on the finger action dims (everything after the six root dims).

    Penalises finger chatter (first difference of the raw policy output); canonical layout [base(6), fingers]."""
    am = env.action_manager
    a, pa = am.action, am.prev_action
    out = torch.zeros(a.shape[0], device=a.device)
    for off, n in _hand_action_slices(env, hands):
        out = out + torch.sum(
            torch.square(a[:, off + n_base : off + n] - pa[:, off + n_base : off + n]), dim=1
        )
    return out


def finger_action_acc_l2(env: ManagerBasedRlEnv, hands: tuple[str, ...], n_base: int = 6):
    """Sum over hands of |a_t - 2 a_{t-1} + a_{t-2}|^2 on the finger action dims (second difference: jerk)."""
    am = env.action_manager
    a, pa, ppa = am.action, am.prev_action, am.prev_prev_action
    out = torch.zeros(a.shape[0], device=a.device)
    for off, n in _hand_action_slices(env, hands):
        sl = slice(off + n_base, off + n)
        out = out + torch.sum(torch.square(a[:, sl] - 2.0 * pa[:, sl] + ppa[:, sl]), dim=1)
    return out


def base_action_rate_l2(env: ManagerBasedRlEnv, hands: tuple[str, ...], n_base: int = 6):
    """Sum over hands of |a_t - a_{t-1}|^2 on the six base (root) action dims only.

    mjlab tracking's ``action_rate_l2`` restricted to the root; raw policy output, canonical layout
    ([base(6), fingers(16)] per hand, in action-term order)."""
    am = env.action_manager
    a, pa = am.action, am.prev_action
    dims = dict(zip(am.active_terms, am.action_term_dim))
    out = torch.zeros(a.shape[0], device=a.device)
    off = 0
    for name in am.active_terms:
        if name in hands:
            out = out + torch.sum(
                torch.square(a[:, off : off + n_base] - pa[:, off : off + n_base]), dim=1
            )
        off += dims[name]
    return out
