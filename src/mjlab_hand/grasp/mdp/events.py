from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz

from mjlab_hand.grasp.mdp.reset_geometry import object_lowest_offset
from mjlab_hand.grasp.mdp.table_utils import get_table_reference_pos_w, get_table_top_height

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "reset_hand_base",
    "reset_object_on_table",
]


def reset_hand_base(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    pose_range: dict,
    velocity_range: dict,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Reset hand base pose with randomization.

    For floating hands, we reset the base joint positions (base_x, base_y, base_z,
    base_roll, base_pitch, base_yaw) instead of the root pose.
    """
    hand: Entity = env.scene[asset_cfg.name]

    n = len(env_ids)

    default_joint_pos = hand.data.default_joint_pos[env_ids].clone()

    base_joint_names = [
        "base_x",
        "base_y",
        "base_z",
        "base_roll",
        "base_pitch",
        "base_yaw",
    ]
    base_indices = []
    for name in base_joint_names:
        if name in hand.joint_names:
            base_indices.append(hand.joint_names.index(name))

    if len(base_indices) == 6:
        if "x" in pose_range:
            default_joint_pos[:, base_indices[0]] += (
                torch.rand(n, device=env.device) * (pose_range["x"][1] - pose_range["x"][0])
                + pose_range["x"][0]
            )
        if "y" in pose_range:
            default_joint_pos[:, base_indices[1]] += (
                torch.rand(n, device=env.device) * (pose_range["y"][1] - pose_range["y"][0])
                + pose_range["y"][0]
            )
        if "z" in pose_range:
            default_joint_pos[:, base_indices[2]] += (
                torch.rand(n, device=env.device) * (pose_range["z"][1] - pose_range["z"][0])
                + pose_range["z"][0]
            )

        if "roll" in pose_range:
            default_joint_pos[:, base_indices[3]] += (
                torch.rand(n, device=env.device) * (pose_range["roll"][1] - pose_range["roll"][0])
                + pose_range["roll"][0]
            )
        if "pitch" in pose_range:
            default_joint_pos[:, base_indices[4]] += (
                torch.rand(n, device=env.device) * (pose_range["pitch"][1] - pose_range["pitch"][0])
                + pose_range["pitch"][0]
            )
        if "yaw" in pose_range:
            default_joint_pos[:, base_indices[5]] += (
                torch.rand(n, device=env.device) * (pose_range["yaw"][1] - pose_range["yaw"][0])
                + pose_range["yaw"][0]
            )

        hand.write_joint_state_to_sim(
            default_joint_pos, torch.zeros_like(default_joint_pos), env_ids=env_ids
        )
    else:
        # For non-floating hands, fall back to root pose reset
        default_root_state = hand.data.default_root_state
        default_pos = default_root_state[:, :3]
        default_quat = default_root_state[:, 3:7]

        pos = default_pos[env_ids].clone()
        if "x" in pose_range:
            pos[:, 0] += (
                torch.rand(n, device=env.device) * (pose_range["x"][1] - pose_range["x"][0])
                + pose_range["x"][0]
            )
        if "y" in pose_range:
            pos[:, 1] += (
                torch.rand(n, device=env.device) * (pose_range["y"][1] - pose_range["y"][0])
                + pose_range["y"][0]
            )
        if "z" in pose_range:
            pos[:, 2] += (
                torch.rand(n, device=env.device) * (pose_range["z"][1] - pose_range["z"][0])
                + pose_range["z"][0]
            )

        quat = default_quat[env_ids]
        pose = torch.cat([pos, quat], dim=-1)
        hand.write_root_link_pose_to_sim(pose, env_ids=env_ids)

        vel = torch.zeros(n, 6, device=env.device)
        hand.write_root_link_velocity_to_sim(vel, env_ids=env_ids)


def reset_object_on_table(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    object_name: str,
    table_name: str,
    position_range: dict,
    orientation_range: dict,
    clearance: float = 0.001,
) -> None:
    """Sample the same XY/orientation distribution, clear of the table surface."""
    if clearance < 0:
        raise ValueError("Object clearance must be nonnegative")
    if len(env_ids) == 0:
        return
    obj: Entity = env.scene[object_name]
    table: Entity = env.scene[table_name]

    table_pos = get_table_reference_pos_w(table)[env_ids]

    n = len(env_ids)

    pos = table_pos.clone()

    if "x" in position_range:
        pos[:, 0] += (
            torch.rand(n, device=env.device) * (position_range["x"][1] - position_range["x"][0])
            + position_range["x"][0]
        )
    if "y" in position_range:
        pos[:, 1] += (
            torch.rand(n, device=env.device) * (position_range["y"][1] - position_range["y"][0])
            + position_range["y"][0]
        )

    roll = torch.zeros(n, device=env.device)
    pitch = torch.zeros(n, device=env.device)
    yaw = torch.zeros(n, device=env.device)

    if "roll" in orientation_range:
        roll = (
            torch.rand(n, device=env.device)
            * (orientation_range["roll"][1] - orientation_range["roll"][0])
            + orientation_range["roll"][0]
        )
    if "pitch" in orientation_range:
        pitch = (
            torch.rand(n, device=env.device)
            * (orientation_range["pitch"][1] - orientation_range["pitch"][0])
            + orientation_range["pitch"][0]
        )
    if "yaw" in orientation_range:
        yaw = (
            torch.rand(n, device=env.device)
            * (orientation_range["yaw"][1] - orientation_range["yaw"][0])
            + orientation_range["yaw"][0]
        )

    quat = quat_from_euler_xyz(roll, pitch, yaw)
    pos[:, 2] = (
        get_table_top_height(table)[env_ids]
        + clearance
        - object_lowest_offset(env, object_name, quat)
    )

    pose = torch.cat([pos, quat], dim=-1)
    obj.write_root_link_pose_to_sim(pose, env_ids=env_ids)

    vel = torch.zeros(n, 6, device=env.device)
    obj.write_root_link_velocity_to_sim(vel, env_ids=env_ids)
