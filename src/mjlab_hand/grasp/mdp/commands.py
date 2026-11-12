from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply

from mjlab_hand.grasp.mdp.table_utils import get_table_reference_pos_w

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

__all__ = [
    "GraspMetricsCommand",
    "GraspMetricsCommandCfg",
    "PoseCommand",
    "PoseCommandCfg",
]


class GraspMetricsCommand(CommandTerm):
    """No-op command that logs grasp diagnostics as episode metrics."""

    cfg: "GraspMetricsCommandCfg"

    def __init__(self, cfg: "GraspMetricsCommandCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)

        self._hand: Entity = env.scene[cfg.hand_name]
        self._object: Entity = env.scene[cfg.object_name]
        self._table: Entity = env.scene[cfg.table_name]

        cfg.asset_cfg.resolve(env.scene)
        body_ids = cfg.asset_cfg.body_ids
        self._hand_ref_body_idx: int = 0 if isinstance(body_ids, slice) else int(body_ids[0])

        cfg.keypoints_cfg.resolve(env.scene)
        keypoint_ids = cfg.keypoints_cfg.body_ids
        self._keypoint_indices: list[int] = (
            list(range(self._hand.num_bodies))
            if isinstance(keypoint_ids, slice)
            else [int(i) for i in keypoint_ids]
        )

        self._blank_command = torch.zeros(self.num_envs, 1, device=self.device)

        self._height_sum = torch.zeros(self.num_envs, device=self.device)
        self._distance_sum = torch.zeros(self.num_envs, device=self.device)
        self._step_count = torch.zeros(self.num_envs, device=self.device)

        self.metrics["object_height_beyond_table"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["hand_object_distance"] = torch.zeros(self.num_envs, device=self.device)

        self._pc_viz_local: torch.Tensor | None = None
        self._pc_viz_region: torch.Tensor | None = None
        self._pc_viz_num_regions: int = 0
        if cfg.contactdb_object_name is not None and cfg.debug_vis_object_pointcloud:
            from mjlab_hand.mdp.contact_explorer import get_cached_contact_db_object

            cdb = get_cached_contact_db_object(
                cfg.contactdb_object_name,
                scale=cfg.contactdb_scale,
            )
            pc = cdb.canonical_pointcloud
            regions = cdb.point_to_region
            m = int(pc.shape[0])
            max_p = max(1, int(cfg.object_pointcloud_max_points))
            if m > max_p:
                idx = torch.linspace(0, m - 1, max_p, device=pc.device).long().clamp(0, m - 1)
                pc = pc[idx]
                regions = regions[idx]
            self._pc_viz_local = pc.cpu()
            self._pc_viz_region = regions.cpu()
            self._pc_viz_num_regions = int(cdb.num_regions_effective)

    @property
    def command(self) -> torch.Tensor:
        return self._blank_command

    def _update_metrics(self) -> None:
        hand_pos = self._hand.data.body_link_pos_w[:, self._hand_ref_body_idx]
        obj_pos = self._object.data.root_link_pos_w
        table_pos = get_table_reference_pos_w(self._table)

        object_height = obj_pos[:, 2] - table_pos[:, 2]
        hand_object_distance = torch.norm(obj_pos - hand_pos, dim=-1)

        self._height_sum += object_height
        self._distance_sum += hand_object_distance
        self._step_count += 1.0

        denom = torch.clamp(self._step_count, min=1.0)
        self.metrics["object_height_beyond_table"] = self._height_sum / denom
        self.metrics["hand_object_distance"] = self._distance_sum / denom

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return

        self._height_sum[env_ids] = 0.0
        self._distance_sum[env_ids] = 0.0
        self._step_count[env_ids] = 0.0
        self.metrics["object_height_beyond_table"][env_ids] = 0.0
        self.metrics["hand_object_distance"][env_ids] = 0.0
        self._blank_command[env_ids] = 0.0

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        """Draw keypoint frames and object-attached pointcloud in viewer debug vis."""
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        if len(self._keypoint_indices) > 0:
            for env_idx in env_indices:
                for body_idx, body_name in zip(
                    self._keypoint_indices,
                    self.cfg.keypoints_cfg.body_names,
                    strict=True,
                ):
                    pos = self._hand.data.body_link_pos_w[env_idx, body_idx]
                    quat = self._hand.data.body_link_quat_w[env_idx, body_idx]
                    rotm = matrix_from_quat(quat)
                    visualizer.add_frame(
                        position=pos.cpu().numpy(),
                        rotation_matrix=rotm.cpu().numpy(),
                        scale=self.cfg.frame_scale,
                        label=f"keypoint_{body_name}_{env_idx}",
                        axis_radius=self.cfg.axis_radius,
                    )

        if self._pc_viz_local is None or not self.cfg.debug_vis_object_pointcloud:
            return

        pc_local = self._pc_viz_local.to(device=self.device, dtype=torch.float32)
        regions = self._pc_viz_region
        k_reg = max(1, self._pc_viz_num_regions)
        r = float(self.cfg.object_pointcloud_point_radius)

        for env_idx in env_indices:
            p_w = self._object.data.root_link_pos_w[env_idx : env_idx + 1]
            q_w = self._object.data.root_link_quat_w[env_idx : env_idx + 1]
            n_pts = pc_local.shape[0]
            q_rep = q_w.expand(n_pts, -1)
            pts_w = quat_apply(q_rep, pc_local) + p_w.expand(n_pts, -1)
            for i in range(n_pts):
                center = pts_w[i]
                if self.cfg.debug_vis_object_pointcloud_color_by_region and regions is not None:
                    rid = int(regions[i].item())
                    color = _region_id_to_rgba(rid, k_reg, self.cfg.object_pointcloud_alpha)
                else:
                    c = self.cfg.object_pointcloud_rgba
                    color = (c[0], c[1], c[2], c[3])
                visualizer.add_sphere(
                    center=center.cpu().numpy(),
                    radius=r,
                    color=color,
                    label=f"obj_pc_{env_idx}_{i}",
                )


def _region_id_to_rgba(
    region_id: int, num_regions: int, alpha: float
) -> tuple[float, float, float, float]:
    """Distinct hue per region (HSV), shared saturation/value."""
    if num_regions <= 0:
        return (0.2, 0.85, 1.0, alpha)
    h = (region_id % num_regions) / float(num_regions)
    s, v = 0.75, 0.95
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return (float(r), float(g), float(b), float(alpha))


@dataclass(kw_only=True)
class GraspMetricsCommandCfg(CommandTermCfg):
    """Config for blank grasp metric command."""

    hand_name: str
    object_name: str
    table_name: str
    frame_scale: float = 0.03
    axis_radius: float = 0.005

    asset_cfg: SceneEntityCfg
    keypoints_cfg: SceneEntityCfg

    contactdb_object_name: str | None = "cylinder_medium"
    contactdb_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    use_goal_state_feature: bool = True
    debug_vis_object_pointcloud: bool = True
    object_pointcloud_max_points: int = 256
    object_pointcloud_point_radius: float = 0.002
    object_pointcloud_rgba: tuple[float, float, float, float] = (0.2, 0.85, 1.0, 0.85)
    object_pointcloud_alpha: float = 0.85
    debug_vis_object_pointcloud_color_by_region: bool = True

    def build(self, env: "ManagerBasedRlEnv") -> GraspMetricsCommand:
        return GraspMetricsCommand(self, env)


class PoseCommand(CommandTerm):
    """Command that samples a target goal pose for the object.

    The goal position is sampled in world frame relative to the table with
    configurable ranges. The goal orientation is fixed to identity (no rotation).
    The policy receives this goal as an observation and is rewarded for moving
    the object toward it.
    """

    cfg: "PoseCommandCfg"

    def __init__(self, cfg: "PoseCommandCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)

        self._object: Entity = env.scene[cfg.object_name]
        self._table: Entity = env.scene[cfg.table_name]

        self.goal_pos = torch.zeros(self.num_envs, 3, device=self.device)

        self._goal_quat_identity = (
            torch.tensor(
                [1.0, 0.0, 0.0, 0.0],
                device=self.device,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .expand(self.num_envs, -1)
        )

    @property
    def command(self) -> torch.Tensor:
        return self.goal_pos

    @property
    def goal_quat(self) -> torch.Tensor:
        """Goal orientation in world frame.

        ObjectGoalCommand samples only positions, so this returns identity
        quaternions (no rotational goal).
        """
        return self._goal_quat_identity

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return

        cfg = self.cfg
        n = len(env_ids)

        table_pos = get_table_reference_pos_w(self._table)

        rel_x = torch.rand(n, device=self.device) * (cfg.pos_x_max - cfg.pos_x_min) + cfg.pos_x_min
        rel_y = torch.rand(n, device=self.device) * (cfg.pos_y_max - cfg.pos_y_min) + cfg.pos_y_min

        rel_z = torch.rand(n, device=self.device) * (cfg.pos_z_max - cfg.pos_z_min) + cfg.pos_z_min

        self.goal_pos[env_ids, 0] = table_pos[env_ids, 0] + rel_x
        self.goal_pos[env_ids, 1] = table_pos[env_ids, 1] + rel_y
        self.goal_pos[env_ids, 2] = table_pos[env_ids, 2] + rel_z

        table_z = table_pos[env_ids, 2]
        min_height = table_z + cfg.min_height_above_table
        self.goal_pos[env_ids, 2] = torch.maximum(self.goal_pos[env_ids, 2], min_height)

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    def _update_metrics(self) -> None:
        pass

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        """Draw sampled goal position as a sphere and world-aligned coordinate frame."""
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        cfg = self.cfg
        identity_rot = torch.eye(3, device=self.device, dtype=torch.float32)
        for env_idx in env_indices:
            goal_pos = self.goal_pos[env_idx]

            visualizer.add_sphere(
                center=goal_pos.cpu().numpy(),
                radius=cfg.goal_sphere_radius,
                color=cfg.goal_sphere_color,
                label=f"goal_sphere_{env_idx}",
            )

            visualizer.add_frame(
                position=goal_pos.cpu().numpy(),
                rotation_matrix=identity_rot.cpu().numpy(),
                scale=cfg.goal_frame_scale,
                label=f"goal_frame_{env_idx}",
                axis_radius=cfg.goal_axis_radius,
            )


@dataclass(kw_only=True)
class PoseCommandCfg(CommandTermCfg):
    """Configuration for pose command.

    The goal position is sampled relative to the table. The orientation
    is fixed to identity (no rotational goal).
    """

    class_type: type = PoseCommand

    object_name: str = "object"

    table_name: str = "table"

    pos_x_min: float = -0.15

    pos_x_max: float = 0.15

    pos_y_min: float = -0.15

    pos_y_max: float = 0.15

    pos_z_min: float = 0.1

    pos_z_max: float = 0.5

    min_height_above_table: float = 0.05

    resampling_time_range: tuple[float, float] = (10.0, 15.0)

    debug_vis: bool = False

    goal_sphere_radius: float = 0.02

    goal_sphere_color: tuple[float, float, float, float] = (0.2, 1.0, 0.2, 0.8)

    goal_frame_scale: float = 0.05

    goal_axis_radius: float = 0.003

    def build(self, env: "ManagerBasedRlEnv") -> PoseCommand:
        return PoseCommand(self, env)
