from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
    quat_error_magnitude,
    random_orientation,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

__all__ = ["RotationCommand", "RotationCommandCfg"]


def _is_visual_geom(geom: mujoco.MjsGeom) -> bool:
    return geom.contype == 0 and geom.conaffinity == 0


def _build_scene_entity_ghost_model(
    scene_spec: mujoco.MjSpec,
    scene_model: mujoco.MjModel,
    entity_name: str,
    target_color: tuple[float, float, float, float],
    ghost_alpha: float,
) -> mujoco.MjModel:
    ghost_spec = scene_spec.copy()
    entity_prefix = f"{entity_name}/"

    for geom in ghost_spec.geoms:
        rgba = tuple(geom.rgba)
        if geom.name and geom.name.startswith(entity_prefix) and _is_visual_geom(geom):
            geom.rgba = (*target_color[:3], ghost_alpha)
        else:
            geom.rgba = (rgba[0], rgba[1], rgba[2], 0.0)

    ghost_model = ghost_spec.compile()
    if ghost_model.nq != scene_model.nq:
        raise ValueError("Ghost model must match the full scene nq")

    return ghost_model


class RotationCommand(CommandTerm):
    cfg: RotationCommandCfg

    def __init__(self, cfg: RotationCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        cfg.hand_asset_cfg.resolve(env.scene)
        self.hand: Entity = env.scene[cfg.hand_asset_cfg.name]
        self._hand_base_body_ids = [int(i) for i in cfg.hand_asset_cfg.body_ids]
        self.object: Entity = env.scene[cfg.entity_name]
        self._object_free_joint_qpos_adr = self.object.indexing.free_joint_q_adr.clone()
        if self._object_free_joint_qpos_adr.numel() != 7:
            raise ValueError("Rotation target ghost visualization requires a free-joint object")
        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.target_quat[:, 0] = 1.0  # identity quaternion (w,x,y,z)
        self._goal_pos_offset_from_hand_base = torch.tensor(
            cfg.goal_pos_offset_from_hand_base,
            device=self.device,
            dtype=torch.float32,
        )
        # Object init offset: use specified value or fall back to goal offset
        obj_init_offset = cfg.obj_init_pos_offset_from_hand_base
        if obj_init_offset is None:
            obj_init_offset = cfg.goal_pos_offset_from_hand_base
        self._obj_init_pos_offset_from_hand_base = torch.tensor(
            obj_init_offset,
            device=self.device,
            dtype=torch.float32,
        )
        self.cfg.ghost_model = _build_scene_entity_ghost_model(
            scene_spec=env.scene.spec,
            scene_model=env.sim.mj_model,
            entity_name=cfg.entity_name,
            target_color=cfg.viz.target_color,
            ghost_alpha=cfg.viz.ghost_alpha,
        )

        self.metrics["rot_dist"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_dist"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)
        self._episode_success = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.target_quat

    @property
    def goal_pos(self) -> torch.Tensor:
        return self.target_pos

    @property
    def goal_quat(self) -> torch.Tensor:
        return self.target_quat

    def _update_metrics(self) -> None:
        obj_pos_w = self.object.data.root_link_pos_w
        obj_quat_w = self.object.data.root_link_quat_w
        hand_base_pos = self.hand.data.body_link_pos_w[:, self._hand_base_body_ids[0]]
        goal_pos = hand_base_pos + self._goal_pos_offset_from_hand_base
        self.target_pos[:] = goal_pos

        rot_dist = quat_error_magnitude(obj_quat_w, self.target_quat)
        goal_dist = torch.norm(obj_pos_w - goal_pos, dim=-1)
        at_goal = (rot_dist < self.cfg.success_tolerance).float()

        self._episode_success = torch.maximum(self._episode_success, at_goal)

        self.metrics["rot_dist"] = rot_dist
        self.metrics["goal_dist"] = goal_dist
        self.metrics["at_goal"] = at_goal
        self.metrics["episode_success"] = self._episode_success

    def compute_success(self) -> torch.Tensor:
        return self.metrics["rot_dist"] < self.cfg.success_tolerance

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)

        self._episode_success[env_ids] = 0.0

        new_target_quat = random_orientation(n, self.device)
        self.target_quat[env_ids] = new_target_quat

        hand_base_pos = self.hand.data.body_link_pos_w[env_ids, self._hand_base_body_ids[0]]
        goal_pos = hand_base_pos + self._goal_pos_offset_from_hand_base
        self.target_pos[env_ids] = goal_pos

        # Object init position can be different from goal position
        obj_init_pos = hand_base_pos + self._obj_init_pos_offset_from_hand_base

        new_obj_quat = random_orientation(n, self.device)
        pose = torch.cat([obj_init_pos, new_obj_quat], dim=-1)
        velocity = torch.zeros(n, 6, device=self.device)

        self.object.write_root_link_pose_to_sim(pose, env_ids=env_ids)
        self.object.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        if self.cfg.ghost_model is None:
            return

        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        for env_idx in env_indices:
            qpos = self._env.sim.data.qpos[env_idx].clone()
            qpos[self._object_free_joint_qpos_adr] = torch.cat(
                (self.target_pos[env_idx], self.target_quat[env_idx]), dim=0
            )
            visualizer.add_ghost_mesh(
                qpos=qpos.cpu().numpy(),
                model=self.cfg.ghost_model,
                alpha=self.cfg.viz.ghost_alpha,
                label=f"rotation_target_{env_idx}",
            )


@dataclass(kw_only=True)
class RotationCommandCfg(CommandTermCfg):
    entity_name: str
    success_tolerance: float = 0.1
    fall_dist: float = 0.24
    goal_pos_offset_from_hand_base: tuple[float, float, float] = (0.0, 0.0, 0.1)
    obj_init_pos_offset_from_hand_base: tuple[float, float, float] | None = None
    """Object init position offset. If None, uses goal_pos_offset_from_hand_base."""
    ghost_model: mujoco.MjModel | None = None

    hand_asset_cfg: SceneEntityCfg
    """SceneEntityCfg for the hand base body. ``RotationCommand`` resolves it at init
  and indexes ``body_link_pos_w`` via ``body_ids``."""

    @dataclass
    class VizCfg:
        target_color: tuple[float, float, float, float] = (0.0, 1.0, 0.5, 0.3)
        ghost_alpha: float = 0.4

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: ManagerBasedRlEnv) -> RotationCommand:
        return RotationCommand(self, env)
