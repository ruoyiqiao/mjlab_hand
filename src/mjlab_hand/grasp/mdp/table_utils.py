"""Table queries that remain valid while reset events write new poses."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.entity import Entity
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply, quat_mul

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _fixed_surface_pose(table: Entity) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Compose a rigid table's geom pose from current mocap state, without FK.

    Geom/body world poses lag writes until sim.forward(). The table's fixed
    local geometry can be cached, but its mocap pose must be read on every call.
    """
    if not (table.is_fixed_base and table.is_mocap and not table.is_articulated):
        return None
    data = table.data
    cached = getattr(table, "_grasp_surface_local_pose", None)
    if cached is None:
        geom_idx = table.geom_names.index("table_surface")
        geom_id = data.indexing.geoms[geom_idx].id
        model = data.model
        pos = model.geom_pos[:, geom_id].clone()
        quat = model.geom_quat[:, geom_id].clone()
        body = int(model.geom_bodyid[geom_id])
        root = data.indexing.root_body_id
        while body != root:
            if body == 0:
                raise ValueError("Table surface must belong to the table mocap root")
            pos = quat_apply(model.body_quat[:, body], pos) + model.body_pos[:, body]
            quat = quat_mul(model.body_quat[:, body], quat)
            body = int(model.body_parentid[body])
        cached = (pos, quat)
        table._grasp_surface_local_pose = cached
    mocap_id = data.indexing.mocap_id
    pos_w = data.data.mocap_pos[:, mocap_id]
    quat_w = data.data.mocap_quat[:, mocap_id]
    return (
        pos_w + quat_apply(quat_w, cached[0].expand_as(pos_w)),
        quat_mul(quat_w, cached[1].expand_as(quat_w)),
    )


def get_table_reference_pos_w(table: Entity) -> torch.Tensor:
    """Return the table geom center; preserve existing reward/goal references."""
    if "table_surface" in table.geom_names:
        pose = _fixed_surface_pose(table)
        if pose is not None:
            return pose[0]
    if hasattr(table.data, "geom_pos_w") and table.data.geom_pos_w.shape[1] > 0:
        geom_idx = (
            table.geom_names.index("table_surface") if "table_surface" in table.geom_names else 0
        )
        return table.data.geom_pos_w[:, geom_idx, :]
    if table.is_fixed_base and table.is_mocap and table.data.body_link_pos_w.shape[1] > 1:
        return table.data.body_link_pos_w[:, 1, :]
    if table.data.body_link_pos_w.shape[1] > 0:
        return table.data.body_link_pos_w[:, 0, :]
    return table.data.root_link_pos_w


def get_table_top_height(table: Entity) -> torch.Tensor:
    """Top of the box table, including its thickness and current orientation."""
    idx = table.geom_names.index("table_surface")
    geom = table.data.indexing.geoms[idx]
    if geom.type != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError("Grasp initialization requires a box table_surface geom")
    pose = _fixed_surface_pose(table)
    if pose is None:
        pos, quat = table.data.geom_pos_w[:, idx], table.data.geom_quat_w[:, idx]
    else:
        pos, quat = pose
    half_size = table.data.model.geom_size[:, geom.id]
    # Exact world-Z support of a box. Existing task tables are horizontal.
    extent = (matrix_from_quat(quat)[:, 2, :].abs() * half_size).sum(dim=-1)
    return pos[:, 2] + extent


def get_table_reference_height(env: ManagerBasedRlEnv, table_name: str) -> torch.Tensor:
    """Return table reference height (center), as used by existing rewards."""
    return get_table_reference_pos_w(env.scene[table_name])[:, 2]
