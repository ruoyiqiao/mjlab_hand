"""Collision support for placing rigid grasp objects above a table."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import matrix_from_quat

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def object_lowest_offset(
    env: ManagerBasedRlEnv, object_name: str, quat: torch.Tensor
) -> torch.Tensor:
    """Minimum world-Z offset of the rotated collision mesh from the root.

    Cache geometry from the compiled model once. Resets only rotate the support
    direction and reduce over vertices on the simulation device; no RNG or
    simulator forward call is needed. Supports rigid mesh/box ContactDB objects
    with fixed geometry (the grasp tasks do not randomize object mesh scale).
    """
    obj = env.scene[object_name]
    vertices = getattr(obj, "_grasp_collision_vertices", None)
    if vertices is None:
        if obj.is_articulated:
            raise ValueError("Grasp table placement requires a rigid object")
        model = env.sim.mj_model
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        root = obj.data.indexing.root_body_id
        root_pos = data.xpos[root]
        root_rot = data.xmat[root].reshape(3, 3)
        points = []
        for geom in obj.data.indexing.geoms:
            gid = geom.id
            if not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
                continue
            kind = model.geom_type[gid]
            if kind == mujoco.mjtGeom.mjGEOM_MESH:
                mid = model.geom_dataid[gid]
                start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                local = model.mesh_vert[start : start + count]
            elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                local = np.asarray(list(product((-1, 1), repeat=3))) * model.geom_size[gid]
            else:
                raise ValueError(f"Unsupported grasp collision geom: {geom.name} ({kind})")
            world = local @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
            points.append((world - root_pos) @ root_rot)
        if not points:
            raise ValueError("Grasp object has no collision mesh or box")
        vertices = torch.as_tensor(np.concatenate(points), dtype=quat.dtype, device=quat.device)
        obj._grasp_collision_vertices = vertices
    # Row 2 of R maps root-local vertices to their world-Z offsets.
    direction = matrix_from_quat(quat)[:, 2, :]
    return (direction @ vertices.T).amin(dim=-1)
