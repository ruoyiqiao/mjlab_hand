"""Object keypoints and pose errors (DexPBT ``allegro_kuka_base.py`` semantics).

DexPBT (B:709-713) defines keypoint offsets in the object frame as::

    offset_k = kp_k * object_size * keypointScale / 2

with ``objectBaseSize=0.05`` m times the per-object scale (``object_size`` here is
the full edge length vector in metres) and ``keypointScale=1.5``. Reorientation /
two-arms use the corner set ``[[1,1,1],[1,1,-1],[-1,-1,1],[-1,-1,-1]]``; throw and
regrasping use the object center ``[[0,0,0]]``. The keypoint error is the **max**
over keypoints of the L2 distance (B:1073).

Quaternions are mjlab/MuJoCo **wxyz**. IsaacGym uses xyzw; convert explicitly with
:func:`xyzw_to_wxyz` (never by reinterpretation).
"""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_conjugate, quat_mul

CORNER_KEYPOINTS: tuple[tuple[float, float, float], ...] = (
    (1.0, 1.0, 1.0),
    (1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0),
    (-1.0, -1.0, -1.0),
)
CENTER_KEYPOINT: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),)
DEXPBT_KEYPOINT_SCALE = 1.5


def keypoint_offsets(
    pattern: tuple[tuple[float, float, float], ...],
    object_size: tuple[float, float, float],
    keypoint_scale: float = DEXPBT_KEYPOINT_SCALE,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Object-frame keypoint offsets, shape (K, 3)."""
    kp = torch.tensor(pattern, dtype=torch.float32, device=device)
    size = torch.tensor(object_size, dtype=torch.float32, device=device)
    return kp * size * keypoint_scale / 2.0


def keypoints_world(pos: torch.Tensor, quat: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """World keypoints (N, K, 3) for poses (N,3)/(N,4 wxyz) and offsets (K,3) or (N,K,3)."""
    n = pos.shape[0]
    if offsets.dim() == 2:
        offsets = offsets.unsqueeze(0).expand(n, -1, 3)
    k = offsets.shape[1]
    q = quat.unsqueeze(1).expand(n, k, 4).reshape(-1, 4)
    o = offsets.reshape(-1, 3)
    return (quat_apply(q, o).reshape(n, k, 3)) + pos.unsqueeze(1)


def keypoint_max_dist(kp_a: torch.Tensor, kp_b: torch.Tensor) -> torch.Tensor:
    """max_k ||a_k - b_k||, shape (N,)."""
    return torch.linalg.norm(kp_a - kp_b, dim=-1).amax(dim=-1)


def quat_angle_error(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Rotation angle (rad, in [0, pi]) between two wxyz quaternions; q == -q."""
    q1 = q1 / torch.linalg.norm(q1, dim=-1, keepdim=True)
    q2 = q2 / torch.linalg.norm(q2, dim=-1, keepdim=True)
    d = quat_mul(q1, quat_conjugate(q2))
    w = d[..., 0].abs().clamp(max=1.0)
    vec = torch.linalg.norm(d[..., 1:], dim=-1)
    return 2.0 * torch.atan2(vec, w)


def xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., 3:4], q[..., 0:3]], dim=-1)


def wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., 1:4], q[..., 0:1]], dim=-1)


def random_quat(n: int, device, generator: torch.Generator | None = None) -> torch.Tensor:
    """Uniform random unit quaternions (wxyz), Shoemake / DexPBT get_random_quat."""
    u = torch.rand(n, 3, device=device, generator=generator)
    a = torch.sqrt(1 - u[:, 0])
    b = torch.sqrt(u[:, 0])
    t1 = 2 * torch.pi * u[:, 1]
    t2 = 2 * torch.pi * u[:, 2]
    # Shoemake returns (x, y, z, w) = (a sin t1, a cos t1, b sin t2, b cos t2).
    xyzw = torch.stack(
        [a * torch.sin(t1), a * torch.cos(t1), b * torch.sin(t2), b * torch.cos(t2)], -1
    )
    return xyzw_to_wxyz(xyzw)
