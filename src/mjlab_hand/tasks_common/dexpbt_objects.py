"""DexPBT object set (cuboids) as an mjlab per-world variant entity.

Re-implements IsaacGymEnvs ``tasks/allegro_kuka/generate_cuboids.py`` and
``AllegroKukaBase._box_asset_files_and_scales`` / ``_main_object_assets_and_scales``
at revision aeed298 (BSD-3-Clause, NVIDIA):

* base size ``objectBaseSize`` = 0.05 m, density 400 kg/m^3 (URDF template);
* sets: default cube; small cuboids (scales {50,66,75,90,100,110,125,150,175,200,
  250,300} %, volume ratio 1-2.5); big cuboids ({100..350} %, 2.5-15, thin plates
  removed); sticks ({50,75,100,200..600} %, 2.5-6, elongated only). All (x, y, z)
  orderings are kept, as in the original;
* file names ``{idx:03d}_cube_{x}_{y}_{z}.urdf`` with ``idx`` restarting per set
  (the default cube and the first small cuboid share a name and collapse to one),
  sorted by name, shuffled with ``np.random.default_rng(42)``, env ``i`` gets
  object ``i % len``.

Throw (``env/throw.yaml``) uses the default cube + small cuboids only.

Each variant is one free body with a box *mesh* collision geom (mjlab variants
vary mesh geoms only) and six thin visual-only colored face meshes so the
orientation is visible; inertia is the analytic solid-box inertia.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import mujoco
import numpy as np
from mjlab.entity.variants import VariantEntityCfg

BASE_SIZE = 0.05
DENSITY = 400.0

SETS = {
    "cube": ([100], 1.0, 1.0, ()),
    "small": ([100, 50, 66, 75, 90, 110, 125, 150, 175, 200, 250, 300], 1.0, 2.5, ()),
    "big": ([100, 125, 150, 200, 250, 300, 350], 2.5, 15.0, ("thin",)),
    "sticks": ([100, 50, 75, 200, 300, 400, 500, 600], 2.5, 6.0, ("thin", "non_elongated")),
}
ALL_SETS = ("cube", "small", "big", "sticks")
THROW_SETS = ("cube", "small")


def _filtered(s: list[int], filters: Iterable[str]) -> bool:
    s = sorted(s)
    for f in filters:
        if f == "thin" and s[0] * 3 <= s[1]:
            return True
        if f == "non_elongated" and (s[2] <= s[0] * 3 or s[2] <= s[1] * 3):
            return True
    return False


def dexpbt_object_list(
    sets: Iterable[str] = ALL_SETS, seed: int = 42
) -> list[tuple[str, tuple[int, int, int]]]:
    """(file-name stem, (x%, y%, z%)) in the original per-env assignment order."""
    files: dict[str, tuple[int, int, int]] = {}
    for name in ("cube", "small", "big", "sticks"):  # original generation order
        if name not in sets:
            continue
        scales, vmin, vmax, filters = SETS[name]
        idx = 0
        for x in scales:
            for y in scales:
                for z in scales:
                    v = x * y * z / 1e6
                    if v > vmax or v < vmin or _filtered([x, y, z], filters):
                        continue
                    files[f"{idx:03d}_cube_{x}_{y}_{z}"] = (x, y, z)
                    idx += 1
    ordered = [(k, files[k]) for k in sorted(files)]
    rng = np.random.default_rng(seed)
    perm = list(ordered)
    rng.shuffle(perm)
    return perm


# Visual: one solid box (same mesh size as the collision box) plus small axis
# marks on the +x / +y / +z face centres (red / green / blue) to show orientation.
_BODY_RGBA = (0.82, 0.80, 0.74, 1.0)
_FACE_MARKS = (
    ((1, 0, 0), (0.85, 0.15, 0.15, 1)),
    ((0, 1, 0), (0.15, 0.65, 0.2, 1)),
    ((0, 0, 1), (0.15, 0.35, 0.9, 1)),
)


def _box_verts(center: np.ndarray, half: np.ndarray) -> list[float]:
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    return (corners * half + center).reshape(-1).tolist()


@dataclass(frozen=True)
class ObjectPhysics:
    friction: tuple[float, float, float] = (1.0, 0.002, 0.002)
    condim: int = 6
    priority: int = 1
    density: float = DENSITY


def make_cuboid_spec(
    size: tuple[float, float, float], phys: ObjectPhysics = ObjectPhysics()
) -> mujoco.MjSpec:
    size_a = np.asarray(size, float)
    half = size_a / 2.0
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="object")
    body.add_freejoint(name="object_joint")
    mass = phys.density * float(np.prod(size_a))
    ixx = mass / 12.0 * (size_a[1] ** 2 + size_a[2] ** 2)
    iyy = mass / 12.0 * (size_a[0] ** 2 + size_a[2] ** 2)
    izz = mass / 12.0 * (size_a[0] ** 2 + size_a[1] ** 2)
    body.mass = mass
    body.inertia = [ixx, iyy, izz]
    body.ipos = [0, 0, 0]
    body.explicitinertial = True
    m = spec.add_mesh(name="box")
    m.uservert = _box_verts(np.zeros(3), half)
    g = body.add_geom(name="object_collision", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="box")
    g.friction = list(phys.friction)
    g.condim = phys.condim
    g.priority = phys.priority
    g.rgba = [0.85, 0.85, 0.8, 1]
    g.group = 3
    g.mass = 0.0
    g.density = 0.0
    vm = spec.add_mesh(name="box_visual")
    vm.uservert = _box_verts(np.zeros(3), half)
    vg = body.add_geom(name="object_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="box_visual")
    vg.contype = 0
    vg.conaffinity = 0
    vg.group = 2
    vg.rgba = list(_BODY_RGBA)
    vg.mass = 0.0
    vg.density = 0.0
    for k, (d, rgba) in enumerate(_FACE_MARKS):
        d = np.asarray(d, float)
        side = 0.45 * float(half[d == 0].min())  # square mark, 45% of the smaller half side
        fh = np.where(d != 0, 0.0008, side)
        center = d * (half + 0.0006)
        fm = spec.add_mesh(name=f"mark{k}")
        fm.uservert = _box_verts(center, fh)
        fg = body.add_geom(
            name=f"object_mark{k}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"mark{k}"
        )
        fg.contype = 0
        fg.conaffinity = 0
        fg.group = 2
        fg.rgba = list(rgba)
        fg.mass = 0.0
        fg.density = 0.0
    return spec


def make_dexpbt_object_cfg(
    sets: Iterable[str] = ALL_SETS, phys: ObjectPhysics = ObjectPhysics(), seed: int = 42
) -> tuple[VariantEntityCfg, list[tuple[str, tuple[float, float, float]]]]:
    """Variant entity + the ordered (name, size_m) table matching variant indices."""
    objs = dexpbt_object_list(tuple(sets), seed)
    table = [(f"o{stem}", tuple(s / 100.0 * BASE_SIZE for s in sc)) for stem, sc in objs]
    variants = {name: (lambda sz=sz: make_cuboid_spec(sz, phys)) for name, sz in table}
    n_var = len(table)
    cfg = VariantEntityCfg(variants=variants, assignment=lambda n: [i % n_var for i in range(n)])
    return cfg, table
