"""Primitive scene objects for the Allegro DexPBT-style tasks.

* Cube: a single box collision geom (exact size and analytic inertia), DexPBT
  default ``objectBaseSize=0.05`` m. Visual-only colored face markers make the
  orientation visible in videos; they never collide.
* Bucket: an open-top container made of box geoms (floor + 4 walls); a real
  collision asset, fixed-base (mjlab wraps it in a mocap body so it can be moved
  per env at goal resets).
* Table: a box named ``table_surface`` so ``grasp/mdp/table_utils`` helpers apply.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
from mjlab.entity import EntityCfg


@dataclass(frozen=True)
class CubeSpec:
    edge: float = 0.05
    mass: float = 0.1
    friction: tuple[float, float, float] = (1.0, 0.002, 0.002)
    condim: int = 6
    priority: int = 1

    @property
    def size(self) -> tuple[float, float, float]:
        return (self.edge, self.edge, self.edge)

    @property
    def inertia(self) -> tuple[float, float, float]:
        i = self.mass * (2 * self.edge**2) / 12.0
        return (i, i, i)

    def to_dict(self) -> dict:
        return {
            "edge_m": self.edge,
            "mass_kg": self.mass,
            "diag_inertia_kgm2": self.inertia,
            "friction": self.friction,
            "condim": self.condim,
            "priority": self.priority,
        }


_FACE_MARKERS = (
    # (pos direction, rgba) visual-only markers to show orientation.
    ((1, 0, 0), "0.9 0.1 0.1 1"),
    ((-1, 0, 0), "0.5 0.0 0.0 1"),
    ((0, 1, 0), "0.1 0.8 0.1 1"),
    ((0, -1, 0), "0.0 0.4 0.0 1"),
    ((0, 0, 1), "0.1 0.2 0.9 1"),
    ((0, 0, -1), "0.0 0.0 0.45 1"),
)


def make_cube_spec(cube: CubeSpec) -> mujoco.MjSpec:
    h = cube.edge / 2.0
    markers = []
    for (dx, dy, dz), rgba in _FACE_MARKERS:
        sx = 0.0005 if dx else h * 0.6
        sy = 0.0005 if dy else h * 0.6
        sz = 0.0005 if dz else h * 0.6
        markers.append(
            f'<geom type="box" size="{sx} {sy} {sz}" pos="{dx * h} {dy * h} {dz * h}" '
            f'rgba="{rgba}" contype="0" conaffinity="0" group="2" mass="0"/>'
        )
    ixx = cube.inertia[0]
    xml = f"""
<mujoco model="cube">
  <worldbody>
    <body name="object">
      <freejoint name="object_joint"/>
      <inertial pos="0 0 0" mass="{cube.mass}" diaginertia="{ixx} {ixx} {ixx}"/>
      <geom name="object_collision" type="box" size="{h} {h} {h}" rgba="0.85 0.85 0.8 1"
            friction="{cube.friction[0]} {cube.friction[1]} {cube.friction[2]}"
            condim="{cube.condim}" priority="{cube.priority}" mass="0"/>
      {"".join(markers)}
    </body>
  </worldbody>
</mujoco>
"""
    return mujoco.MjSpec.from_string(xml)


def make_cube_entity_cfg(cube: CubeSpec) -> EntityCfg:
    return EntityCfg(spec_fn=lambda: make_cube_spec(cube))


@dataclass(frozen=True)
class BucketSpec:
    """Round flared bucket with the original DexPBT bucket.obj dimensions.

    IsaacGymEnvs ``assets/urdf/objects/meshes/bucket.obj`` (12-sided, measured):
    height 0.198 m, floor top at 0.010 m with inner radius ~0.070 m, rim inner
    radius ~0.103 m / outer ~0.120 m. Body origin = bucket bottom center, as in
    the original (goal = bucket position + 0.05 m z).

    MuJoCo collides a mesh as its convex hull (a solid bucket), so the hollow
    shape is built from ``segments`` thin tilted box walls plus a cylinder floor.
    """

    floor_inner_radius: float = 0.070
    rim_inner_radius: float = 0.103
    height: float = 0.198
    floor_thickness: float = 0.010
    wall_thickness: float = 0.014
    segments: int = 24
    friction: tuple[float, float, float] = (0.8, 0.005, 0.0001)
    rgba: tuple[float, float, float, float] = (0.80, 0.82, 0.86, 1.0)

    def inner_radius_at(self, z: float) -> float:
        t = (z - self.floor_thickness) / (self.height - self.floor_thickness)
        t = min(max(t, 0.0), 1.0)
        return self.floor_inner_radius + t * (self.rim_inner_radius - self.floor_inner_radius)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def make_bucket_spec(b: BucketSpec) -> mujoco.MjSpec:
    n = b.segments
    t = b.wall_thickness
    z0, z1 = b.floor_thickness, b.height
    r0 = b.floor_inner_radius + t / 2
    r1 = b.rim_inner_radius + t / 2
    dz, dr = z1 - z0, r1 - r0
    length = math.hypot(dz, dr)
    tilt = math.atan2(dr, dz)  # outward lean of the wall
    rm = (r0 + r1) / 2
    half_w = (rm + t) * math.tan(math.pi / n) * 1.02
    f = f'friction="{b.friction[0]} {b.friction[1]} {b.friction[2]}"'
    rgba = 'rgba="{} {} {} {}"'.format(*b.rgba)
    geoms = [
        f'<geom name="bucket_floor" type="cylinder" size="{b.floor_inner_radius + t} {z0 / 2}" '
        f'pos="0 0 {z0 / 2}" {f} {rgba}/>'
    ]
    for i in range(n):
        th = 2 * math.pi * i / n
        # q = Rz(th) * Ry(tilt): local x radial, local z along the wall slant.
        cz, sz = math.cos(th / 2), math.sin(th / 2)
        cy, sy = math.cos(tilt / 2), math.sin(tilt / 2)
        q = (cz * cy, -sz * sy, cz * sy, sz * cy)
        px, py = rm * math.cos(th), rm * math.sin(th)
        geoms.append(
            f'<geom name="bucket_wall_{i:02d}" type="box" size="{t / 2} {half_w} {length / 2}" '
            f'pos="{px:.6f} {py:.6f} {(z0 + z1) / 2:.6f}" '
            f'quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}" {f} {rgba}/>'
        )
    xml = f"""
<mujoco model="bucket">
  <worldbody>
    <body name="bucket">
      {"".join(geoms)}
    </body>
  </worldbody>
</mujoco>
"""
    return mujoco.MjSpec.from_string(xml)


def make_bucket_entity_cfg(b: BucketSpec, pos: tuple[float, float, float]) -> EntityCfg:
    return EntityCfg(
        spec_fn=lambda: make_bucket_spec(b),
        init_state=EntityCfg.InitialStateCfg(pos=pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )


@dataclass(frozen=True)
class TableSpec:
    center: tuple[float, float, float] = (0.0, 0.0, 0.4)
    half_size: tuple[float, float, float] = (0.3, 0.3, 0.02)
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)

    @property
    def top_z(self) -> float:
        return self.center[2] + self.half_size[2]

    def to_dict(self) -> dict:
        return {
            "center": self.center,
            "half_size": self.half_size,
            "top_z": self.top_z,
            "friction": self.friction,
        }


def make_table_spec(t: TableSpec) -> mujoco.MjSpec:
    cx, cy, cz = t.center
    sx, sy, sz = t.half_size
    xml = f"""
<mujoco model="table">
  <worldbody>
    <body name="table" pos="{cx} {cy} {cz}">
      <geom name="table_surface" type="box" size="{sx} {sy} {sz}"
            rgba="0.5 0.35 0.2 1" friction="{t.friction[0]} {t.friction[1]} {t.friction[2]}"
            solref="0.005 1" solimp="0.99 0.99 0.001"/>
    </body>
  </worldbody>
</mujoco>
"""
    return mujoco.MjSpec.from_string(xml)


def make_table_entity_cfg(t: TableSpec) -> EntityCfg:
    return EntityCfg(
        spec_fn=lambda: make_table_spec(t),
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )
