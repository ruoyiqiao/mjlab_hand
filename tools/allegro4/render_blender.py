"""Render a record_scene.py .npz with Blender Cycles (white studio look, soft floor shadow, translucent goal ghost).

Usage: blender -b -P render_blender.py -- <scene.npz> <out_dir> [--res 960x624] [--samples 32] [--az 135] [--el 22]
           [--fov 34] [--margin 1.15] [--frames 0:200] [--exclude table] [--look-dz 0.0] [--floor-z auto]

Writes <out_dir>/frame_0000.png ... (RGBA: transparent background, floor shadow as alpha; make_webp.py composites them
on white). The camera keeps a fixed azimuth / elevation and follows the smoothed centroid of everything that moves (all
geoms except --exclude entities), at the distance that keeps that subject in view, so every clip is framed the same
way. Hands with saturated model colours get a studio two-tone palette (--palette model keeps them).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector


def args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("out")
    ap.add_argument("--res", default="960x624")
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument(
        "--az", type=float, default=245.0, help="camera azimuth (deg, around world z; 0 = from +x)"
    )
    ap.add_argument("--el", type=float, default=32.0, help="camera elevation (deg)")
    ap.add_argument("--fov", type=float, default=34.0, help="horizontal field of view (deg)")
    ap.add_argument("--margin", type=float, default=0.95)
    ap.add_argument("--frames", default="", help="start:stop[:step] of recorded frames")
    ap.add_argument("--exclude", default="table", help="comma list: entities ignored for framing")
    ap.add_argument("--look-dz", type=float, default=0.0)
    ap.add_argument(
        "--fit-pct",
        type=float,
        default=0.0,
        help="framing percentile trimmed at each end (fixed camera)",
    )
    ap.add_argument(
        "--ghost-side",
        type=float,
        default=0.0,
        help="draw the goal ghost this far (m) to the camera-right of the object instead of at the goal "
        "position (orientation-only goals, e.g. in-hand rotation); 0 = at the goal pose",
    )
    ap.add_argument(
        "--follow",
        type=float,
        default=1.0,
        help="camera follows the subject, smoothing window (s); 0 = fixed camera framing the whole clip",
    )
    ap.add_argument(
        "--floor-z",
        default="auto",
        help="'auto' = table top if there is a table, else the lowest point",
    )
    ap.add_argument("--palette", default="studio", choices=("studio", "model"), help="hand colours")
    ap.add_argument(
        "--trail",
        type=float,
        default=0.0,
        help="seconds of fading path drawn behind every ball (shows a juggling pattern's shape); 0 = none",
    )
    ap.add_argument(
        "--hide-bodies",
        default=r"(lh|rh)_forearm$",
        help="regex on body names not drawn (default: Shadow's forearm housing)",
    )
    ap.add_argument(
        "--hide",
        default="table",
        help="comma list: entities not drawn (the table becomes the floor)",
    )
    return ap.parse_args(argv)


TABLE_RGBA = (0.93, 0.93, 0.92, 1.0)
OBJECT_RGBA = (0.22, 0.45, 0.85, 1.0)  # one blue for every manipulated object / ball


SHELL, DARK, METAL = (0.86, 0.85, 0.83, 1.0), (0.13, 0.13, 0.14, 1.0), (0.58, 0.59, 0.61, 1.0)
ROUGH = {SHELL: 0.78, DARK: 0.62, METAL: 0.55}


def is_object(entity: str) -> bool:
    return entity in ("object", "plate") or entity.startswith("ball")


def quarter_marker(ob):
    """Plate: one quarter of the disc in the lighter tone, so its spin is visible. The quarter is taken
    in the two in-plane axes of the mesh (MuJoCo re-frames a mesh to its principal axes, so the plate
    normal is whichever local axis the mesh is thinnest along)."""
    me = ob.data
    v = np.array([p.co[:] for p in me.vertices])
    a, b = sorted(np.argsort(np.ptp(v, axis=0))[1:])
    me.materials.append(material(f"{ob.name}_quarter", OBJECT_RGBA2, rough=0.6))
    k = len(me.materials) - 1
    for p_ in me.polygons:
        c = p_.center
        if c[a] > 0.0 and c[b] > 0.0:
            p_.material_index = k


def studio_palette(geoms):
    """Hand colours -> one studio look for every hand: warm off-white shells, graphite joints / links, satin grey parts.
    Hands with saturated model colours (Wuji's blue / beige) map by how many geoms use a colour (most -> shell,
    next -> graphite); neutral hands map by brightness (light -> shell, dark -> graphite, mid -> grey)."""
    import collections

    cnt = collections.Counter(
        tuple(round(c, 3) for c in g["rgba"]) for g in geoms if g["entity"].startswith("hand")
    )
    if any(max(c[:3]) - min(c[:3]) > 0.15 for c in cnt):
        order = (SHELL, DARK, METAL)
        return {c: order[min(k, 2)] for k, (c, _) in enumerate(cnt.most_common())}
    lum = {c: 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2] for c in cnt}
    return {c: SHELL if lum[c] > 0.55 else DARK if lum[c] < 0.3 else METAL for c in cnt}


BALL_RGBA = (
    (0.22, 0.45, 0.85, 1.0),
    (0.93, 0.45, 0.38, 1.0),
    (0.95, 0.70, 0.25, 1.0),
)  # blue, coral, amber
EDGE_RGBA = (0.10, 0.12, 0.16, 1.0)


def is_marker(v):
    """Face marker decals of the task objects: a flat patch (< 3 mm thick)."""
    ext = v.max(0) - v.min(0)
    return bool(ext.min() < 0.003 and ext.max() < 0.03)


def object_rgba(entity: str):
    if entity.startswith("ball"):
        k = {"ball_a": 0, "ball_b": 1, "ball_0": 0, "ball_1": 1, "ball_2": 2}.get(entity, 0)
        return BALL_RGBA[k]
    return OBJECT_RGBA


def planar_faces(ob):
    """Merge coplanar triangles (so the outline shows the object's edges, not its triangulation)."""
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(1.0), verts=bm.verts, edges=bm.edges)
    bm.to_mesh(ob.data)
    bm.free()


OBJECT_RGBA2 = (
    0.66,
    0.78,
    0.96,
    1.0,
)  # lighter half of curved objects, so their rotation is visible


def n_faces_after_dissolve(ob):
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(1.0), verts=bm.verts, edges=bm.edges)
    n = len(bm.faces)
    bm.free()
    return n


def two_tone(ob, alpha=None):
    """Curved object (cylinder, sphere-like): faces whose local normal points to +x get a lighter blue."""
    me = ob.data
    me.materials.append(material(f"{ob.name}_half", OBJECT_RGBA2, rough=0.6, alpha=alpha))
    k = len(me.materials) - 1
    for p_ in me.polygons:
        if p_.normal[0] > 0.0:
            p_.material_index = k


def outline(ob, thickness=0.0016):
    """Crisp dark edges for the translucent goal (a wireframe copy that follows the goal), readable over hand and object."""
    me = ob.data.copy()
    me.materials.clear()
    me.materials.append(material(f"{ob.name}_edge", EDGE_RGBA, rough=0.8))
    edges = bpy.data.objects.new(f"{ob.name}_edges", me)
    bpy.context.scene.collection.objects.link(edges)
    mod = edges.modifiers.new("wire", "WIREFRAME")
    mod.thickness = thickness
    mod.use_replace = True
    mod.use_even_offset = True
    edges.parent = ob
    return edges


def srgb_to_linear(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def material(name, rgba, rough=0.5, alpha=None):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    lin = srgb_to_linear(rgba[:3])
    bsdf.inputs["Base Color"].default_value = (*lin, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    for name, val in (
        ("Specular IOR Level", 0.25),
        ("Sheen Weight", 0.15),
        ("Sheen Roughness", 0.6),
    ):
        if name in bsdf.inputs:
            bsdf.inputs[name].default_value = val
    a = rgba[3] if alpha is None else alpha
    if a < 1.0:
        bsdf.inputs["Alpha"].default_value = a
    return mat


def mesh_object(name, verts, faces, mat):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts.tolist(), [], faces.tolist())
    me.update()
    for p in me.polygons:
        p.use_smooth = False
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def pose(p, m9):
    R = np.asarray(m9, float).reshape(3, 3)
    M = Matrix(
        (
            (R[0, 0], R[0, 1], R[0, 2], p[0]),
            (R[1, 0], R[1, 1], R[1, 2], p[1]),
            (R[2, 0], R[2, 1], R[2, 2], p[2]),
            (0, 0, 0, 1),
        )
    )
    return M


def setup_scene(a, W, H):
    """Cycles on the GPU, transparent film (white is composited later), AgX, soft white world light."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.render.resolution_x, sc.render.resolution_y = W, H
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.view_settings.view_transform = "AgX"
    try:
        sc.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass
    sc.cycles.samples = a.samples
    sc.cycles.use_denoising = True
    sc.cycles.film_transparent_glass = False
    sc.cycles.filter_width = 1.2
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for kind in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = kind
                prefs.get_devices()
                if any(dv.type == kind for dv in prefs.devices):
                    for dv in prefs.devices:
                        dv.use = dv.type == kind
                    sc.cycles.device = "GPU"
                    break
            except TypeError:
                continue
    except KeyError:
        pass
    world = bpy.data.worlds.new("white")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (1, 1, 1, 1)
    bg.inputs["Strength"].default_value = 0.55
    sc.world = world
    return sc


def build_geoms(a, d, geoms, pos, f0, f1):
    """One Blender mesh per recorded geom, with the studio palette; hidden: --hide entities, --hide-bodies, object
    marker decals and parked (unused) balls."""
    hidden = set(a.hide.split(",")) if a.hide else set()
    hide_re = re.compile(a.hide_bodies) if a.hide_bodies else None
    hand_map = studio_palette(geoms) if a.palette == "studio" else {}
    mats, objs = {}, []
    for i, g in enumerate(geoms):
        rgba = list(object_rgba(g["entity"])) if is_object(g["entity"]) else list(g["rgba"])
        if g["entity"].startswith("hand"):
            rgba = list(hand_map.get(tuple(round(c, 3) for c in rgba), rgba))
        key = tuple(round(c, 3) for c in rgba)
        if key not in mats:
            mats[key] = material(
                f"m{len(mats)}", rgba, rough=0.6 if is_object(g["entity"]) else ROUGH.get(key, 0.7)
            )
        v = d[f"v_{g['mesh']}"]
        ob = mesh_object(f"g{i}_{g['name'] or g['body']}", v, d[f"f_{g['mesh']}"], mats[key])
        parked = (
            g["entity"].startswith("ball") and float(np.ptp(pos[f0:f1, i], axis=0).max()) < 0.02
        )
        ob.hide_render = (
            g["entity"] in hidden
            or (is_object(g["entity"]) and is_marker(v))
            or bool(hide_re and hide_re.search(g["body"]))
            or parked
        )
        if (
            g["entity"] == "object" and not ob.hide_render and n_faces_after_dissolve(ob) > 8
        ):  # curved, not a box
            two_tone(ob)
        if g["entity"] == "plate":
            quarter_marker(ob)
        objs.append(ob)
    return objs, hidden


def build_ghost(sc, d, meta, geoms, pos, mat9, f0):
    """Translucent goal: the object's own meshes at the goal pose (orientation goals), else a 2.5 cm sphere."""
    gk = meta.get("goal_kind", "box" if meta.get("has_goal") else "")
    if not (gk and len(d["goal_pos"])):
        return None
    body = [
        i
        for i, g in enumerate(geoms)
        if is_object(g["entity"]) and not is_marker(d[f"v_{g['mesh']}"])
    ]
    if not (gk == "box" and body):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.025)
        ghost = bpy.context.active_object
        ghost.data.materials.append(material("ghost", OBJECT_RGBA, rough=0.3, alpha=0.30))
        for p_ in ghost.data.polygons:
            p_.use_smooth = True
        return ghost
    gmat_ = material("ghost", OBJECT_RGBA, rough=0.3, alpha=0.28)
    ghost = bpy.data.objects.new("goal", None)
    sc.collection.objects.link(ghost)
    # The goal is a pose of the object's root body (what the task checks). Place each object geom relative to that
    # body frame; a mesh geom's own frame can be rotated against it (120 deg for the DexPBT cuboids).
    assert len(d["obj_pos"]), "recording has no object root pose: re-record with record_scene.py"
    B = pose(d["obj_pos"][f0], d["obj_mat"][f0]).inverted()
    for i in body:
        part = mesh_object(
            f"goal_{i}", d[f"v_{geoms[i]['mesh']}"], d[f"f_{geoms[i]['mesh']}"], gmat_
        )
        part.parent = ghost
        part.matrix_parent_inverse = Matrix.Identity(4)
        part.matrix_basis = B @ pose(pos[f0, i], mat9[f0, i])
        if (
            n_faces_after_dissolve(part) > 8
        ):  # curved: same two tones, no wireframe (it would show facets)
            two_tone(part, alpha=0.28)
        else:
            planar_faces(part)
            outline(part)
    return ghost


TRAIL_CHUNKS = 8


def build_trails(sc, geoms, objs, n_frames):
    """Per visible ball: TRAIL_CHUNKS poly-curves in the ball's colour, older chunks thinner and more transparent.
    Returns [(geom index, [(curve object, first frame offset)...])]; update_trails moves the points every frame."""
    trails = []
    per = max(2, int(math.ceil(n_frames / TRAIL_CHUNKS)))
    for i, g in enumerate(geoms):
        if not g["entity"].startswith("ball") or objs[i].hide_render:
            continue
        chunks = []
        for j in range(TRAIL_CHUNKS):  # j = 0 newest
            age = j / TRAIL_CHUNKS
            cu = bpy.data.curves.new(f"trail_{i}_{j}", "CURVE")
            cu.dimensions, cu.bevel_depth, cu.bevel_resolution = "3D", 0.006 * (1 - 0.6 * age), 3
            sp = cu.splines.new("POLY")
            sp.points.add(per)  # per + 1 points: chunks share their end points
            ob = bpy.data.objects.new(f"trail_{i}_{j}", cu)
            ob.data.materials.append(
                material(
                    f"trail_{i}_{j}", object_rgba(g["entity"]), rough=0.5, alpha=0.55 * (1 - age)
                )
            )
            sc.collection.objects.link(ob)
            chunks.append((ob, j * per))
        trails.append((i, chunks, per))
    return trails


def update_trails(trails, pos, t, f0):
    for i, chunks, per in trails:
        for ob, off in chunks:
            sp = ob.data.splines[0]
            for k in range(per + 1):
                f = max(f0, t - off - k)
                sp.points[k].co = (*pos[f, i].tolist(), 1.0)
            ob.hide_render = t - off <= f0  # nothing recorded that far back yet


def frame_camera(a, meta, pos, sel, gpos_draw, has_ghost, f0, f1, W, H):
    """Camera target per frame (``track``, None for a fixed camera), mean target, distance and view direction.

    Follow camera: the look-at follows the smoothed centroid of the moving geoms (+ goal), at the distance that keeps
    them in view. Fixed camera: the projected outline of every recorded point (all balls, apexes included) fits the
    frame, so a tall juggling pattern fills a wide frame."""
    pts = pos[f0:f1][:, sel].reshape(-1, 3)
    if has_ghost:
        pts = np.concatenate([pts, gpos_draw[f0:f1]])
    lo = (
        np.percentile(pts, a.fit_pct, axis=0) - 0.04
    )  # --fit-pct trims rare extremes; 0 = every point
    hi = np.percentile(pts, 100 - a.fit_pct, axis=0) + 0.04
    centre = (lo + hi) / 2 + np.array([0, 0, a.look_dz])
    radius = 0.5 * np.linalg.norm(hi - lo)
    track = None
    if a.follow:
        track = pos[:, sel].mean(1)
        if has_ghost:
            track = (track * len(sel) + gpos_draw) / (len(sel) + 1)
        k = max(1, int(round(a.follow * meta["fps"])))
        padded = np.concatenate([np.repeat(track[:1], k, 0), track, np.repeat(track[-1:], k, 0)])
        kern = np.ones(2 * k + 1) / (2 * k + 1)
        track = np.stack([np.convolve(padded[:, j], kern, mode="valid") for j in range(3)], 1)
        rel = pos[f0:f1][:, sel] - track[f0:f1, None, :]
        radius = float(np.percentile(np.linalg.norm(rel, axis=-1), 99)) + 0.02
        if has_ghost:  # the goal ghost (object size) must stay in view too
            radius = max(
                radius, float(np.linalg.norm(gpos_draw[f0:f1] - track[f0:f1], axis=-1).max() + 0.06)
            )
        centre = track[f0:f1].mean(0) + np.array([0, 0, a.look_dz])
    az, el = math.radians(a.az), math.radians(a.el)
    fov_h = math.radians(a.fov)
    fov_v = 2 * math.atan(math.tan(fov_h / 2) * H / W)
    dist = a.margin * radius / math.sin(min(fov_h, fov_v) / 2)
    direction = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    if not a.follow:
        fwd = -direction
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        part = (
            0.06  # part size around each recorded geom origin (fingers reach past the link origins)
        )
        rel = pts - centre
        x, y, z = rel @ right, rel @ up, rel @ fwd
        cx, cy = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2
        centre = centre + cx * right + cy * up
        x, y = x - cx, y - cy
        th, tv = math.tan(fov_h / 2), math.tan(fov_v / 2)
        dist = (
            float(np.max(np.maximum((np.abs(x) + part) / th, (np.abs(y) + part) / tv) + z))
            * a.margin
        )
    return track, centre, dist, direction, fov_h


def look_at(ob, target):
    ob.rotation_euler = (Vector(target.tolist()) - ob.location).to_track_quat("-Z", "Y").to_euler()


def add_floor(a, d, geoms, pos, mat9, sel, hidden, centre, f0, f1):
    """Shadow catcher (transparent in the render, shadow kept as alpha): a visible table's top, else just under the
    lowest point the subject reaches."""
    tab = [i for i, g in enumerate(geoms) if g["entity"] == "table"]
    if a.floor_z != "auto":
        fz = float(a.floor_z)
    elif tab and "table" not in hidden:
        fz = max(
            float(
                (d[f"v_{geoms[i]['mesh']}"] @ mat9[f0, i].reshape(3, 3).T + pos[f0, i])[:, 2].max()
            )
            for i in tab
        )
    else:
        fz = float(np.percentile(pos[f0:f1][:, sel].reshape(-1, 3)[:, 2], 0.5)) - 0.04
    bpy.ops.mesh.primitive_plane_add(size=40, location=(centre[0], centre[1], fz))
    bpy.context.active_object.is_shadow_catcher = True


def add_lights(sc, centre, az):
    """Soft key light above, weak fill opposite, rim light from behind (azimuths relative to the camera)."""
    lights = []
    for name, energy, size, az_off, el_deg in (
        ("key", 150.0, 1.8, 0.4, 72),
        ("fill", 40.0, 3.0, -1.3, 22),
        ("rim", 90.0, 1.2, math.pi, 35),
    ):
        li = bpy.data.lights.new(name, "AREA")
        li.energy, li.size = energy, size
        ob = bpy.data.objects.new(name, li)
        sc.collection.objects.link(ob)
        e = math.radians(el_deg)
        dv = np.array(
            [math.cos(e) * math.cos(az + az_off), math.cos(e) * math.sin(az + az_off), math.sin(e)]
        )
        ob.location = Vector((centre + 2.2 * dv).tolist())
        look_at(ob, centre)
        lights.append(ob)
    return lights


def main():
    a = args()
    d = np.load(a.npz)
    meta = json.loads(str(d["meta"]))
    pos, mat9 = d["pos"], d["mat"]
    T = pos.shape[0]
    fs = [int(x) for x in a.frames.split(":")] if a.frames else [0, T]
    f0, f1, step = fs[0], min(fs[1], T), fs[2] if len(fs) > 2 else 1
    W, H = (int(x) for x in a.res.split("x"))
    geoms = meta["geoms"]

    sc = setup_scene(a, W, H)
    objs, hidden = build_geoms(a, d, geoms, pos, f0, f1)
    ghost = build_ghost(sc, d, meta, geoms, pos, mat9, f0)
    az = math.radians(a.az)
    gpos_draw = d["goal_pos"]
    if (
        ghost is not None and a.ghost_side and len(d["obj_pos"])
    ):  # goal ghost beside the object, camera-right
        gpos_draw = d["obj_pos"] + a.ghost_side * np.array([-math.sin(az), math.cos(az), 0.0])
    excl = set(a.exclude.split(",")) if a.exclude else set()
    sel = [i for i, g in enumerate(geoms) if g["entity"] not in excl and not objs[i].hide_render]
    track, centre, dist, direction, fov_h = frame_camera(
        a, meta, pos, sel, gpos_draw, ghost is not None, f0, f1, W, H
    )

    cam_data = bpy.data.cameras.new("cam")
    cam_data.angle = fov_h
    cam_data.sensor_fit = "HORIZONTAL"
    cam = bpy.data.objects.new("cam", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam
    add_floor(a, d, geoms, pos, mat9, sel, hidden, centre, f0, f1)
    lights = add_lights(sc, centre, az)
    trails = build_trails(sc, geoms, objs, int(round(a.trail * meta["fps"]))) if a.trail > 0 else []

    target = centre
    for k, t in enumerate(range(f0, f1, step)):
        for i, ob in enumerate(objs):
            ob.matrix_world = pose(pos[t, i], mat9[t, i])
        update_trails(trails, pos, t, f0)
        if ghost is not None:
            ghost.matrix_world = pose(
                gpos_draw[t], d["goal_mat"][t] if len(d["goal_mat"]) else np.eye(3).ravel()
            )
            bpy.context.view_layer.update()
        if track is not None:  # camera and lights move with the subject
            new = track[t] + np.array([0, 0, a.look_dz])
            for li in lights:
                li.location = li.location + Vector((new - target).tolist())
                look_at(li, new)
            target = new
        cam.location = Vector((target + dist * direction).tolist())
        look_at(cam, target)
        sc.render.filepath = f"{a.out}/frame_{k:04d}.png"
        bpy.ops.render.render(write_still=True)
    print(
        json.dumps(
            {
                "rendered": len(range(f0, f1, step)),
                "fps": meta["fps"],
                "camera_dist": float(dist),
                "centre": [float(c) for c in centre],
            }
        )
    )


if __name__ == "__main__":
    main()
