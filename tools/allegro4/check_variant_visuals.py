"""Check per-world object variants: physics model vs renderer vs standalone compile.

For several envs: compile that env's cuboid alone, then compare
(a) geom size / pos / dataid in the simulation's per-world model (sim.model),
(b) the host MjModel after the renderer's sync,
(c) the standalone compile,
for the collision box and the six face plates. Also reports the vertical extent of
the collision geom actually resting on the table vs the expected half size.

Usage: python check_variant_visuals.py <task_id> <out.json> [--device cuda:0]
"""

from __future__ import annotations

import argparse
import json

import mujoco
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.viewer.model_sync import VIEWER_MODEL_FIELDS, sync_model_fields

    import mjlab_hand  # noqa: F401
    from mjlab_hand.tasks_common.dexpbt_objects import make_cuboid_spec

    cfg = load_env_cfg(a.task, play=True)
    cfg.scene.num_envs = 8
    env = ManagerBasedRlEnv(cfg, device=a.device)
    env.reset()
    t = env.command_manager.get_term(next(iter(env.command_manager.active_terms)))
    host = env.sim.mj_model
    sim_model = env.sim.model
    obj = env.scene["object"]
    gids = [g.id for g in obj.data.indexing.geoms]
    gnames = [mujoco.mj_id2name(host, mujoco.mjtObj.mjOBJ_GEOM, g) for g in gids]
    exp = sorted(env.sim.expanded_fields)
    rep = {"task": a.task, "object_geoms": gnames, "expanded_fields": exp, "envs": []}
    for i in range(8):
        size = t.obj_size[i].tolist()
        ref = make_cuboid_spec(tuple(size)).compile()
        ref_geoms = [ref.geom(k) for k in range(ref.ngeom)]
        sync_model_fields(host, sim_model, set(exp) & VIEWER_MODEL_FIELDS, i)
        row = {"env": i, "size": [round(x, 4) for x in size], "geoms": []}
        for k, g in enumerate(gids):
            sim_pos = (
                np.asarray(sim_model.geom_pos[i, g].cpu())
                if sim_model.geom_pos.dim() == 3
                else np.asarray(sim_model.geom_pos[0, g].cpu())
            )
            host_pos = host.geom_pos[g]
            r = ref_geoms[k] if k < len(ref_geoms) else None
            did = (
                int(sim_model.geom_dataid[i, g])
                if sim_model.geom_dataid.dim() == 2
                else int(sim_model.geom_dataid[g])
            )
            # mesh extent of the mesh actually assigned in this world
            if did >= 0:
                v0, nv = host.mesh_vertadr[did], host.mesh_vertnum[did]
                ext = (
                    (host.mesh_vert[v0 : v0 + nv].max(0) - host.mesh_vert[v0 : v0 + nv].min(0))
                    .round(4)
                    .tolist()
                )
            else:
                ext = None
            row["geoms"].append(
                {
                    "name": gnames[k],
                    "dataid": did,
                    "sim_pos": np.round(sim_pos, 4).tolist(),
                    "host_pos_after_sync": np.round(host_pos, 4).tolist(),
                    "ref_pos": np.round(r.pos, 4).tolist() if r is not None else None,
                    "mesh_extent": ext,
                    "ref_mesh_extent": None
                    if r is None or int(r.dataid[0]) < 0
                    else (
                        np.ptp(
                            ref.mesh_vert[
                                int(ref.mesh_vertadr[int(r.dataid[0])]) : int(
                                    ref.mesh_vertadr[int(r.dataid[0])]
                                )
                                + int(ref.mesh_vertnum[int(r.dataid[0])])
                            ],
                            axis=0,
                        )
                        .round(4)
                        .tolist()
                    ),
                }
            )
        rep["envs"].append(row)
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1)
    for row in rep["envs"][:3]:
        print(row["env"], row["size"])
        for gg in row["geoms"]:
            print(
                "  ",
                gg["name"],
                "did",
                gg["dataid"],
                "sim",
                gg["sim_pos"],
                "host",
                gg["host_pos_after_sync"],
                "ref",
                gg["ref_pos"],
                "ext",
                gg["mesh_extent"],
                "ref_ext",
                gg["ref_mesh_extent"],
            )
    print("expanded:", exp)


if __name__ == "__main__":
    main()
