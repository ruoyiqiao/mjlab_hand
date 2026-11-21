"""Close-up render of several object variants resting on the table (visual check)."""

import argparse
import os

import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.viewer.model_sync import VIEWER_MODEL_FIELDS, sync_model_fields
from PIL import Image, ImageDraw

import mjlab_hand  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("task")
ap.add_argument("out")
ap.add_argument("--n", type=int, default=8)
a = ap.parse_args()
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

cfg = load_env_cfg(a.task, play=True)
cfg.scene.num_envs = a.n
env = ManagerBasedRlEnv(cfg, device="cuda:0")
env.reset()
for _ in range(40):
    env.step(torch.full((a.n, env.action_manager.total_action_dim), -1.0, device="cuda:0"))
t = env.command_manager.get_term(next(iter(env.command_manager.active_terms)))
m = env.sim.mj_model
d = mujoco.MjData(m)
r = mujoco.Renderer(m, 400, 400)
tiles = []
for i in range(a.n):
    sync_model_fields(m, env.sim.model, set(env.sim.expanded_fields) & VIEWER_MODEL_FIELDS, i)
    sd = env.sim.data
    d.qpos[:] = sd.qpos[i].cpu().numpy()
    d.mocap_pos[:] = sd.mocap_pos[i].cpu().numpy()
    d.mocap_quat[:] = sd.mocap_quat[i].cpu().numpy()
    mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = t.obj.data.root_link_pos_w[i].cpu().numpy()
    cam.distance = 0.45
    cam.azimuth = 135
    cam.elevation = -30
    for grp3 in (0, 1):
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 1
        opt.geomgroup[3] = grp3
        opt.geomgroup[2] = 1 - grp3
        r.update_scene(d, camera=cam, scene_option=opt)
        im = Image.fromarray(r.render())
        ImageDraw.Draw(im).text(
            (6, 6),
            f"env{i} {[round(x, 3) for x in t.obj_size[i].tolist()]} {'collision' if grp3 else 'visual'}",
            fill=(255, 255, 255),
        )
        tiles.append(im)
W = 400
cols = 4
rows = (len(tiles) + cols - 1) // cols
sheet = Image.new("RGB", (cols * W, rows * W))
for k, im in enumerate(tiles):
    sheet.paste(im, ((k % cols) * W, (k // cols) * W))
sheet.save(a.out)
print("saved", a.out)
