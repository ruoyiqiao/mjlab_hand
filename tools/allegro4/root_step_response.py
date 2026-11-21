"""Check the root action/motor parameters of a DualHand task and measure the step response.

Prints, per hand-0 base joint: compiled actuator kp/kv/force limit, action scale/offset (absolute mode),
the PD's analytic wn/zeta from the measured effective inertia, and the response of the real env to a
constant action of +0.5 for 1 s (peak joint speed, 90% rise time, overshoot). Also a=0 -> hold home.
Usage: python root_step_response.py <task_id> [--device cuda:0]
"""

import argparse
import math

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from mjlab_hand.registration import register_tasks

ap = argparse.ArgumentParser()
ap.add_argument("task")
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()

register_tasks()

cfg = load_env_cfg(a.task, play=True)
cfg.scene.num_envs = 4
env = ManagerBasedRlEnv(cfg, device=a.device)
env.reset()
term = env.action_manager.get_term("hand_left")
m = env.sim.mj_model
names = term.canonical_joint_names[:6]
print("mode", term.cfg.base_mode, "dt", env.step_dt)
for k, n in enumerate(names):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"hand_left/{n}")
    act = [
        i
        for i in range(m.nu)
        if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT and m.actuator_trnid[i, 0] == jid
    ][0]
    kp, kv = m.actuator_gainprm[act, 0], -m.actuator_biasprm[act, 2]
    fr = m.actuator_forcerange[act]
    dof = m.jnt_dofadr[jid]
    d = mujoco.MjData(m)
    d.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
    mujoco.mj_forward(m, d)
    col = np.zeros(m.nv)
    col[dof] = 1
    out = np.zeros(m.nv)
    mujoco.mj_crb(m, d)
    mujoco.mj_mulM(m, d, out, col)
    inertia = out[dof]
    wn = math.sqrt(kp / inertia)
    z = kv / (2 * math.sqrt(kp * inertia))
    sc = float(term._abs_scale[0, k]) if term.cfg.base_mode == "absolute" else float("nan")
    print(
        f"{n:11s} kp {kp:9.2f} kv {kv:7.3f} force +-{fr[1]:8.1f} | I_eff {inertia:.4f} wn {wn / 2 / math.pi:5.2f} Hz zeta {z:4.2f}"
        f" | scale {sc:.3f} full-action load {kp * sc / fr[1]:.2f} of limit"
    )

dev = env.device


def run(act_val, joint, steps=60):
    # also +-: sign check done by calling with negative act_val
    env.reset()
    act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=dev)
    act[:, joint] = act_val
    # fingers absolute: hold the post-reset finger target by sending zeros -> offset; fine for a root check
    ids = term.joint_ids_canonical[joint]
    qs, vs = [], []
    for _ in range(steps):
        env.step(act)
        qs.append(float(env.scene["hand_left"].data.joint_pos[0, ids]))
        vs.append(float(env.scene["hand_left"].data.joint_vel[0, ids]))
    return np.array(qs), np.array(vs)


for k in range(6):
    pass
for k in range(6):
    n = names[k]
    q, v = run(0.5, k)
    qn, _ = run(-0.5, k)
    print(f"  {n}: +0.5 -> final {q[-1]:+.4f}   -0.5 -> final {qn[-1]:+.4f}")
    tgt = 0.5 * float(term._abs_scale[0, k]) if term.cfg.base_mode == "absolute" else float("nan")
    q0 = q[0]
    if term.cfg.base_mode == "absolute":
        rise = next((i for i, x in enumerate(q) if x >= 0.9 * tgt), None)
        print(
            f"step {n}: target {tgt:.3f}  final {q[-1]:.3f}  err {tgt - q[-1]:+.4f}  peak speed {np.abs(v).max():.2f}  "
            f"rise90 {None if rise is None else round((rise + 1) * env.step_dt, 3)} s  overshoot {max(0.0, q.max() - tgt):.4f}"
        )
    else:
        print(f"delta {n}: final {q[-1]:.3f} peak speed {np.abs(v).max():.2f}")
env.close()
