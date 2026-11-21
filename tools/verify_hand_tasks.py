"""Small GPU integration check; outputs PPO, reset, render and resume evidence."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from PIL import Image

import mjlab_hand  # noqa: F401


def assert_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_state_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for va, vb in zip(a, b, strict=True):
            assert_state_equal(va, vb)
    else:
        assert a == b


def verify(task, out):
    start = time.monotonic()
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = 8
    cfg.seed = 42
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
    try:
        obs, _ = env.reset()
        obs_shapes = {k: list(v.shape) for k, v in obs.items()}
        action_dim = env.action_manager.total_action_dim
        for _ in range(32):
            actions = torch.rand((8, action_dim), device="cuda:0") * 2 - 1
            obs, rew, terminated, truncated, info = env.step(actions)
            assert torch.isfinite(rew).all()
            assert all(torch.isfinite(v).all() for v in obs.values())
            assert torch.isfinite(env.sim.data.qpos).all()
            assert torch.isfinite(env.sim.data.qvel).all()
        for ids in ([0], [1, 3, 5]):
            untouched = [i for i in range(8) if i not in ids]
            qpos, qvel = env.sim.data.qpos[untouched].clone(), env.sim.data.qvel[untouched].clone()
            env.reset(env_ids=torch.tensor(ids, device="cuda:0"))
            torch.testing.assert_close(env.sim.data.qpos[untouched], qpos, atol=0, rtol=0)
            torch.testing.assert_close(env.sim.data.qvel[untouched], qvel, atol=0, rtol=0)
        frame = env.render()
        assert frame is not None and frame.ndim == 3 and frame.shape[-1] == 3
        assert np.isfinite(frame).all() and frame.std() > 1
        Image.fromarray(frame).save(out / f"{task}.png")
        agent = load_rl_cfg(task)
        agent.logger = "tensorboard"
        agent.upload_model = False
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        cls = load_runner_cls(task) or MjlabOnPolicyRunner
        runner = cls(wrapped, asdict(agent), str(out / task), "cuda:0")
        runner.learn(num_learning_iterations=2, init_at_random_ep_len=False)
        checkpoint = out / task / "verified.pt"
        runner.save(str(checkpoint))
        saved = runner.alg.save()
        steps = env.common_step_counter
        iteration = runner.current_learning_iteration
    finally:
        env.close()
    # Build a fresh environment and runner, rather than loading back into the writer.
    restored_env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    try:
        restored = cls(
            RslRlVecEnvWrapper(restored_env, clip_actions=agent.clip_actions),
            asdict(agent),
            str(out / task / "resumed"),
            "cuda:0",
        )
        restored.load(str(checkpoint))
        assert_state_equal(saved, restored.alg.save())
        assert restored.current_learning_iteration == iteration
        assert restored_env.common_step_counter == steps
        restored.learn(num_learning_iterations=1, init_at_random_ep_len=False)
        for model in (restored.alg.actor, restored.alg.critic):
            assert all(torch.isfinite(p).all() for p in model.parameters())
        result = dict(
            task=task,
            num_envs=8,
            action_dim=action_dim,
            sim_dt=cfg.sim.mujoco.timestep,
            decimation=cfg.decimation,
            obs_shapes=obs_shapes,
            random_steps=32,
            ppo_iterations=2,
            resumed_iterations=1,
            partial_reset=True,
            fresh_runner_checkpoint_equal=True,
            render_shape=list(frame.shape),
            elapsed_s=time.monotonic() - start,
        )
        (out / f"{task}.json").write_text(json.dumps(result, indent=2) + "\n")
        print("VERIFY_PASS", json.dumps(result), flush=True)
    finally:
        restored_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        verify(task, args.output)
