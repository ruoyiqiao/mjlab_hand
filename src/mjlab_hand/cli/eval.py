#!/usr/bin/env python
"""Evaluate a single trained policy on a grasp or in-hand-rotation task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trained policies")
    parser.add_argument("--task", type=str, required=True, help="Task ID (e.g. Grasp-Allegro)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to local .pt checkpoint")
    parser.add_argument("--wandb-run-path", type=str, default=None, help="WandB run path")
    parser.add_argument(
        "--wandb-checkpoint-name",
        type=str,
        default=None,
        help="Checkpoint name inside WandB run",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML eval config (CLI flags override config values)",
    )
    parser.add_argument(
        "--num-envs", type=int, default=None, help="Number of parallel environments"
    )
    parser.add_argument("--num-steps", type=int, default=None, help="Total simulation steps to run")
    parser.add_argument("--device", type=str, default=None, help="Torch device")
    parser.add_argument(
        "--object", type=str, default=None, help="ContactDB object name (e.g. mug, bowl)"
    )
    parser.add_argument(
        "--scale", type=str, default=None, help="Object scale as comma-separated values"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Optional JSON file to write metrics"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser


def main():
    parser = _build_parser()

    # Late imports keep CLI startup lightweight.
    from mjlab_hand.eval import (  # noqa: E402
        EvalConfig,
        default_eval_steps,
        load_config,
        load_policy,
        make_runner,
        register_cli_args,
        resolve_checkpoint,
        run_eval,
        select_evaluator,
        setup_eval_env,
        write_metrics_json,
    )

    register_cli_args(parser)
    args = parser.parse_args()

    base = load_config(args.config)
    cli_overrides = {
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "device": args.device,
        "seed": args.seed,
        "object": args.object,
        "scale": args.scale,
        "grasp_success_tolerance": getattr(args, "grasp_success_tolerance", None),
        "grasp_success_steps": getattr(args, "grasp_success_steps", None),
        "rotation_success_tolerance": getattr(args, "rotation_success_tolerance", None),
        "rotation_success_steps": getattr(args, "rotation_success_steps", None),
    }
    cfg: EvalConfig = base.merge(cli_overrides)

    # Side-effect import: registers Mjlab tasks with the gym registry.
    import mjlab_hand

    _ = mjlab_hand

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    ckpt_path = resolve_checkpoint(
        args.checkpoint,
        args.wandb_run_path,
        args.wandb_checkpoint_name,
        args.task,
        cfg.wandb_cache_root,
    )
    print(f"[INFO] Task: {args.task}")
    print(f"[INFO] Checkpoint: {ckpt_path}")
    print(f"[INFO] Envs: {cfg.num_envs} | Device: {device} | Seed: {cfg.seed}")

    env, rl_cfg = setup_eval_env(args.task, cfg, device)

    EvaluatorCls = select_evaluator(env)
    print(f"[INFO] Detected task type: {EvaluatorCls.name}")

    if cfg.num_steps is not None:
        num_steps = cfg.num_steps
    elif hasattr(EvaluatorCls, "default_steps"):
        num_steps = EvaluatorCls.default_steps(env)
    else:
        num_steps = default_eval_steps(EvaluatorCls.name)
    print(f"[INFO] Eval steps: {num_steps}")

    args.grasp_success_tolerance = cfg.grasp_success_tolerance
    args.grasp_success_steps = cfg.grasp_success_steps
    args.rotation_success_tolerance = cfg.rotation_success_tolerance
    args.rotation_success_steps = cfg.rotation_success_steps
    evaluator = EvaluatorCls(env, args, device)

    runner = make_runner(args.task, env, rl_cfg, device)
    policy = load_policy(runner, ckpt_path, device)
    print("[INFO] Policy loaded.")

    metrics = run_eval(env, policy, evaluator, num_steps, device)

    evaluator.report(metrics)
    print(json.dumps(metrics, indent=2))

    if args.output is not None:
        write_metrics_json(metrics, args.task, ckpt_path, Path(args.output))

    env.close()


if __name__ == "__main__":
    main()
