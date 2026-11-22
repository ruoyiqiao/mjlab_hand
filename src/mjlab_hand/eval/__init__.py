"""Task-agnostic evaluation framework for mjlab_hand (configs in configs/eval, entry point cli/eval.py)."""

from mjlab_hand.eval.base import (
    TaskEvaluator,
    register_cli_args,
    register_evaluator,
    run_eval,
    select_evaluator,
    write_metrics_json,
)
from mjlab_hand.eval.checkpoint import (
    discover_checkpoints,
    get_eval_cache_path,
    load_cached_eval,
    read_training_config,
    resolve_checkpoint,
    write_eval_cache,
)
from mjlab_hand.eval.config import EvalConfig, add_config_arg, load_config

# DexPBT-style Allegro tasks: registered first so the legacy command-name
# matchers below can never claim these envs.
from mjlab_hand.eval.dexpbt_tasks import DexPBTTaskEvaluator  # noqa: E402,F401
from mjlab_hand.eval.env_setup import (
    default_eval_steps,
    load_policy,
    make_runner,
    setup_eval_env,
)
from mjlab_hand.eval.grasp import GraspEvaluator
from mjlab_hand.eval.rotation import RotationEvaluator

__all__ = [
    "TaskEvaluator",
    "register_cli_args",
    "register_evaluator",
    "run_eval",
    "select_evaluator",
    "write_metrics_json",
    "DexPBTTaskEvaluator",
    "GraspEvaluator",
    "RotationEvaluator",
    "EvalConfig",
    "add_config_arg",
    "load_config",
    "discover_checkpoints",
    "get_eval_cache_path",
    "load_cached_eval",
    "read_training_config",
    "resolve_checkpoint",
    "write_eval_cache",
    "setup_eval_env",
    "make_runner",
    "load_policy",
    "default_eval_steps",
]
