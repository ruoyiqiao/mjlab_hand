"""Diagnostic only: run mjlab_hand.cli.eval with the play config's tolerance scale overridden.

The frozen evaluation uses tolerance scale 1.0 (final tolerance). For diagnosing a policy that
is still inside the tolerance curriculum, this evaluates it at the curriculum level it trains at,
so per-direction / per-goal failure modes are visible. Never used for acceptance numbers.

Usage: python eval_at_tolerance.py <tol_scale> <mjlab_hand.cli.eval args...>
"""

from __future__ import annotations

import sys


def main():
    scale = float(sys.argv[1])
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    from mjlab.tasks import registry

    import mjlab_hand  # noqa: F401  (registers tasks)

    task = sys.argv[sys.argv.index("--task") + 1]
    play = registry._REGISTRY[task].play_env_cfg
    for term in play.commands.values():
        if hasattr(term, "tolerance_scale_init"):
            term.tolerance_scale_init = scale
            print(f"[DIAG] {task}: tolerance_scale_init -> {scale}")
    from mjlab_hand.cli.eval import main as eval_main

    eval_main()


if __name__ == "__main__":
    main()
