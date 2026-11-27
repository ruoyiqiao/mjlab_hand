"""Registers the single-hand tasks (Grasp-Throw, Object-Reorientation, Regrasp) for every hand except Allegro.

Allegro's ids are registered by the task packages themselves (``<task>/config/allegro``). Ids are ``<Task>-<Suffix>`` with the
suffixes below; the configs are the Allegro ones with ``hand=`` changed (same rewards, tolerances, timing, objects).
"""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from mjlab_hand.grasp_throw.grasp_throw_env_cfg import make_grasp_throw_spec
from mjlab_hand.object_reorientation.object_reorientation_env_cfg import (
    make_object_reorientation_spec,
)
from mjlab_hand.regrasp.regrasp_env_cfg import make_regrasp_spec
from mjlab_hand.tasks_common.env_builder import make_single_hand_env_cfg
from mjlab_hand.tasks_common.rl import make_ppo_cfg

HAND_SUFFIX = {
    "allegro": "Allegro",
    "leap": "LEAP",
    "wuji": "Wuji",
    "sharpa": "Sharpa",
    "xhand": "XHand",
    "shadow": "Shadow",
}
SINGLE_TASKS = {
    "Grasp-Throw": ("grasp_throw", make_grasp_throw_spec),
    "Object-Reorientation": ("object_reorientation", make_object_reorientation_spec),
    "Regrasp": ("regrasp", make_regrasp_spec),
}
HANDS = tuple(h for h in HAND_SUFFIX if h != "allegro")
# Default run length per (task, hand) where training longer scored lower on the held-out seeds: the reported
# Sharpa and Shadow reorientation checkpoints come from 3000-iteration runs (Sharpa s43: 69.5 % after 3000
# iterations, 39.9 % after 9000). The other single-hand tasks default to 10000.
MAX_ITERATIONS = {
    ("Object-Reorientation", "sharpa"): 3000,
    ("Object-Reorientation", "shadow"): 3000,
}


def _register(hand: str) -> None:
    for task, (exp, make_spec) in SINGLE_TASKS.items():

        def cfg(play=False, make_spec=make_spec, hand=hand):
            return make_single_hand_env_cfg(make_spec(hand), play=play)

        register_mjlab_task(
            task_id=f"{task}-{HAND_SUFFIX[hand]}",
            env_cfg=cfg(),
            play_env_cfg=cfg(play=True),
            rl_cfg=make_ppo_cfg(
                f"{hand}_{exp}", max_iterations=MAX_ITERATIONS.get((task, hand), 10000)
            ),
            runner_cls=None,
        )


for _h in HANDS:
    _register(_h)
