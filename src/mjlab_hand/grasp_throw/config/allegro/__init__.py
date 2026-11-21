from mjlab.tasks.registry import register_mjlab_task

from mjlab_hand.grasp_throw.grasp_throw_env_cfg import make_grasp_throw_spec
from mjlab_hand.tasks_common.env_builder import make_single_hand_env_cfg
from mjlab_hand.tasks_common.rl import make_ppo_cfg


def allegro_grasp_throw_env_cfg(play: bool = False):
    return make_single_hand_env_cfg(make_grasp_throw_spec(), play=play)


register_mjlab_task(
    task_id="Grasp-Throw-Allegro",
    env_cfg=allegro_grasp_throw_env_cfg(),
    play_env_cfg=allegro_grasp_throw_env_cfg(play=True),
    rl_cfg=make_ppo_cfg("allegro_grasp_throw", max_iterations=10000),
    runner_cls=None,
)
