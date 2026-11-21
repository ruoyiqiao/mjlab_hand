from mjlab.tasks.registry import register_mjlab_task

from mjlab_hand.regrasp.regrasp_env_cfg import make_regrasp_spec
from mjlab_hand.tasks_common.env_builder import make_single_hand_env_cfg
from mjlab_hand.tasks_common.rl import make_ppo_cfg


def allegro_regrasp_env_cfg(play: bool = False):
    return make_single_hand_env_cfg(make_regrasp_spec(), play=play)


register_mjlab_task(
    task_id="Regrasp-Allegro",
    env_cfg=allegro_regrasp_env_cfg(),
    play_env_cfg=allegro_regrasp_env_cfg(play=True),
    rl_cfg=make_ppo_cfg("allegro_regrasp", max_iterations=10000),
    runner_cls=None,
)
