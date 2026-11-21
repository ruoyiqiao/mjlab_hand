from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import xhand_grasp_env_cfg
from .rl_cfg import xhand_grasp_ppo_runner_cfg

register_mjlab_task(
    task_id="Grasp-XHand",
    env_cfg=xhand_grasp_env_cfg(),
    play_env_cfg=xhand_grasp_env_cfg(play=True),
    rl_cfg=xhand_grasp_ppo_runner_cfg(),
    runner_cls=None,
)
