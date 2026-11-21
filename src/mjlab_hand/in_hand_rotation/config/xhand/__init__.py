from mjlab.tasks.registry import register_mjlab_task

from mjlab_hand.in_hand_rotation.runner import InHandRotationOnPolicyRunner

from .env_cfgs import xhand_rotation_env_cfg
from .rl_cfg import xhand_rotation_ppo_runner_cfg

register_mjlab_task(
    task_id="InHand-Rotation-XHand",
    env_cfg=xhand_rotation_env_cfg(),
    play_env_cfg=xhand_rotation_env_cfg(play=True),
    rl_cfg=xhand_rotation_ppo_runner_cfg(),
    runner_cls=InHandRotationOnPolicyRunner,
)

# Explicit candidate for comparison; the existing task retains its palm-up pose.
register_mjlab_task(
    task_id="InHand-Rotation-XHand-Tilt30",
    env_cfg=xhand_rotation_env_cfg(palm_tilt_deg=30.0),
    play_env_cfg=xhand_rotation_env_cfg(play=True, palm_tilt_deg=30.0),
    rl_cfg=xhand_rotation_ppo_runner_cfg(),
    runner_cls=InHandRotationOnPolicyRunner,
)
