from mjlab.tasks.registry import register_mjlab_task

from mjlab_hand.dual_hand_reorientation.dual_hand_reorientation_env_cfg import (
    make_dual_hand_env_cfg,
)
from mjlab_hand.tasks_common.rl import make_ppo_cfg

_PPO = "allegro_dual_hand_reorientation_dexpbt"

# Original layout (regular PPO preset); the hands can collide in this layout.
register_mjlab_task(
    task_id="DualHand-Reorientation-Allegro",
    env_cfg=make_dual_hand_env_cfg(),
    play_env_cfg=make_dual_hand_env_cfg(play=True),
    rl_cfg=make_ppo_cfg("allegro_dual_hand_reorientation", max_iterations=30000),
    runner_cls=None,
)

# Separated layout: hands cannot touch (collision termination only monitors). Handover through the table is allowed.
register_mjlab_task(
    task_id="DualHand-Reorientation-Allegro-Sep",
    env_cfg=make_dual_hand_env_cfg(layout="sep"),
    play_env_cfg=make_dual_hand_env_cfg(play=True, layout="sep"),
    rl_cfg=make_ppo_cfg(_PPO, preset="dexpbt", max_iterations=30000),
    runner_cls=None,
)

# Direct handover (default): far layout, a handover through the table ends the episode, +-30 handover shaping in
# training only; evaluation uses the play config.
register_mjlab_task(
    task_id="DualHand-Reorientation-Allegro-Direct",
    env_cfg=make_dual_hand_env_cfg(layout="far", handoff_shaping=True),
    play_env_cfg=make_dual_hand_env_cfg(play=True, layout="far"),
    rl_cfg=make_ppo_cfg(_PPO, preset="dexpbt", max_iterations=30000),
    runner_cls=None,
)

# Direct handover with the mjlab-tracking-style root action (absolute target, slew-limited, G1-recipe PD, action-rate
# penalty).
register_mjlab_task(
    task_id="DualHand-Reorientation-Allegro-TrackSlew",
    env_cfg=make_dual_hand_env_cfg(layout="far", handoff_shaping=True, track=True),
    play_env_cfg=make_dual_hand_env_cfg(play=True, layout="far", track=True),
    rl_cfg=make_ppo_cfg(_PPO, preset="dexpbt", max_iterations=30000, init_std=0.5),
    runner_cls=None,
)
