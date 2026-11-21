"""PPO runner configs for the Allegro DexPBT-style tasks.

Default ("mjlab_grasp"): the repository's Allegro grasp PPO settings
(grasp/config/allegro/rl_cfg.py: MLP 512-256-128, obs normalization, lr 3e-4,
adaptive KL, 24 steps/env). Fallback ("dexpbt"): closer to IsaacGymEnvs
AllegroKukaPPO.yaml (MLP 1024-1024-512-512, lr 1e-4, desired KL 0.016,
clip 0.1, 16 steps, 4 epochs, value coef 4). The choice is recorded per run.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.rl.config import RslRlModelCfg


def make_ppo_cfg(
    experiment_name: str,
    preset: str = "mjlab_grasp",
    max_iterations: int = 3000,
    init_std: float = 1.0,
) -> RslRlOnPolicyRunnerCfg:
    if preset == "mjlab_grasp":
        hidden = (512, 256, 128)
        algo = RslRlPpoAlgorithmCfg(
            entropy_coef=0.0,
            value_loss_coef=2.0,
            schedule="adaptive",
            use_clipped_value_loss=True,
            learning_rate=3e-4,
        )
        steps = 24
    elif preset == "dexpbt":
        hidden = (1024, 1024, 512, 512)
        algo = RslRlPpoAlgorithmCfg(
            entropy_coef=0.0,
            value_loss_coef=4.0,
            schedule="adaptive",
            desired_kl=0.016,
            use_clipped_value_loss=True,
            learning_rate=1e-4,
            clip_param=0.1,
            num_learning_epochs=4,
            num_mini_batches=4,
            max_grad_norm=1.0,
        )
        steps = 16
    else:
        raise ValueError(preset)
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=hidden,
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": init_std,
                "std_type": "log",
            },
        ),
        critic=RslRlModelCfg(hidden_dims=hidden, obs_normalization=True),
        algorithm=algo,
        clip_actions=1.0,
        experiment_name=experiment_name,
        max_iterations=max_iterations,
        num_steps_per_env=steps,
        save_interval=100,
        logger="tensorboard",
    )
