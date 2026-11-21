from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.rl.config import RslRlModelCfg


def xhand_grasp_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "log",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            entropy_coef=0.0,
            value_loss_coef=2.0,
            schedule="adaptive",
            use_clipped_value_loss=True,
            learning_rate=3e-4,
        ),
        clip_actions=1.0,
        experiment_name="xhand_grasp",
        max_iterations=20_000,
        num_steps_per_env=24,
        save_interval=100,
    )
