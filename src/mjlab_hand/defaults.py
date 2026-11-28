"""Defaults shared by every task."""

# Parallel environments a task trains with unless --env.scene.num-envs says otherwise (every reported run used
# 2048). Play configs keep a single environment.
NUM_TRAIN_ENVS = 2048
