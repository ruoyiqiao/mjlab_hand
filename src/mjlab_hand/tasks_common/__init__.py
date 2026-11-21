"""Shared building blocks for the Allegro DexPBT-style tasks.

Grasp-Throw, Object-Reorientation, DualHand-Reorientation and Regrasp share:

* :mod:`workspace` - hand-root anchor / bounds description (no arm),
* :mod:`actions` - bounded floating-hand action (base deltas + finger targets),
* :mod:`keypoints` - DexPBT object keypoints and quaternion errors,
* :mod:`objects` - primitive cube, open-top bucket and table specs,
* :mod:`tracker` - per-env DexPBT task state (lifted flag, best distances,
  hold timer, success counter, per-goal timer) as a command term,
* :mod:`rewards`, :mod:`observations`, :mod:`terminations`, :mod:`events`,
  :mod:`sensors`.

Reference: IsaacGymEnvs allegro_kuka tasks, revision
aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3 (BSD-3-Clause, NVIDIA). Formulas are
re-implemented, not copied; see docs/allegro_four_tasks/reward_mapping.md.
"""
