"""Task registration entry point for mjlab_hand."""

from importlib import import_module

_TASK_MODULES = (
    "mjlab_hand.grasp.config.allegro",
    "mjlab_hand.grasp.config.leap",
    "mjlab_hand.grasp.config.shadow",
    "mjlab_hand.grasp.config.sharpa",
    "mjlab_hand.grasp.config.wuji",
    "mjlab_hand.grasp.config.xhand",
    "mjlab_hand.in_hand_rotation.config.allegro",
    "mjlab_hand.in_hand_rotation.config.leap",
    "mjlab_hand.in_hand_rotation.config.shadow",
    "mjlab_hand.in_hand_rotation.config.sharpa",
    "mjlab_hand.in_hand_rotation.config.wuji",
    "mjlab_hand.in_hand_rotation.config.xhand",
    # Allegro DexPBT-style tasks (tasks_common/).
    "mjlab_hand.grasp_throw.config.allegro",
    "mjlab_hand.object_reorientation.config.allegro",
    "mjlab_hand.regrasp.config.allegro",
    "mjlab_hand.dual_hand_reorientation.config.allegro",
    # The same single-hand tasks on the other hands.
    "mjlab_hand.tasks_common.hand_tasks",
    # Allegro toss juggling (juggling/).
    "mjlab_hand.juggling.config.allegro",
    "mjlab_hand.juggling.config.hands",
    # Palm-up spinning tasks on every hand: Baoding balls, plate spinning.
    "mjlab_hand.palm_tasks",
)


def register_tasks() -> None:
    """Import task modules so their mjlab registry side effects run."""
    for module_name in _TASK_MODULES:
        import_module(module_name)


__all__ = ["register_tasks"]
