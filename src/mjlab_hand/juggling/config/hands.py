"""Juggle1/2 (one hand, one / two balls) on the other hands.

Same spec as ``Juggle{1,2}-OneHand-Allegro`` (juggling/config/allegro: rewards, timing, hold limit, start states, fixed
finger cup + 6-D base action) with ``hand=`` changed. Per hand only the measured cup is new: the fixed finger pose and
the ball rest point on the palm (hand_params/juggle.json, tools/allegro4/juggle_cup_probe.py). Base gains are
Allegro's juggling gains; hands with ``match_base_dynamics`` get them scaled by their base inertia (env_builder).
"""

import json

from mjlab.tasks.registry import register_mjlab_task

from mjlab_hand.juggling.config.allegro import juggle_rl_spec
from mjlab_hand.juggling.juggle_env_cfg import make_juggle_env_cfg
from mjlab_hand.tasks_common.hand_tasks import HAND_SUFFIX
from mjlab_hand.tasks_common.rl import make_ppo_cfg

JUGGLE_HANDS = ("leap", "wuji", "sharpa", "xhand", "shadow")


def _spec(hand: str, num_balls: int):
    sp = juggle_rl_spec(num_balls)
    sp.hand = hand
    if hand == "leap" and num_balls == 2:
        sp.leap_pd = "menagerie"  # LEAP_HAND_MENAGERIE
    return sp


for _h in JUGGLE_HANDS:
    for _nb in (1, 2):
        register_mjlab_task(
            task_id=f"Juggle{_nb}-OneHand-{HAND_SUFFIX[_h]}",
            env_cfg=make_juggle_env_cfg(_spec(_h, _nb)),
            play_env_cfg=make_juggle_env_cfg(_spec(_h, _nb), play=True),
            rl_cfg=make_ppo_cfg(f"{_h}_juggle{_nb}_onehand", max_iterations=20000, init_std=0.5),
            runner_cls=None,
        )


# Juggle3 (two hands, three balls, cascade): Allegro's CascadeSpec with hand= and the measured hand_y
# (hand_params/juggle3.json, tools/allegro4/juggle3_probe.py). Registered only for hands that have been measured.
def _register_juggle3():
    from mjlab_hand.juggling.cascade import _PARAMS3, CascadeSpec, make_cascade_env_cfg

    measured = json.loads(_PARAMS3.read_text()) if _PARAMS3.exists() else {}
    for h in JUGGLE_HANDS:
        if h not in measured:
            continue
        hy = float(measured[h]["hand_y"])
        prior = CascadeSpec(hand=h, hand_y=hy)
        residual = CascadeSpec(hand=h, hand_y=hy, random_phase=True)
        for name, train, play in (("Prior", prior, prior), ("Residual", residual, prior)):
            register_mjlab_task(
                task_id=f"Juggle3-{HAND_SUFFIX[h]}-Cascade-{name}",
                env_cfg=make_cascade_env_cfg(train),
                play_env_cfg=make_cascade_env_cfg(play, play=True),
                rl_cfg=make_ppo_cfg(f"{h}_juggle3_cascade", max_iterations=30000, init_std=0.3),
                runner_cls=None,
            )
        # the other four patterns (tennis, half-shower, shower, box), as Juggle3-Allegro-<Pattern>
        from mjlab_hand.juggling.patterns import PATTERN_ORDER, PatternSpec, make_pattern_env_cfg

        for pat in PATTERN_ORDER[1:]:
            ps = PatternSpec(patterns=(pat,), hand=h, hand_y=hy)
            register_mjlab_task(
                task_id=f"Juggle3-{HAND_SUFFIX[h]}-{''.join(w.capitalize() for w in pat.split('_'))}",
                env_cfg=make_pattern_env_cfg(ps),
                play_env_cfg=make_pattern_env_cfg(ps, play=True),
                rl_cfg=make_ppo_cfg(f"{h}_juggle3_patterns", max_iterations=30000, init_std=0.3),
                runner_cls=None,
            )


_register_juggle3()


# Juggle2 -Full (all hands, Allegro included; the frozen Juggle2 above is unchanged): whole-hand action = 6 palm dims +
# the hand's finger targets (delta mode: zero action holds the current pose) and the Juggle3 ball (63.5 mm / 130 g soft).
# The fingers start in the measured Juggle3 grip, which holds this ball (hand_params/juggle3.json; Allegro: its cascade
# grip pose), and the ball rest point is that grip's. Timing, rewards, start states and evaluation are Juggle2's.
def full_spec(hand: str):
    from mjlab_hand.juggling.cascade import SIDES, CascadeSpec, finger_pose, hand_cascade_params
    from mjlab_hand.tasks_common.hand_profile import get_profile

    sp = _spec(hand, 2)
    ball = CascadeSpec()
    sp.ball_radius, sp.ball_mass, sp.ball_solref = (
        ball.ball_radius,
        ball.ball_mass,
        ball.ball_solref,
    )
    sp.finger_mode = "delta"
    if hand == "allegro":
        sp.cup_pose = finger_pose(
            ball.hold_pose
        )  # rest point: the palm-face rule of the Allegro command
    else:
        hp = hand_cascade_params(hand)
        side = next(s for s in SIDES if hp["sides"][s]["model_side"] == get_profile(hand).side)
        sp.cup_pose = dict(hp["poses"][side]["grip"])
        sp.cup_rel_palm = tuple(hp["cup_rel_palm"][side])
    return sp


for _h in ("allegro", *JUGGLE_HANDS):
    register_mjlab_task(
        task_id=f"Juggle2-OneHand-{HAND_SUFFIX[_h]}-Full",
        env_cfg=make_juggle_env_cfg(full_spec(_h)),
        play_env_cfg=make_juggle_env_cfg(full_spec(_h), play=True),
        rl_cfg=make_ppo_cfg(f"{_h}_juggle2_full", max_iterations=20000, init_std=0.5),
        runner_cls=None,
    )
