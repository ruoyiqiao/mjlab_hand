"""Juggling task registration (Allegro).

Juggle1/2: one hand, one / two balls (Ploeger et al., CoRL 2020 setting), closed-loop PPO.
Juggle3-Allegro-*: two hands, three balls, paper-style designed prior + RL residual (docs/allegro_four_tasks/juggling_plan.md).

One-hand pattern constants from first principles (Shannon): throw interval T, dwell D, vacant V = T - D, flight
F = 2V + D, release speed v0 = g F / 2, apex g F^2 / 8. T = 0.40 s, D = 0.15 s -> F = 0.65 s, v0 = 3.19 m/s, apex 0.52 m.
"""

from mjlab.tasks.registry import register_mjlab_task

from mjlab_hand.juggling.cascade import CascadeSpec, make_cascade_env_cfg
from mjlab_hand.juggling.juggle_env_cfg import JuggleSpec, make_juggle_env_cfg
from mjlab_hand.juggling.patterns import PATTERN_ORDER, PatternSpec, make_pattern_env_cfg
from mjlab_hand.tasks_common.rl import make_ppo_cfg

G = 9.81
T_HALF, DWELL, COL = 0.40, 0.15, 0.08
VAC = T_HALF - DWELL
FLIGHT = 2 * VAC + DWELL
V0 = G * FLIGHT / 2
APEX = G * FLIGHT**2 / 8


def juggle_rl_spec(num_balls: int) -> JuggleSpec:
    """Track B (closed-loop PPO). Paper binary 'alive' reward + event / shaping terms,
    hold limit, fixed finger cup (6-D base action, paper: rigid funnel), and a
    reference-state start consistent with the steady pattern: ball A at rest in the
    cup at column A, ball B on its steady parabola landing on column B after
    tau ~ U(T - 0.03, T + 0.10) at speed v0. Deviations: juggling_plan.md."""
    return JuggleSpec(
        num_balls=num_balls,
        column_offset=(0.0, COL, 0.0),
        half_cycle_s=T_HALF,
        ref_tau_range=(T_HALF - 0.03, T_HALF + 0.10) if num_balls > 1 else None,
        ref_land_speed=V0,
        max_hold_s=DWELL + 0.25,
        target_apex=APEX,
        finger_mode="fixed",
        lin_vel_max=6.0,
        ang_vel_max=6.0,
        base_gains=(5000.0, 140.0, 400.0),
        reward_alive=0.05,
        reward_catch=1.0,
        reward_failure=-5.0,
        reward_too_high=-1.0,
        airborne_prob=0.5 if num_balls > 1 else 0.0,
        throw_speed=V0,
        vacant_s=VAC,
        reward_throw=0.0,
        reward_throw_quality=1.0,
        reward_catch_align=0.02,
        reward_ball_ball=-0.05,
    )


def _register(task_id: str, spec: JuggleSpec, num_balls: int):
    """Training spec `spec`; play (evaluation) config is always the frozen base spec of the same ball count."""
    register_mjlab_task(
        task_id=task_id,
        env_cfg=make_juggle_env_cfg(spec),
        play_env_cfg=make_juggle_env_cfg(juggle_rl_spec(num_balls), play=True),
        rl_cfg=make_ppo_cfg(
            f"allegro_juggle{num_balls}_onehand", max_iterations=20000, init_std=0.5
        ),
        runner_cls=None,
    )


def _entry(**overrides) -> JuggleSpec:
    """Training-only variants of Juggle2 (evaluation always uses the frozen play config): paper starts, entry deadline,
    penalties for lost holds / stalls / too-high throws, and a cap on the actual flight height."""
    sp = juggle_rl_spec(2)
    sp.airborne_prob = 0.3
    sp.ref_tau_range = (T_HALF - 0.03, T_HALF + 0.20)
    for k, v in overrides.items():
        setattr(sp, k, v)
    return sp


_E3 = dict(
    reward_lost_hold=-0.5, entry_deadline_s=0.6
)  # Entry3: entry deadline, held -> free penalty
_E4 = dict(
    _E3, alive_recent_throw_s=0.9, reward_stall=-5.0, reward_too_high=-3.0
)  # Entry4: alive only while throwing
_E6 = dict(_E4, max_throw_apex=1.6 * APEX, cap_actual_apex=True)  # Entry6: cap actual flight height

_register("Juggle1-OneHand-Allegro", juggle_rl_spec(1), 1)
_register("Juggle2-OneHand-Allegro", juggle_rl_spec(2), 2)
_register("Juggle2-OneHand-Allegro-Entry3", _entry(**_E3), 2)
_register("Juggle2-OneHand-Allegro-Entry4", _entry(**_E4), 2)
_register("Juggle2-OneHand-Allegro-Entry6", _entry(**_E6), 2)

# Two hands, three balls: paper-style cascade (fixed start; prior = designed controller, zero action).
PRIOR, LEARN, RESIDUAL = CascadeSpec(), CascadeSpec(learner=True), CascadeSpec(random_phase=True)
for _id, _train, _play in (
    ("Juggle3-Allegro-Cascade-Prior", PRIOR, PRIOR),
    ("Juggle3-Allegro-Cascade-Learner", LEARN, LEARN),
    ("Juggle3-Allegro-Cascade-Residual", RESIDUAL, PRIOR),
):
    register_mjlab_task(
        task_id=_id,
        env_cfg=make_cascade_env_cfg(_train),
        play_env_cfg=make_cascade_env_cfg(_play, play=True),
        rl_cfg=make_ppo_cfg("allegro_juggle3_cascade", max_iterations=30000, init_std=0.3),
        runner_cls=None,
    )

# All five patterns (cascade, tennis, half-shower, shower, box): one task per pattern, plus -All (pattern sampled per episode).
for _name, _pats in [("All", PATTERN_ORDER)] + [
    ("".join(w.capitalize() for w in _p.split("_")), (_p,)) for _p in PATTERN_ORDER
]:
    _sp = PatternSpec(patterns=tuple(_pats))
    register_mjlab_task(
        task_id=f"Juggle3-Allegro-{_name}",
        env_cfg=make_pattern_env_cfg(_sp),
        play_env_cfg=make_pattern_env_cfg(_sp, play=True),
        rl_cfg=make_ppo_cfg("allegro_juggle3_patterns", max_iterations=30000, init_std=0.3),
        runner_cls=None,
    )
