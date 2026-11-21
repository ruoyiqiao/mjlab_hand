"""Two-hand, three-ball juggling patterns from siteswap (cascade, tennis, half-shower, shower, box).

Hands: 0 = right (y < 0), 1 = left (y > 0). Throws sit on a beat grid t_k = k T (async: one throw per beat, hands
alternating; sync: both hands every 2 T). A throw of value v at beat k is thrown again at beat k + v (by the same hand
if v is even, else the other; sync "x" throws cross). Flight F = v T - D_next (D_next = dwell of the ball's next throw),
apex g F^2 / 8 above the release point; the lateral path runs from the release to the catch offset (+ = away from the
other hand). The patterns actually used are built in ``patterns.pattern_def`` (paper beat, searched dwells).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

G = 9.81
HANDS = ("right", "left")


@dataclass(frozen=True)
class Throw:
    beat: int  # beat index within the period
    hand: int  # 0 right, 1 left
    value: int  # siteswap value (async beats)
    cross: bool  # lands in the other hand
    release: str  # "in" / "out" / "mid"
    catch: str  # "in" / "out" / "mid"
    dwell: float  # dwell (catch -> this throw) of the ball thrown here, s
    rise: float = 0.0  # release height above the cup home (m): a hard throw after a short dwell


@dataclass(frozen=True)
class Pattern:
    name: str
    beat_s: float
    period_beats: int
    throws: tuple[Throw, ...]
    sync: bool = False
    num_balls: int = 3
    notes: str = ""


LATERAL = {"in": -0.04, "out": 0.04, "mid": 0.0}  # + = away from the other hand


def _cascade(T=0.30, D=0.30) -> Pattern:
    return Pattern(
        "cascade",
        T,
        2,
        (
            Throw(0, 0, 3, True, "in", "out", D),
            Throw(1, 1, 3, True, "in", "out", D),
        ),
        notes="siteswap 3",
    )


def _tennis(T=0.30, D=0.30) -> Pattern:
    # Ball thrown at beats 0, 3 (mod 6) is the "tennis" ball: released and caught outside, so
    # it travels over the top of the other two.
    th = []
    for b in range(6):
        over = b % 3 == 0
        th.append(Throw(b, b % 2, 3, True, "out" if over else "in", "out", D))
    return Pattern("tennis", T, 6, tuple(th), notes="siteswap 3, one ball over the top")


def _half_shower(T=0.30, d_right=0.42, d_left=0.20) -> Pattern:
    # Right throws over (from outside, high), left throws under (from inside, low).
    return Pattern(
        "half_shower",
        T,
        2,
        (
            Throw(0, 0, 3, True, "out", "out", d_right),
            Throw(1, 1, 3, True, "in", "in", d_left),
        ),
        notes="siteswap 3, right over / left under",
    )


def _shower(T=0.22, d_right=0.10, d_left=0.20, rise=0.20) -> Pattern:
    # The right hand catches the fast pass almost at rest and has d_right to reach ~4.4 m/s:
    # at constant acceleration that takes ~v D / 2 = 0.2 m of travel, so the 5 is released
    # 0.2 m above the catch point (a release at the catch height would need the cup to dip
    # first, i.e. accelerate down faster than g while holding the ball).
    return Pattern(
        "shower",
        T,
        2,
        (
            Throw(0, 0, 5, True, "out", "out", d_right, rise),
            Throw(1, 1, 1, True, "in", "in", d_left),
        ),
        notes="siteswap 51",
    )


def _box(T=0.25, D=0.25) -> Pattern:
    # sync (4,2x)(2x,4): beats 0 and 2 (sync beat = 2 T).
    return Pattern(
        "box",
        T,
        4,
        (
            Throw(0, 0, 4, False, "out", "out", D),
            Throw(0, 1, 2, True, "in", "in", D),
            Throw(2, 0, 2, True, "in", "in", D),
            Throw(2, 1, 4, False, "out", "out", D),
        ),
        sync=True,
        notes="sync (4,2x)(2x,4)",
    )


@dataclass
class Unrolled:
    """Throw events of one pattern over a horizon (arrays of length E)."""

    pattern: str
    t: np.ndarray  # throw time, s
    hand: np.ndarray  # throwing hand
    to_hand: np.ndarray  # catching hand
    ball: np.ndarray  # ball id
    flight: np.ndarray  # flight time, s
    catch_t: np.ndarray  # = t + flight
    release_lat: np.ndarray  # lateral offset (+ away from the other hand), m
    catch_lat: np.ndarray
    value: np.ndarray
    extra: dict = field(default_factory=dict)


def unroll(p: Pattern, horizon_s: float, lateral: dict | None = None) -> Unrolled:
    """Unroll pattern p from t = 0 to horizon_s, assigning ball ids and flight times.

    ``lateral`` overrides the module's LATERAL offsets ("in"/"out"/"mid" -> m, + away from the other hand)."""
    lat = LATERAL if lateral is None else lateral
    n_per = int(np.ceil(horizon_s / (p.period_beats * p.beat_s))) + 3
    ev = []  # (beat, hand, throw)
    for k in range(n_per):
        for th in p.throws:
            ev.append((th.beat + k * p.period_beats, th.hand, th))
    ev.sort(key=lambda e: (e[0], e[1]))
    index = {(b, h): i for i, (b, h, _) in enumerate(ev)}
    nxt = []
    for b, h, th in ev:
        to = 1 - h if th.cross else h
        nxt.append(index.get((b + th.value, to)))
    # Ball ids: an event that is nobody's "next" starts a new ball.
    has_prev = set(j for j in nxt if j is not None)
    ball = [-1] * len(ev)
    nb = 0
    for i in range(len(ev)):
        if i not in has_prev:
            ball[i] = nb
            nb += 1
        if nxt[i] is not None:
            ball[nxt[i]] = ball[i]
    assert nb == p.num_balls, (p.name, nb)
    rows = []
    for i, (b, h, th) in enumerate(ev):
        if nxt[i] is None:
            continue
        nt = ev[nxt[i]][2]
        t = b * p.beat_s
        flight = th.value * p.beat_s - nt.dwell
        assert flight > 0.05, (p.name, th, flight)
        rows.append(
            (
                t,
                h,
                1 - h if th.cross else h,
                ball[i],
                flight,
                lat[th.release],
                lat[th.catch],
                th.value,
                th.rise,
            )
        )
    a = np.array(rows, dtype=np.float64)
    a = a[a[:, 0] <= horizon_s + 1e-9]
    return Unrolled(
        p.name,
        a[:, 0],
        a[:, 1].astype(int),
        a[:, 2].astype(int),
        a[:, 3].astype(int),
        a[:, 4],
        a[:, 0] + a[:, 4],
        a[:, 5],
        a[:, 6],
        a[:, 7].astype(int),
        extra={"rise": a[:, 8]},
    )
