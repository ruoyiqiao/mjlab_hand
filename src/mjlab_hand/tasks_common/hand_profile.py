"""Per-hand description for the shared task layer (single-hand, dual-hand, juggling tasks).

A :class:`HandProfile` bundles what the task code needs to know about one floating hand model:
names (finger joints in canonical order, tips, palm, root body, thumb), and the measured frame data
in ``hand_params/frames.json`` (written by ``tools/allegro4/hand_frame_probe.py``): anchor quaternions that
put the palm down / up with the fingers toward +x / -x, the palm origin in the attachment frame, and the
canonical base-joint axes. After the canonical-axis fix every hand's base joints act like Allegro's:
``base_x / base_roll`` about hand-frame -up, ``base_y / base_pitch`` about lat, ``base_z / base_yaw`` about the
finger direction, hinge pivots at the palm-body origin, so one set of task layouts serves all hands. Allegro's
own model already is canonical (no fix applied, its configs stay bit-identical).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from mjlab_hand.robot.hand_spec import HandSpec

_FRAMES = json.loads((Path(__file__).parent / "hand_params" / "frames.json").read_text())

ALLEGRO_FINGERS = tuple(f"{f}j{i}" for f in ("ff", "mf", "rf", "th") for i in range(4))
PALM_DOWN_FINGERS_PX = (0.0, 1.0, 0.0, 0.0)
PALM_DOWN_FINGERS_NX = (0.0, 0.0, 1.0, 0.0)


def _rotate(
    q: tuple[float, float, float, float], v: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Rotate v by the unit quaternion q (wxyz)."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    vx, vy, vz = v
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


@dataclass(frozen=True)
class HandProfile:
    hand: str
    side: str
    """Chirality of the loaded model (may differ from the requested side when only one exists)."""
    spec: HandSpec
    finger_joints: tuple[str, ...]
    """Actuated finger joints in canonical order (the policy's finger action order)."""
    thumb_tip: int
    """Index of the thumb in ``spec.keypoint_bodies``."""
    thumb_geom: str
    """Substring identifying the thumb tip collision geom (contact-sensor column lookup)."""
    root_body: str
    """Entity root body; hand-subtree contact sensors use it, so the whole hand is covered."""
    canonical_axes: bool
    """True when the model's base joints need the canonical-axis fix (everything except Allegro)."""
    tendon_targets: tuple[str, ...] = ()
    """Tendon-length action targets (Shadow: J2 + J1 coupling of ff/mf/rf/lf), appended after the finger joints."""
    fixed_joints: tuple[tuple[str, float], ...] = ()
    """Actuated joints held at a fixed target and excluded from the action (Shadow wrist, redundant with the base)."""
    coupled_joints: tuple[str, ...] = ()
    """Joints moved only through a tendon: observed, not actuated."""
    match_base_dynamics: bool = False
    """Give the floating root Allegro's closed-loop dynamics: base joint armature/damping/friction set to Allegro's and
    every base gain (the XML default or a task's Allegro-tuned gains) scaled by I_hand / I_allegro per joint, so the
    natural frequency, damping ratio and gravity sag equal Allegro's. Set where the shipped floating base deviates
    strongly (tools/allegro4/base_sag_probe.py, 2026-11-17): shadow (3.8 kg, kp 300: sag 11 cm vs 2.1 cm),
    sharpa (slide armature 10, hinge kp 10: hinge w_n 9.5 vs 69 rad/s), xhand (armature 10, kp 1000/100: w_n 9.5 / 30).
    LEAP / Wuji are within 25 % of Allegro's w_n and keep their XML base."""

    @property
    def key(self) -> str:
        return f"{self.hand}_{self.side}"

    @property
    def palm_body(self) -> str:
        return self.spec.palm_body

    @property
    def tip_bodies(self) -> tuple[str, ...]:
        return tuple(self.spec.keypoint_bodies)

    @property
    def tip_geoms(self) -> str:
        return self.spec.keypoint_geom_pattern

    @property
    def n_tips(self) -> int:
        return len(self.spec.keypoint_bodies)

    @property
    def n_fingers(self) -> int:
        """Finger action dimensions per hand: joint targets plus tendon targets."""
        return len(self.finger_joints) + len(self.tendon_targets)

    @property
    def obs_joints(self) -> tuple[str, ...]:
        """Finger joints in the joint observations / reset noise / velocity penalty (actuated + tendon-coupled)."""
        return tuple(self.finger_joints) + tuple(self.coupled_joints)

    @property
    def frames(self) -> dict:
        return _FRAMES[self.key]

    def anchor_rot(
        self, down: bool = True, fingers_px: bool = True
    ) -> tuple[float, float, float, float]:
        """Anchor quaternion (wxyz): palm facing down (up) with the fingers toward +x (-x)."""
        if self.hand == "allegro" and down:
            return PALM_DOWN_FINGERS_PX if fingers_px else PALM_DOWN_FINGERS_NX
        fr = self.frames
        if down:
            return tuple(fr["anchor_down_px" if fingers_px else "anchor_down_nx"])  # type: ignore[return-value]
        if not fingers_px:
            raise NotImplementedError("palm-up with fingers toward -x is not measured")
        return tuple(fr["anchor_up_px"])  # type: ignore[return-value]

    def anchor_pos(
        self, palm_pos: tuple[float, float, float], rot: tuple[float, float, float, float]
    ) -> tuple[float, float, float]:
        """Attachment-frame origin that puts the palm-body origin at ``palm_pos`` for anchor rotation ``rot``."""
        o = self.frames["palm_origin_attach"]
        if not any(o):
            return tuple(palm_pos)  # type: ignore[return-value]
        r = _rotate(rot, tuple(o))  # type: ignore[arg-type]
        return tuple(float(p - d) for p, d in zip(palm_pos, r))  # type: ignore[return-value]

    @property
    def base_fix(self) -> dict | None:
        """Per base joint: canonical axis / pivot in the joint's body frame (None for Allegro)."""
        return self.frames["base_joint_fix"] if self.canonical_axes else None


def _p(
    hand, side, spec, joints, thumb_tip, thumb_geom, root, canonical=True, match_base=False
) -> HandProfile:
    return HandProfile(
        hand=hand,
        side=side,
        spec=spec,
        finger_joints=tuple(joints),
        thumb_tip=thumb_tip,
        thumb_geom=thumb_geom,
        root_body=root,
        canonical_axes=canonical,
        match_base_dynamics=match_base,
    )


def _build() -> dict[str, HandProfile]:
    from mjlab_hand.robot.allegro import ALLEGRO_HAND_LEFT, ALLEGRO_HAND_RIGHT
    from mjlab_hand.robot.leap import LEAP_HAND, LEAP_HAND_LEFT
    from mjlab_hand.robot.shadow_hand import SHADOW_HAND_LEFT, SHADOW_HAND_RIGHT
    from mjlab_hand.robot.shadow_hand.shadow_constants import (
        SHADOW_COUPLED_JOINTS,
        SHADOW_FINGER_JOINT_TARGETS,
        SHADOW_TENDON_TARGETS,
        SHADOW_WRIST_JOINTS,
        for_side,
    )
    from mjlab_hand.robot.sharpa_wave import get_sharpa_hand_spec
    from mjlab_hand.robot.wuji import WUJI_HAND
    from mjlab_hand.robot.xhand import XHAND_HAND

    def fingers4(names):
        return tuple(f"{f}_{j}" for f in names for j in ("mcp", "rot", "pip", "dip"))

    leap = fingers4(("if", "mf", "rf")) + ("th_cmc", "th_axl", "th_mcp", "th_ipl")
    wuji = tuple(f"right_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5))

    def sharpa(side):
        j = [f"{side}_thumb_{n}" for n in ("CMC_FE", "CMC_AA", "MCP_FE", "MCP_AA", "IP")]
        for f in ("index", "middle", "ring"):
            j += [f"{side}_{f}_{n}" for n in ("MCP_FE", "MCP_AA", "PIP", "DIP")]
        j += [f"{side}_pinky_CMC"] + [
            f"{side}_pinky_{n}" for n in ("MCP_FE", "MCP_AA", "PIP", "DIP")
        ]
        return tuple(j)

    xhand = (
        "index_bend_joint",
        "index_joint1",
        "index_joint2",
        "mid_joint1",
        "mid_joint2",
        "pinky_joint1",
        "pinky_joint2",
        "ring_joint1",
        "ring_joint2",
        "thumb_bend_joint",
        "thumb_rota_joint1",
        "thumb_rota_joint2",
    )
    xhand = tuple(f"left_hand_{j}" for j in xhand)

    def shadow(side, spec):
        return HandProfile(
            hand="shadow",
            side=side,
            spec=spec,
            finger_joints=for_side(SHADOW_FINGER_JOINT_TARGETS, side),
            thumb_tip=4,
            thumb_geom="thdistal",
            root_body=for_side("lh_forearm", side),
            canonical_axes=True,
            tendon_targets=for_side(SHADOW_TENDON_TARGETS, side),
            fixed_joints=tuple((j, 0.0) for j in for_side(SHADOW_WRIST_JOINTS, side)),
            coupled_joints=for_side(SHADOW_COUPLED_JOINTS, side),
            match_base_dynamics=True,
        )

    profiles = [
        _p(
            "allegro",
            "left",
            ALLEGRO_HAND_LEFT,
            ALLEGRO_FINGERS,
            3,
            "th_tip",
            "palm",
            canonical=False,
        ),
        _p(
            "allegro",
            "right",
            ALLEGRO_HAND_RIGHT,
            ALLEGRO_FINGERS,
            3,
            "th_tip",
            "palm",
            canonical=False,
        ),
        _p("leap", "right", LEAP_HAND, leap, 3, "th_tip", "palm"),
        _p("leap", "left", LEAP_HAND_LEFT, leap, 3, "th_tip", "palm"),
        shadow("left", SHADOW_HAND_LEFT),
        shadow("right", SHADOW_HAND_RIGHT),
        _p("wuji", "right", WUJI_HAND, wuji, 0, "right_finger1", "right_palm_link"),
        _p(
            "sharpa",
            "left",
            get_sharpa_hand_spec("left"),
            sharpa("left"),
            0,
            "left_thumb",
            "left_sharpa_wave_with_float_base",
            match_base=True,
        ),
        _p(
            "sharpa",
            "right",
            get_sharpa_hand_spec("right"),
            sharpa("right"),
            0,
            "right_thumb",
            "right_sharpa_wave_with_float_base",
            match_base=True,
        ),
        _p(
            "xhand", "left", XHAND_HAND, xhand, 4, "thumb_rota", "xhand_float_base", match_base=True
        ),
    ]
    return {p.key: p for p in profiles}


_PROFILES: dict[str, HandProfile] | None = None
DEFAULT_SIDE = {
    "allegro": "left",
    "leap": "right",
    "wuji": "right",
    "sharpa": "left",
    "xhand": "left",
    "shadow": "left",
}


def hand_profiles() -> dict[str, HandProfile]:
    global _PROFILES
    if _PROFILES is None:
        _PROFILES = _build()
    return _PROFILES


def get_profile(hand: str, side: str | None = None) -> HandProfile:
    """Profile of ``hand``; ``side`` defaults to the hand's default side, and falls back to the side that exists."""
    ps = hand_profiles()
    side = side or DEFAULT_SIDE[hand]
    if f"{hand}_{side}" in ps:
        return ps[f"{hand}_{side}"]
    other = "right" if side == "left" else "left"
    if f"{hand}_{other}" in ps:
        return ps[f"{hand}_{other}"]
    raise KeyError(f"no floating model for hand {hand!r}")


def profile_by_key(key: str) -> HandProfile:
    """Profile from its ``"<hand>_<side>"`` key (what task configs store)."""
    return hand_profiles()[key]


__all__ = [
    "ALLEGRO_FINGERS",
    "DEFAULT_SIDE",
    "HandProfile",
    "get_profile",
    "hand_profiles",
    "profile_by_key",
]
