"""Make a floating-base MJCF (six base joints + six base position actuators) from a menagerie fixed-base hand XML.

Usage: python make_floating_xml.py <leap|shadow> <left|right> <in.xml> <out.xml>

Applies exactly the edits that produced the repo's existing floating files (leap right_hand_floating.xml,
shadow_hand left_hand_floating.xml), so every floating hand has the same base: slides base_x/y/z (kp 300, kv 40, +-200 N),
hinges base_roll/pitch/yaw (kp 60, kv 8, +-80 Nm) on the root body, same names and the same ``base_slide`` / ``base_hinge``
default classes. Every edit is anchored on exact text and asserted (it fails instead of silently producing a different model).
Used for: leap left, shadow right (their fixed menagerie models are the source; no mirroring).
"""

from __future__ import annotations

import re
import sys

SLIDE = (
    ' damping="3.0" armature="0.02" frictionloss="0.0"',
    'kp="300.0" kv="40.0"',
    'forcerange="-200 200"',
)
BASE_JOINTS = (
    ("base_x", "slide", "1 0 0"),
    ("base_y", "slide", "0 1 0"),
    ("base_z", "slide", "0 0 1"),
    ("base_roll", "hinge", "1 0 0"),
    ("base_pitch", "hinge", "0 1 0"),
    ("base_yaw", "hinge", "0 0 1"),
)


def _sub1(pattern: str, repl: str, text: str, flags: int = 0) -> str:
    out, n = re.subn(pattern, repl, text, count=1, flags=flags)
    assert n == 1, f"anchor not found: {pattern!r}"
    return out


def convert(hand: str, text: str) -> str:
    leap = hand == "leap"
    slide_rng = 'range="-0.50 0.50"' if leap else 'range="-0.30 0.30"'
    ctrl_slide = "-0.50 0.50" if leap else "-0.30 0.30"
    hinge_lim = "-1.5708 1.5708" if leap else "-0.7853 0.7853"
    defaults = (
        '    <default class="base_slide">\n'
        f'      <joint type="slide" limited="true" damping="3.0" armature="0.02" frictionloss="0.0"/>\n'
        f'      <position kp="300.0" kv="40.0" ctrlrange="{ctrl_slide}" forcerange="-200 200"/>\n'
        "    </default>\n"
        '    <default class="base_hinge">\n'
        f'      <joint type="hinge" limited="true" damping="0.5" armature="0.01" frictionloss="0.0"/>\n'
        f'      <position kp="60.0" kv="8.0" ctrlrange="-0.7853 0.7853" forcerange="-80 80"/>\n'
        "    </default>\n"
    )
    if leap:
        # inside the top-level <default>, after the generic joint default
        text = _sub1(
            r'(<joint damping="0\.03" frictionloss="0\.001"\s*/>\s*\n)',
            lambda m: m.group(1) + defaults,
            text,
        )
        root = r'(<body name="palm"[^>]*>\s*\n)'
    else:
        # inside the hand default class (left_hand / right_hand), after the generic position default
        defaults = defaults.replace("    <default", "      <default").replace(
            "    </default", "      </default"
        )
        defaults = defaults.replace(
            '      <joint type="slide" limited="true"',
            '        <joint type="slide" limited="true" ' + slide_rng,
        )
        defaults = defaults.replace(
            '      <joint type="hinge" limited="true"',
            '        <joint type="hinge" limited="true" range="-0.7853 0.7853"',
        )
        defaults = defaults.replace("      <position", "        <position")
        text = _sub1(
            r'(<position forcerange="-1 1"\s*/>\s*\n)', lambda m: m.group(1) + defaults, text
        )
        root = r'(<body name="[lr]h_forearm"[^>]*>\s*\n)'
    joints = (
        "".join(
            f'      <joint name="{n}" class="base_{"slide" if t == "slide" else "hinge"}" axis="{a}"'
            + (
                f" {slide_rng if leap else ''}"
                if (t == "slide" and leap)
                else (f' range="{hinge_lim}"' if leap else "")
            )
            + "/>\n"
            for n, t, a in BASE_JOINTS
        )
        + "\n"
    )
    text = _sub1(root, lambda m: m.group(1) + joints, text)
    acts = (
        "".join(
            f'    <position name="{n}_act" joint="{n}" class="base_{"slide" if t == "slide" else "hinge"}"/>\n'
            for n, t, _ in BASE_JOINTS
        )
        + "\n"
    )
    text = _sub1(r"(<actuator>\s*\n)", lambda m: m.group(1) + acts, text)
    # velocity sensors on the finger joints are not used (the sim reads joint state directly)
    text = re.sub(r"[ \t]*<jointvel [^>]*/>[ \t]*\n", "", text)
    return text


def main():
    hand, side, src, dst = sys.argv[1:5]
    assert hand in ("leap", "shadow") and side in ("left", "right")
    text = open(src).read()
    out = convert(hand, text)
    for n, _, _ in BASE_JOINTS:
        assert f'name="{n}"' in out and f'name="{n}_act"' in out, n
    open(dst, "w").write(out)
    print(f"wrote {dst} ({len(out)} bytes) from {src}")


if __name__ == "__main__":
    main()
