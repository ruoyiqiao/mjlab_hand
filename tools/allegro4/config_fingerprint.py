"""Fingerprint the registered task configs, to prove a refactor leaves them unchanged.

Usage: python config_fingerprint.py [--match REGEX] [--out fp.json] [--compare other.json]

For every registered task id matching ``--match`` (default: all ids containing "Allegro") hashes
repr(train env cfg), repr(play env cfg) and repr(rl cfg); memory addresses in reprs are masked.
``--mask`` removes fields that a refactor added with Allegro defaults (e.g. ', hand_profiles=\\([^)]*\\)').
``--compare`` prints the ids whose hashes differ from a previous run (exit status 1 if any).
CPU only; builds configs, does not create environments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys

_ADDR = re.compile(r"0x[0-9a-fA-F]+")
_MASKS: list[re.Pattern] = []


def _h(obj) -> str:
    text = _ADDR.sub("0x?", repr(obj))
    for m in _MASKS:
        text = m.sub("", text)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="Allegro")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", default=None)
    ap.add_argument(
        "--mask",
        action="append",
        default=[],
        help="regex removed from every repr before hashing (fields added by a refactor with default values)",
    )
    a = ap.parse_args()
    _MASKS.extend(re.compile(m) for m in a.mask)

    from mjlab_hand.registration import register_tasks

    register_tasks()
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

    rx = re.compile(a.match)
    out = {}
    for tid in sorted(t for t in list_tasks() if rx.search(t)):
        out[tid] = {
            "train": _h(load_env_cfg(tid, play=False)),
            "play": _h(load_env_cfg(tid, play=True)),
            "rl": _h(load_rl_cfg(tid)),
        }
    print(f"fingerprinted {len(out)} task ids")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, sort_keys=True)
    if a.compare:
        ref = json.load(open(a.compare))
        bad = sorted(t for t in ref if ref.get(t) != out.get(t))
        new = sorted(t for t in out if t not in ref)
        for t in bad:
            print("DIFF", t, ref.get(t), "->", out.get(t))
        if new:
            print(
                f"{len(new)} new ids (not in the reference): {', '.join(new[:6])}{' ...' if len(new) > 6 else ''}"
            )
        print(
            f"IDENTICAL on all {len(ref)} reference ids"
            if not bad
            else f"{len(bad)} reference ids differ"
        )
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
