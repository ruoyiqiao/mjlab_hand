"""Turn rendered RGBA frames (render_blender.py) into a looping animated WebP on a white background.

Usage: python make_webp.py <frames_dir> <out.webp> --fps 25 [--width 960] [--quality 80] [--speed 1.0]
"""

from __future__ import annotations

import argparse
import glob

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("out")
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--width", type=int, default=960, help="0 = keep the rendered width")
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed factor (>1 = faster)")
    a = ap.parse_args()
    files = sorted(glob.glob(f"{a.frames}/frame_*.png"))
    out = []
    for f in files:
        im = Image.open(f).convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im).convert("RGB")
        if a.width > 0 and im.width != a.width:
            im = im.resize((a.width, round(im.height * a.width / im.width)), Image.LANCZOS)
        out.append(im)
    ms = round(1000.0 / (a.fps * a.speed))
    out[0].save(
        a.out,
        save_all=True,
        append_images=out[1:],
        duration=ms,
        loop=0,
        quality=a.quality,
        method=6,
    )
    print(f"{a.out}: {len(out)} frames, {ms} ms/frame")


if __name__ == "__main__":
    main()
