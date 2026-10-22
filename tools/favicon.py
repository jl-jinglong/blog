"""Generate the site favicon set: a dark, sci-fi mark.

The theme ships a cartoon ant, so this replaces the whole set under
assets/img/favicons/ (site assets override the gem's copies). The mark is a
crisp hexagonal "core" on a near-black rounded square, which stays legible down
to 16px where anything detailed turns to mush.
"""

import math
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MASTER = 2048                      # supersampled canvas, downscaled at the end
CX = CY = 512.0                    # geometry is authored in a 1024 grid
R_HEX = 296.0
STROKE = 62.0
R_CORE = 88.0
CORNER = 224.0

BG_TOP = (16, 26, 40)              # #101a28
BG_BOT = (4, 6, 10)                # #04060a
CYAN = (74, 226, 255)              # #4ae2ff
CORE = (240, 253, 255)


def gradient(size, top, bot):
    """Diagonal gradient as an RGBA array."""
    y, x = np.mgrid[0:size, 0:size]
    t = ((x + y) / (2.0 * (size - 1))).astype(np.float32)
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(3):
        arr[..., i] = (top[i] + (bot[i] - top[i]) * t).astype(np.uint8)
    return Image.fromarray(arr, "RGB").convert("RGBA")


def hex_points(cx, cy, r):
    """Pointy-top hexagon vertices."""
    return [(cx + r * math.cos(math.radians(-90 + 60 * k)),
             cy + r * math.sin(math.radians(-90 + 60 * k))) for k in range(6)]


def draw_glyph(size, scale, rounded=True):
    """Render the mark at `size`; `scale` maps the 1024 grid onto it."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    # --- background
    bg = gradient(size, BG_TOP, BG_BOT)
    mask = Image.new("L", (size, size), 0)
    md = ImageDraw.Draw(mask)
    if rounded:
        md.rounded_rectangle([0, 0, size - 1, size - 1], radius=CORNER * scale, fill=255)
    else:
        md.rectangle([0, 0, size - 1, size - 1], fill=255)
    img.paste(bg, (0, 0), mask)

    # --- glyph layer (drawn bright, then blurred for the glow)
    glyph = Image.new("L", (size, size), 0)
    gd = ImageDraw.Draw(glyph)
    pts = hex_points(CX * scale, CY * scale, R_HEX * scale)
    gd.line(pts + [pts[0]], fill=255, width=int(STROKE * scale), joint="curve")
    r = R_CORE * scale
    gd.ellipse([CX * scale - r, CY * scale - r, CX * scale + r, CY * scale + r], fill=255)

    glow = glyph.filter(ImageFilter.GaussianBlur(max(1.0, 34 * scale)))
    glow_img = Image.new("RGBA", (size, size), CYAN + (0,))
    glow_img.putalpha(glow.point(lambda v: int(v * 0.85)))
    img = Image.alpha_composite(img, glow_img)

    # --- crisp mark on top (cyan ring, near-white core)
    ring = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    rd = ImageDraw.Draw(ring)
    rd.line(pts + [pts[0]], fill=CYAN + (255,), width=int(STROKE * scale), joint="curve")
    core = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    cd = ImageDraw.Draw(core)
    cd.ellipse([CX * scale - r, CY * scale - r, CX * scale + r, CY * scale + r], fill=CORE + (255,))
    img = Image.alpha_composite(img, ring)
    img = Image.alpha_composite(img, core)
    return img


def svg() -> str:
    pts = hex_points(CX, CY, R_HEX)
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" role="img" aria-label="JLong">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#101a28"/>
      <stop offset="1" stop-color="#04060a"/>
    </linearGradient>
    <radialGradient id="glow" cx="50%" cy="50%" r="50%">
      <stop offset="0" stop-color="#4ae2ff" stop-opacity="0.42"/>
      <stop offset="1" stop-color="#4ae2ff" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect width="1024" height="1024" rx="{CORNER:g}" fill="url(#bg)"/>
  <circle cx="512" cy="512" r="430" fill="url(#glow)"/>
  <polygon points="{poly}" fill="none" stroke="#4ae2ff" stroke-width="{STROKE:g}"
           stroke-linejoin="round"/>
  <circle cx="512" cy="512" r="{R_CORE:g}" fill="#f0fdff"/>
</svg>
"""


def main(out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    master = draw_glyph(MASTER, MASTER / 1024.0, rounded=True)
    square = draw_glyph(MASTER, MASTER / 1024.0, rounded=False)

    def save(img, name, size):
        img.resize((size, size), Image.Resampling.LANCZOS).save(out / name)
        print(f"  {name:34} {size}x{size}")

    save(master, "favicon-96x96.png", 96)
    save(master, "apple-touch-icon.png", 180)
    save(square, "web-app-manifest-192x192.png", 192)
    save(square, "web-app-manifest-512x512.png", 512)

    ico = master.resize((256, 256), Image.Resampling.LANCZOS)
    ico.save(out / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (256, 256)])
    print("  favicon.ico                        multi-size")

    (out / "favicon.svg").write_text(svg(), encoding="utf-8")
    print("  favicon.svg                        vector")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
