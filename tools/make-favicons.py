"""Regenerate the blog's favicon set from a single photo.

The theme (jekyll-theme-chirpy) links, in order: favicon-96x96.png, favicon.svg,
favicon.ico, apple-touch-icon.png, plus the two web-app-manifest PNGs. Browsers
prefer the SVG, so it has to be regenerated too — otherwise the old vector logo
stays visible.

A wide starfield averages to near-black at 16-32px, so the crop is centred on the
brightest star instead of the frame centre; that keeps the icon recognisable in
a tab.

Run:  python tools/make-favicons.py <source-image>
"""

import base64
import io
import pathlib
import sys

from PIL import Image, ImageFilter

sys.reconfigure = getattr(sys.stdout, "reconfigure", None)
if sys.reconfigure:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MASTER = 512          # master square every size is derived from
CROP_FRAC = 0.34      # side length of the crop, as a fraction of the short edge


def brightest_square(im: Image.Image) -> Image.Image:
    """Crop a square centred on the most prominent bright blob."""
    w, h = im.size
    grey = im.convert("L").filter(ImageFilter.GaussianBlur(9))
    px = grey.load()
    _, cx, cy = max((px[x, y], x, y) for y in range(0, h, 3) for x in range(0, w, 3))
    side = int(min(w, h) * CROP_FRAC)
    x0 = min(max(cx - side // 2, 0), w - side)
    y0 = min(max(cy - side // 2, 0), h - side)
    crop = im.crop((x0, y0, x0 + side, y0 + side))
    print(f"source {w}x{h}; brightest blob at ({cx},{cy}); crop {side}x{side} at ({x0},{y0})")
    return crop.resize((MASTER, MASTER), Image.Resampling.LANCZOS)


def main() -> None:
    repo = pathlib.Path(r"D:\Desktop\课件\申请\blog")
    fdir = repo / "assets/img/favicons"
    with Image.open(pathlib.Path(sys.argv[1])) as im:
        base = brightest_square(im.convert("RGB"))

    for name, size in {
        "favicon-96x96.png": 96,
        "apple-touch-icon.png": 180,
        "web-app-manifest-192x192.png": 192,
        "web-app-manifest-512x512.png": 512,
    }.items():
        base.resize((size, size), Image.Resampling.LANCZOS).save(fdir / name, "PNG", optimize=True)
        print(f"  {name:32} {size}x{size}")

    base.resize((256, 256), Image.Resampling.LANCZOS).save(
        fdir / "favicon.ico", format="ICO",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64)])
    print(f"  {'favicon.ico':32} 16/24/32/48/64")

    buf = io.BytesIO()
    base.resize((96, 96), Image.Resampling.LANCZOS).save(buf, "PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    (fdir / "favicon.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96" role="img" aria-label="JLong">\n'
        f'  <image width="96" height="96" href="data:image/png;base64,{b64}"/>\n'
        "</svg>\n",
        encoding="utf-8")
    print(f"  {'favicon.svg':32} 96x96 embedded")


if __name__ == "__main__":
    main()
