"""Render the card cover for the RL note (part 1).

The blog's other notes use a padded paper figure as the post-list card image,
but those figures turn into an unreadable grey mush at card size. This cover is
a deliberately simple concept diagram — three stages, three arrows — so the
topic still reads when scaled down to a ~380px-wide card.
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BLUE = "#1f5da8"
TINT = "#eef4fc"
GREY = "#6b7280"
INK = "#1f2937"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

W, H = 1059, 661


def render(out: pathlib.Path) -> None:
    """Optional alternative: a designed concept chain (kept for reference)."""
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    bw, bh, y = 26.0, 34.0, 33.0      # box size / bottom edge (vertically centred)
    xs = [3.0, 37.0, 71.0]
    cy = y + bh / 2

    nodes = [("策略梯度", "奖励不可导\n怎么还能训练", False),
             ("Actor-Critic", "基线从哪来", False),
             ("PPO", "数据复用\n+ 更新限速", True)]

    for x, (name, sub, solid) in zip(xs, nodes):
        ax.add_patch(FancyBboxPatch(
            (x, y), bw, bh,
            boxstyle="round,pad=0,rounding_size=2.8",
            linewidth=2.0, edgecolor=BLUE,
            facecolor=BLUE if solid else TINT, zorder=2))
        ax.text(x + bw / 2, cy + 5.4, name,
                ha="center", va="center", fontsize=18 if solid else 17,
                color="white" if solid else INK,
                fontweight="bold", zorder=3)
        ax.text(x + bw / 2, cy - 8.0, sub,
                ha="center", va="center", fontsize=9.5,
                color="#dce8f7" if solid else GREY,
                linespacing=1.6, zorder=3)

    for i in range(2):
        ax.add_patch(FancyArrowPatch(
            (xs[i] + bw + 0.9, cy), (xs[i + 1] - 0.9, cy),
            arrowstyle="-|>", mutation_scale=20, linewidth=2.4,
            color=BLUE, shrinkA=0, shrinkB=0, zorder=1))

    png = out / "rl-cover-alt.png"
    fig.savefig(png, dpi=100, facecolor="white")
    plt.close(fig)
    with Image.open(png) as im:
        im.convert("RGB").save(out / "rl-cover-alt.webp", quality=94, method=6)
        print(f"rl-cover-alt.webp {im.size}")
    png.unlink()


def pad_alt(out: pathlib.Path) -> None:
    """Card cover: the PPO pipeline diagram, padded to the blog's cover ratio.

    Chosen over the PPO-vs-GRPO figure because part 1 does not cover GRPO, and
    over a designed text graphic because type that small is illegible on a card.
    """
    src = out / "ppo-pipeline.webp"
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((979, 581), Image.Resampling.LANCZOS)
        cover = Image.new("RGB", (W, H), "white")
        cover.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
        cover.save(out / "rl-cover.webp", quality=94, method=6)
        print(f"rl-cover.webp {cover.size} (from ppo-pipeline)")


if __name__ == "__main__":
    d = pathlib.Path(sys.argv[1])
    render(d)
    pad_alt(d)
