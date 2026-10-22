"""Draw the conceptual diagrams used by the RL note.

These are original vector-style figures rendered with matplotlib, because the
source papers contain mostly experiment curves rather than the explanatory
diagrams a beginner needs. Styling matches the note's blue/red palette.

Layout rule of thumb: at the chosen figsize, one x-unit is roughly 7 points, so
a CJK glyph of fontsize `fs` occupies about `fs / 7` x-units.
"""

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Match the note palette (assets/css/jekyll-theme-chirpy.scss).
BLUE = "#1f5da8"
RED = "#ad3038"
TINT = "#f5f8fc"
GREY = "#6b7280"
GREEN = "#2f7d4f"
INK = "#1f2937"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fc="white", ec=BLUE, tc=INK, fs=10, bold=False, lw=1.6):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0,rounding_size=2.0",
            linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2,
        )
    )
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fs, color=tc, zorder=3,
        fontweight="bold" if bold else "normal", linespacing=1.6,
    )


def arrow(ax, p0, p1, color=GREY, lw=1.6, ls="-", rad=0.0, zorder=1):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=13, linewidth=lw,
            color=color, linestyle=ls, zorder=zorder,
            connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2,
        )
    )


def title(ax, text, y=94, fs=12.5, color=BLUE):
    ax.text(50, y, text, ha="center", va="center", fontsize=fs, color=color, fontweight="bold")


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.2)
    plt.close(fig)
    print(f"{name}.png")




# --------------------------------------------------------- 3. PPO 四模型四阶段
def fig_ppo_pipeline(out):
    fig, ax = canvas(11.0, 5.6)
    title(ax, "PPO 的四个模型与四个阶段", y=94)

    models = [
        ("Policy Model\n(Actor)", "训练中更新", GREEN, TINT),
        ("Value Model\n(Critic)", "训练中更新", GREEN, TINT),
        ("Reference Model\n(冻结的 SFT)", "全程冻结", GREY, "#f3f4f6"),
        ("Reward Model\n(冻结的打分器)", "全程冻结", GREY, "#f3f4f6"),
    ]
    for i, (name, status, col, fc) in enumerate(models):
        x = 2 + i * 24.5
        box(ax, x, 70, 21, 15, name, fc=fc, ec=col, fs=9.5)
        ax.text(x + 10.5, 67, status, ha="center", va="top", fontsize=9, color=col, fontweight="bold")

    stages = [
        ("阶段 0\nReward Modeling\n(只做一次)", GREY),
        ("阶段 1\nRollout\nActor 生成回答", BLUE),
        ("阶段 2\nEvaluation\nReward 打分", BLUE),
        ("阶段 3\nOptimization\n更新 Actor + Critic", BLUE),
    ]
    for i, (name, col) in enumerate(stages):
        x = 2 + i * 24.5
        box(ax, x, 32, 21, 17, name, fc="white", ec=col, fs=9.5, tc=INK)

    for i in range(3):
        x0 = 2 + i * 24.5 + 21
        arrow(ax, (x0 + 0.5, 40.5), (x0 + 3.0, 40.5), color=GREY)

    # Feedback loop from stage 3 back to stage 1.
    arrow(ax, (87.5, 32), (87.5, 22), color=RED, lw=1.5)
    arrow(ax, (87.5, 22), (12.5, 22), color=RED, lw=1.5)
    arrow(ax, (12.5, 22), (12.5, 32), color=RED, lw=1.5)
    ax.text(50, 18.5, "回到阶段 1：同一批数据可以多轮更新", ha="center", va="center", fontsize=9.5, color=RED)

    ax.text(50, 6, "阶段 0 造自动打分器，阶段 3 多训一个 Critic —— 代价是显存和工程复杂度",
            ha="center", va="center", fontsize=9.5, color=GREY)
    save(fig, out, "ppo-pipeline")



# --------------------------------------------------------- 5. GRPO 组内比较
def fig_grpo_group(out):
    fig, ax = canvas(10.6, 5.2)
    title(ax, "GRPO：同一道题采样一组回答，用组内平均当基线", y=95)

    box(ax, 2, 52, 14, 14, "问题 q", fc=TINT, ec=BLUE, fs=11)
    arrow(ax, (16, 59), (22, 59), color=GREY, lw=1.7)

    data = [
        ("o1", "答对", "1.0", "+1.00", GREEN),
        ("o2", "答错", "0.0", "−1.00", RED),
        ("o3", "答对", "1.0", "+1.00", GREEN),
        ("o4", "答错", "0.0", "−1.00", RED),
    ]
    for i, (name, verdict, r, adv, col) in enumerate(data):
        y = 76 - i * 12
        box(ax, 23, y, 10, 10, name, fc="white", ec=BLUE, fs=10)
        ax.text(35.5, y + 5, verdict, ha="left", va="center", fontsize=10, color=col)
        ax.text(45, y + 5, f"奖励 {r}", ha="left", va="center", fontsize=9.5, color=GREY)
        ax.text(62, y + 5, f"优势 {adv}", ha="left", va="center", fontsize=10, color=col, fontweight="bold")

    ax.text(81, 59, "mean = 0.5\nstd = 0.5", ha="left", va="center", fontsize=9.5, color=GREY, linespacing=1.7)
    box(ax, 23, 28, 50, 11, "组内标准化：Â = (r − mean) / std", fc=TINT, ec=BLUE, fs=10.5)

    ax.text(50, 15, "同一条回答里所有 token 共享同一个优势值（奖励是整段给的）",
            ha="center", va="center", fontsize=10, color=RED)
    ax.text(50, 4, "全对或全错时 std = 0 → 优势全为 0 → 这道题不产生梯度",
            ha="center", va="center", fontsize=9.5, color=GREY)
    save(fig, out, "grpo-group")


# --------------------------------------------------- 6. DAPO Clip-Higher
def fig_clip_higher(out):
    """Clipping bounds the importance ratio, not the token probability.

    For an old probability p the ratio ceiling 1+eps would allow p*(1+eps), but
    a probability cannot exceed 1. With p=0.9 the ratio can only reach 1/0.9 =
    1.11 before the probability saturates, so a 1.2 ceiling never binds; with
    p=0.01 the ceiling binds immediately and the gain is negligible.
    """
    fig, ax = canvas(10.8, 5.4)
    title(ax, "DAPO Clip-Higher：卡住的是比率，不是概率", y=95)

    S = 18.0  # x-units per unit of probability
    ax.text(13, 84, "旧概率 π_old", ha="center", fontsize=10.5, color=GREY)
    ax.text(62, 84, "比率到 1+ε 时的概率  π_old × (1+ε)", ha="center", fontsize=10.5, color=GREY)

    rows = [
        ("高概率 token", 0.90, 1.08, BLUE, "比率最多 1/0.9 ≈ 1.11，碰不到 1.2 → 上界形同虚设"),
        ("低概率 token", 0.01, 0.012, RED, "一碰 1.2 就封顶 → 0.012，约等于没涨"),
    ]
    for i, (name, old, want, col, note) in enumerate(rows):
        y = 64 - i * 28
        ax.text(2, y + 9, name, ha="left", va="center", fontsize=10, color=INK, fontweight="bold")

        # Old probability bar.
        ax.add_patch(plt.Rectangle((2, y - 3), max(old * S, 0.6), 7,
                                   facecolor=col, alpha=0.35, edgecolor=col, lw=1.2))
        ax.text(2 + max(old * S, 0.6) + 1.2, y + 0.5, f"{old:g}",
                ha="left", va="center", fontsize=9.5, color=col)

        # Probability the ratio ceiling would allow (capped at 1).
        draw = min(want, 1.0)
        ax.add_patch(plt.Rectangle((50, y - 3), max(draw * S, 0.6), 7,
                                   facecolor=col, alpha=0.8, edgecolor=col, lw=1.2))
        if want > 1.0:
            # The part above 1 is impossible; show it as a dashed ghost.
            ax.add_patch(plt.Rectangle((50 + S, y - 3), (want - 1.0) * S, 7,
                                       facecolor="none", edgecolor=col, lw=1.2, ls=(0, (3, 2))))
            ax.text(50 + want * S + 1.2, y + 0.5, f"{want:g} > 1 → 封顶 1",
                    ha="left", va="center", fontsize=9.5, color=col)
        else:
            ax.text(50 + max(draw * S, 0.6) + 1.2, y + 0.5, f"{want:g}",
                    ha="left", va="center", fontsize=9.5, color=col)

        ax.text(50, y - 7, note, ha="left", va="top", fontsize=9, color=col)

    ax.text(50, 11, "→ 探索型 token 涨不上来，熵迅速掉到 0。放宽上界（0.2 → 0.28）即可缓解",
            ha="center", va="center", fontsize=9.5, color=GREEN)
    save(fig, out, "clip-higher")


# ------------------------------------------------- 7. GSPO token vs 序列
def fig_gspo_levels(out):
    fig, ax = canvas(10.8, 4.8)
    title(ax, "GRPO 与 GSPO 的差别：比率算在 token 级还是序列级", y=93)

    # GRPO: per-token ratios.
    ax.text(25, 79, "GRPO：每个 token 一个比率", ha="center", fontsize=11, color=RED, fontweight="bold")
    for i in range(5):
        x = 6 + i * 8
        box(ax, x, 60, 6.5, 10, f"y{i+1}", fc="white", ec=RED, fs=9.5)
        ax.text(x + 3.25, 55, f"r{i+1}", ha="center", va="center", fontsize=9.5, color=RED)
    ax.text(25, 45, "每个 next-token 分布只采 1 个样本\n重要性权重起不到修正作用",
            ha="center", va="top", fontsize=9.5, color=GREY, linespacing=1.7)

    # GSPO: one sequence ratio.
    ax.text(75, 79, "GSPO：整段序列一个比率", ha="center", fontsize=11, color=GREEN, fontweight="bold")
    for i in range(5):
        x = 56 + i * 8
        box(ax, x, 60, 6.5, 10, f"y{i+1}", fc="white", ec=GREEN, fs=9.5)
    ax.plot([54.5, 99], [55, 55], color=GREEN, lw=2.2)
    ax.plot([76.5, 76.5], [55, 50], color=GREEN, lw=2.2)
    ax.text(76.5, 45, "s = (πθ(y) / π_old(y)) ^ (1/|y|)", ha="center", va="top", fontsize=10, color=GREEN)

    ax.text(50, 26, "原则：优化目标的单元，应该和奖励的单元一致", ha="center", va="center",
            fontsize=11, color=BLUE, fontweight="bold")
    ax.text(50, 14, "奖励是整段给的 → 比率也应该按整段算", ha="center", va="center", fontsize=9.5, color=GREY)
    ax.text(50, 5, "GSPO 顺带解决了 MoE 训练必须依赖 Routing Replay 的问题",
            ha="center", va="center", fontsize=9.5, color=GREY)
    save(fig, out, "gspo-levels")


FIGURES = {
    "ppo-pipeline": (fig_ppo_pipeline, "PPO 的四个模型（含是否更新）与四个阶段（阶段 0 一次，阶段 1–3 循环）。"),
    "grpo-group": (fig_grpo_group, "GRPO 组内比较：同一道题 4 条回答的奖励标准化为优势。"),
    "clip-higher": (fig_clip_higher, "DAPO Clip-Higher：同一乘数下高/低概率 token 的上升空间对比。"),
    "gspo-levels": (fig_gspo_levels, "GRPO（token 级比率）与 GSPO（序列级比率）的对比。"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=pathlib.Path)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--sources", type=pathlib.Path, default=None,
                        help="existing sources.json to merge the diagram records into")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    drawn = []
    for name, (fn, note) in FIGURES.items():
        if args.only and name not in args.only:
            continue
        fn(args.output_dir)
        with Image.open(args.output_dir / f"{name}.png") as im:
            size = list(im.size)
        drawn.append({
            "asset": f"{name}.webp",
            "origin": "original diagram drawn for this article (tools/rl-diagrams.py)",
            "pixels": size,
            "note": note,
        })

    if args.sources and args.sources.exists():
        records = json.loads(args.sources.read_text(encoding="utf-8"))
        records = [r for r in records if r.get("origin") is None] + drawn
        args.sources.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        print(f"merged {len(drawn)} diagram records into {args.sources.name}")


if __name__ == "__main__":
    main()
