"""논문용 method 그림 넷을 그린다. PNG·SVG·PDF 로 같이 낸다.

슬라이드에 붙일 때 PNG, PowerPoint/LaTeX 에 벡터로 넣을 때 SVG·PDF 를 쓴다.
라벨은 영어다 -- 논문 그림이고, 서버에 한글 폰트가 없어 깨질 일도 없앤다.

    python make_method_figures.py [출력 디렉터리]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures"
os.makedirs(OUT, exist_ok=True)

INK, MUTED, FAINT = "#1A1C21", "#585C64", "#8A8E96"
RULE, ACC = "#C4BFB2", "#1F4E5F"
RISK, STABLE = "#A33A22", "#3C6B4A"
ACC_S, RISK_S, STABLE_S = "#E2ECEE", "#F6E8E3", "#E6EFE8"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "text.color": INK, "axes.edgecolor": RULE,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})


def save(fig, name):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=300, transparent=True)
    plt.close(fig)
    print(f"  {name}.png / .svg / .pdf")


def box(ax, x, y, w, h, title, sub=None, face="white", edge=RULE, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor=face, edgecolor=edge, linewidth=lw, zorder=2))
    ax.text(x + 0.16, y + h - (0.34 if sub else h / 2 + 0.1), title,
            fontsize=9.5, va="top" if sub else "center", zorder=3)
    if sub:
        ax.text(x + 0.16, y + h - 0.78, sub, fontsize=8, color=MUTED, va="top", zorder=3)


def arrow(ax, p, q, color=RULE, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9,
                                 color=color, linewidth=1.0, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))


def blank(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_axis_off()
    return fig, ax


# ---------------------------------------------------------------- fig 1
def fig_pipeline():
    fig, ax = blank(10.4, 3.1)
    ax.set_xlim(0, 20); ax.set_ylim(0, 6)
    for x, t in ((0.2, "MEASURE"), (4.9, "DERIVE"), (9.4, "LABEL"),
                 (13.6, "MAP"), (17.4, "DISTIL")):
        ax.text(x, 5.72, t, fontsize=7.5, color=ACC, letterspacing=1.4)
    ax.plot([0.2, 19.8], [5.5, 5.5], color=RULE, lw=0.7, ls=(0, (1, 3)))

    box(ax, 0.2, 2.3, 4.1, 1.5, "Measured compression damage",
        "per-task eval / operator counts")
    arrow(ax, (4.3, 3.4), (4.9, 4.5), rad=-0.25)
    arrow(ax, (4.3, 2.7), (4.9, 1.6), rad=0.25)

    box(ax, 4.9, 4.0, 4.6, 1.1, "Per-task ceiling and floor", face=ACC_S, edge=ACC)
    ax.plot([9.5, 16.6], [4.55, 4.55], color=ACC, lw=1.0, ls=(0, (4, 3)))
    arrow(ax, (16.6, 4.55), (17.2, 4.55), color=ACC)

    box(ax, 4.9, 1.0, 4.6, 1.1, "5 checks, signs, weights")
    arrow(ax, (9.5, 1.55), (10.0, 1.55))
    box(ax, 10.0, 1.0, 3.3, 1.1, "VLM writes grades 1-5")
    arrow(ax, (13.3, 1.55), (13.8, 1.55))
    box(ax, 13.8, 1.0, 2.8, 1.1, "confidence 0-1")
    arrow(ax, (16.6, 1.55), (17.2, 3.6), rad=-0.3)

    box(ax, 17.2, 3.1, 2.6, 1.5, "Playback ratio", "position inside the band",
        face=ACC_S, edge=ACC)
    box(ax, 17.2, 1.0, 2.6, 1.1, "Student gate")
    arrow(ax, (18.5, 3.1), (18.5, 2.1))

    ax.text(9.7, 3.55, "the ceiling comes from measurement", fontsize=8, color=ACC)
    ax.text(9.7, 0.45, "the checks decide how much of it to use", fontsize=8, color=MUTED)
    save(fig, "fig1_pipeline")


# ---------------------------------------------------------------- fig 2
def fig_pools():
    dmg = [-.28, -.22, -.18, -.18, -.16, -.16, -.14, -.12, -.10, -.10, -.08, -.08,
           -.06, -.02, -.02, 0, 0, 0, .02, .04, .04, .08, .14, .18]
    fig, ax = blank(10.4, 3.0)
    ax.set_xlim(-.325, .225); ax.set_ylim(-1.35, 1.15)
    ax.axhspan(-.05, .62, xmin=0, xmax=(-.155 + .325) / .55, color=RISK_S, zorder=0)
    ax.axhspan(-.05, .62, xmin=(-.025 + .325) / .55, xmax=1, color=STABLE_S, zorder=0)
    ax.plot([-.30, .20], [0, 0], color=RULE, lw=0.8, zorder=1)
    seen = {}
    for d in dmg:
        seen[d] = seen.get(d, 0) + 1
        c = RISK if d <= -.16 else (STABLE if d >= -.02 else RULE)
        ax.plot(d, 0.10 + (seen[d] - 1) * 0.15, "o", ms=5.2, color=c, zorder=3)
    for v in (-.30, -.15, 0, .18):
        ax.text(v, -0.20, f"{v:+.2f}".replace("+0.00", "0.00"), fontsize=7.5,
                color=FAINT, ha="center")
    ax.text(-.30, -0.44, "change in success rate under K2 compression, 50 episodes per task",
            fontsize=8, color=MUTED)
    ax.text(-.30, .82, "RISK POOL  6", fontsize=8.5, color=RISK)
    ax.text(-.135, .82, "neutral 3", fontsize=8.5, color=FAINT)
    ax.text(-.018, .82, "STABLE POOL  15", fontsize=8.5, color=STABLE)

    ax.add_patch(FancyBboxPatch((-.315, -1.22), .23, .58,
                                boxstyle="round,pad=0,rounding_size=0.008",
                                facecolor=RISK_S, edgecolor=RISK, lw=1.0))
    ax.text(-.302, -0.76, "penalty checks", fontsize=9, color=INK)
    ax.text(-.302, -0.99, "A 0.667   B 0.333", fontsize=8, color=MUTED,
            family="DejaVu Sans Mono")
    ax.add_patch(FancyBboxPatch((-.03, -1.22), .245, .58,
                                boxstyle="round,pad=0,rounding_size=0.008",
                                facecolor=STABLE_S, edgecolor=STABLE, lw=1.0))
    ax.text(-.017, -0.76, "bonus checks", fontsize=9, color=INK)
    ax.text(-.017, -0.99, "C 0.267  D 0.333  E 0.400", fontsize=8, color=MUTED,
            family="DejaVu Sans Mono")
    arrow(ax, (-.23, -.50), (-.23, -.62))
    arrow(ax, (.06, -.50), (.06, -.62))
    ax.text(.215, -0.76, "sign from the pool a check came from,\n"
                         "weight from how many of its tasks it covers",
            fontsize=7.5, color=FAINT, ha="right", va="top")
    save(fig, "fig2_pools")


# ---------------------------------------------------------------- fig 3
def fig_bands():
    cells = [("Rotate Box", 1.5, 2.0, 1), ("Bring PolyBag", 1.5, 2.5, 1),
             ("Rotate PolyBag", 2.0, 2.5, 1), ("Bring Box", 2.0, 3.0, 0),
             ("Pass Box", 2.0, 3.0, 0), ("Pass PolyBag", 2.0, 3.0, 0)]
    fig, ax = blank(10.4, 2.9)
    ax.set_xlim(0.72, 3.45); ax.set_ylim(-1.0, len(cells) - 0.3)
    for k, (nm, lo, hi, risky) in enumerate(cells):
        y = len(cells) - 1 - k
        c = RISK if risky else STABLE
        ax.text(0.75, y, nm, fontsize=9, va="center")
        ax.plot([1.0, 3.0], [y, y], color=RULE, lw=0.7, ls=(0, (1, 3)), zorder=1)
        ax.plot([lo, hi], [y, y], color=c, lw=5.5, alpha=.28,
                solid_capstyle="round", zorder=2)
        for q in (.08, .3, .52, .74, .94):
            ax.plot(lo + q * (hi - lo), y, "o", ms=3.6, color=c, zorder=3)
        ax.text(hi + .06, y, f"{lo:g}-{hi:g}x", fontsize=8, color=MUTED,
                va="center", family="DejaVu Sans Mono")
    ax.plot([1.0, 3.0], [-0.55, -0.55], color=RULE, lw=0.8)
    for v in (1.0, 1.5, 2.0, 2.5, 3.0):
        ax.plot([v, v], [-0.62, -0.48], color=RULE, lw=0.8)
        ax.text(v, -0.82, f"{v:g}x", fontsize=7.5, color=FAINT, ha="center")
    ax.text(0.75, -0.55, "playback", fontsize=8, color=MUTED, va="center")
    ax.text(3.42, len(cells) - 0.55,
            "band from measurement;\ndots are chunks ranked by confidence",
            fontsize=7.5, color=FAINT, ha="right", va="top")
    save(fig, "fig3_bands")


# ---------------------------------------------------------------- fig 4
def fig_causality():
    fig, ax = blank(10.4, 1.9)
    ax.set_xlim(0, 20); ax.set_ylim(0, 4)
    ax.add_patch(FancyBboxPatch((2.6, 1.9), 10.2, 1.0,
                                boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor=ACC_S, edgecolor=ACC, lw=1.0))
    ax.add_patch(FancyBboxPatch((13.4, 1.9), 5.6, 1.0,
                                boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor=RISK_S, edgecolor=RISK, lw=1.0, ls=(0, (4, 3))))
    ax.plot([1.2, 19.4], [1.55, 1.55], color=RULE, lw=0.8)
    ax.plot([13.1, 13.1], [1.2, 3.3], color=INK, lw=1.6)
    ax.text(13.1, 3.45, "f", fontsize=9.5, ha="center", family="DejaVu Sans Mono")
    ax.text(13.5, 3.45, "the observation being judged", fontsize=8, color=MUTED)
    ax.text(2.85, 2.62, "goes in", fontsize=9)
    ax.text(2.85, 2.16, "state[f]   action[f-16 : f]", fontsize=8, color=MUTED,
            family="DejaVu Sans Mono")
    ax.text(13.7, 2.62, "never goes in", fontsize=9, color=RISK)
    ax.text(13.7, 2.16, "action[f] and later", fontsize=8, color=MUTED,
            family="DejaVu Sans Mono")
    ax.text(2.6, 1.15, "f-16", fontsize=7.5, color=FAINT, ha="center",
            family="DejaVu Sans Mono")
    ax.text(19.0, 1.15, "f+16", fontsize=7.5, color=FAINT, ha="center",
            family="DejaVu Sans Mono")
    ax.text(1.2, 0.55, "at inference the history records only commands already sent to "
                       "env.step, never the freshly planned chunk",
            fontsize=8, color=MUTED)
    save(fig, "fig4_causality")


print(f"-> {OUT}")
fig_pipeline(); fig_pools(); fig_bands(); fig_causality()
