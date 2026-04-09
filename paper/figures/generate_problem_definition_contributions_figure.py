from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "problem_definition_contributions_figure.png"


def add_box(ax, xy, w, h, text, fc, ec="#333333", fontsize=10, weight="normal"):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        wrap=True,
    )


def add_arrow(ax, start, end, color="#555555"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.4,
            color=color,
        )
    )


def main():
    fig, ax = plt.subplots(figsize=(15.5, 5.2), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    panel_w = 0.295
    gap = 0.03
    panels = [
        (0.02, 0.07, panel_w, 0.86, "#f7efe5"),
        (0.3525, 0.07, panel_w, 0.86, "#e9f3ee"),
        (0.685, 0.07, panel_w, 0.86, "#edf1f7"),
    ]
    titles = [
        "A. Fragmented SR Research Workflow",
        "B. SREvoLab Core Architecture",
        "C. What the Platform Enables",
    ]
    title_colors = ["#9c5b1a", "#2f6b45", "#355d8c"]

    for (x, y, w, h, fc), title, color in zip(panels, titles, title_colors):
        add_box(ax, (x, y), w, h, "", fc, ec=color)
        ax.text(x + 0.015, y + h - 0.05, title, fontsize=12, weight="bold", color=color, va="center")

    # Panel A
    x, y, w, h, _ = panels[0]
    add_box(ax, (x + 0.03, y + 0.55), 0.10, 0.22, "Method\nRepos\n\nPySR\nDRSR\nLLMSR\nDSO\nTPSR", "#fff8f0", fontsize=10, weight="bold")
    add_box(ax, (x + 0.165, y + 0.55), 0.10, 0.22, "Dataset\nFamilies\n\nSRBench\nLLM-SRBench\nSRSD", "#fff8f0", fontsize=10, weight="bold")
    add_box(ax, (x + 0.08, y + 0.22), 0.17, 0.19, "Pain points\n\nincompatible envs\ninconsistent splits\nexpression mismatch\nweak OOD protocol", "#fffaf5", fontsize=9.5)
    add_arrow(ax, (x + 0.13, y + 0.55), (x + 0.16, y + 0.41), "#9c5b1a")
    add_arrow(ax, (x + 0.215, y + 0.55), (x + 0.19, y + 0.41), "#9c5b1a")
    ax.text(x + 0.03, y + 0.10, "Fragmentation slows reproducible SR research.", fontsize=10.3, color="#6e4216", weight="bold")

    # Panel B
    x, y, w, h, _ = panels[1]
    add_box(ax, (x + 0.06, y + 0.63), 0.18, 0.11, "Unified Method Substrate\nwrappers | subprocess | env isolation", "#f6fcf8", fontsize=9.5, weight="bold")
    add_box(ax, (x + 0.06, y + 0.48), 0.18, 0.11, "Canonical Dataset Substrate\ntrain/valid/id/OOD | metadata | formula", "#f6fcf8", fontsize=9.5, weight="bold")
    add_box(ax, (x + 0.06, y + 0.33), 0.18, 0.11, "Skill Layer\nonboard | homogenize | validate | compile", "#f6fcf8", fontsize=9.5, weight="bold")
    add_box(ax, (x + 0.06, y + 0.16), 0.18, 0.12, "Audited Evolution Loop\nAdapt -> Execute -> Audit -> Review", "#f6fcf8", fontsize=9.5, weight="bold")
    add_arrow(ax, (x + 0.15, y + 0.63), (x + 0.15, y + 0.59), "#2f6b45")
    add_arrow(ax, (x + 0.15, y + 0.48), (x + 0.15, y + 0.44), "#2f6b45")
    add_arrow(ax, (x + 0.15, y + 0.33), (x + 0.15, y + 0.28), "#2f6b45")
    ax.text(x + 0.04, y + 0.08, "A common, auditable operating layer for SR.", fontsize=10.2, color="#224e34", weight="bold")

    # Panel C
    x, y, w, h, _ = panels[2]
    add_box(ax, (x + 0.05, y + 0.57), 0.20, 0.18, "Outputs\n\nresult.json\ncanonical equations\naudit report\npaper-ready tables", "#f8fbff", fontsize=10, weight="bold")
    add_box(ax, (x + 0.05, y + 0.29), 0.20, 0.20, "Enables\n\nreproducible execution\nformula-aware validation\nOOD-aware evaluation\nbounded improvement", "#f8fbff", fontsize=10)
    add_arrow(ax, (x + 0.15, y + 0.57), (x + 0.15, y + 0.49), "#355d8c")
    ax.text(x + 0.04, y + 0.12, "Not a new SR model,\nbut a research operating substrate.", fontsize=10.3, color="#27486e", weight="bold")

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
