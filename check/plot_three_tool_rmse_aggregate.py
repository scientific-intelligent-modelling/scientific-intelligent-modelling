from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


# 这组数据来自 three_tools_3seeds_2h_blt35_20260413_013856 的 3-seed 平均结果。
AGGREGATED = {
    "oscillator1": {
        "pysr": {"id_rmse": 8.525178525162321e-05, "ood_rmse": 0.014875716664903157},
        "llmsr": {"id_rmse": 1.1183058813958016e-04, "ood_rmse": 0.013867992341112455},
        "drsr": {"id_rmse": 2.412758128025635e-07, "ood_rmse": 5.305664135294504e-04},
    },
    "oscillator2": {
        "pysr": {"id_rmse": 0.1499851006977979, "ood_rmse": 0.8975503545194906},
        "llmsr": {"id_rmse": 7.177632610048211e-04, "ood_rmse": 0.036106877532357244},
        "drsr": {"id_rmse": 0.006545238148414508, "ood_rmse": 0.083179664355484},
    },
    "stressstrain": {
        "pysr": {"id_rmse": 0.049472240592466826, "ood_rmse": 0.046868925512781136},
        "llmsr": {"id_rmse": 0.05385985868234768, "ood_rmse": 0.0411818028471401},
        "drsr": {"id_rmse": 0.040139358794782136, "ood_rmse": 0.04169850193045501},
    },
}

DATASETS = ["oscillator1", "oscillator2", "stressstrain"]
TOOLS = ["drsr", "llmsr", "pysr"]
COLORS = {"drsr": "#1f77b4", "llmsr": "#ff7f0e", "pysr": "#2ca02c"}
MARKERS = {"drsr": "o", "llmsr": "s", "pysr": "^"}


def plot_metric(metric: str, ylabel: str, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=180)
    x = list(range(len(DATASETS)))

    for tool in TOOLS:
        y = [AGGREGATED[dataset][tool][metric] for dataset in DATASETS]
        ax.plot(
            x,
            y,
            label=tool.upper(),
            color=COLORS[tool],
            marker=MARKERS[tool],
            linewidth=2.0,
            markersize=6.0,
        )

    ax.set_xticks(x, DATASETS)
    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Dataset")
    ax.set_title(title)
    ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    figures_dir = Path("paper/figures")
    plot_metric(
        metric="id_rmse",
        ylabel="ID RMSE (log scale)",
        title="3-Seed Average ID RMSE",
        output_path=figures_dir / "three_tool_id_rmse_avg.png",
    )
    plot_metric(
        metric="ood_rmse",
        ylabel="OOD RMSE (log scale)",
        title="3-Seed Average OOD RMSE",
        output_path=figures_dir / "three_tool_ood_rmse_avg.png",
    )


if __name__ == "__main__":
    main()
