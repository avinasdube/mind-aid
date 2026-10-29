"""Generate performance charts from metrics.json."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def _apply_chart_theme() -> None:
    """Use a consistent professional chart style."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "#f8fafc",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#0f172a",
            "axes.titleweight": "normal",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "grid.color": "#e2e8f0",
            "grid.linestyle": "--",
            "grid.alpha": 0.75,
            "legend.frameon": False,
            "legend.fontsize": 8,
        }
    )


def _rounded_bar(ax: plt.Axes, x: float, height: float, *, width: float, color: str, edge_color: str) -> None:
    rect = FancyBboxPatch(
        (x - (width / 2), 0),
        width,
        height,
        boxstyle="round,pad=0.0,rounding_size=0.03",
        linewidth=1.0,
        edgecolor=edge_color,
        facecolor=color,
        mutation_aspect=1,
    )
    ax.add_patch(rect)


def _annotate_value(ax: plt.Axes, x: float, height: float, *, dy: float = 0.015) -> None:
    ax.text(x, height + dy, f"{height:.3f}", ha="center", va="bottom", fontsize=7, color="#0f172a")


def _load_metrics(metrics_path: Path) -> dict[str, object]:
    """Load and validate metric values from a JSON file."""
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    required = {"accuracy", "precision", "recall", "f1"}
    missing = required.difference(metrics.keys())
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required metrics: {missing_str}")

    return {
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "baseline_comparison": metrics.get("baseline_comparison", {}),
    }


def _save_accuracy_chart(accuracy: float, output_dir: Path) -> Path:
    """Create and save a single-bar accuracy chart."""
    _apply_chart_theme()
    output_path = output_dir / "accuracy_bar_chart.png"

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _rounded_bar(ax, 0.0, accuracy, width=0.55, color="#0ea5e9", edge_color="#0369a1")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Model Accuracy")
    ax.set_xlim(-0.75, 0.75)
    ax.set_xticks([0.0])
    ax.set_xticklabels(["Accuracy"])
    ax.grid(axis="y")
    _annotate_value(ax, 0.0, accuracy)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def _save_prf_chart(precision: float, recall: float, f1: float, output_dir: Path) -> Path:
    """Create and save precision/recall/f1 comparison chart."""
    _apply_chart_theme()
    output_path = output_dir / "precision_recall_f1_chart.png"

    labels = ["Precision", "Recall", "F1"]
    values = [precision, recall, f1]
    colors = [("#14b8a6", "#0f766e"), ("#f59e0b", "#b45309"), ("#8b5cf6", "#6d28d9")]

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    positions = list(range(len(labels)))
    bar_width = 0.58
    for idx, value in enumerate(values):
        color, edge_color = colors[idx]
        _rounded_bar(ax, float(positions[idx]), value, width=bar_width, color=color, edge_color=edge_color)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.grid(axis="y")

    for idx, value in enumerate(values):
        _annotate_value(ax, float(positions[idx]), value, dy=0.02)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def _save_baseline_chart(baseline_comparison: dict[str, dict[str, float]], output_dir: Path) -> Path | None:
    """Create and save a baseline comparison chart when comparison data exists."""
    if not baseline_comparison:
        return None

    _apply_chart_theme()

    output_path = output_dir / "baseline_comparison_chart.png"
    models = list(baseline_comparison.keys())
    accuracies = [float(baseline_comparison[name].get("accuracy", 0.0)) for name in models]
    balanced = [float(baseline_comparison[name].get("balanced_accuracy", 0.0)) for name in models]

    x = list(range(len(models)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    acc_positions = [float(idx - width / 2) for idx in x]
    bal_positions = [float(idx + width / 2) for idx in x]
    for pos, value in zip(acc_positions, accuracies):
        _rounded_bar(ax, pos, value, width=width, color="#0ea5e9", edge_color="#0369a1")
    for pos, value in zip(bal_positions, balanced):
        _rounded_bar(ax, pos, value, width=width, color="#14b8a6", edge_color="#0f766e")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Baseline Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(["\n".join(textwrap.wrap(name.replace("_", " ").title(), width=14)) for name in models])
    ax.grid(axis="y")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="#0ea5e9", lw=6, label="Accuracy"),
            plt.Line2D([0], [0], color="#14b8a6", lw=6, label="Balanced Accuracy"),
        ],
        loc="lower right",
    )

    for pos, value in zip(acc_positions, accuracies):
        _annotate_value(ax, pos, value)
    for pos, value in zip(bal_positions, balanced):
        _annotate_value(ax, pos, value)

    fig.tight_layout(pad=1.2)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def generate_metrics_charts(
    metrics_path: Path = Path("metrics.json"),
    output_dir: Path = Path("outputs/charts"),
) -> dict[str, str]:
    """Read metrics.json and generate requested performance charts."""
    metrics = _load_metrics(metrics_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    accuracy_path = _save_accuracy_chart(metrics["accuracy"], output_dir)
    prf_path = _save_prf_chart(metrics["precision"], metrics["recall"], metrics["f1"], output_dir)
    baseline_path = _save_baseline_chart(metrics.get("baseline_comparison", {}), output_dir)

    created = {
        "accuracy_chart": str(accuracy_path),
        "precision_recall_f1_chart": str(prf_path),
    }

    if baseline_path is not None:
        created["baseline_comparison_chart"] = str(baseline_path)

    return created


def main() -> None:
    """Run chart generation from default paths."""
    created = generate_metrics_charts()
    print("Charts generated:")
    for name, path in created.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
