"""Generate polished visual reports from training and research evaluation artifacts."""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

from .utils import ensure_parent_dir


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _metric_value(block: dict[str, Any] | None, key: str) -> float:
    if not block:
        return 0.0
    value = block.get(key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _apply_chart_theme() -> None:
    """Set a modern, consistent theme for generated charts."""
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
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 8,
        }
    )


def _rounded_bars(ax: Any, x_positions: list[float], heights: list[float], *, width: float, color: str, edge_color: str) -> None:
    """Draw rounded bars with clean edges."""
    for x, h in zip(x_positions, heights):
        rect = FancyBboxPatch(
            (x - (width / 2), 0),
            width,
            h,
            boxstyle="round,pad=0.0,rounding_size=0.03",
            linewidth=1.0,
            edgecolor=edge_color,
            facecolor=color,
            mutation_aspect=1,
        )
        ax.add_patch(rect)


def _annotate_values(ax: Any, x_positions: list[float], heights: list[float], *, dy: float = 0.012) -> None:
    for x, h in zip(x_positions, heights):
        ax.text(x, h + dy, f"{h:.3f}", ha="center", va="bottom", fontsize=7, color="#0f172a")


def _save_generalization_chart(
    metrics_data: dict[str, Any],
    evaluation_data: dict[str, Any],
    output_path: Path,
) -> Path:
    _apply_chart_theme()
    holdout = metrics_data.get("holdout_metrics", {})
    external = (evaluation_data.get("external_validation") or {}).get("text_only_model", {})

    labels = ["Accuracy", "F1", "ROC-AUC"]
    holdout_values = [
        _metric_value(holdout, "accuracy"),
        _metric_value(holdout, "f1"),
        _metric_value(holdout, "roc_auc"),
    ]
    external_values = [
        _metric_value(external, "accuracy"),
        _metric_value(external, "f1"),
        _metric_value(external, "roc_auc"),
    ]

    x = list(range(len(labels)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    holdout_x = [idx - width / 2 for idx in x]
    external_x = [idx + width / 2 for idx in x]

    _rounded_bars(ax, holdout_x, holdout_values, width=width, color="#22c55e", edge_color="#15803d")
    _rounded_bars(ax, external_x, external_values, width=width, color="#3b82f6", edge_color="#1d4ed8")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Generalization: Holdout vs Unseen Validation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y")
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    _annotate_values(ax, holdout_x, holdout_values)
    _annotate_values(ax, external_x, external_values)
    ax.legend(["Holdout", "External"], loc="lower right")

    fig.tight_layout()
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _save_robustness_chart(evaluation_data: dict[str, Any], output_path: Path) -> Path | None:
    _apply_chart_theme()
    robustness = (evaluation_data.get("internal_ablation") or {}).get("robustness_checks", {})
    variants = robustness.get("variants", {})
    if not variants:
        return None

    names = list(variants.keys())
    flip_rates = [float(variants[name].get("prediction_flip_rate", 0.0)) for name in names]
    accuracies = [float(variants[name].get("accuracy", 0.0)) for name in names]

    x = list(range(len(names)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    flip_x = [idx - width / 2 for idx in x]
    acc_x = [idx + width / 2 for idx in x]

    _rounded_bars(ax, flip_x, flip_rates, width=width, color="#f97316", edge_color="#c2410c")
    _rounded_bars(ax, acc_x, accuracies, width=width, color="#06b6d4", edge_color="#0e7490")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Robustness Under Text Perturbations")
    ax.set_xticks(x)
    ax.set_xticklabels(["\n".join(textwrap.wrap(name.replace("_", " ").title(), width=14)) for name in names])
    ax.grid(axis="y")
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    _annotate_values(ax, flip_x, flip_rates)
    _annotate_values(ax, acc_x, accuracies)
    ax.legend(["Flip Rate", "Accuracy"], loc="upper right")

    fig.tight_layout()
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _write_markdown_summary(
    metrics_data: dict[str, Any],
    evaluation_data: dict[str, Any],
    output_path: Path,
) -> Path:
    holdout = metrics_data.get("holdout_metrics", {})
    external = (evaluation_data.get("external_validation") or {}).get("text_only_model", {})
    internal_block = (evaluation_data.get("internal_diagnostic_in_sample") or evaluation_data.get("internal_ablation") or {})
    ece = _metric_value((internal_block or {}).get("text_only_model", {}), "expected_calibration_error")

    content = [
        "# MIND-AID Analysis Summary",
        "",
        "## Core Results",
        "",
        "| Metric | Holdout | External Validation |",
        "|---|---:|---:|",
        f"| Accuracy | {_metric_value(holdout, 'accuracy'):.4f} | {_metric_value(external, 'accuracy'):.4f} |",
        f"| F1 | {_metric_value(holdout, 'f1'):.4f} | {_metric_value(external, 'f1'):.4f} |",
        f"| ROC-AUC | {_metric_value(holdout, 'roc_auc'):.4f} | {_metric_value(external, 'roc_auc'):.4f} |",
        f"| Brier Score | {_metric_value(holdout, 'brier_score'):.4f} | {_metric_value(external, 'brier_score'):.4f} |",
        "",
        "## Calibration",
        "",
        f"- Expected Calibration Error (ECE): {ece:.4f}",
        "- Lower ECE means predicted probabilities better match observed outcomes.",
        "",
        "## How To Explain To Reviewers",
        "",
        "- External validation scores are the main evidence of generalization.",
        "- Internal diagnostics are in-sample and can look optimistic; do not treat them as generalization evidence.",
        "- Holdout scores show in-distribution behavior and are supportive, not the main claim.",
        "- Calibration metrics show that confidence values are meaningful and not only ranking scores.",
        "- Robustness checks show sensitivity to text perturbations and improve trust in deployment claims.",
    ]

    ensure_parent_dir(output_path)
    with output_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(content))
    return output_path


def _write_html_report(
    metrics_data: dict[str, Any],
    evaluation_data: dict[str, Any],
    output_path: Path,
    charts: dict[str, Path | None],
) -> Path:
    holdout = metrics_data.get("holdout_metrics", {})
    external = (evaluation_data.get("external_validation") or {}).get("text_only_model", {})
    internal_block = (evaluation_data.get("internal_diagnostic_in_sample") or evaluation_data.get("internal_ablation") or {})
    ece = _metric_value((internal_block or {}).get("text_only_model", {}), "expected_calibration_error")

    def rel(path: Path | None) -> str:
        if path is None:
            return ""
        relative = os.path.relpath(path, output_path.parent)
        return relative.replace("\\", "/")

    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>MIND-AID Visual Report</title>
    <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">
    <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>
    <link href=\"https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&display=swap\" rel=\"stylesheet\">
  <style>
    :root {{
            --bg: #eef6ff;
      --ink: #0f172a;
      --muted: #475569;
      --card: #ffffff;
            --accent: #2563eb;
            --accent2: #06b6d4;
            --accent3: #22c55e;
    }}
    body {{
      margin: 0;
            font-family: "Manrope", "Segoe UI", sans-serif;
      color: var(--ink);
            background: radial-gradient(circle at 100% 0%, #bfdbfe 0%, var(--bg) 45%);
    }}
    .wrap {{ max-width: 1080px; margin: 0 auto; padding: 28px; }}
    .hero {{
            background: linear-gradient(120deg, #0b1226, #1d4ed8 55%, #06b6d4);
      color: #fff;
            border-radius: 20px;
      padding: 26px;
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.22);
    }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; margin-top: 16px; }}
        .card {{ background: var(--card); border-radius: 16px; padding: 16px; box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08); border: 1px solid #dbeafe; }}
    .label {{ color: var(--muted); font-size: 13px; margin-bottom: 6px; }}
        .value {{ font-size: 28px; font-weight: 800; }}
    .section {{ margin-top: 22px; }}
        h2 {{ margin: 0 0 10px; font-size: 22px; }}
        img {{ width: 100%; border-radius: 16px; border: 1px solid #cbd5e1; background: #fff; }}
    .two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
        .panel {{ background: #fff; border-radius: 16px; border: 1px solid #dbeafe; padding: 14px; box-shadow: 0 6px 16px rgba(15, 23, 42, 0.06); }}
        .meta {{ color: #334155; line-height: 1.6; margin-top: 12px; font-size: 14px; }}
    @media (max-width: 900px) {{ .two {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"hero\">
      <h1>MIND-AID Visual Analysis Report</h1>
      <p>Reviewer-friendly summary of generalization, calibration, and robustness.</p>
    </div>

    <div class=\"grid\">
      <div class=\"card\"><div class=\"label\">Holdout Accuracy</div><div class=\"value\">{_metric_value(holdout, 'accuracy'):.3f}</div></div>
      <div class=\"card\"><div class=\"label\">External Accuracy</div><div class=\"value\">{_metric_value(external, 'accuracy'):.3f}</div></div>
      <div class=\"card\"><div class=\"label\">External ROC-AUC</div><div class=\"value\">{_metric_value(external, 'roc_auc'):.3f}</div></div>
      <div class=\"card\"><div class=\"label\">Calibration (ECE)</div><div class=\"value\">{ece:.3f}</div></div>
    </div>

    <div class=\"section\">
            <div class=\"panel\">
                <h2>Generalization Evidence</h2>
                <img src=\"{rel(charts.get('generalization'))}\" alt=\"Generalization chart\" />
            </div>
    </div>

    <div class=\"section two\">
            <div class=\"panel\">
        <h2>Ablation</h2>
        <img src=\"{rel(charts.get('ablation'))}\" alt=\"Ablation chart\" />
      </div>
            <div class=\"panel\">
        <h2>Calibration</h2>
        <img src=\"{rel(charts.get('calibration'))}\" alt=\"Calibration chart\" />
      </div>
    </div>

    <div class=\"section\">
            <div class=\"panel\">
                <h2>Robustness</h2>
                <img src=\"{rel(charts.get('robustness'))}\" alt=\"Robustness chart\" />
            </div>
                        <p class="meta">Interpretation tip: external validation is the primary proof of generalization. Internal diagnostics are in-sample and can look optimistic. Calibration shows score trustworthiness, and robustness shows stability under noisy text.</p>
    </div>
  </div>
</body>
</html>
"""

    ensure_parent_dir(output_path)
    with output_path.open("w", encoding="utf-8") as file:
        file.write(html)
    return output_path


def generate_visual_report(
    metrics_path: Path = Path("metrics.json"),
    evaluation_path: Path = Path("outputs/reports/research_evaluation.json"),
    charts_dir: Path = Path("outputs/charts"),
    report_dir: Path = Path("outputs/reports"),
) -> dict[str, str]:
    """Generate polished charts and summary reports for presentation and review."""
    metrics_data = _load_json(metrics_path)
    evaluation_data = _load_json(evaluation_path)

    generalization_chart = _save_generalization_chart(
        metrics_data,
        evaluation_data,
        charts_dir / "generalization_comparison.png",
    )

    # Create a styled ablation chart for visual consistency in the HTML report.
    ablation_source = evaluation_data.get("internal_diagnostic_in_sample") or evaluation_data.get("internal_ablation") or {}
    ablation_keys = [
        "text_only_model",
        "text_threshold_045",
        "text_threshold_050",
        "fixed_text_face_fusion",
        "confidence_aware_fusion",
    ]
    ablation_labels: list[str] = []
    ablation_values: list[float] = []
    for key in ablation_keys:
        block = ablation_source.get(key, {})
        if isinstance(block, dict) and block.get("status") != "skipped":
            ablation_labels.append(key.replace("_", " ").title())
            ablation_values.append(_metric_value(block, "f1"))

    styled_ablation_chart = None
    if ablation_labels:
        _apply_chart_theme()
        fig, ax = plt.subplots(figsize=(10.2, 5.2))
        y = list(range(len(ablation_labels)))
        bar_height = 0.58
        for idx, value in enumerate(ablation_values):
            rect = FancyBboxPatch(
                (0, idx - bar_height / 2),
                value,
                bar_height,
                boxstyle="round,pad=0.0,rounding_size=0.03",
                linewidth=1.0,
                edgecolor="#6d28d9",
                facecolor="#8b5cf6",
                mutation_aspect=1,
            )
            ax.add_patch(rect)
            ax.text(value + 0.012, idx, f"{value:.3f}", va="center", fontsize=7, color="#0f172a")

        ax.set_xlim(0, 1.08)
        ax.set_xlabel("F1 Score")
        ax.set_title("Ablation Comparison")
        ax.set_yticks(y)
        ax.set_yticklabels(["\n".join(textwrap.wrap(label, width=16)) for label in ablation_labels])
        ax.grid(axis="x")
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.legend(handles=[Patch(facecolor="#8b5cf6", edgecolor="#6d28d9", label="F1")], loc="lower right")
        styled_ablation_chart = charts_dir / "ablation_comparison_styled.png"
        ensure_parent_dir(styled_ablation_chart)
        fig.tight_layout(pad=1.2)
        fig.savefig(styled_ablation_chart, dpi=150)
        plt.close(fig)

    robustness_chart = _save_robustness_chart(
        evaluation_data,
        charts_dir / "robustness_perturbation_chart.png",
    )

    markdown_report = _write_markdown_summary(
        metrics_data,
        evaluation_data,
        report_dir / "analysis_summary.md",
    )

    html_report = _write_html_report(
        metrics_data,
        evaluation_data,
        report_dir / "visual_report.html",
        charts={
            "generalization": generalization_chart,
            "ablation": styled_ablation_chart or Path(str(evaluation_data.get("ablation_chart", charts_dir / "ablation_comparison.png"))),
            "calibration": Path(str(evaluation_data.get("calibration_chart", charts_dir / "calibration_curve.png"))),
            "robustness": robustness_chart,
        },
    )

    return {
        "generalization_chart": str(generalization_chart),
        "robustness_chart": str(robustness_chart) if robustness_chart else "",
        "markdown_summary": str(markdown_report),
        "html_visual_report": str(html_report),
    }


def main() -> None:
    """Run visual report generation from default project paths."""
    created = generate_visual_report()
    print("Visual report artifacts generated:")
    for name, path in created.items():
        if path:
            print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
