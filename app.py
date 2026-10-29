"""Application entry point for the MIND-AID project."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.explain import explain_prediction
from src.explainability_audit import run_explainability_audit
from src.shap_explainability import run_shap_audit
from src.distilbert_explainability import run_distilbert_shap_audit, train_distilbert
from src.model_benchmark import run_model_benchmark
from src.dataset_split import split_labeled_dataset
from src.predict_text import predict_emotion
from src.preprocess import preprocess_dataset
from src.generate_visual_report import generate_visual_report
from src.research_eval import run_research_evaluation
from src.train_model import train_model
from src.validate_dataset import validate_external_dataset


def build_parser() -> argparse.ArgumentParser:
    """Create and return the CLI argument parser."""
    parser = argparse.ArgumentParser(description="MIND-AID command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_cmd = subparsers.add_parser("preprocess", help="Preprocess raw dataset")
    preprocess_cmd.add_argument("--input", type=Path, required=True, help="Path to raw CSV")
    preprocess_cmd.add_argument("--output", type=Path, required=True, help="Path to processed CSV")

    split_cmd = subparsers.add_parser(
        "split-dataset",
        help="Split a labeled dataset into train and unseen validation CSV files",
    )
    split_cmd.add_argument("--input", type=Path, required=True, help="Path to labeled CSV")
    split_cmd.add_argument("--train-output", type=Path, required=True, help="Path for train CSV")
    split_cmd.add_argument(
        "--validation-output",
        type=Path,
        required=True,
        help="Path for unseen validation CSV",
    )
    split_cmd.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows to reserve for validation",
    )
    split_cmd.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the split",
    )

    train_cmd = subparsers.add_parser("train", help="Train emotion model")
    train_cmd.add_argument("--data", type=Path, required=True, help="Path to processed CSV")
    train_cmd.add_argument(
        "--model",
        type=Path,
        default=Path("models/text_model.pkl"),
        help="Model output path",
    )
    train_cmd.add_argument(
        "--vectorizer",
        type=Path,
        default=Path("models/vectorizer.pkl"),
        help="Vectorizer output path",
    )
    train_cmd.add_argument(
        "--metrics",
        type=Path,
        default=Path("metrics.json"),
        help="Metrics JSON output path",
    )
    train_cmd.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the train/test diagnostic split and baseline CV",
    )

    predict_cmd = subparsers.add_parser("predict", help="Predict emotion for text")
    predict_cmd.add_argument("--model", type=Path, required=True, help="Path to trained model")
    predict_cmd.add_argument(
        "--vectorizer",
        type=Path,
        default=Path("models/vectorizer.pkl"),
        help="Path to trained vectorizer",
    )
    predict_cmd.add_argument("--text", type=str, required=True, help="Text to classify")

    explain_cmd = subparsers.add_parser("explain", help="Explain a text prediction")
    explain_cmd.add_argument("--model", type=Path, required=True, help="Path to trained model")
    explain_cmd.add_argument(
        "--vectorizer",
        type=Path,
        default=Path("models/vectorizer.pkl"),
        help="Path to trained vectorizer",
    )
    explain_cmd.add_argument("--text", type=str, required=True, help="Text to explain")

    explainability_cmd = subparsers.add_parser(
        "explainability-audit",
        help="Audit explanation stability and masking sensitivity",
    )
    explainability_cmd.add_argument("--data", type=Path, required=True, help="Path to labeled CSV")
    explainability_cmd.add_argument("--model", type=Path, required=True, help="Path to trained model")
    explainability_cmd.add_argument(
        "--vectorizer",
        type=Path,
        required=True,
        help="Path to trained vectorizer",
    )
    explainability_cmd.add_argument("--output", type=Path, required=True, help="Output JSON path")

    shap_cmd = subparsers.add_parser(
        "shap-audit",
        help="Generate SHAP attributions for the TF-IDF linear model",
    )
    shap_cmd.add_argument("--data", type=Path, required=True, help="Path to labeled CSV")
    shap_cmd.add_argument("--model", type=Path, required=True, help="Path to trained model")
    shap_cmd.add_argument("--vectorizer", type=Path, required=True, help="Path to trained vectorizer")
    shap_cmd.add_argument("--output", type=Path, required=True, help="Output JSON path")
    shap_cmd.add_argument("--max-samples", type=int, default=25, help="Number of rows to explain")
    shap_cmd.add_argument("--background-size", type=int, default=100, help="SHAP background row count")

    distilbert_train_cmd = subparsers.add_parser(
        "train-distilbert",
        help="Fine-tune DistilBERT and evaluate on an external CSV",
    )
    distilbert_train_cmd.add_argument("--data", type=Path, required=True, help="Training CSV")
    distilbert_train_cmd.add_argument("--external-data", type=Path, required=True, help="External test CSV")
    distilbert_train_cmd.add_argument("--output-dir", type=Path, required=True, help="Model directory")
    distilbert_train_cmd.add_argument("--metrics", type=Path, required=True, help="Metrics JSON path")
    distilbert_train_cmd.add_argument("--epochs", type=int, default=2)
    distilbert_train_cmd.add_argument("--batch-size", type=int, default=8)
    distilbert_train_cmd.add_argument("--max-length", type=int, default=256)

    distilbert_shap_cmd = subparsers.add_parser(
        "distilbert-shap-audit",
        help="Generate token SHAP values for a fine-tuned DistilBERT model",
    )
    distilbert_shap_cmd.add_argument("--data", type=Path, required=True)
    distilbert_shap_cmd.add_argument("--model-dir", type=Path, required=True)
    distilbert_shap_cmd.add_argument("--output", type=Path, required=True)
    distilbert_shap_cmd.add_argument("--max-samples", type=int, default=10)
    distilbert_shap_cmd.add_argument("--max-evals", type=int, default=200)

    benchmark_cmd = subparsers.add_parser(
        "benchmark-model",
        help="Benchmark repeated seeds and measured model resources",
    )
    benchmark_cmd.add_argument("--data", type=Path, required=True, help="Path to training CSV")
    benchmark_cmd.add_argument("--external-data", type=Path, required=True, help="Path to external CSV")
    benchmark_cmd.add_argument("--output", type=Path, required=True, help="Output JSON path")

    evaluate_cmd = subparsers.add_parser(
        "evaluate",
        help="Run internal ablation and optional external validation",
    )
    evaluate_cmd.add_argument("--data", type=Path, required=True, help="Path to labeled evaluation CSV")
    evaluate_cmd.add_argument(
        "--external-data",
        type=Path,
        default=None,
        help="Optional unseen labeled CSV for external validation",
    )
    evaluate_cmd.add_argument(
        "--model",
        type=Path,
        default=Path("models/text_model.pkl"),
        help="Path to trained model",
    )
    evaluate_cmd.add_argument(
        "--vectorizer",
        type=Path,
        default=Path("models/vectorizer.pkl"),
        help="Path to trained vectorizer",
    )
    evaluate_cmd.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/research_evaluation.json"),
        help="Output path for the research evaluation report",
    )
    evaluate_cmd.add_argument(
        "--chart",
        type=Path,
        default=Path("outputs/charts/ablation_comparison.png"),
        help="Output path for the ablation comparison chart",
    )

    validate_cmd = subparsers.add_parser(
        "validate-external",
        help="Validate external evaluation dataset schema, overlap risk, and image readiness",
    )
    validate_cmd.add_argument(
        "--external-data",
        type=Path,
        required=True,
        help="Path to external CSV",
    )
    validate_cmd.add_argument(
        "--training-data",
        type=Path,
        default=Path("data/splits/train.csv"),
        help="Path to training CSV used for overlap checks",
    )
    validate_cmd.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/external_dataset_validation.json"),
        help="Output path for validation report JSON",
    )
    validate_cmd.add_argument(
        "--require-images",
        action="store_true",
        help="Fail validation when image paths are missing or invalid",
    )

    visual_report_cmd = subparsers.add_parser(
        "visual-report",
        help="Generate polished visual and markdown reports from latest evaluation outputs",
    )
    visual_report_cmd.add_argument(
        "--metrics",
        type=Path,
        default=Path("metrics.json"),
        help="Path to metrics JSON",
    )
    visual_report_cmd.add_argument(
        "--evaluation",
        type=Path,
        default=Path("outputs/reports/research_evaluation.json"),
        help="Path to research evaluation JSON",
    )
    visual_report_cmd.add_argument(
        "--charts-dir",
        type=Path,
        default=Path("outputs/charts"),
        help="Directory to save generated charts",
    )
    visual_report_cmd.add_argument(
        "--report-dir",
        type=Path,
        default=Path("outputs/reports"),
        help="Directory to save generated report files",
    )

    return parser


def main() -> None:
    """Run the selected command."""
    args = build_parser().parse_args()

    if args.command == "preprocess":
        preprocess_dataset(args.input, args.output)
    elif args.command == "split-dataset":
        split_info = split_labeled_dataset(
            args.input,
            args.train_output,
            args.validation_output,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        print(f"Dataset split completed. Train: {split_info['train_output']}, Validation: {split_info['validation_output']}")
    elif args.command == "train":
        metrics = train_model(
            args.data,
            args.model,
            args.vectorizer,
            args.metrics,
            random_state=args.random_state,
        )
        print(f"Training completed. Metrics: {metrics}")
    elif args.command == "predict":
        label = predict_emotion(args.model, args.vectorizer, args.text)
        print(f"Predicted emotion: {label}")
    elif args.command == "explain":
        explanation = explain_prediction(args.model, args.text, args.vectorizer)
        print(explanation)
    elif args.command == "explainability-audit":
        report = run_explainability_audit(
            args.data,
            args.model,
            args.vectorizer,
            args.output,
        )
        print(f"Explainability audit completed. Samples: {report['sample_count']}")
    elif args.command == "shap-audit":
        report = run_shap_audit(
            args.data,
            args.model,
            args.vectorizer,
            args.output,
            max_samples=args.max_samples,
            background_size=args.background_size,
        )
    elif args.command == "train-distilbert":
        metrics = train_distilbert(
            args.data,
            args.external_data,
            args.output_dir,
            args.metrics,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        print(f"DistilBERT training completed. Accuracy: {metrics['accuracy']:.4f}")
    elif args.command == "distilbert-shap-audit":
        report = run_distilbert_shap_audit(
            args.data,
            args.model_dir,
            args.output,
            max_samples=args.max_samples,
            max_evals=args.max_evals,
        )
        print(f"DistilBERT SHAP audit completed. Samples: {report['sample_count']}")
    elif args.command == "benchmark-model":
        report = run_model_benchmark(args.data, args.external_data, args.output)
        print(f"Model benchmark completed. Runs: {len(report['runs'])}")
    elif args.command == "evaluate":
        report = run_research_evaluation(
            args.data,
            model_path=args.model,
            vectorizer_path=args.vectorizer,
            output_path=args.output,
            chart_path=args.chart,
            external_data_path=args.external_data,
        )
        print(f"Research evaluation completed. Report saved to: {args.output}")
        if report.get("ablation_chart"):
            print(f"Ablation chart saved to: {report['ablation_chart']}")
    elif args.command == "visual-report":
        generated = generate_visual_report(
            metrics_path=args.metrics,
            evaluation_path=args.evaluation,
            charts_dir=args.charts_dir,
            report_dir=args.report_dir,
        )
        print("Visual report generation completed.")
        for name, path in generated.items():
            if path:
                print(f"- {name}: {path}")
    elif args.command == "validate-external":
        report = validate_external_dataset(
            args.external_data,
            training_data_path=args.training_data,
            output_path=args.output,
            require_images=args.require_images,
        )
        print(f"External dataset validation status: {report['status']}")
        print(f"Validation report saved to: {args.output}")
        if report.get("warnings"):
            print("Warnings:")
            for warning in report["warnings"]:
                print(f"- {warning}")
        if report.get("errors"):
            print("Errors:")
            for error in report["errors"]:
                print(f"- {error}")


if __name__ == "__main__":
    main()
