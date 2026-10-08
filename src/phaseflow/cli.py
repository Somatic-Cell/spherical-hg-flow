"""Command-line adapters; all file formats and physical assumptions are explicit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .data import PhasePointCloud
from .model import ModelConfig
from .synthetic import make_demo
from .training import (
    TrainingConfig,
    atomic_json,
    evaluate_model,
    load_model_checkpoint,
    train_model,
)


def load_config(path: str | Path) -> tuple[ModelConfig, TrainingConfig]:
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    unknown = set(config).difference({"schema_version", "model", "training"})
    if unknown:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")
    if config.get("schema_version") != 1:
        raise ValueError("config requires schema_version=1")
    if "model" not in config or "training" not in config:
        raise ValueError("config requires explicit model and training objects")
    return ModelConfig.from_dict(config["model"]), TrainingConfig.from_dict(config["training"])


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _progress(entry: dict[str, Any]) -> None:
    report = {name: entry[name] for name in ("global_step", "stage", "loss")}
    if "metrics" in entry:
        report["train_nll"] = entry["metrics"]["train"]["mean_nll"]
        validation = entry["metrics"].get("validation")
        if validation is not None:
            report["validation_nll"] = validation["mean_nll"]
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Conditional HG-base spherical normalizing flow")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect-rainbow", help="validate one Rainbow CDF record")
    inspect.add_argument("--record", required=True, type=Path)
    rainbow_train = commands.add_parser(
        "train-rainbow", help="train a circular/interval RQS flow from one Rainbow CDF and its g"
    )
    rainbow_train.add_argument("--record", required=True, type=Path)
    rainbow_train.add_argument("--config", required=True, type=Path)
    rainbow_train.add_argument("--output", "--output-dir", required=True, type=Path)
    rainbow_train.add_argument("--resume", type=Path)
    rainbow_train.add_argument("--max-steps-this-run", type=int)
    rainbow_train.add_argument("--device")
    rainbow_train.add_argument(
        "--no-plots", action="store_true", help="explicitly skip plots after completed training"
    )
    rainbow_train.add_argument("--quiet", action="store_true")
    rainbow_eval = commands.add_parser(
        "evaluate-rainbow", help="independent same-condition NLL, forward KL and importance ESS"
    )
    rainbow_eval.add_argument("--record", required=True, type=Path)
    rainbow_eval.add_argument("--checkpoint", required=True, type=Path)
    rainbow_eval.add_argument("--samples", type=int, default=65536)
    rainbow_eval.add_argument("--proposal-samples", type=int)
    rainbow_eval.add_argument("--seed", type=int, default=2026)
    rainbow_eval.add_argument("--batch-size", type=int, default=4096)
    rainbow_eval.add_argument("--device", default="cuda")
    rainbow_eval.add_argument("--cpu-threads", type=int, default=1)
    rainbow_eval.add_argument("--output", type=Path)
    rainbow_plot = commands.add_parser(
        "plot-rainbow", help="aligned CDF PDF, CDF samples and NF PDF for one checkpoint"
    )
    rainbow_plot.add_argument("--record", required=True, type=Path)
    rainbow_plot.add_argument("--checkpoint", required=True, type=Path)
    rainbow_plot.add_argument("--output", "--output-dir", required=True, type=Path)
    rainbow_plot.add_argument("--device", default="cuda")
    rainbow_plot.add_argument("--samples", type=int, default=32768)
    rainbow_plot.add_argument("--seed", type=int, default=2027)
    rainbow_plot.add_argument("--batch-size", type=int, default=16384)
    rainbow_plot.add_argument("--dpi", type=int, default=160)
    rainbow_plot.add_argument("--write-pdf", action="store_true")
    rainbow_plot.add_argument("--cpu-threads", type=int, default=1)
    demo = commands.add_parser(
        "demo",
        aliases=["make-demo"],
        help="generate analytic implementation-test data, not rainbow data",
    )
    demo.add_argument("--output", required=True, type=Path)
    demo.add_argument("--mode", choices=("target_samples", "quadrature"), default="target_samples")
    demo.add_argument("--points-per-condition", type=int, default=4096)
    demo.add_argument("--seed", type=int, default=17)
    demo.add_argument(
        "--wavelengths", nargs="+", type=float, default=[400.0, 550.0, 700.0], metavar="NM"
    )
    demo.add_argument(
        "--incident-cosines", nargs="+", type=float, default=[-0.75, -0.25, 0.25, 0.75]
    )
    demo.add_argument(
        "--n-mu", type=int, default=48, help="Gauss-Legendre points for quadrature mode"
    )
    demo.add_argument("--n-phi", type=int, default=64, help="azimuthal quadrature points")
    train = commands.add_parser(
        "train",
        help="HG moment warmup, frozen-HG residual likelihood, optional explicit joint stage",
    )
    train.add_argument("--data", required=True, type=Path)
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--output", "--output-dir", required=True, type=Path)
    train.add_argument("--resume", type=Path)
    train.add_argument(
        "--max-steps-this-run",
        type=int,
        help="stop at an exact update boundary and preserve the original plan for resume",
    )
    train.add_argument(
        "--device",
        help="explicitly override the configured device; an exact resume requires the original device setting",
    )
    train.add_argument("--quiet", action="store_true")
    evaluate = commands.add_parser(
        "evaluate", help="condition-balanced NLL and HG/flow moment diagnostics"
    )
    evaluate.add_argument("--data", required=True, type=Path)
    evaluate.add_argument("--checkpoint", required=True, type=Path)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--split", choices=("all", "train", "validation"), default="all")
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--batch-size", type=int, default=8192)
    evaluate.add_argument(
        "--sample-count",
        type=int,
        default=0,
        help="additional model samples per condition for final-moment and sample/PDF checks",
    )
    evaluate.add_argument("--seed", type=int, default=2025)
    export = commands.add_parser(
        "export", help="export checkpoint weights and C++/CUDA inference parameters"
    )
    export.add_argument("--checkpoint", required=True, type=Path)
    export.add_argument("--output", "--output-dir", required=True, type=Path)
    export.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.command in (
        "inspect-rainbow", "train-rainbow", "evaluate-rainbow", "plot-rainbow"
    ):
        return _rainbow_main(arguments)
    if arguments.command in ("demo", "make-demo"):
        if arguments.output.exists():
            parser.error(f"{arguments.output} already exists; choose a new output path")
        cloud = make_demo(
            mode=arguments.mode,
            points_per_condition=arguments.points_per_condition,
            seed=arguments.seed,
            wavelengths_nm=tuple(arguments.wavelengths),
            incident_cosines=tuple(arguments.incident_cosines),
            n_mu=arguments.n_mu,
            n_phi=arguments.n_phi,
        )
        cloud.save_npz(arguments.output)
        _print_json({"output": str(arguments.output.resolve()), "data": cloud.summary()})
        return 0
    if arguments.command == "train":
        model_config, training_config = load_config(arguments.config)
        if arguments.device is not None:
            values = training_config.to_dict()
            values["device"] = arguments.device
            training_config = TrainingConfig.from_dict(values)
        result = train_model(
            PhasePointCloud.load_npz(arguments.data),
            model_config,
            training_config,
            arguments.output,
            resume=arguments.resume,
            max_steps_this_run=arguments.max_steps_this_run,
            callback=None if arguments.quiet else _progress,
        )
        _print_json(
            {
                "checkpoint": str(result.checkpoint_path.resolve()),
                "completed": result.completed,
                "metrics": result.metrics,
            }
        )
        return 0
    if arguments.command == "evaluate":
        model, checkpoint = load_model_checkpoint(arguments.checkpoint, device=arguments.device)
        cloud = PhasePointCloud.load_npz(arguments.data)
        groups = None
        scope = "all_supplied_conditions"
        if arguments.split != "all":
            if cloud.fingerprint() != checkpoint["dataset_fingerprint"]:
                raise ValueError(
                    "checkpoint condition splits require the original dataset; use --split all for another dataset"
                )
            groups = checkpoint["split"][arguments.split]
            if not groups:
                raise ValueError(
                    "this run has no held-out validation conditions; do not label in-sample metrics as validation"
                )
            scope = (
                "held_out_conditions" if arguments.split == "validation" else "in_sample_conditions"
            )
        report = evaluate_model(
            model,
            cloud,
            groups=groups,
            batch_size=arguments.batch_size,
            sample_count=arguments.sample_count,
            seed=arguments.seed,
            scope=scope,
        )
        report["checkpoint"] = str(arguments.checkpoint.resolve())
        report["dataset_fingerprint"] = cloud.fingerprint()
        if arguments.output is not None:
            atomic_json(arguments.output, report)
        _print_json(report)
        return 0
    if arguments.command == "export":
        from .export import export_model

        model, _ = load_model_checkpoint(arguments.checkpoint, device=arguments.device)
        export_model(model, arguments.output)
        _print_json({"output_directory": str(arguments.output.resolve())})
        return 0
    parser.error("unknown command")
    return 2


def _rainbow_main(arguments: argparse.Namespace) -> int:
    from .rainbow import RainbowReference
    from .single_condition import (
        FAMILY,
        SingleTrainingConfig,
        evaluate_single_condition,
        load_single_checkpoint,
        train_single_condition,
    )
    from .sphere_model import SphereFlowConfig

    def progress(entry: dict[str, Any]) -> None:
        value = {key: entry[key] for key in ("global_step", "loss") if key in entry}
        if "validation" in entry:
            value["validation_nll"] = entry["validation"]["nll"]
            value["validation_forward_kl"] = entry["validation"]["forward_kl_estimate"]
        print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)

    with RainbowReference(arguments.record) as reference:
        if arguments.command == "inspect-rainbow":
            _print_json(reference.summary())
            return 0
        if arguments.command == "train-rainbow":
            from .plotting import RainbowPlotConfig

            with arguments.config.open(encoding="utf-8") as handle:
                config = json.load(handle)
            if (
                not isinstance(config, dict)
                or not {"schema_version", "family", "model", "training"} <= set(config)
                or set(config) - {"schema_version", "family", "model", "training", "visualization"}
                or config["schema_version"] != 2
                or config["family"] != FAMILY
            ):
                raise ValueError(
                    "Rainbow config requires schema_version=2, family="
                    + FAMILY
                    + ", and explicit model/training objects"
                )
            model_config = SphereFlowConfig.from_dict(config["model"])
            training_values = config["training"].copy()
            if arguments.device is not None:
                training_values["device"] = arguments.device
            training_config = SingleTrainingConfig.from_dict(training_values)
            visualization = config.get("visualization", {})
            if not isinstance(visualization, dict):
                raise ValueError("visualization config must be an object")
            visualization = visualization.copy()
            enabled = visualization.pop("enabled", True)
            if type(enabled) is not bool:
                raise ValueError("visualization.enabled must be boolean")
            make_plots = enabled and not arguments.no_plots
            plot_config = RainbowPlotConfig.from_dict(visualization)
            result = train_single_condition(
                reference,
                model_config,
                training_config,
                arguments.output,
                resume=arguments.resume,
                max_steps_this_run=arguments.max_steps_this_run,
                callback=None if arguments.quiet else progress,
                make_plots=make_plots,
                plot_config=plot_config,
            )
            _print_json(
                {
                    "checkpoint": str(result.checkpoint_path.resolve()),
                    "best_checkpoint": str(result.best_path.resolve()),
                    "global_step": result.global_step,
                    "complete": result.complete,
                    "metrics": result.metrics,
                    "plots": (
                        str((arguments.output / "plots" / "plots.json").resolve())
                        if result.complete and make_plots else None
                    ),
                }
            )
            return 0
        import torch

        if arguments.cpu_threads < 1:
            raise ValueError("cpu-threads must be positive")
        torch.set_num_threads(arguments.cpu_threads)
        model, checkpoint = load_single_checkpoint(arguments.checkpoint, device=arguments.device)
        if checkpoint["dataset_fingerprint"] != reference.fingerprint():
            raise ValueError("evaluation record differs from the model's single-condition teacher")
        if arguments.command == "plot-rainbow":
            from .plotting import RainbowPlotConfig, plot_rainbow_comparison

            report = plot_rainbow_comparison(
                model,
                reference,
                arguments.output,
                config=RainbowPlotConfig(
                    cdf_samples=arguments.samples,
                    seed=arguments.seed,
                    eval_batch_size=arguments.batch_size,
                    dpi=arguments.dpi,
                    write_pdf=arguments.write_pdf,
                ),
                checkpoint_path=arguments.checkpoint,
                selected_step=(
                    checkpoint["global_step"] if checkpoint["kind"] == "inference" else None
                ),
                title=(
                    "Validation-selected model" if checkpoint["kind"] == "inference"
                    else "Training checkpoint model"
                ) + f" (step {checkpoint['global_step']})",
            )
            _print_json(report)
            return 0
        report = evaluate_single_condition(
            model,
            reference,
            samples=arguments.samples,
            seed=arguments.seed,
            batch_size=arguments.batch_size,
            proposal_samples=arguments.proposal_samples,
        )
        report["checkpoint"] = str(arguments.checkpoint.resolve())
        report["checkpoint_step"] = checkpoint["global_step"]
        report["training_code_fingerprint"] = checkpoint["code_fingerprint"]
        if arguments.output is not None:
            atomic_json(arguments.output, report)
        _print_json(report)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
