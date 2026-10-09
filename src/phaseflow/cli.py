"""Command-line adapters; all file formats and physical assumptions are explicit."""

from __future__ import annotations

import argparse
import hashlib
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
    rainbow_sweep = commands.add_parser(
        "sweep-rainbow", help="compare learning rates, then point counts with validation selection"
    )
    rainbow_sweep.add_argument("--record", required=True, type=Path)
    rainbow_sweep.add_argument("--config", required=True, type=Path)
    rainbow_sweep.add_argument("--sweep-config", required=True, type=Path)
    rainbow_sweep.add_argument("--output", "--output-dir", required=True, type=Path)
    rainbow_sweep.add_argument("--device")
    rainbow_sweep.add_argument("--max-trials-this-run", type=int)
    rainbow_sweep.add_argument("--no-plots", action="store_true")
    rainbow_sweep.add_argument("--quiet", action="store_true")
    capacity_sweep = commands.add_parser(
        "sweep-capacity-rainbow", help="compare spline bins along shared optimization milestones"
    )
    capacity_sweep.add_argument("--record", required=True, type=Path)
    capacity_sweep.add_argument("--config", required=True, type=Path)
    capacity_sweep.add_argument("--sweep-config", required=True, type=Path)
    capacity_sweep.add_argument("--output", required=True, type=Path)
    capacity_sweep.add_argument("--device")
    capacity_sweep.add_argument("--max-milestones-this-run", type=int)
    capacity_sweep.add_argument("--no-plots", action="store_true")
    capacity_sweep.add_argument("--quiet", action="store_true")
    sample_sweep = commands.add_parser(
        "sweep-samples-rainbow", help="recheck training pool sizes for a completed capacity winner"
    )
    sample_sweep.add_argument("--record", required=True, type=Path)
    sample_sweep.add_argument("--capacity", required=True, type=Path)
    sample_sweep.add_argument("--output", required=True, type=Path)
    sample_sweep.add_argument("--device")
    sample_sweep.add_argument("--train-samples", nargs="+", type=int,
                              default=[65536, 262144, 1048576])
    sample_sweep.add_argument("--max-trials-this-run", type=int)
    sample_sweep.add_argument("--quiet", action="store_true")
    sampling_audit = commands.add_parser(
        "audit-sampling-rainbow", help="compare a verified training pool to exact CDF region masses"
    )
    sampling_audit.add_argument("--record", required=True, type=Path)
    sampling_audit.add_argument("--run", required=True, type=Path)
    sampling_audit.add_argument("--output", required=True, type=Path)
    sampling_audit.add_argument("--device", default="cuda")
    sampling_audit.add_argument("--cpu-threads", type=int, default=1)
    sampling_audit.add_argument("--u-bins", type=int, default=32)
    sampling_audit.add_argument("--phi-bins", type=int, default=64)
    sampling_audit.add_argument("--no-plots", action="store_true")
    rainbow_replot = commands.add_parser(
        "replot-sweep-rainbow", help="redraw completed sweep maps with every verified training point"
    )
    rainbow_replot.add_argument("--record", required=True, type=Path)
    rainbow_replot.add_argument("--sweep", required=True, type=Path)
    rainbow_replot.add_argument("--output", type=Path)
    rainbow_replot.add_argument("--device", default="cuda")
    rainbow_replot.add_argument("--cpu-threads", type=int, default=1)
    rainbow_replot.add_argument("--quiet", action="store_true")
    rainbow_diagnose = commands.add_parser(
        "diagnose-rainbow", help="angular band mass and PDF profiles for an existing selected model"
    )
    rainbow_diagnose.add_argument("--record", required=True, type=Path)
    rainbow_diagnose.add_argument("--run", required=True, type=Path)
    rainbow_diagnose.add_argument("--sweep-config", required=True, type=Path)
    rainbow_diagnose.add_argument("--output", type=Path)
    rainbow_diagnose.add_argument("--device", default="cuda")
    rainbow_eval = commands.add_parser(
        "evaluate-rainbow", help="independent same-condition NLL, forward KL and importance ESS"
    )
    rainbow_eval.add_argument("--record", required=True, type=Path)
    rainbow_eval.add_argument("--checkpoint", required=True, type=Path)
    rainbow_eval.add_argument("--samples", type=int, default=65536)
    rainbow_eval.add_argument("--proposal-samples", type=int)
    rainbow_eval.add_argument(
        "--uniform-samples", type=int,
        help="independent solid-angle log-error queries; version-5 default from objective",
    )
    rainbow_eval.add_argument("--seed", type=int, default=2026)
    rainbow_eval.add_argument("--batch-size", type=int, default=4096)
    rainbow_eval.add_argument("--device", default="cuda")
    rainbow_eval.add_argument("--cpu-threads", type=int, default=1)
    rainbow_eval.add_argument("--output", type=Path)
    rainbow_plot = commands.add_parser(
        "plot-rainbow", help="aligned CDF PDF, complete training pool and NF PDF for one checkpoint"
    )
    rainbow_plot.add_argument("--record", required=True, type=Path)
    rainbow_plot.add_argument("--checkpoint", required=True, type=Path)
    rainbow_plot.add_argument("--output", "--output-dir", required=True, type=Path)
    rainbow_plot.add_argument("--device", default="cuda")
    rainbow_plot.add_argument("--scatter", choices=("training", "independent"), default="training")
    rainbow_plot.add_argument("--training-run", type=Path)
    rainbow_plot.add_argument("--samples", type=int, help="point count for --scatter independent only")
    rainbow_plot.add_argument("--seed", type=int, help="sample seed for --scatter independent only")
    rainbow_plot.add_argument("--batch-size", type=int, default=16384)
    rainbow_plot.add_argument("--dpi", type=int, default=160)
    rainbow_plot.add_argument("--write-pdf", action="store_true")
    rainbow_plot.add_argument("--cpu-threads", type=int, default=1)
    history_plot = commands.add_parser(
        "plot-history", help="regenerate learning curves from saved JSON/JSONL, including live logs"
    )
    history_plot.add_argument("--history", required=True, type=Path)
    history_plot.add_argument("--output", required=True, type=Path)
    history_plot.add_argument("--dpi", type=int, default=160)
    precision = commands.add_parser(
        "precision-rainbow", help="paired FP32/FP64 and FP16 weight-rounding diagnostics"
    )
    precision.add_argument("--record", required=True, type=Path)
    precision.add_argument("--checkpoint", required=True, type=Path)
    precision.add_argument("--output", type=Path)
    precision.add_argument("--samples", type=int, default=4096)
    precision.add_argument("--seed", type=int, default=2028)
    precision.add_argument("--batch-size", type=int, default=4096)
    precision.add_argument("--device", default="cuda")
    precision.add_argument("--cpu-threads", type=int, default=1)
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
        "inspect-rainbow", "train-rainbow", "evaluate-rainbow", "plot-rainbow", "precision-rainbow",
        "sweep-rainbow", "diagnose-rainbow", "replot-sweep-rainbow",
        "sweep-capacity-rainbow", "sweep-samples-rainbow", "audit-sampling-rainbow",
    ):
        return _rainbow_main(arguments)
    if arguments.command == "plot-history":
        from .monitoring import plot_training_history

        _print_json(plot_training_history(
            arguments.history, arguments.output, dpi=arguments.dpi,
        ))
        return 0
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


def _read_rainbow_config(
    path: Path, device: str | None = None, *, include_objective: bool = False,
):
    from .log_objective import LogObjectiveConfig
    from .plotting import RainbowPlotConfig
    from .single_condition import FAMILY, SingleTrainingConfig
    from .sphere_model import SphereFlowConfig

    with path.open(encoding="utf-8-sig") as handle:
        config = json.load(handle)
    allowed = {"schema_version", "family", "model", "training", "visualization"}
    objective = None
    version = config.get("schema_version") if isinstance(config, dict) else None
    if include_objective and version == 3:
        allowed.add("objective")
        objective = LogObjectiveConfig.from_dict(config.get("objective"))
    if (
        not isinstance(config, dict)
        or not {"schema_version", "family", "model", "training"} <= set(config)
        or set(config) - allowed
        or (version != 2 and not (include_objective and version == 3))
        or config["family"] != FAMILY
    ):
        raise ValueError(
            "Rainbow config requires schema_version=2 (NLL), or an explicitly supported "
            "schema_version=3 objective workflow, family="
            + FAMILY
            + ", and explicit model/training objects"
        )
    model_config = SphereFlowConfig.from_dict(config["model"])
    if not isinstance(config["training"], dict):
        raise ValueError("training config must be an object")
    training_values = config["training"].copy()
    if device is not None:
        training_values["device"] = device
    training_config = SingleTrainingConfig.from_dict(training_values)
    visualization = config.get("visualization", {})
    if not isinstance(visualization, dict):
        raise ValueError("visualization config must be an object")
    visualization = visualization.copy()
    enabled = visualization.pop("enabled", True)
    if type(enabled) is not bool:
        raise ValueError("visualization.enabled must be boolean")
    values = (model_config, training_config, enabled, RainbowPlotConfig.from_dict(visualization))
    return (*values, objective) if include_objective else values


def _read_sweep_config(path: Path):
    from .angular_diagnostics import AngularDiagnosticConfig
    from .sweep import SweepConfig

    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if (
        not isinstance(config, dict)
        or config.get("schema_version") != 1
        or not {"schema_version", "sweep", "diagnostics"} == set(config)
        or not isinstance(config["diagnostics"], dict)
    ):
        raise ValueError("sweep config requires schema_version=1, sweep and diagnostics objects")
    sweep_config = SweepConfig.from_dict(config["sweep"])
    diagnostics = config["diagnostics"].copy()
    enabled = diagnostics.pop("enabled", True)
    if type(enabled) is not bool:
        raise ValueError("diagnostics.enabled must be boolean")
    diagnostic_config = AngularDiagnosticConfig.from_dict(diagnostics)
    return sweep_config, enabled, diagnostic_config


def _read_capacity_config(path: Path):
    from .angular_diagnostics import AngularDiagnosticConfig
    from .capacity_sweep import CapacitySweepConfig

    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if (
        not isinstance(config, dict)
        or config.get("schema_version") != 1
        or set(config) != {"schema_version", "sweep", "diagnostics"}
        or not isinstance(config["diagnostics"], dict)
    ):
        raise ValueError("capacity config requires schema_version=1, sweep and diagnostics")
    sweep_config = CapacitySweepConfig.from_dict(config["sweep"])
    diagnostics = config["diagnostics"].copy()
    enabled = diagnostics.pop("enabled", True)
    if type(enabled) is not bool:
        raise ValueError("diagnostics.enabled must be boolean")
    return sweep_config, enabled, AngularDiagnosticConfig.from_dict(diagnostics)


def _rainbow_main(arguments: argparse.Namespace) -> int:
    from .rainbow import RainbowReference
    from .single_condition import (
        evaluate_single_condition,
        load_single_checkpoint,
        train_single_condition,
    )

    def progress(entry: dict[str, Any]) -> None:
        value = {key: entry[key] for key in (
            "global_step", "loss", "nll", "log_mse", "beta", "nll_weight",
            "gradient_norm_before_clip", "examples_seen", "effective_passes"
        ) if key in entry}
        if "validation" in entry:
            value["validation_nll"] = entry["validation"]["nll"]
            value["validation_forward_kl"] = entry["validation"]["forward_kl_estimate"]
            if "log_rmse" in entry["validation"]:
                value["validation_log_rmse"] = entry["validation"]["log_rmse"]
        if "train_monitor" in entry:
            value["train_fixed_subset_nll"] = entry["train_monitor"]["nll"]
        print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)

    with RainbowReference(arguments.record) as reference:
        if arguments.command == "inspect-rainbow":
            _print_json(reference.summary())
            return 0
        if arguments.command == "audit-sampling-rainbow":
            import torch

            from .sampling_audit import audit_training_sampling

            if arguments.cpu_threads < 1:
                raise ValueError("cpu-threads must be positive")
            torch.set_num_threads(arguments.cpu_threads)
            _print_json(audit_training_sampling(
                reference, arguments.run, arguments.output,
                device=arguments.device, u_bins=arguments.u_bins, phi_bins=arguments.phi_bins,
                plot=not arguments.no_plots,
            ))
            return 0
        if arguments.command == "sweep-samples-rainbow":
            from . import angular_diagnostics
            from .capacity_sweep import CAPACITY_SCHEMA, run_capacity_sample_sweep

            with (arguments.capacity / "manifest.json").open(encoding="utf-8") as handle:
                parent = json.load(handle)
            if not isinstance(parent, dict) or parent.get("schema") != CAPACITY_SCHEMA:
                raise ValueError("--capacity must identify a completed bin/step sweep")
            recorded_diagnostics = parent.get("diagnostics")
            diagnostic_config = diagnostic_identity = None
            if recorded_diagnostics is not None:
                if (
                    not isinstance(recorded_diagnostics, dict)
                    or recorded_diagnostics.get("id") != "phaseflow.angular_diagnostics.v1"
                ):
                    raise ValueError("unsupported saved angular diagnostic configuration")
                diagnostic_config = angular_diagnostics.AngularDiagnosticConfig.from_dict(
                    recorded_diagnostics["configuration"],
                )
                diagnostic_identity = {
                    "id": "phaseflow.angular_diagnostics.v1",
                    "configuration": diagnostic_config.to_dict(),
                    "implementation_sha256": hashlib.sha256(
                        Path(angular_diagnostics.__file__).read_bytes()
                    ).hexdigest(),
                }

            def sample_diagnostics(model, source, trial_directory):
                _, trial_config, _, _ = _read_rainbow_config(trial_directory / "config.json")
                return angular_diagnostics.evaluate_angular_diagnostics(
                    model, source, trial_directory / "diagnostics",
                    train_samples=trial_config.train_samples,
                    batch_size=trial_config.batch_size,
                    config=diagnostic_config,
                    checkpoint_path=trial_directory / "best.pt",
                )

            def sample_progress(entry):
                print(json.dumps(entry, sort_keys=True, allow_nan=False), flush=True)

            report = run_capacity_sample_sweep(
                reference, arguments.capacity, arguments.output,
                train_samples=tuple(arguments.train_samples), device=arguments.device,
                callback=None if arguments.quiet else sample_progress,
                max_trials_this_run=arguments.max_trials_this_run,
                diagnostic_callback=sample_diagnostics if diagnostic_config is not None else None,
                diagnostic_config=diagnostic_identity,
            )
            _print_json({
                "complete": report["status"] == "complete", "status": report["status"],
                "output_directory": str(arguments.output.resolve()),
                "summary": str((arguments.output / "summary.json").resolve()),
                "table": str((arguments.output / "summary.csv").resolve()),
                "report": str((arguments.output / "summary.md").resolve()),
            })
            return 0
        if arguments.command == "sweep-capacity-rainbow":
            from . import angular_diagnostics
            from .capacity_sweep import run_capacity_sweep

            model_config, training_config, enabled, plot_config = _read_rainbow_config(
                arguments.config, arguments.device,
            )
            sweep_config, diagnose, diagnostic_config = _read_capacity_config(
                arguments.sweep_config,
            )

            def capacity_diagnostics(model, source, trial_directory):
                _, trial_config, _, _ = _read_rainbow_config(trial_directory / "config.json")
                return angular_diagnostics.evaluate_angular_diagnostics(
                    model, source, trial_directory / "diagnostics",
                    train_samples=trial_config.train_samples,
                    batch_size=trial_config.batch_size,
                    config=diagnostic_config,
                    checkpoint_path=trial_directory / "best.pt",
                )

            def capacity_progress(entry):
                print(json.dumps(entry, sort_keys=True, allow_nan=False), flush=True)

            report = run_capacity_sweep(
                reference, model_config, training_config, arguments.output,
                sweep_config=sweep_config, make_plots=enabled and not arguments.no_plots,
                plot_config=plot_config,
                callback=None if arguments.quiet else capacity_progress,
                max_milestones_this_run=arguments.max_milestones_this_run,
                diagnostic_callback=capacity_diagnostics if diagnose else None,
                diagnostic_config=(
                    {
                        "id": "phaseflow.angular_diagnostics.v1",
                        "configuration": diagnostic_config.to_dict(),
                        "implementation_sha256": hashlib.sha256(
                            Path(angular_diagnostics.__file__).read_bytes()
                        ).hexdigest(),
                    } if diagnose else None
                ),
            )
            _print_json({
                "complete": report["status"] == "complete", "status": report["status"],
                "output_directory": str(arguments.output.resolve()),
                "summary": str((arguments.output / "summary.json").resolve()),
                "table": str((arguments.output / "summary.csv").resolve()),
                "report": str((arguments.output / "summary.md").resolve()),
                "plot": str((arguments.output / "summary.png").resolve()),
            })
            return 0
        if arguments.command == "replot-sweep-rainbow":
            import torch

            from .replot import replot_sweep_training_points

            if arguments.cpu_threads < 1:
                raise ValueError("cpu-threads must be positive")
            torch.set_num_threads(arguments.cpu_threads)

            def replot_progress(entry):
                print(json.dumps(entry, sort_keys=True, allow_nan=False), flush=True)

            _print_json(replot_sweep_training_points(
                reference, arguments.sweep, arguments.output,
                device=arguments.device,
                callback=None if arguments.quiet else replot_progress,
            ))
            return 0
        if arguments.command in ("train-rainbow", "sweep-rainbow"):
            parsed = _read_rainbow_config(
                arguments.config, arguments.device,
                include_objective=arguments.command == "train-rainbow",
            )
            model_config, training_config, enabled, plot_config = parsed[:4]
            make_plots = enabled and not arguments.no_plots
            if arguments.command == "sweep-rainbow":
                from . import angular_diagnostics
                from .sweep import run_single_condition_sweep

                sweep_config, diagnose, diagnostic_config = _read_sweep_config(
                    arguments.sweep_config,
                )

                def diagnostics(model, source, trial_directory):
                    _, trial_config, _, _ = _read_rainbow_config(trial_directory / "config.json")
                    return angular_diagnostics.evaluate_angular_diagnostics(
                        model, source, trial_directory / "diagnostics",
                        train_samples=trial_config.train_samples,
                        batch_size=trial_config.batch_size,
                        config=diagnostic_config,
                        checkpoint_path=trial_directory / "best.pt",
                    )

                def sweep_progress(entry):
                    print(json.dumps(entry, sort_keys=True, allow_nan=False), flush=True)

                report = run_single_condition_sweep(
                    reference, model_config, training_config, arguments.output,
                    sweep_config=sweep_config,
                    make_plots=make_plots,
                    plot_config=plot_config,
                    callback=None if arguments.quiet else sweep_progress,
                    max_trials_this_run=arguments.max_trials_this_run,
                    diagnostic_callback=diagnostics if diagnose else None,
                    diagnostic_config=(
                        {
                            "id": "phaseflow.angular_diagnostics.v1",
                            "configuration": diagnostic_config.to_dict(),
                            "implementation_sha256": hashlib.sha256(
                                Path(angular_diagnostics.__file__).read_bytes()
                            ).hexdigest(),
                        } if diagnose else None
                    ),
                )
                _print_json({
                    "complete": report["status"] == "complete",
                    "status": report["status"],
                    "output_directory": str(arguments.output.resolve()),
                    "summary": str((arguments.output / "summary.json").resolve()),
                    "table": str((arguments.output / "summary.csv").resolve()),
                    "report": str((arguments.output / "summary.md").resolve()),
                    "plot": str((arguments.output / "summary.png").resolve()),
                })
                return 0
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
                objective_config=parsed[4],
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
        if arguments.command == "diagnose-rainbow":
            import torch

            from .angular_diagnostics import evaluate_angular_diagnostics

            model_config, training_config, _, _, objective = _read_rainbow_config(
                arguments.run / "config.json", include_objective=True,
            )
            _, _, diagnostic_config = _read_sweep_config(arguments.sweep_config)
            torch.set_num_threads(training_config.cpu_threads)
            checkpoint_path = arguments.run / "best.pt"
            model, checkpoint = load_single_checkpoint(checkpoint_path, device=arguments.device)
            if checkpoint["dataset_fingerprint"] != reference.fingerprint():
                raise ValueError("diagnostic record differs from the model's single-condition teacher")
            if (
                checkpoint["kind"] != "inference"
                or checkpoint["model_config"] != model_config.to_dict()
                or checkpoint["dtype"] != training_config.dtype
            ):
                raise ValueError("selected checkpoint and saved run configuration disagree")
            _print_json(evaluate_angular_diagnostics(
                model, reference, arguments.output or arguments.run / "diagnostics",
                train_samples=training_config.train_samples,
                batch_size=training_config.batch_size,
                config=diagnostic_config,
                checkpoint_path=checkpoint_path,
                objective_config=objective,
            ))
            return 0
        import torch

        if arguments.cpu_threads < 1:
            raise ValueError("cpu-threads must be positive")
        torch.set_num_threads(arguments.cpu_threads)
        if arguments.command == "precision-rainbow":
            from .precision import evaluate_precision

            _print_json(evaluate_precision(
                arguments.checkpoint, reference, arguments.output,
                device=arguments.device, samples=arguments.samples,
                seed=arguments.seed, batch_size=arguments.batch_size,
            ))
            return 0
        model, checkpoint = load_single_checkpoint(arguments.checkpoint, device=arguments.device)
        if checkpoint["dataset_fingerprint"] != reference.fingerprint():
            raise ValueError("evaluation record differs from the model's single-condition teacher")
        if arguments.command == "plot-rainbow":
            from .plotting import RainbowPlotConfig, plot_rainbow_comparison

            if arguments.scatter == "training" and (
                arguments.samples is not None or arguments.seed is not None
            ):
                raise ValueError(
                    "training scatter uses the saved training count/data seed; "
                    "--samples and --seed require --scatter independent"
                )
            if arguments.scatter == "independent" and arguments.training_run is not None:
                raise ValueError("--training-run requires --scatter training")
            report = plot_rainbow_comparison(
                model,
                reference,
                arguments.output,
                config=RainbowPlotConfig(
                    cdf_samples=32768 if arguments.samples is None else arguments.samples,
                    seed=2027 if arguments.seed is None else arguments.seed,
                    eval_batch_size=arguments.batch_size,
                    dpi=arguments.dpi,
                    write_pdf=arguments.write_pdf,
                ),
                checkpoint_path=arguments.checkpoint,
                scatter_mode=arguments.scatter,
                training_run_directory=arguments.training_run,
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
        uniform_samples = arguments.uniform_samples
        if uniform_samples is None:
            uniform_samples = checkpoint.get("objective_config", {}).get("test_uniform_samples", 0)
        report = evaluate_single_condition(
            model,
            reference,
            samples=arguments.samples,
            seed=arguments.seed,
            batch_size=arguments.batch_size,
            proposal_samples=arguments.proposal_samples,
            uniform_samples=uniform_samples,
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
