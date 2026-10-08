"""Compare spline bins along exact, predeclared optimization trajectories.

The trainer and its checkpoint/resume contract remain unchanged. Each bin count
plans the same final number of optimizer updates, pauses at specified evaluation
boundaries, and archives its validation-selected model before continuing. The
archives are independent of the mutable continuation directory. They represent
prefixes of one trajectory, not separately initialized experimental repetitions.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import torch

from .monitoring import check_tensorboard, plot_training_history
from .plotting import RainbowPlotConfig, plot_rainbow_comparison
from .rainbow import RainbowReference
from .single_condition import (
    SingleTrainingConfig,
    _array_hash,
    _code_hash,
    _points,
    _resolve_device,
    _runtime,
    load_single_checkpoint,
    read_single_checkpoint,
    train_single_condition,
)
from .sphere_model import SingleConditionSphereFlow, SphereFlowConfig
from .sweep import _atomic_bytes, _check_checkpoint, _file_hash, _manifest_hash, _read_json
from .training import _setup_runtime
from .training_scatter import _state_equal

CAPACITY_SCHEMA = "phaseflow.capacity_sweep.v1"
SAMPLE_SCHEMA = "phaseflow.capacity_sample_sweep.v1"
_RECEIPT = "milestone_completed.json"
_CORE_FILES = (
    "checkpoint.pt", "best.pt", "config.json", "data_summary.json",
    "sample_split.json", "history.json", "metrics.json",
)
DiagnosticCallback = Callable[
    [SingleConditionSphereFlow, RainbowReference, Path], dict[str, Any]
]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_bytes(path, contents.encode("utf-8"))


@dataclass(frozen=True)
class CapacitySweepConfig:
    num_bins: tuple[int, ...] = (16, 32, 64, 128)
    milestones: tuple[int, ...] = (2000, 6000, 10000, 20000)

    def __post_init__(self) -> None:
        for name in ("num_bins", "milestones"):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)) or not values:
                raise ValueError(f"{name} must be a nonempty list or tuple")
            if any(type(value) is not int or value < 1 for value in values):
                raise ValueError(f"{name} must contain positive integers")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must be unique")
            if list(values) != sorted(values):
                raise ValueError(f"{name} must be in increasing order")
            object.__setattr__(self, name, tuple(values))

    def to_dict(self) -> dict[str, Any]:
        return {name: list(value) for name, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> CapacitySweepConfig:
        if not isinstance(values, dict):
            raise ValueError("capacity sweep configuration must be an object")
        try:
            return cls(**values)
        except TypeError as error:
            raise ValueError(f"Invalid capacity sweep configuration: {error}") from error


def _implementation() -> dict[str, str]:
    root = Path(__file__).parent
    return {name: _file_hash(root / name) for name in (
        "capacity_sweep.py", "sweep.py", "plotting.py", "training_scatter.py",
    )}


def _prepare_runtime(cfg: SingleTrainingConfig) -> torch.device:
    if cfg.tensorboard:
        check_tensorboard()
    if torch.device(cfg.device).type == "cuda" and cfg.deterministic:
        workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if workspace not in (":4096:8", ":16:8"):
            raise ValueError("deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
    device = _resolve_device(cfg.device)
    _setup_runtime(cfg)
    return device


def _streams(reference: RainbowReference, cfg: SingleTrainingConfig,
             counts: tuple[int, ...] | None = None) -> dict[str, Any]:
    seed = cfg.seed if cfg.data_seed is None else cfg.data_seed
    sizes = sorted(set(counts or (cfg.train_samples,)))
    largest = _points(reference, sizes[-1], seed, "train")
    prefixes = {str(n): _array_hash(*(array[:n] for array in largest)) for n in sizes}
    del largest
    return {
        "data_seed": seed,
        "training_prefix_sha256": prefixes,
        "validation_points_sha256": _array_hash(
            *_points(reference, cfg.validation_samples, seed, "validation")
        ),
        "test_points_sha256": _array_hash(
            *_points(reference, cfg.test_samples, seed, "test")
        ),
        "nesting": "training pools are prefixes of the same deterministic CDF stream",
    }


def _safe_tree(root: Path) -> None:
    # Generated paths never need symlinks. Check ancestors too: an otherwise
    # ordinary output path can point into an existing result through a link.
    for path in (root, *root.parents):
        if path.is_symlink():
            raise ValueError(f"sweep output paths must not use symbolic links: {path}")
    if root.exists():
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"sweep output must not contain symbolic links: {path}")


def _open_output(output: Path, manifest: dict[str, Any]) -> None:
    _safe_tree(output)
    path = output / "manifest.json"
    if path.exists():
        if _read_json(path) != manifest:
            raise ValueError("sweep data/configuration/code/runtime changed; use a new output directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("sweep output is not empty and has no manifest; choose a new folder")
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(path, manifest)


def _trial_manifest(manifest: dict[str, Any], bins: int) -> dict[str, Any]:
    return {**manifest, "model": manifest["models"][str(bins)]}


def _rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = SingleTrainingConfig.from_dict(manifest["base_training"])
    plan = CapacitySweepConfig.from_dict(manifest["sweep"])
    return [{
        "trial_id": f"bins_{bins}",
        "num_bins": bins,
        "milestone_step": step,
        "planned_steps": cfg.steps,
        "learning_rate": cfg.learning_rate,
        "train_samples": cfg.train_samples,
        "batch_size": cfg.batch_size,
        "run_directory": f"bins/k_{bins}/milestones/updates_{step}",
        "training_directory": f"bins/k_{bins}/training",
        "status": "pending",
    } for bins in plan.num_bins for step in plan.milestones]


def _snapshot_fields(path: Path, row: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    cfg = SingleTrainingConfig.from_dict(manifest["base_training"])
    checkpoint = read_single_checkpoint(path / "checkpoint.pt")
    _check_checkpoint(checkpoint, cfg, _trial_manifest(manifest, row["num_bins"]))
    step = row["milestone_step"]
    if checkpoint["global_step"] != step:
        raise ValueError("milestone checkpoint does not end at the recorded update")
    if type(checkpoint["best_step"]) is not int or not 0 <= checkpoint["best_step"] <= step:
        raise ValueError("selected checkpoint step must be within the archived training prefix")
    metrics, best = _read_json(path / "metrics.json"), read_single_checkpoint(path / "best.pt")
    complete = step == cfg.steps
    if (
        metrics.get("complete") is not complete
        or metrics.get("global_step") != step
        or metrics.get("selected_step") != checkpoint["best_step"]
        or metrics.get("best_validation") != checkpoint["best_validation"]
        or metrics.get("dataset_fingerprint") != manifest["dataset_fingerprint"]
    ):
        raise ValueError("milestone metrics and training checkpoint disagree")
    validation = checkpoint["best_validation"]
    latest = checkpoint["latest_validation"]
    if not math.isfinite(validation["nll"]) or not math.isfinite(latest["nll"]):
        raise ValueError("milestone has no finite validation result")
    if (
        best["kind"] != "inference"
        or best["global_step"] != checkpoint["best_step"]
        or best.get("validation") != validation
        or any(best.get(key) != checkpoint.get(key) for key in (
            "model_config", "physics", "dtype", "dataset_fingerprint", "code_fingerprint",
            "runtime",
        ))
        or not _state_equal(best["model_state"], checkpoint["best_state"])
    ):
        raise ValueError("milestone best.pt differs from the validation-selected checkpoint")
    expected_config = {
        "schema_version": 2, "family": checkpoint["family"],
        "model": manifest["models"][str(row["num_bins"])], "training": cfg.to_dict(),
        "visualization": {"enabled": False, **manifest["plot_configuration"]},
    }
    if _read_json(path / "config.json") != expected_config:
        raise ValueError("milestone saved configuration differs from the sweep")
    if _read_json(path / "sample_split.json") != checkpoint["sample_split"]:
        raise ValueError("milestone sample split differs from its checkpoint")
    history = _read_json(path / "history.json")
    if history != checkpoint["history"]:
        raise ValueError("milestone history differs from its checkpoint")
    if (
        len(history) != step + 1
        or any(type(entry.get("global_step")) is not int for entry in history)
        or [entry["global_step"] for entry in history] != list(range(step + 1))
    ):
        raise ValueError("milestone history must contain exactly one ordered entry per update")
    evaluated = [entry for entry in history if "validation" in entry]
    if [entry["global_step"] for entry in evaluated] != list(range(0, step + 1, cfg.eval_every)):
        raise ValueError("milestone history does not match the scheduled validation evaluations")
    if any(not math.isfinite(entry["validation"]["nll"]) for entry in evaluated):
        raise ValueError("milestone history contains a nonfinite validation NLL")
    first_minimum = min(evaluated, key=lambda entry: entry["validation"]["nll"])
    if first_minimum["global_step"] != checkpoint["best_step"] or (
        first_minimum["validation"] != validation
    ):
        raise ValueError("selected checkpoint is not the first minimum validation NLL in its prefix")
    if not history or history[-1]["global_step"] != step or (
        history[-1].get("validation") != latest
    ):
        raise ValueError("milestone must coincide with a scheduled validation evaluation")
    selected_history = next(
        (entry for entry in history if entry["global_step"] == checkpoint["best_step"]), None
    )
    if selected_history is None or selected_history.get("validation") != validation:
        raise ValueError("selected validation checkpoint is absent from milestone history")
    test = metrics.get("test")
    if complete:
        if not isinstance(test, dict) or test.get("seed") != manifest["fixed_streams"]["data_seed"] or (
            test.get("sample_count") != cfg.test_samples
        ):
            raise ValueError("final milestone must report the common independent test stream")
    elif test is not None:
        raise ValueError("an intermediate milestone must not evaluate final test data")
    parameter_count = sum(
        value.numel() for name, value in best["model_state"].items()
        if name.endswith("weight") or name.endswith("bias")
    )
    monitor = selected_history.get("train_monitor")
    results = {
        "global_step": step,
        "complete_training_plan": complete,
        "selected_step": checkpoint["best_step"],
        "processed_examples": step * cfg.batch_size,
        "effective_passes": step * cfg.batch_size / cfg.train_samples,
        "validation_nll": validation["nll"],
        "validation_kl": validation["forward_kl_estimate"],
        "validation_kl_standard_error": validation["forward_kl_standard_error"],
        "validation_hg_kl": validation["hg_forward_kl_estimate"],
        "latest_validation_nll": latest["nll"],
        "latest_validation_kl": latest["forward_kl_estimate"],
        "latest_validation_kl_standard_error": latest["forward_kl_standard_error"],
        "train_monitor_nll_at_selected_step": None if monitor is None else monitor["nll"],
        "test_nll": None if test is None else test["nll"],
        "test_kl": None if test is None else test["forward_kl_estimate"],
        "test_kl_standard_error": None if test is None else test["forward_kl_standard_error"],
        "test_relative_ess": None if test is None else (test.get("proposal") or {}).get("relative_ess"),
        "parameter_count": parameter_count,
        "parameter_bytes": parameter_count * (4 if cfg.dtype == "float32" else 8),
        "checkpoint_bytes": (path / "best.pt").stat().st_size,
        "sample_split": checkpoint["sample_split"],
    }
    if manifest["diagnostics"] is not None:
        results["diagnostics"] = _read_json(path / "diagnostics.json")
    return results


def _snapshot_identity(row: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": CAPACITY_SCHEMA,
        "manifest_sha256": _manifest_hash(manifest),
        "num_bins": row["num_bins"],
        "milestone_step": row["milestone_step"],
        "planned_steps": row["planned_steps"],
        "scope": "validation-selected prefix of one predeclared optimization trajectory",
    }


def _verify_snapshot(path: Path, row: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    receipt = _read_json(path / _RECEIPT)
    if receipt.get("identity") != _snapshot_identity(row, manifest):
        raise ValueError("milestone receipt belongs to another experiment or update")
    hashes = receipt.get("files_sha256", {})
    required = {*_CORE_FILES, "snapshot.json"}
    if manifest["visualization_enabled"]:
        required.update((
            "learning_curves.png", "plots/plots.json", "plots/reference_pdf.png",
            "plots/cdf_samples.png", "plots/nf_pdf.png", "plots/comparison.png",
            "plots/training_scatter.npz",
        ))
    if manifest["diagnostics"] is not None:
        required.add("diagnostics.json")
    if not isinstance(hashes, dict) or not required <= hashes.keys():
        raise ValueError("milestone receipt is missing required artifact hashes")
    actual = {item.relative_to(path).as_posix() for item in path.rglob("*")
              if item.is_file() and item.name != _RECEIPT}
    if actual != set(hashes):
        raise ValueError("milestone artifact set changed after completion")
    for relative, expected in hashes.items():
        target = path / relative
        if not target.resolve().is_relative_to(path.resolve()) or target.is_symlink():
            raise ValueError("invalid path in milestone receipt")
        if not target.is_file() or _file_hash(target) != expected:
            raise ValueError(f"milestone file changed or missing: {target}")
    metadata = _read_json(path / "snapshot.json")
    if metadata.get("identity") != receipt["identity"] or (
        metadata.get("source_files_sha256") != {name: hashes[name] for name in _CORE_FILES}
    ):
        raise ValueError("milestone source identity differs from archived files")
    fields = _snapshot_fields(path, row, manifest)
    if fields != receipt.get("results"):
        raise ValueError("milestone results differ from their completion receipt")
    return {**fields, "elapsed_seconds": receipt["elapsed_seconds"]}


def _archive_milestone(
    source: Path, destination: Path, row: dict[str, Any], manifest: dict[str, Any],
    model: SingleConditionSphereFlow, reference: RainbowReference,
    plotting: RainbowPlotConfig, diagnostic_callback: DiagnosticCallback | None,
) -> dict[str, Any]:
    source_hashes = {name: _file_hash(source / name) for name in _CORE_FILES}
    metadata = {
        "identity": _snapshot_identity(row, manifest),
        "source_directory": row["training_directory"],
        "source_files_sha256": source_hashes,
        "checkpoint_plan": "training_config.steps remains the common final milestone",
        "selection": "minimum validation NLL among scheduled evaluations up to this milestone",
    }
    metadata_path = destination / "snapshot.json"
    if metadata_path.exists():
        if _read_json(metadata_path) != metadata:
            raise ValueError("unfinished milestone snapshot differs from the saved continuation state")
    else:
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError("nonempty milestone directory has no snapshot identity")
        destination.mkdir(parents=True, exist_ok=True)
        _atomic_json(metadata_path, metadata)
    for name, expected in source_hashes.items():
        path = destination / name
        if path.exists():
            if _file_hash(path) != expected:
                raise ValueError(f"unfinished milestone source file changed: {path}")
        else:
            _atomic_bytes(path, (source / name).read_bytes())
    if manifest["visualization_enabled"]:
        plot_training_history(destination / "history.json", destination / "learning_curves.png")
        plot_rainbow_comparison(
            model, reference, destination / "plots", config=plotting,
            checkpoint_path=destination / "best.pt", training_run_directory=destination,
            title=(f"K = {row['num_bins']}; up to {row['milestone_step']:,} updates; "
                   f"selected step {_selected_step(destination)}"),
            selected_step=_selected_step(destination), scatter_mode="training",
        )
    if diagnostic_callback is not None:
        diagnostics = diagnostic_callback(model, reference, destination)
        if not isinstance(diagnostics, dict):
            raise ValueError("diagnostic callback must return a JSON object")
        _atomic_json(destination / "diagnostics.json", diagnostics)
    # Check the copied scientific state independently before making it reusable.
    return _snapshot_fields(destination, row, manifest)


def _selected_step(directory: Path) -> int:
    return read_single_checkpoint(directory / "best.pt")["global_step"]


def _complete_snapshot(path: Path, row: dict[str, Any], manifest: dict[str, Any],
                       results: dict[str, Any], elapsed: float) -> dict[str, Any]:
    receipt = {
        "identity": _snapshot_identity(row, manifest),
        "files_sha256": {
            item.relative_to(path).as_posix(): _file_hash(item)
            for item in sorted(path.rglob("*")) if item.is_file() and item.name != _RECEIPT
        },
        "results": results,
        "elapsed_seconds": elapsed,
        "timing_scope": "cumulative recorded attempts including CDF loading/evaluation/plots; not GPU kernel time",
    }
    _atomic_json(path / _RECEIPT, receipt)
    return {**results, "elapsed_seconds": elapsed}


_COLUMNS = (
    "trial_id", "num_bins", "train_samples", "learning_rate", "batch_size", "status",
    "milestone_step", "planned_steps", "global_step", "selected_step",
    "processed_examples", "effective_passes", "validation_nll", "validation_kl",
    "validation_kl_standard_error", "latest_validation_nll", "latest_validation_kl",
    "latest_validation_kl_standard_error", "validation_hg_kl", "test_nll", "test_kl",
    "test_kl_standard_error", "test_relative_ess", "elapsed_seconds", "parameter_count",
    "parameter_bytes", "checkpoint_bytes", "run_directory", "training_directory", "reuse_of",
)


def _summary_plot(output: Path, summary: dict[str, Any]) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    figure = Figure(figsize=(12, 4.8), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    rows = [row for row in summary["trials"] if row["status"] == "complete"]
    for bins in sorted({row["num_bins"] for row in rows}):
        group = sorted((row for row in rows if row["num_bins"] == bins),
                       key=lambda row: row["milestone_step"])
        for axis, metric, title in zip(axes, ("validation_kl", "latest_validation_kl"), (
            "Best validation checkpoint up to each update", "Last iterate at each update",
        ), strict=True):
            axis.errorbar(
                [row["milestone_step"] for row in group], [row[metric] for row in group],
                yerr=[1.96 * row[metric + "_standard_error"] for row in group],
                marker="o", capsize=3, label=f"K = {bins}",
            )
            axis.set_title(title)
    for axis in axes:
        axis.set_xlabel("Optimizer updates")
        axis.set_ylabel("Validation forward KL [nats]")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.25)
        if rows:
            axis.axhline(rows[0]["validation_hg_kl"], color="0.5", ls="--", label="HG")
            axis.legend(fontsize=8)
        else:
            axis.text(0.5, 0.5, "No completed milestones yet", ha="center", transform=axis.transAxes)
    figure.suptitle("One training trajectory per K; shared fixed points\n"
                    "Bars: ±1.96 Monte Carlo SE, excluding seed and selection uncertainty", fontsize=10)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=140)
    _atomic_bytes(output / "summary.png", buffer.getvalue())
    figure.clear()


def _write_table(output: Path, summary: dict[str, Any]) -> None:
    _atomic_json(output / "summary.json", summary)
    diagnostic_keys = sorted({
        key for row in summary["trials"] for key, value in row.get("diagnostics", {}).items()
        if value is None or type(value) in (bool, int, float, str)
    })
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=[*_COLUMNS, *(f"diag_{key}" for key in diagnostic_keys)],
        extrasaction="ignore", lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows({**row, **{
        f"diag_{key}": row.get("diagnostics", {}).get(key) for key in diagnostic_keys
    }} for row in summary["trials"])
    _atomic_bytes(output / "summary.csv", buffer.getvalue().encode("utf-8-sig"))


def _write_summary(output: Path, summary: dict[str, Any]) -> None:
    _write_table(output, summary)

    def number(value: Any) -> str:
        return "-" if value is None else f"{value:.6g}"

    lines = [
        "# Spline bins and optimizer updates", "", f"Status: **{summary['status']}**.", "",
        "Each K follows one trajectory to the predeclared common final update.",
        "Milestones archive best-so-far validation checkpoints; prefixes are not retrained.",
        "Bin selection uses only the final milestone's best validation NLL; exact ties use smaller K.",
        "Intermediate milestones have no final test evaluation. Final test is reporting only.", "",
        "| K | Updates | Best step | N | Status | Best validation KL | Latest validation KL | Test KL |",
        "|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in summary["trials"]:
        link = f"[{row['milestone_step']}]({row['run_directory']}/metrics.json)"
        lines.append(
            f"| {row['num_bins']} | {link} | {row.get('selected_step', '-')} | "
            f"{row['train_samples']} | {row['status']} | {number(row.get('validation_kl'))} | "
            f"{number(row.get('latest_validation_kl'))} | {number(row.get('test_kl'))} |"
        )
    if summary.get("selection"):
        lines.extend(["", f"Selected bins: **{summary['selection']['selected_num_bins']}**."])
    lines.extend([
        "", "![Validation results](summary.png)", "",
        "Best-so-far validation loss cannot increase along a trajectory by construction;",
        "the latest-iterate column and complete learning histories show optimization behavior.",
        "Error bars describe Monte Carlo uncertainty for each model, not differences between",
        "models, repeated-seed uncertainty, or the uncertainty introduced by selection.",
        "The same seeds and fixed point pools are used. Different model shapes do not imply",
        "identical parameter initializations. Every enabled scatter shows its full training pool.",
        "Checkpoint training_config.steps stays at the common final update in every archive.",
        "Archives are not modified when optimization continues in the training directory.",
        "No loss weighting, smoothing, target modification, precision change, or CPU fallback is used.",
    ])
    if summary.get("error"):
        lines.extend(["", "## Stopped milestone", "", summary["error"]])
    _atomic_bytes(output / "summary.md", ("\n".join(lines) + "\n").encode("utf-8"))
    _summary_plot(output, summary)


def _selection(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    final = manifest["base_training"]["steps"]
    candidates = [row for row in rows if row["milestone_step"] == final]
    if not candidates or any(row["status"] != "complete" for row in candidates):
        raise ValueError("capacity selection requires all candidates at the common final update")
    winner = min(candidates, key=lambda row: (row["validation_nll"], row["num_bins"]))
    return {
        "schema": CAPACITY_SCHEMA,
        "manifest_sha256": _manifest_hash(manifest),
        "selected_num_bins": winner["num_bins"],
        "selected_trial_id": winner["trial_id"],
        "selected_run_directory": winner["run_directory"],
        "selected_validation_nll": winner["validation_nll"],
        "selected_step": winner["selected_step"],
        "common_final_update": final,
        "rule": manifest["selection"],
        "test_used": False,
        "candidates": [{key: row[key] for key in (
            "trial_id", "num_bins", "validation_nll", "validation_kl", "selected_step",
            "milestone_step", "run_directory",
        )} for row in candidates],
    }


def run_capacity_sweep(
    reference: RainbowReference,
    model_config: SphereFlowConfig,
    training_config: SingleTrainingConfig,
    output_directory: str | Path,
    *,
    sweep_config: CapacitySweepConfig | None = None,
    make_plots: bool = True,
    plot_config: RainbowPlotConfig | None = None,
    callback: Callable[[dict[str, Any]], None] | None = None,
    max_milestones_this_run: int | None = None,
    diagnostic_callback: DiagnosticCallback | None = None,
    diagnostic_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run or exactly resume a GPU-default bins/updates experiment.

    ``training_config.steps`` must already equal the final planned milestone;
    neither that field nor the optimizer/RNG/source/runtime resume checks change.
    ``max_milestones_this_run`` stops after that many newly archived milestones.
    A process interrupted after optimization but before archival recovers the
    same update by a zero-update trainer resume, then completes its snapshot.
    All callbacks for angular diagnostics are reporting-only.
    """
    cfg = training_config
    plan = CapacitySweepConfig() if sweep_config is None else sweep_config
    plotting = RainbowPlotConfig() if plot_config is None else plot_config
    if not isinstance(plan, CapacitySweepConfig) or not isinstance(plotting, RainbowPlotConfig):
        raise ValueError("invalid capacity or plotting configuration")
    if cfg.steps != plan.milestones[-1]:
        raise ValueError("training.steps must equal the final capacity milestone from the start")
    if any(step % cfg.eval_every for step in plan.milestones):
        raise ValueError("each milestone must be a multiple of training.eval_every")
    if type(make_plots) is not bool:
        raise ValueError("make_plots must be boolean")
    if max_milestones_this_run is not None and (
        type(max_milestones_this_run) is not int or max_milestones_this_run < 0
    ):
        raise ValueError("max_milestones_this_run must be a nonnegative integer")
    if (diagnostic_callback is None) != (diagnostic_config is None):
        raise ValueError("diagnostic_callback and diagnostic_config must be supplied together")
    if diagnostic_config is not None and not isinstance(diagnostic_config, dict):
        raise ValueError("diagnostic_config must be a JSON object")
    diagnostics = json.loads(json.dumps(diagnostic_config, allow_nan=False))
    models = {str(bins): replace(model_config, num_bins=bins).to_dict() for bins in plan.num_bins}
    device = _prepare_runtime(cfg)
    manifest = {
        "schema": CAPACITY_SCHEMA,
        "base_model": model_config.to_dict(), "models": models,
        "base_training": cfg.to_dict(), "sweep": plan.to_dict(),
        "visualization_enabled": make_plots, "plot_configuration": plotting.to_dict(),
        "trainer_automatic_plots": False,
        "diagnostics": diagnostics,
        "dataset_fingerprint": reference.fingerprint(), "data_summary": reference.summary(),
        "training_code_fingerprint": _code_hash(), "implementation_sha256": _implementation(),
        "runtime": _runtime(device, cfg.dtype), "fixed_streams": _streams(reference, cfg),
        "selection": "minimum best validation NLL at common final update; exact ties use smaller bins",
        "test_policy": "only the final milestone evaluates test; never used for selection or stopping",
        "initialization": "common seeds, not identical parameter tensors across different shapes",
    }
    output = Path(output_directory)
    _open_output(output, manifest)
    rows = _rows(manifest)
    summary = {
        "schema": CAPACITY_SCHEMA, "manifest_sha256": _manifest_hash(manifest),
        "status": "running", "trials": rows, "selection": None,
        "selection_metric": "validation_nll_at_common_final_update",
        "test_used_for_selection": False,
        "scope": "one condition and one seed; milestone rows share their training trajectory",
    }
    executed = 0

    def notify(event: dict[str, Any]) -> None:
        if callback is not None:
            callback(event)

    # Verify every existing immutable archive before changing any summary or
    # continuation state. An edited later archive must not be noticed only after
    # retraining an earlier missing one.
    for row in rows:
        archive = output / row["run_directory"]
        if (archive / _RECEIPT).exists():
            row.update(_verify_snapshot(archive, row, manifest), status="complete")

    try:
        for row in rows:
            if row["status"] == "complete":
                notify({"event": "milestone_reused", "trial_id": row["trial_id"],
                        "num_bins": row["num_bins"], "milestone_step": row["milestone_step"]})
                continue
            if max_milestones_this_run is not None and executed >= max_milestones_this_run:
                summary["status"] = "paused"
                _write_summary(output, summary)
                return summary
            live, archive = output / row["training_directory"], output / row["run_directory"]
            checkpoint = live / "checkpoint.pt"
            current_step = 0
            resume = None
            if checkpoint.exists():
                saved = read_single_checkpoint(checkpoint)
                _check_checkpoint(saved, cfg, _trial_manifest(manifest, row["num_bins"]))
                current_step = saved["global_step"]
                resume = checkpoint
                del saved
            elif live.exists() and any(live.iterdir()):
                raise FileExistsError(f"continuation directory has no resumable checkpoint: {live}")
            if current_step > row["milestone_step"]:
                raise ValueError("continuation passed an unarchived milestone; its selected weights cannot be recovered")
            row["status"] = "running"
            _write_summary(output, summary)
            notify({"event": "milestone_started", "trial_id": row["trial_id"],
                    "num_bins": row["num_bins"], "milestone_step": row["milestone_step"],
                    "from_step": current_step, "train_samples": cfg.train_samples,
                    "learning_rate": cfg.learning_rate, "resumed": resume is not None})
            attempt_path = output / "attempts" / f"{row['trial_id']}.json"
            attempts = _read_json(attempt_path) if attempt_path.exists() else []
            started = time.perf_counter()
            succeeded = False
            executed += 1
            result = selected_model = None
            try:
                # A completed trainer return already has a consistent metrics
                # file. Keep those exact bytes when recovering an unfinished
                # archive; even a zero-update save can reserialize a checkpoint.
                source_metrics = live / "metrics.json"
                recovered = (
                    resume is not None and current_step == row["milestone_step"]
                    and source_metrics.is_file()
                    and _read_json(source_metrics).get("global_step") == current_step
                )
                if recovered:
                    _snapshot_fields(live, row, {**manifest, "diagnostics": None})
                    selected_model, _ = load_single_checkpoint(live / "best.pt", device=device)
                else:
                    result = train_single_condition(
                        reference, SphereFlowConfig.from_dict(models[str(row["num_bins"])]),
                        cfg, live, resume=resume,
                        max_steps_this_run=row["milestone_step"] - current_step,
                        make_plots=False, plot_config=plotting,
                        callback=lambda entry: notify({
                            **entry, "trial_id": row["trial_id"], "num_bins": row["num_bins"],
                            "milestone_step": row["milestone_step"],
                        }),
                    )
                    if result.global_step != row["milestone_step"]:
                        raise ValueError("trainer did not reach the requested milestone")
                    selected_model = result.model
                fields = _archive_milestone(
                    live, archive, row, manifest, selected_model, reference, plotting,
                    diagnostic_callback,
                )
                elapsed = time.perf_counter() - started
                row.update(_complete_snapshot(
                    archive, row, manifest, fields,
                    sum(attempt["elapsed_seconds"] for attempt in attempts) + elapsed,
                ), status="complete")
                succeeded = True
            finally:
                attempts.append({
                    "from_step": current_step, "milestone_step": row["milestone_step"],
                    "elapsed_seconds": time.perf_counter() - started,
                    "snapshot_completed": succeeded, "resumed": resume is not None,
                })
                _atomic_json(attempt_path, attempts)
                del result, selected_model
            notify({"event": "milestone_completed", "trial_id": row["trial_id"],
                    "num_bins": row["num_bins"], "milestone_step": row["milestone_step"],
                    "selected_step": row["selected_step"], "validation_kl": row["validation_kl"],
                    "latest_validation_kl": row["latest_validation_kl"]})
            _write_summary(output, summary)
        selection = _selection(rows, manifest)
        selection_path = output / "selection.json"
        if selection_path.exists() and _read_json(selection_path) != selection:
            raise ValueError("saved capacity selection differs from verified validation results")
        _atomic_json(selection_path, selection)
        summary.update(status="complete", selection=selection)
        _write_summary(output, summary)
        notify({"event": "capacity_sweep_completed", **selection})
        return summary
    except BaseException as error:
        for row in rows:
            if row["status"] == "running":
                row["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        summary.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                       error=f"{type(error).__name__}: {error}")
        _write_summary(output, summary)
        raise


def _verified_completed_capacity(reference: RainbowReference, directory: Path):
    """Read a complete recorded experiment, without rewriting its artifacts."""
    _safe_tree(directory)
    manifest = _read_json(directory / "manifest.json")
    if manifest.get("schema") != CAPACITY_SCHEMA:
        raise ValueError("source must be a completed capacity sweep")
    if manifest.get("dataset_fingerprint") != reference.fingerprint():
        raise ValueError("capacity source uses a different Rainbow record")
    summary = _read_json(directory / "summary.json")
    if summary.get("status") != "complete" or summary.get("manifest_sha256") != _manifest_hash(manifest):
        raise ValueError("capacity source must have a complete, consistent summary")
    rows = _rows(manifest)
    if len(rows) != len(summary.get("trials", [])):
        raise ValueError("capacity source has an unexpected milestone count")
    for row, recorded in zip(rows, summary["trials"], strict=True):
        row.update(_verify_snapshot(directory / row["run_directory"], row, manifest), status="complete")
        if row != recorded:
            raise ValueError("capacity source summary differs from verified milestone receipts")
    selection = _selection(rows, manifest)
    if summary.get("selection") != selection or _read_json(directory / "selection.json") != selection:
        raise ValueError("capacity source selection differs from verified validation results")
    selected = next(row for row in rows if row["run_directory"] == selection["selected_run_directory"])
    return manifest, summary, selected


def _sample_summary(output: Path, summary: dict[str, Any]) -> None:
    _write_table(output, summary)
    lines = [
        "# Fixed training pool after bin selection", "", f"Status: **{summary['status']}**.", "",
        "The completed capacity experiment selects K using validation at the common final update.",
        "Its original training-pool result is reused without retraining or changing that experiment.",
        "Other N values start independent optimization runs with the same K, seeds, update count,",
        "batch size, learning rate, model, precision, and validation/test point streams.",
        "Training pools are nested prefixes. Changing N changes minibatch indices and repeats.",
        "Selection uses validation NLL only; exact ties use smaller N. Test is reporting only.", "",
        "| N | K | Status | Best validation KL | Test KL | Selected step | Source |",
        "|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in summary["trials"]:
        validation = "-" if row.get("validation_kl") is None else f"{row['validation_kl']:.6g}"
        test = "-" if row.get("test_kl") is None else f"{row['test_kl']:.6g}"
        source = "reused capacity result" if row.get("reuse_of") else "new fixed-pool run"
        lines.append(
            f"| {row['train_samples']} | {row['num_bins']} | {row['status']} | {validation} | "
            f"{test} | {row.get('selected_step', '-')} | {source} |"
        )
    if summary.get("selection"):
        lines.extend(["", f"Selected pool size: **{summary['selection']['selected_train_samples']}**."])
    lines.extend([
        "", "![Validation versus training pool](summary.png)", "",
        "CSV/JSON include exact source paths and reuse provenance. Reused elapsed time belongs",
        "to the original capacity trajectory and is not an additional optimization cost.",
        "The fixed update comparison does not establish convergence or equal wall-clock cost.",
        "Each enabled scatter shows all entries in its actual fixed training pool once.",
    ])
    if summary.get("error"):
        lines.extend(["", "## Stopped trial", "", summary["error"]])
    _atomic_bytes(output / "summary.md", ("\n".join(lines) + "\n").encode("utf-8"))

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import NullLocator

    figure = Figure(figsize=(7, 4.5), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    completed = [row for row in summary["trials"] if row["status"] == "complete"]
    if completed:
        axis.errorbar(
            [row["train_samples"] for row in completed],
            [row["validation_kl"] for row in completed],
            yerr=[1.96 * row["validation_kl_standard_error"] for row in completed],
            marker="o", capsize=4, label="Validation KL (±1.96 MC SE)",
        )
        axis.axhline(completed[0]["validation_hg_kl"], ls="--", color="0.5", label="HG")
        axis.set_xscale("log")
        axis.set_xticks([row["train_samples"] for row in completed],
                       [f"{row['train_samples']:,}" for row in completed])
        axis.xaxis.set_minor_locator(NullLocator())
        axis.legend(fontsize=8)
    axis.set_xlabel("Fixed training pool size")
    axis.set_ylabel("Validation forward KL [nats]")
    axis.set_title(f"K = {summary['selected_num_bins']}; fixed optimizer updates")
    axis.grid(alpha=0.25)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=140)
    _atomic_bytes(output / "summary.png", buffer.getvalue())
    figure.clear()


def run_capacity_sample_sweep(
    reference: RainbowReference,
    capacity_directory: str | Path,
    output_directory: str | Path,
    *,
    train_samples: tuple[int, ...] = (65536, 262144, 1048576),
    device: str | None = None,
    make_plots: bool | None = None,
    plot_config: RainbowPlotConfig | None = None,
    callback: Callable[[dict[str, Any]], None] | None = None,
    max_trials_this_run: int | None = None,
    diagnostic_callback: DiagnosticCallback | None = None,
    diagnostic_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare fixed-pool sizes after a verified, completed bins experiment.

    The exact selected capacity baseline is reused. Every other N is a singleton
    capacity experiment at the same final update, preserving all existing trainer
    and archive gates. Source and destination trees must be separate. No source
    capacity file is modified. Only validation chooses the reported pool size.
    """
    source, output = Path(capacity_directory), Path(output_directory)
    source_root, output_root = source.resolve(), output.resolve()
    if output_root.is_relative_to(source_root) or source_root.is_relative_to(output_root):
        raise ValueError("sample sweep output and source capacity trees must be disjoint")
    parent, parent_summary, baseline = _verified_completed_capacity(reference, source)
    cfg = SingleTrainingConfig.from_dict(parent["base_training"])
    if device is not None:
        cfg = replace(cfg, device=device)
    selected_device = _prepare_runtime(cfg)
    if parent["runtime"] != _runtime(selected_device, cfg.dtype) or (
        parent["training_code_fingerprint"] != _code_hash()
        or parent["implementation_sha256"] != _implementation()
    ):
        raise ValueError("sample comparison requires the capacity experiment's training implementation and runtime")
    if not isinstance(train_samples, (list, tuple)) or not train_samples or any(
        type(n) is not int or n < 1 for n in train_samples
    ) or list(train_samples) != sorted(set(train_samples)):
        raise ValueError("train_samples must contain unique positive integers in increasing order")
    counts = tuple(train_samples)
    if cfg.train_samples not in counts:
        raise ValueError("train_samples must include the original capacity training-pool size")
    if max_trials_this_run is not None and (
        type(max_trials_this_run) is not int or max_trials_this_run < 0
    ):
        raise ValueError("max_trials_this_run must be a nonnegative integer")
    plots = parent["visualization_enabled"] if make_plots is None else make_plots
    plotting = (RainbowPlotConfig.from_dict(parent["plot_configuration"])
                if plot_config is None else plot_config)
    if type(plots) is not bool or not isinstance(plotting, RainbowPlotConfig):
        raise ValueError("invalid sample-sweep plotting configuration")
    if (diagnostic_callback is None) != (diagnostic_config is None):
        raise ValueError("diagnostic_callback and diagnostic_config must be supplied together")
    if diagnostic_config is not None and not isinstance(diagnostic_config, dict):
        raise ValueError("diagnostic_config must be a JSON object")
    diagnostics = json.loads(json.dumps(diagnostic_config, allow_nan=False))
    fixed_streams = _streams(reference, cfg, counts)
    for key in ("validation_points_sha256", "test_points_sha256"):
        if fixed_streams[key] != parent["fixed_streams"][key]:
            raise ValueError("sample comparison evaluation stream differs from its capacity source")
    original_count = str(cfg.train_samples)
    if fixed_streams["training_prefix_sha256"][original_count] != (
        parent["fixed_streams"]["training_prefix_sha256"][original_count]
    ):
        raise ValueError("regenerated baseline pool differs from the capacity source")
    selected_bins = parent_summary["selection"]["selected_num_bins"]
    model = SphereFlowConfig.from_dict(parent["models"][str(selected_bins)])
    manifest = {
        "schema": SAMPLE_SCHEMA, "capacity_directory": str(source_root),
        "capacity_manifest_sha256": _file_hash(source / "manifest.json"),
        "capacity_selection_sha256": _file_hash(source / "selection.json"),
        "capacity_selection": parent_summary["selection"],
        "baseline_receipt_sha256": _file_hash(source / baseline["run_directory"] / _RECEIPT),
        "base_training": cfg.to_dict(), "model": model.to_dict(), "train_samples": list(counts),
        "fixed_streams": fixed_streams, "dataset_fingerprint": reference.fingerprint(),
        "training_code_fingerprint": _code_hash(), "implementation_sha256": _implementation(),
        "runtime": _runtime(selected_device, cfg.dtype),
        "visualization_enabled": plots, "plot_configuration": plotting.to_dict(),
        "diagnostics": diagnostics,
        "selection": "minimum best validation NLL at the common final update; exact ties use smaller N",
        "test_policy": "reported after each completed plan; never used for selection or stopping",
    }
    _open_output(output, manifest)
    rows = []
    for n in counts:
        row = {
            "trial_id": f"n_{n}", "num_bins": selected_bins, "train_samples": n,
            "learning_rate": cfg.learning_rate, "batch_size": cfg.batch_size,
            "milestone_step": cfg.steps, "planned_steps": cfg.steps, "status": "pending",
            "reuse_of": None,
            "run_directory": f"n_{n}/bins/k_{selected_bins}/milestones/updates_{cfg.steps}",
            "training_directory": f"n_{n}/bins/k_{selected_bins}/training",
        }
        if n == cfg.train_samples:
            row.update(baseline)
            row.update(
                trial_id=f"n_{n}", reuse_of=f"capacity:{baseline['trial_id']}",
                run_directory=str((source_root / baseline["run_directory"]).resolve()),
                training_directory=str((source_root / baseline["training_directory"]).resolve()),
            )
        rows.append(row)
    summary = {
        "schema": SAMPLE_SCHEMA, "manifest_sha256": _manifest_hash(manifest),
        "status": "running", "trials": rows, "selection": None,
        "selected_num_bins": selected_bins, "capacity_selection": parent_summary["selection"],
        "test_used_for_selection": False,
        "scope": "fixed-update, same-condition sample-count comparison after capacity selection",
    }
    executed = 0

    def notify(event: dict[str, Any]) -> None:
        if callback is not None:
            callback(event)

    try:
        for row in rows:
            if row["reuse_of"]:
                notify({"event": "sample_baseline_reused", "train_samples": row["train_samples"],
                        "run_directory": row["run_directory"]})
                continue
            child = output / f"n_{row['train_samples']}"
            child_cfg = replace(cfg, train_samples=row["train_samples"])
            child_summary_path = child / "summary.json"
            if child_summary_path.exists() and _read_json(child_summary_path).get("status") == "complete":
                child_manifest, child_report, child_row = _verified_completed_capacity(reference, child)
                if child_manifest["base_training"] != child_cfg.to_dict() or (
                    child_manifest["base_model"] != model.to_dict()
                    or child_manifest["sweep"] != {
                        "num_bins": [selected_bins], "milestones": [cfg.steps],
                    }
                    or child_manifest["training_code_fingerprint"] != manifest["training_code_fingerprint"]
                    or child_manifest["implementation_sha256"] != manifest["implementation_sha256"]
                    or child_manifest["runtime"] != manifest["runtime"]
                    or child_manifest["diagnostics"] != diagnostics
                    or child_manifest["visualization_enabled"] != plots
                    or child_manifest["plot_configuration"] != plotting.to_dict()
                ):
                    raise ValueError("completed sample trial differs from its recorded experiment")
            else:
                if max_trials_this_run is not None and executed >= max_trials_this_run:
                    summary["status"] = "paused"
                    _sample_summary(output, summary)
                    return summary
                row["status"] = "running"
                _sample_summary(output, summary)
                executed += 1
                child_report = run_capacity_sweep(
                    reference, model, child_cfg, child,
                    sweep_config=CapacitySweepConfig(num_bins=(selected_bins,), milestones=(cfg.steps,)),
                    make_plots=plots, plot_config=plotting,
                    callback=lambda event: notify({**event, "sample_trial_id": row["trial_id"],
                                                    "train_samples": row["train_samples"]}),
                    diagnostic_callback=diagnostic_callback, diagnostic_config=diagnostics,
                )
                child_row = child_report["trials"][0]
            row.update({key: value for key, value in child_row.items() if key not in (
                "trial_id", "run_directory", "training_directory", "reuse_of",
            )})
            if row["sample_split"]["training_points_sha256"] != (
                fixed_streams["training_prefix_sha256"][str(row["train_samples"])]
            ) or row["sample_split"]["validation_points_sha256"] != (
                fixed_streams["validation_points_sha256"]
            ):
                raise ValueError("sample trial point streams differ from the common nested pools")
            _sample_summary(output, summary)
        # Parent data and selected archives remain authoritative throughout the
        # separate sample-count comparison. Reading them never changes the parent.
        _verified_completed_capacity(reference, source)
        winner = min(rows, key=lambda row: (row["validation_nll"], row["train_samples"]))
        selection = {
            "schema": SAMPLE_SCHEMA, "manifest_sha256": _manifest_hash(manifest),
            "selected_num_bins": selected_bins,
            "selected_train_samples": winner["train_samples"],
            "selected_trial_id": winner["trial_id"], "selected_run_directory": winner["run_directory"],
            "selected_validation_nll": winner["validation_nll"], "selected_step": winner["selected_step"],
            "rule": manifest["selection"], "test_used": False,
            "candidates": [{key: row[key] for key in (
                "trial_id", "train_samples", "validation_nll", "validation_kl", "selected_step", "reuse_of",
            )} for row in rows],
        }
        path = output / "selection.json"
        if path.exists() and _read_json(path) != selection:
            raise ValueError("saved sample selection differs from verified validation results")
        _atomic_json(path, selection)
        summary.update(status="complete", selection=selection)
        _sample_summary(output, summary)
        notify({"event": "sample_sweep_completed", **selection})
        return summary
    except BaseException as error:
        for row in rows:
            if row["status"] == "running":
                row["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        summary.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                       error=f"{type(error).__name__}: {error}")
        _sample_summary(output, summary)
        raise
