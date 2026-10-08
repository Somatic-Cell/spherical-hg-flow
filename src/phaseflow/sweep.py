"""Sequential learning-rate and point-count experiments for one Rainbow CDF.

Stage A chooses a learning rate using validation NLL only. Stage B changes the
fixed training pool, keeping that learning rate and the update/batch budget.
The existing single-condition trainer supplies the model, objective, checkpoint
selection, independent test evaluation and comparison maps without modification.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import torch

from .monitoring import check_tensorboard
from .plotting import RainbowPlotConfig
from .rainbow import RainbowReference
from .single_condition import (
    CHECKPOINT_VERSION,
    SingleTrainingConfig,
    _array_hash,
    _code_hash,
    _points,
    _resolve_device,
    _runtime,
    read_single_checkpoint,
    train_single_condition,
)
from .sphere_model import SingleConditionSphereFlow, SphereFlowConfig
from .training import _setup_runtime, atomic_json

SWEEP_SCHEMA = "phaseflow.sequential_sweep.v1"
_RECEIPT = "sweep_completed.json"


@dataclass(frozen=True)
class SweepConfig:
    learning_rates: tuple[float, ...] = (3e-4, 1e-3, 3e-3)
    train_samples: tuple[int, ...] = (4096, 16384, 65536, 262144)
    baseline_train_samples: int = 65536

    def __post_init__(self) -> None:
        for name in ("learning_rates", "train_samples"):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or not values:
                raise ValueError(f"{name} must be a nonempty list or tuple")
            object.__setattr__(self, name, tuple(values))
        if any(
            type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0
            for rate in self.learning_rates
        ):
            raise ValueError("learning rates must be finite and positive")
        if len(set(self.learning_rates)) != len(self.learning_rates):
            raise ValueError("learning rates must be unique")
        if any(type(n) is not int or n < 1 for n in self.train_samples):
            raise ValueError("training sample counts must be positive integers")
        if len(set(self.train_samples)) != len(self.train_samples):
            raise ValueError("training sample counts must be unique")
        if type(self.baseline_train_samples) is not int or self.baseline_train_samples < 1:
            raise ValueError("baseline_train_samples must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "learning_rates": list(self.learning_rates),
            "train_samples": list(self.train_samples),
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> SweepConfig:
        if not isinstance(values, dict):
            raise ValueError("sweep configuration must be an object")
        try:
            return cls(**values)
        except TypeError as error:
            raise ValueError(f"Invalid sweep configuration: {error}") from error


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _atomic_bytes(path: Path, contents: bytes) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _fixed_streams(reference: RainbowReference, cfg: SingleTrainingConfig,
                   sweep: SweepConfig) -> dict[str, Any]:
    seed = cfg.seed if cfg.data_seed is None else cfg.data_seed
    counts = sorted(set((*sweep.train_samples, sweep.baseline_train_samples)))
    largest = _points(reference, counts[-1], seed, "train")
    prefixes = {str(n): _array_hash(*(a[:n] for a in largest)) for n in counts}
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
        "nesting": "each training pool equals the prefix of the largest pool; hashes checked",
    }


def _trial(stage: str, key: str, cfg: SingleTrainingConfig) -> dict[str, Any]:
    return {
        "trial_id": f"{stage}_{key}",
        "stage": stage,
        "run_directory": f"stage_{stage.lower()}/{key}",
        "learning_rate": cfg.learning_rate,
        "train_samples": cfg.train_samples,
        "training_config": cfg.to_dict(),
        "status": "pending",
        "reuse_of": None,
    }


def _check_checkpoint(payload: dict[str, Any], cfg: SingleTrainingConfig,
                      manifest: dict[str, Any]) -> None:
    expected = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "kind": "training",
        "model_config": manifest["model"],
        "training_config": cfg.to_dict(),
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "code_fingerprint": manifest["training_code_fingerprint"],
        "runtime": manifest["runtime"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"sweep checkpoint differs in {key}; choose a new output directory")
    step = payload["global_step"]
    if type(step) is not int or not 0 <= step <= cfg.steps:
        raise ValueError("invalid sweep checkpoint step")
    split = payload.get("sample_split", {})
    streams = manifest["fixed_streams"]
    for key, value in {
        "seed": cfg.seed,
        "data_seed": streams["data_seed"],
        "train_samples": cfg.train_samples,
        "validation_samples": cfg.validation_samples,
        "test_samples": cfg.test_samples,
        "training_points_sha256": streams["training_prefix_sha256"][str(cfg.train_samples)],
        "validation_points_sha256": streams["validation_points_sha256"],
    }.items():
        if split.get(key) != value:
            raise ValueError(f"sweep checkpoint point stream differs in {key}")


def _state_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _state_equal(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _state_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _result_fields(path: Path, cfg: SingleTrainingConfig,
                   manifest: dict[str, Any]) -> dict[str, Any]:
    checkpoint = read_single_checkpoint(path / "checkpoint.pt")
    _check_checkpoint(checkpoint, cfg, manifest)
    metrics = _read_json(path / "metrics.json")
    if not metrics.get("complete") or checkpoint["global_step"] != cfg.steps:
        raise ValueError("completed sweep trial has an incomplete training checkpoint")
    if (
        metrics["global_step"] != cfg.steps
        or metrics["selected_step"] != checkpoint["best_step"]
        or metrics["best_validation"] != checkpoint["best_validation"]
        or metrics["dataset_fingerprint"] != manifest["dataset_fingerprint"]
    ):
        raise ValueError("trial metrics and training checkpoint disagree")
    validation, test = metrics["best_validation"], metrics["test"]
    if test is None or not math.isfinite(validation["nll"]):
        raise ValueError("completed sweep trial has no finite validation result/final test")
    best = read_single_checkpoint(path / "best.pt")
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
        raise ValueError("best.pt does not match the validation-selected trial checkpoint")
    expected_config = {
        "schema_version": 2, "family": checkpoint["family"], "model": manifest["model"],
        "training": cfg.to_dict(), "visualization": manifest["visualization"],
    }
    if _read_json(path / "config.json") != expected_config:
        raise ValueError("saved trial configuration differs from the sweep configuration")
    if _read_json(path / "sample_split.json") != checkpoint["sample_split"]:
        raise ValueError("saved sample split differs from the training checkpoint")
    if test.get("seed") != manifest["fixed_streams"]["data_seed"] or (
        test.get("sample_count") != cfg.test_samples
    ):
        raise ValueError("trial final test does not use the common test stream")
    parameter_count = sum(
        value.numel() for name, value in best["model_state"].items()
        if name.endswith("weight") or name.endswith("bias")
    )
    monitor = next(
        (row.get("train_monitor") for row in checkpoint["history"]
         if row["global_step"] == checkpoint["best_step"]), None
    )
    results = {
        "global_step": cfg.steps,
        "selected_step": checkpoint["best_step"],
        "processed_examples": cfg.steps * cfg.batch_size,
        "effective_passes": cfg.steps * cfg.batch_size / cfg.train_samples,
        "validation_nll": validation["nll"],
        "validation_kl": validation["forward_kl_estimate"],
        "validation_kl_standard_error": validation["forward_kl_standard_error"],
        "validation_hg_kl": validation["hg_forward_kl_estimate"],
        "train_monitor_nll_at_selected_step": None if monitor is None else monitor["nll"],
        "test_nll": test["nll"],
        "test_kl": test["forward_kl_estimate"],
        "test_kl_standard_error": test["forward_kl_standard_error"],
        "test_hg_kl": test["hg_forward_kl_estimate"],
        "test_relative_ess": (test.get("proposal") or {}).get("relative_ess"),
        "parameter_count": parameter_count,
        "parameter_bytes": parameter_count * (4 if cfg.dtype == "float32" else 8),
        "checkpoint_bytes": (path / "best.pt").stat().st_size,
        "sample_split": checkpoint["sample_split"],
    }
    if manifest["diagnostics"] is not None:
        results["diagnostics"] = _read_json(path / "diagnostics.json")
    return results


def _verify_receipt(path: Path, cfg: SingleTrainingConfig,
                    manifest: dict[str, Any]) -> dict[str, Any]:
    receipt = _read_json(path / _RECEIPT)
    if receipt.get("schema") != SWEEP_SCHEMA or receipt.get("manifest_sha256") != (
        _manifest_hash(manifest)
    ):
        raise ValueError("completed trial belongs to a different sweep manifest")
    hashes = receipt.get("files_sha256", {})
    if not {"checkpoint.pt", "best.pt", "metrics.json", "config.json"} <= hashes.keys():
        raise ValueError("completion receipt is missing required artifact hashes")
    for relative, expected in hashes.items():
        target = path / relative
        if target.resolve().is_relative_to(path.resolve()) is False:
            raise ValueError("invalid path in sweep completion receipt")
        if not target.is_file() or _file_hash(target) != expected:
            raise ValueError(f"completed trial file changed or missing: {target}")
    fields = _result_fields(path, cfg, manifest)
    if fields != receipt.get("results"):
        raise ValueError("completed trial results differ from their completion receipt")
    return {**fields, "elapsed_seconds": receipt["elapsed_seconds"]}


def _manifest_hash(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _write_receipt(path: Path, cfg: SingleTrainingConfig, manifest: dict[str, Any],
                   elapsed_seconds: float) -> dict[str, Any]:
    results = _result_fields(path, cfg, manifest)
    required = [
        "checkpoint.pt", "best.pt", "config.json", "data_summary.json",
        "sample_split.json", "history.json", "metrics.json",
    ]
    if manifest["visualization"]["enabled"]:
        required.extend([
            "learning_curves.png", "plots/plots.json", "plots/reference_pdf.png",
            "plots/cdf_samples.png", "plots/nf_pdf.png", "plots/comparison.png",
        ])
    # Immutable numerical/scientific artifacts are checked before reuse. Live
    # JSONL and TensorBoard files remain available to external monitoring tools.
    artifacts = set(required)
    artifacts.update(str(p.relative_to(path)).replace(os.sep, "/")
                     for p in path.glob("plots/*.pdf"))
    if manifest["diagnostics"] is not None:
        artifacts.add("diagnostics.json")
        artifacts.update(p.relative_to(path).as_posix()
                         for p in (path / "diagnostics").rglob("*") if p.is_file())
    receipt = {
        "schema": SWEEP_SCHEMA,
        "manifest_sha256": _manifest_hash(manifest),
        "files_sha256": {name: _file_hash(path / name) for name in sorted(artifacts)},
        "results": results,
        "elapsed_seconds": elapsed_seconds,
        "timing_scope": "all recorded trial attempts, including evaluation/plots; not kernel timing",
    }
    atomic_json(path / _RECEIPT, receipt)
    return {**results, "elapsed_seconds": elapsed_seconds}


_COLUMNS = (
    "stage", "trial_id", "status", "learning_rate", "train_samples", "global_step",
    "selected_step", "processed_examples", "effective_passes", "validation_nll",
    "validation_kl", "validation_kl_standard_error", "validation_hg_kl",
    "train_monitor_nll_at_selected_step", "test_nll", "test_kl",
    "test_kl_standard_error", "test_hg_kl", "test_relative_ess", "elapsed_seconds",
    "parameter_count", "parameter_bytes", "checkpoint_bytes", "reuse_of", "run_directory",
)


def _summary_plot(output: Path, summary: dict[str, Any]) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator, NullLocator

    figure = Figure(figsize=(12, 8), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    for column, stage in enumerate(("A", "B")):
        rows = [r for r in summary["trials"]
                if r["stage"] == stage and r["status"] == "complete"]
        ax, history_ax = axes[0, column], axes[1, column]
        key = "learning_rate" if stage == "A" else "train_samples"
        rows.sort(key=lambda row: row[key])
        if rows:
            ax.errorbar(
                [r[key] for r in rows], [r["validation_kl"] for r in rows],
                yerr=[1.96 * r["validation_kl_standard_error"] for r in rows],
                marker="o", capsize=4, label="Validation KL (±1.96 MC SE)",
            )
            ax.axhline(rows[0]["validation_hg_kl"], ls="--", color="0.5", label="HG")
            selected = (summary.get("selection") or {}).get("selected_trial_id")
            for row in rows:
                if row["trial_id"] == selected:
                    ax.scatter(row[key], row["validation_kl"], marker="*", s=180,
                               color="darkorange", zorder=5, label="Selected learning rate")
                history = _read_json(output / row["run_directory"] / "history.json")
                values = [entry for entry in history if "validation" in entry]
                label = f"lr={row[key]:g}" if stage == "A" else f"N={row[key]:,}"
                if row["reuse_of"]:
                    label += " (A reused)"
                history_ax.plot(
                    [v["global_step"] for v in values],
                    [v["validation"]["forward_kl_estimate"] for v in values], label=label,
                )
            ax.set_xscale("log")
            ax.set_xticks([r[key] for r in rows], [f"{r[key]:g}" for r in rows])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.legend(fontsize=8)
            history_ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No completed trials yet", ha="center", transform=ax.transAxes)
        title = f"Stage {stage}: {'learning rate' if stage == 'A' else 'training points'}"
        if stage == "B" and summary.get("selection"):
            title += f" (lr={summary['selection']['selected_learning_rate']:g})"
        ax.set_title(title)
        ax.set_xlabel("Learning rate" if stage == "A" else "Fixed training pool size")
        ax.set_ylabel("Forward KL [nats], validation-selected weights")
        history_ax.set_xlabel("Optimizer updates")
        history_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        history_ax.set_ylabel("Validation forward KL [nats]")
        for item in (ax, history_ax):
            item.grid(alpha=0.25)
    figure.suptitle(
        "One-condition A -> B sweep: validation selects the learning rate\n"
        "Common fixed validation points; intervals exclude training-seed and selection uncertainty",
        fontsize=11,
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150)
    _atomic_bytes(output / "summary.png", buffer.getvalue())
    figure.clear()


def _write_summary(output: Path, summary: dict[str, Any]) -> None:
    atomic_json(output / "summary.json", summary)
    buffer = io.StringIO(newline="")
    diagnostic_keys = sorted({
        key for row in summary["trials"] for key, value in row.get("diagnostics", {}).items()
        if value is None or type(value) in (bool, int, float, str)
    })
    writer = csv.DictWriter(buffer, fieldnames=[*_COLUMNS, *(f"diag_{k}" for k in diagnostic_keys)],
                            extrasaction="ignore")
    writer.writeheader()
    writer.writerows({**row, **{f"diag_{k}": row.get("diagnostics", {}).get(k)
                              for k in diagnostic_keys}} for row in summary["trials"])
    _atomic_bytes(output / "summary.csv", buffer.getvalue().encode("utf-8-sig"))

    def number(value: Any) -> str:
        return "-" if value is None else f"{value:.6g}"

    lines = [
        "# Single-condition A -> B sweep", "", f"Status: **{summary['status']}**.", "",
        "Stage A selects the minimum validation NLL; exact ties use the smaller learning rate.",
        "Stage B holds that rate, optimizer updates, batch size, model and seeds fixed.",
        "Test metrics are reporting only and never enter the selection rule.", "",
        "| Stage | Trial | N | LR | Status | Validation KL | Test KL | Test rESS | Best step |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary["trials"]:
        suffix = f"; reuse of {row['reuse_of']}" if row["reuse_of"] else ""
        link = f"[{row['trial_id']}]({row['run_directory']}/metrics.json)"
        lines.append(
            f"| {row['stage']} | {link}{suffix} | {row['train_samples']} | "
            f"{number(row['learning_rate'])} | {row['status']} | "
            f"{number(row.get('validation_kl'))} | {number(row.get('test_kl'))} | "
            f"{number(row.get('test_relative_ess'))} | {row.get('selected_step', '-')} |"
        )
    selection = summary.get("selection")
    if selection is not None:
        lines.extend(["", f"Selected learning rate: **{selection['selected_learning_rate']:g}** "
                      f"from `{selection['selected_trial_id']}`."])
    lines.extend([
        "", "![Validation comparison and learning curves](summary.png)", "",
        "Each physical trial contains its existing history, metrics, checkpoints and plots.",
        "A reused B row points to the identical A run; it is not an independent repetition.",
        "CSV/JSON contain Monte Carlo standard errors, elapsed time, parameter bytes and budget.",
        "The reported training-monitor NLL uses the first min(N, train_monitor_samples) points.",
        "Fixed update counts do not establish convergence of every pool size.",
        "A single seed and condition do not establish a universal optimal learning rate.",
        "No loss weighting, PDF floor, smoothing or teacher modification is applied.",
    ])
    if summary.get("error"):
        lines.extend(["", "## Stopped trial", "", str(summary["error"])])
    _atomic_bytes(output / "summary.md", ("\n".join(lines) + "\n").encode("utf-8"))
    _summary_plot(output, summary)


def run_single_condition_sweep(
    reference: RainbowReference,
    model_config: SphereFlowConfig,
    training_config: SingleTrainingConfig,
    output_directory: str | Path,
    *,
    sweep_config: SweepConfig | None = None,
    make_plots: bool = True,
    plot_config: RainbowPlotConfig | None = None,
    callback: Callable[[dict[str, Any]], None] | None = None,
    max_trials_this_run: int | None = None,
    diagnostic_callback: Callable[
        [SingleConditionSphereFlow, RainbowReference, Path], dict[str, Any]
    ] | None = None,
    diagnostic_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run/reuse A trials, select by validation, then run/reuse B sequentially.

    Restart with the identical arguments and output directory after interruption.
    Completed trials are hash-verified; partial trials use exact trainer resume.
    Changed data/configuration/code/runtime is refused before modifying results.
    ``max_trials_this_run`` permits a controlled pause between physical trials.
    """
    if max_trials_this_run is not None and (
        type(max_trials_this_run) is not int or max_trials_this_run < 0
    ):
        raise ValueError("max_trials_this_run must be a nonnegative integer")
    if type(make_plots) is not bool:
        raise ValueError("make_plots must be boolean")
    sweep = sweep_config if sweep_config is not None else SweepConfig()
    plotting = plot_config if plot_config is not None else RainbowPlotConfig()
    if not isinstance(sweep, SweepConfig) or not isinstance(plotting, RainbowPlotConfig):
        raise ValueError("invalid sweep or plotting configuration")
    if (diagnostic_callback is None) != (diagnostic_config is None):
        raise ValueError("diagnostic_callback and diagnostic_config must be supplied together")
    if diagnostic_config is not None and not isinstance(diagnostic_config, dict):
        raise ValueError("diagnostic_config must be a JSON object")
    # A JSON round trip makes the recorded configuration independent of tuple
    # representations and refuses NaN/nonserializable implicit parameters.
    diagnostics = json.loads(json.dumps(diagnostic_config, allow_nan=False))
    cfg = training_config
    if cfg.tensorboard:
        check_tensorboard()
    if torch.device(cfg.device).type == "cuda" and cfg.deterministic:
        workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if workspace not in (":4096:8", ":16:8"):
            raise ValueError("deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
    device = _resolve_device(cfg.device)
    _setup_runtime(cfg)
    manifest = {
        "schema": SWEEP_SCHEMA,
        "model": model_config.to_dict(),
        "base_training": cfg.to_dict(),
        "sweep": sweep.to_dict(),
        "visualization": {"enabled": make_plots, **plotting.to_dict()},
        "diagnostics": diagnostics,
        "dataset_fingerprint": reference.fingerprint(),
        "data_summary": reference.summary(),
        "training_code_fingerprint": _code_hash(),
        "sweep_code_sha256": _file_hash(Path(__file__)),
        "plotting_code_sha256": _file_hash(Path(__file__).with_name("plotting.py")),
        "runtime": _runtime(device, cfg.dtype),
        "fixed_streams": _fixed_streams(reference, cfg, sweep),
        "selection": "minimum best_validation.nll in A; exact ties use smaller learning_rate",
        "test_policy": "reported per completed trial; not used for any selection or stopping",
    }
    output = Path(output_directory)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if _read_json(manifest_path) != manifest:
            raise ValueError(
                "sweep data/configuration/code/runtime changed; use a new output directory"
            )
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("sweep output is not empty and has no manifest; choose a new folder")
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest_path, manifest)
    rows = [
        _trial("A", f"lr_{float(rate)}", replace(
            cfg, learning_rate=rate, train_samples=sweep.baseline_train_samples
        )) for rate in sweep.learning_rates
    ]
    summary: dict[str, Any] = {
        "schema": SWEEP_SCHEMA,
        "manifest_sha256": _manifest_hash(manifest),
        "status": "running", "selection": None, "trials": rows,
        "selection_metric": "validation_nll", "test_used_for_selection": False,
        "scope": "single condition, fixed seed; synthetic inputs are not optical validation",
    }
    executed = 0

    def notify(event: dict[str, Any]) -> None:
        if callback is not None:
            callback(event)

    def run(row: dict[str, Any]) -> bool:
        nonlocal executed
        current = SingleTrainingConfig.from_dict(row["training_config"])
        directory = output / row["run_directory"]
        if (directory / _RECEIPT).exists():
            row.update(_verify_receipt(directory, current, manifest), status="complete")
            notify({"event": "trial_reused", "trial_id": row["trial_id"], "stage": row["stage"]})
            return True
        if max_trials_this_run is not None and executed >= max_trials_this_run:
            return False
        checkpoint = directory / "checkpoint.pt"
        resume = checkpoint if checkpoint.exists() else None
        if resume is not None:
            _check_checkpoint(read_single_checkpoint(resume), current, manifest)
        elif directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"partial trial has no resumable checkpoint: {directory}")
        row["status"] = "running"
        _write_summary(output, summary)
        notify({"event": "trial_resumed" if resume else "trial_started",
                "trial_id": row["trial_id"], "stage": row["stage"],
                "learning_rate": current.learning_rate, "train_samples": current.train_samples,
                "run_directory": str(directory)})
        attempts_path = output / "attempts" / f"{row['trial_id']}.json"
        attempts = _read_json(attempts_path) if attempts_path.exists() else []
        started = time.perf_counter()
        executed += 1
        succeeded = False
        try:
            result = train_single_condition(
                reference, model_config, current, directory, resume=resume,
                make_plots=make_plots, plot_config=plotting,
                callback=lambda event: notify({
                    **event, "trial_id": row["trial_id"], "stage": row["stage"],
                }),
            )
            if result.complete and diagnostic_callback is not None:
                supplemental = diagnostic_callback(result.model, reference, directory)
                if not isinstance(supplemental, dict):
                    raise ValueError("diagnostic callback must return a JSON object")
                atomic_json(directory / "diagnostics.json", supplemental)
            succeeded = result.complete
        finally:
            attempts.append({
                "elapsed_seconds": time.perf_counter() - started,
                "complete": succeeded,
                "resumed": resume is not None,
            })
            atomic_json(attempts_path, attempts)
        row["global_step"] = result.global_step
        if not result.complete:
            row["status"] = "partial"
            del result
            return False
        elapsed = sum(a["elapsed_seconds"] for a in attempts)
        row.update(_write_receipt(directory, current, manifest, elapsed), status="complete")
        del result  # Release the selected GPU model before preparing the next trial.
        notify({"event": "trial_completed", "trial_id": row["trial_id"],
                "stage": row["stage"], "validation_nll": row["validation_nll"],
                "validation_kl": row["validation_kl"]})
        return True

    try:
        for row in rows:
            if not run(row):
                summary["status"] = "paused"
                _write_summary(output, summary)
                return summary
            _write_summary(output, summary)
        winner = min(rows, key=lambda row: (row["validation_nll"], row["learning_rate"]))
        selection = {
            "schema": SWEEP_SCHEMA,
            "manifest_sha256": _manifest_hash(manifest),
            "selected_trial_id": winner["trial_id"],
            "selected_learning_rate": winner["learning_rate"],
            "selected_validation_nll": winner["validation_nll"],
            "rule": manifest["selection"],
            "test_used": False,
            "candidates": [{key: row[key] for key in (
                "trial_id", "learning_rate", "validation_nll", "validation_kl",
                "selected_step", "run_directory",
            )} for row in rows],
        }
        selection_path = output / "selection.json"
        if selection_path.exists() and _read_json(selection_path) != selection:
            raise ValueError("saved A selection differs from verified validation results")
        atomic_json(selection_path, selection)
        summary["selection"] = selection
        stage_b = [_trial("B", f"n_{n}", replace(
            cfg, train_samples=n, learning_rate=winner["learning_rate"]
        )) for n in sweep.train_samples]
        rows.extend(stage_b)
        for row in stage_b:
            if row["training_config"] == winner["training_config"]:
                # The same seed/configuration and budget imply the same trial;
                # its receipt also checks code, runtime, data and saved artifacts.
                fields = _verify_receipt(
                    output / winner["run_directory"],
                    SingleTrainingConfig.from_dict(winner["training_config"]), manifest,
                )
                row.update(fields, status="complete", reuse_of=winner["trial_id"],
                           run_directory=winner["run_directory"])
                notify({"event": "stage_b_reuses_a", "trial_id": row["trial_id"],
                        "reuse_of": winner["trial_id"]})
            elif not run(row):
                summary["status"] = "paused"
                _write_summary(output, summary)
                return summary
            _write_summary(output, summary)
        summary["status"] = "complete"
        _write_summary(output, summary)
        return summary
    except BaseException as error:
        summary["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        for row in rows:
            if row["status"] == "running":
                row["status"] = summary["status"]
        _write_summary(output, summary)
        raise
