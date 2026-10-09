"""Controlled beta experiments for solid-angle log-density regression.

Every candidate uses the same validation log RMSE for checkpoint and sweep
selection. The total objective and final tests never select the winner. This is
separate from the historical NLL-selected capacity/sample sweeps.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

from .angular_diagnostics import AngularDiagnosticConfig, evaluate_angular_diagnostics
from .log_objective import (
    LogObjectiveConfig,
    make_training_pool,
    make_uniform_points,
    verify_positive_teacher,
)
from .monitoring import check_tensorboard
from .plotting import RainbowPlotConfig, plot_rainbow_comparison
from .rainbow import RainbowReference
from .single_condition import (
    FAMILY,
    SingleTrainingConfig,
    _array_hash,
    _code_hash,
    _points,
    _resolve_device,
    _runtime,
    _validate_selections,
    load_single_checkpoint,
    read_single_checkpoint,
    train_single_condition,
)
from .sphere_model import SphereFlowConfig
from .sweep import _atomic_bytes, _file_hash, _manifest_hash, _read_json, _state_equal
from .training import _setup_runtime, atomic_json

LOG_SWEEP_SCHEMA = "phaseflow.log_loss_sweep.v1"
_RECEIPT = "log_sweep_completed.json"
_SELECTION = (
    "minimum validation log_rmse; exact ties use lower beta, then trial_id; "
    "all candidates use validation-log-RMSE-selected checkpoints"
)


@dataclass(frozen=True)
class LogSweepConfig:
    betas: tuple[float, ...] = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)
    include_target_nll_control: bool = True
    include_pure_log_control: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.betas, (list, tuple)) or not self.betas:
            raise ValueError("betas must be a nonempty list or tuple")
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in self.betas):
            raise ValueError("betas must contain finite nonnegative numbers")
        object.__setattr__(self, "betas", tuple(float(v) for v in self.betas))
        if len(set(self.betas)) != len(self.betas):
            raise ValueError("betas must be unique")
        for name in ("include_target_nll_control", "include_pure_log_control"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "betas": list(self.betas)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LogSweepConfig:
        if not isinstance(value, dict):
            raise ValueError("log sweep configuration must be an object")
        try:
            return cls(**value)
        except TypeError as error:
            raise ValueError(f"Invalid log sweep configuration: {error}") from error


def _trial_plan(
    cfg: SingleTrainingConfig, objective: LogObjectiveConfig, sweep: LogSweepConfig,
) -> list[dict[str, Any]]:
    # A supplied saved config controls architecture, precision, data, optimizer
    # and evaluation counts. These explicit experiment axes replace its loss.
    common = replace(objective, nll_weight=1.0, sampling="target_uniform",
                     selection_metric="log_rmse")
    common.validate_training_count(cfg.train_samples)
    choices = []
    if sweep.include_target_nll_control:
        choices.append(("target_nll", "target_nll_control", replace(common, beta=0.0,
                                                                   sampling="target")))
    # Round-trip float text keeps distinct accepted beta values in distinct
    # trial directories; :g's default six digits can silently collide.
    choices.extend((f"beta_{str(beta).removesuffix('.0')}", "beta", replace(common, beta=beta))
                   for beta in sweep.betas)
    if sweep.include_pure_log_control:
        choices.append(("pure_log", "pure_log_control", replace(common, nll_weight=0.0, beta=1.0)))
    return [{
        "trial_id": name, "kind": kind, "status": "pending",
        "run_directory": f"trials/{name}", "beta": loss.beta,
        "nll_weight": loss.nll_weight, "sampling": loss.sampling,
        "selection_metric": loss.selection_metric, "objective_config": loss.to_dict(),
        "learning_rate": cfg.learning_rate, "train_samples": cfg.train_samples,
        "planned_steps": cfg.steps, "batch_size": cfg.batch_size,
    } for name, kind, loss in choices]


def _geometry_dtype(model: SphereFlowConfig, cfg: SingleTrainingConfig) -> torch.dtype:
    return torch.float64 if model.geometry_dtype == "float64" else getattr(torch, cfg.dtype)


def _fixed_streams(
    reference: RainbowReference, model: SphereFlowConfig, cfg: SingleTrainingConfig,
    objective: LogObjectiveConfig, rows: list[dict[str, Any]],
) -> dict[str, Any]:
    seed = cfg.seed if cfg.data_seed is None else cfg.data_seed
    geometry_dtype = _geometry_dtype(model, cfg)
    pools = {}
    for row in rows:
        sampling = row["sampling"]
        if sampling in pools:
            continue
        loss = LogObjectiveConfig.from_dict(row["objective_config"])
        pool = make_training_pool(reference, cfg.train_samples, seed, loss, geometry_dtype)
        pools[sampling] = {
            "training_points_sha256": _array_hash(pool.directions, pool.log_p, pool.components),
            "training_pool": pool.provenance,
        }
    return {
        "data_seed": seed, "training_pools": pools,
        "validation_points_sha256": _array_hash(
            *_points(reference, cfg.validation_samples, seed, "validation")
        ),
        "validation_uniform_points_sha256": _array_hash(*make_uniform_points(
            reference, objective.validation_uniform_samples, seed, "validation", geometry_dtype,
        )),
        "validation_samples": cfg.validation_samples,
        "validation_uniform_samples": objective.validation_uniform_samples,
        "test_samples": cfg.test_samples,
        "test_uniform_samples": objective.test_uniform_samples,
        "test_policy": "independent test streams materialized only for completed trial reporting",
        "sharing": "all mixture betas and pure-log control share identical fixed training queries",
    }


def _check_checkpoint(
    payload: dict[str, Any], row: dict[str, Any], manifest: dict[str, Any],
) -> None:
    for key, expected in {
        "checkpoint_version": 5, "kind": "training", "model_config": manifest["model"],
        "training_config": manifest["base_training"],
        "objective_config": row["objective_config"],
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "code_fingerprint": manifest["training_code_fingerprint"],
        "runtime": manifest["runtime"],
    }.items():
        if payload.get(key) != expected:
            raise ValueError(f"log sweep checkpoint differs in {key}; use a new output directory")
    step = payload.get("global_step")
    if type(step) is not int or not 0 <= step <= manifest["base_training"]["steps"]:
        raise ValueError("invalid log sweep checkpoint step")
    _validate_selections(
        payload.get("selections", {}), payload.get("history", []), step,
        LogObjectiveConfig.from_dict(row["objective_config"]), payload.get("best_step"),
        payload.get("best_validation"), payload.get("best_state"),
    )
    streams = manifest["fixed_streams"]
    split = payload.get("sample_split", {})
    expected_pool = streams["training_pools"][row["sampling"]]
    for key, expected in {
        "seed": manifest["base_training"]["seed"], "data_seed": streams["data_seed"],
        "train_samples": row["train_samples"],
        "validation_samples": streams["validation_samples"],
        "test_samples": streams["test_samples"],
        "training_points_sha256": expected_pool["training_points_sha256"],
        "training_pool": expected_pool["training_pool"],
        "validation_points_sha256": streams["validation_points_sha256"],
        "uniform_validation_points_sha256": streams["validation_uniform_points_sha256"],
        "validation_uniform_samples": streams["validation_uniform_samples"],
    }.items():
        if split.get(key) != expected:
            raise ValueError(f"log sweep checkpoint point stream differs in {key}")


def _selected_fields(directory: Path, row: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    checkpoint = read_single_checkpoint(directory / "checkpoint.pt")
    _check_checkpoint(checkpoint, row, manifest)
    metrics = _read_json(directory / "metrics.json")
    steps = manifest["base_training"]["steps"]
    if not metrics.get("complete") or checkpoint["global_step"] != steps:
        raise ValueError("completed log trial has an incomplete checkpoint")
    for key, expected in {
        "global_step": steps, "selected_step": checkpoint["best_step"],
        "best_validation": checkpoint["best_validation"],
        "dataset_fingerprint": manifest["dataset_fingerprint"],
    }.items():
        if metrics.get(key) != expected:
            raise ValueError(f"log trial metrics disagree with checkpoint in {key}")
    expected_config = {
        "schema_version": 3, "family": FAMILY, "model": manifest["model"],
        "training": manifest["base_training"], "objective": row["objective_config"],
        "visualization": manifest["visualization"],
    }
    if _read_json(directory / "config.json") != expected_config:
        raise ValueError("saved log trial configuration differs from its manifest")
    if _read_json(directory / "sample_split.json") != checkpoint["sample_split"]:
        raise ValueError("saved log trial sample split differs from its checkpoint")
    validation, test = metrics["best_validation"], metrics.get("test")
    if test is None or not math.isfinite(validation["log_rmse"]):
        raise ValueError("completed log trial lacks finite validation log RMSE or final test")
    streams = manifest["fixed_streams"]
    if (
        validation.get("sample_count") != streams["validation_samples"]
        or validation.get("uniform_sample_count") != streams["validation_uniform_samples"]
        or test.get("seed") != streams["data_seed"]
        or test.get("sample_count") != streams["test_samples"]
        or test.get("uniform_sample_count") != streams["test_uniform_samples"]
        or not isinstance(test.get("uniform_test_points_sha256"), str)
        or len(test["uniform_test_points_sha256"]) != 64
    ):
        raise ValueError("log trial evaluation does not match its common independent holdouts")
    scheduled = [entry for entry in checkpoint["history"] if "validation" in entry]
    selected = min(scheduled, key=lambda e: (e["validation"]["log_rmse"], e["global_step"]))
    if selected["global_step"] != checkpoint["best_step"]:
        raise ValueError("log trial was not selected by the first minimum validation log RMSE")
    best = read_single_checkpoint(directory / "best.pt")
    if (
        best["kind"] != "inference" or best["global_step"] != checkpoint["best_step"]
        or best.get("validation") != validation
        or best.get("selection_metric") != "log_rmse"
        or any(best.get(key) != checkpoint.get(key) for key in (
            "model_config", "physics", "dtype", "dataset_fingerprint", "code_fingerprint",
            "runtime", "objective_config",
        ))
        or not _state_equal(best["model_state"], checkpoint["best_state"])
    ):
        raise ValueError("best.pt differs from the validation-log-RMSE-selected checkpoint")
    nll_selected = min(scheduled, key=lambda e: (e["validation"]["nll"], e["global_step"]))
    for metric, filename, selected_entry in (
        ("log_rmse", "best_by_log.pt", selected), ("nll", "best_by_nll.pt", nll_selected),
    ):
        alternative = read_single_checkpoint(directory / filename)
        if (
            alternative.get("kind") != "inference"
            or alternative.get("selection_metric") != metric
            or alternative.get("global_step") != selected_entry["global_step"]
            or alternative.get("validation") != selected_entry["validation"]
            or any(alternative.get(key) != best.get(key) for key in (
                "model_config", "physics", "dtype", "dataset_fingerprint", "objective_config",
            ))
            or not _state_equal(alternative["model_state"],
                                checkpoint["selections"][metric]["model_state"])
        ):
            raise ValueError(f"{filename} differs from its validation selection")
        if metric == "log_rmse" and not _state_equal(alternative["model_state"], best["model_state"]):
            raise ValueError("best_by_log.pt and best.pt weights disagree")
    count = sum(value.numel() for name, value in best["model_state"].items()
                if name.endswith("weight") or name.endswith("bias"))
    split = checkpoint["sample_split"]
    results = {
        "global_step": steps, "selected_step": checkpoint["best_step"],
        "processed_examples": steps * row["batch_size"],
        "effective_passes": steps * row["batch_size"] / row["train_samples"],
        "parameter_count": count, "checkpoint_bytes": (directory / "best.pt").stat().st_size,
        "training_points_sha256": split["training_points_sha256"],
        "validation_points_sha256": split["validation_points_sha256"],
        "validation_uniform_points_sha256": split["uniform_validation_points_sha256"],
        "test_uniform_points_sha256": test["uniform_test_points_sha256"],
        "training_pool": split["training_pool"],
        "nll_selected_step": nll_selected["global_step"],
        "nll_selected_validation_nll": nll_selected["validation"]["nll"],
        "nll_selected_validation_log_rmse": nll_selected["validation"]["log_rmse"],
        "test_relative_ess": (test.get("proposal") or {}).get("relative_ess"),
    }
    for prefix, values in (("validation", validation), ("test", test)):
        for key in ("nll", "log_mse", "log_rmse", "relative_rmse"):
            results[f"{prefix}_{key}"] = values[key]
        results[f"{prefix}_relative_rmse_status"] = values["relative_rmse_status"]
        for output, source in (("kl", "forward_kl_estimate"),
                               ("kl_standard_error", "forward_kl_standard_error"),
                               ("hg_kl", "hg_forward_kl_estimate")):
            results[f"{prefix}_{output}"] = values[source]
    return results


def _receipt_files(directory: Path, manifest: dict[str, Any]) -> list[str]:
    names = {
        "checkpoint.pt", "best.pt", "best_by_log.pt", "best_by_nll.pt", "metrics.json",
        "config.json", "data_summary.json", "sample_split.json", "history.json",
    }
    if manifest["visualization"]["enabled"]:
        names.update(("learning_curves.png", "plots/plots.json", "plots/comparison.png",
                      "plots/training_scatter.npz"))
        names.update(p.relative_to(directory).as_posix() for p in (directory / "plots").rglob("*")
                     if p.is_file())
    if manifest["diagnostics"]["enabled"]:
        names.add("diagnostics/angular_diagnostics.json")
        names.update(p.relative_to(directory).as_posix()
                     for p in (directory / "diagnostics").rglob("*") if p.is_file())
    return sorted(names)


def _verify_receipt(directory: Path, row: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    receipt = _read_json(directory / _RECEIPT)
    if receipt.get("schema") != LOG_SWEEP_SCHEMA or receipt.get("manifest_sha256") != (
        _manifest_hash(manifest)
    ):
        raise ValueError("completed log trial belongs to a different sweep")
    hashes = receipt.get("files_sha256", {})
    if not set(_receipt_files(directory, manifest)) <= hashes.keys():
        raise ValueError("completed log trial receipt lacks required artifacts")
    for name, expected in hashes.items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("invalid file path in completed log trial receipt")
        if not path.is_file() or _file_hash(path) != expected:
            raise ValueError(f"completed log trial file changed or missing: {path}")
    results = _selected_fields(directory, row, manifest)
    if receipt.get("results") != results:
        raise ValueError("completed log trial metrics differ from its receipt")
    return {**results, "elapsed_seconds": receipt["elapsed_seconds"]}


def _write_receipt(
    directory: Path, row: dict[str, Any], manifest: dict[str, Any], elapsed: float,
) -> dict[str, Any]:
    results = _selected_fields(directory, row, manifest)
    atomic_json(directory / _RECEIPT, {
        "schema": LOG_SWEEP_SCHEMA, "manifest_sha256": _manifest_hash(manifest),
        "files_sha256": {name: _file_hash(directory / name)
                          for name in _receipt_files(directory, manifest)},
        "results": results, "elapsed_seconds": elapsed,
        "timing_scope": "recorded trial attempts including teacher setup/evaluation/plots",
    })
    return {**results, "elapsed_seconds": elapsed}


def _pareto_ids(rows: list[dict[str, Any]]) -> list[str]:
    complete = [row for row in rows if row["status"] == "complete"]
    return [row["trial_id"] for row in complete if not any(
        other["validation_log_rmse"] <= row["validation_log_rmse"]
        and other["validation_kl"] <= row["validation_kl"]
        and (other["validation_log_rmse"] < row["validation_log_rmse"]
             or other["validation_kl"] < row["validation_kl"])
        for other in complete
    )]


def _check_common_test_stream(rows: list[dict[str, Any]], additional: str | None = None) -> None:
    hashes = {row["test_uniform_points_sha256"] for row in rows if row["status"] == "complete"}
    if additional is not None:
        hashes.add(additional)
    if len(hashes) > 1:
        raise ValueError("completed log trials used different uniform test streams")


def _selection(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    winner = min(rows, key=lambda row: (row["validation_log_rmse"], row["beta"], row["trial_id"]))
    return {
        "schema": LOG_SWEEP_SCHEMA, "manifest_sha256": _manifest_hash(manifest),
        "selected_trial_id": winner["trial_id"], "selected_kind": winner["kind"],
        "selected_beta": winner["beta"], "selected_nll_weight": winner["nll_weight"],
        "selected_sampling": winner["sampling"],
        "selected_validation_log_rmse": winner["validation_log_rmse"],
        "selected_validation_kl": winner["validation_kl"],
        "selected_step": winner["selected_step"], "run_directory": winner["run_directory"],
        "rule": _SELECTION, "test_used": False, "total_loss_used": False,
        "pareto_trial_ids": _pareto_ids(rows),
    }


_COLUMNS = (
    "trial_id", "kind", "status", "beta", "nll_weight", "sampling", "selection_metric",
    "learning_rate", "train_samples", "batch_size", "planned_steps", "global_step",
    "selected_step", "processed_examples", "effective_passes", "validation_log_rmse",
    "validation_log_mse", "validation_relative_rmse", "validation_nll", "validation_kl",
    "validation_kl_standard_error", "test_log_rmse", "test_relative_rmse", "test_nll",
    "test_kl", "test_kl_standard_error", "test_relative_ess", "nll_selected_step",
    "nll_selected_validation_nll", "nll_selected_validation_log_rmse", "elapsed_seconds",
    "parameter_count", "checkpoint_bytes", "training_points_sha256",
    "validation_points_sha256", "validation_uniform_points_sha256", "test_uniform_points_sha256",
    "validation_relative_rmse_status", "test_relative_rmse_status", "run_directory",
)


def _summary_plot(output: Path, summary: dict[str, Any]) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(13, 9), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    complete = [row for row in summary["trials"] if row["status"] == "complete"]
    beta_rows = sorted((r for r in complete if r["kind"] == "beta"), key=lambda r: r["beta"])
    for ax, key, ylabel in (
        (axes[0, 0], "validation_log_rmse", "Validation log PDF RMSE [nats]"),
        (axes[0, 1], "validation_kl", "Validation forward KL [nats]"),
    ):
        if beta_rows:
            ax.plot([r["beta"] for r in beta_rows], [r[key] for r in beta_rows], "o-",
                    label="Mixture: NLL + beta log-MSE")
            positive = [r["beta"] for r in beta_rows if r["beta"] > 0]
            ax.set_xscale("symlog", linthresh=min(positive) if positive else 0.01)
            ax.set_xticks([r["beta"] for r in beta_rows], [f"{r['beta']:g}" for r in beta_rows])
        for row in complete:
            if row["kind"] != "beta":
                ax.axhline(row[key], linestyle="--", label=row["trial_id"])
        ax.set_xlabel("Beta (zero included)")
        ax.set_ylabel(ylabel)
        if complete:
            ax.legend(fontsize=8)
    for row in complete:
        axes[1, 0].scatter(row["validation_kl"], row["validation_log_rmse"], s=40)
        axes[1, 0].annotate(row["trial_id"],
                            (row["validation_kl"], row["validation_log_rmse"]), fontsize=8)
    axes[1, 0].set_xlabel("Validation forward KL [nats]")
    axes[1, 0].set_ylabel("Validation log PDF RMSE [nats]")
    axes[1, 0].set_title("Both smaller is better; no test-based selection")
    for row in summary["trials"]:
        history_path = output / row["run_directory"] / "history.json"
        if not history_path.is_file():
            continue
        values = [entry for entry in _read_json(history_path) if "validation" in entry]
        if values:
            axes[1, 1].plot([v["global_step"] for v in values],
                            [v["validation"]["log_rmse"] for v in values],
                            label=row["trial_id"])
    axes[1, 1].set_xlabel("Optimizer updates")
    axes[1, 1].set_ylabel("Current validation log PDF RMSE [nats]")
    if axes[1, 1].lines:
        axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    figure.suptitle(
        "Single-condition log-density objective comparison\n"
        "Common holdouts, fixed model/data size/update count; validation log RMSE selects weights",
        fontsize=11,
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150)
    _atomic_bytes(output / "summary.png", buffer.getvalue())
    figure.clear()


def _write_summary(output: Path, summary: dict[str, Any]) -> None:
    summary["pareto_trial_ids"] = _pareto_ids(summary["trials"])
    atomic_json(output / "summary.json", summary)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=_COLUMNS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(summary["trials"])
    _atomic_bytes(output / "summary.csv", buffer.getvalue().encode("utf-8-sig"))

    def number(value: Any) -> str:
        return "-" if value is None else f"{value:.6g}"

    lines = [
        "# Single-condition log-density loss sweep", "", f"Status: **{summary['status']}**.", "",
        "All trials select checkpoints by the same independent validation log RMSE.",
        "Beta changes the training objective; total losses across beta are not ranking scores.",
        "Final test metrics, angular profiles and training errors do not select the winner.", "",
        "| Trial | Status | Beta | NLL weight | Validation log RMSE | Validation KL | Best step | Seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["trials"]:
        lines.append(
            f"| [{row['trial_id']}]({row['run_directory']}/metrics.json) | {row['status']} | "
            f"{row['beta']:g} | {row['nll_weight']:g} | {number(row.get('validation_log_rmse'))} | "
            f"{number(row.get('validation_kl'))} | {row.get('selected_step', '-')} | "
            f"{number(row.get('elapsed_seconds'))} |"
        )
    if summary.get("selection"):
        selected = summary["selection"]
        lines.extend(["", f"Validation log-RMSE winner: **{selected['selected_trial_id']}**.",
                      f"Checkpoint: `{selected['run_directory']}/best.pt`."])
    else:
        lines.extend(["", "No final winner is declared until all planned trials are complete."])
    lines.extend([
        "", "Pareto set for validation KL and log RMSE: "
        + (", ".join(f"`{name}`" for name in summary["pareto_trial_ids"]) or "none yet") + ".",
        "", "![Beta comparison and validation histories](summary.png)", "",
        "Each trial preserves best.pt (= best_by_log.pt), best_by_nll.pt and checkpoint.pt.",
        "The NLL-selected alternative is reporting-only; its step/log RMSE/NLL are in CSV/JSON.",
        "Mixture betas share the same actual pool; target_nll has an intentionally different pool.",
        "Mixed training scatter shows all actual CDF and uniform queries with distinct labels.",
        "Log RMSE uses natural log density ratios and uniform solid angle, without log1p or a floor.",
        "KL Monte Carlo estimates may be negative. One condition/seed does not establish generality.",
        "Elapsed seconds include setup/evaluation/plots and are not GPU kernel timings.",
    ])
    if summary.get("error"):
        lines.extend(["", "## Stopped trial", "", summary["error"]])
    _atomic_bytes(output / "summary.md", ("\n".join(lines) + "\n").encode("utf-8"))
    _summary_plot(output, summary)


def run_log_loss_sweep(
    reference: RainbowReference,
    model_config: SphereFlowConfig,
    training_config: SingleTrainingConfig,
    output_directory: str | Path,
    *,
    objective_config: LogObjectiveConfig | None = None,
    sweep_config: LogSweepConfig | None = None,
    make_plots: bool = True,
    plot_config: RainbowPlotConfig | None = None,
    diagnostic_config: AngularDiagnosticConfig | None = None,
    diagnostics_enabled: bool = True,
    callback: Callable[[dict[str, Any]], None] | None = None,
    max_trials_this_run: int | None = None,
    max_steps_this_trial: int | None = None,
) -> dict[str, Any]:
    """Train independent beta trials; verify immutable results before exact reuse.

    A controlled step limit pauses within a trial without changing its planned
    final update count. Rerun without that limit to complete the same experiment.
    """
    for name, value in (("max_trials_this_run", max_trials_this_run),
                        ("max_steps_this_trial", max_steps_this_trial)):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"{name} must be a nonnegative integer")
    if type(make_plots) is not bool or type(diagnostics_enabled) is not bool:
        raise ValueError("make_plots and diagnostics_enabled must be boolean")
    objective = objective_config or LogObjectiveConfig()
    sweep = sweep_config or LogSweepConfig()
    plotting = plot_config or RainbowPlotConfig()
    diagnostic = diagnostic_config or AngularDiagnosticConfig()
    cfg = training_config
    rows = _trial_plan(cfg, objective, sweep)
    if cfg.tensorboard:
        check_tensorboard()
    if torch.device(cfg.device).type == "cuda" and cfg.deterministic:
        workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if workspace not in (":4096:8", ":16:8"):
            raise ValueError("deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
    device = _resolve_device(cfg.device)
    _setup_runtime(cfg)
    # Fail for a true-zero teacher before writing a manifest or any trial.
    teacher_check = verify_positive_teacher(reference)
    manifest = {
        "schema": LOG_SWEEP_SCHEMA, "model": model_config.to_dict(),
        "base_training": cfg.to_dict(), "base_objective": objective.to_dict(),
        "sweep": sweep.to_dict(), "visualization": {"enabled": make_plots, **plotting.to_dict()},
        "diagnostics": {"enabled": diagnostics_enabled, "configuration": diagnostic.to_dict()},
        "dataset_fingerprint": reference.fingerprint(), "data_summary": reference.summary(),
        "teacher_log_domain_check": teacher_check,
        "training_code_fingerprint": _code_hash(), "sweep_code_sha256": _file_hash(Path(__file__)),
        "plotting_code_sha256": _file_hash(Path(__file__).with_name("plotting.py")),
        "diagnostic_code_sha256": _file_hash(Path(__file__).with_name("angular_diagnostics.py")),
        "runtime": _runtime(device, cfg.dtype),
        "fixed_streams": _fixed_streams(reference, model_config, cfg, objective, rows),
        "selection": _SELECTION, "test_policy": "report only; never used for model selection",
        "trial_objectives": {row["trial_id"]: row["objective_config"] for row in rows},
    }
    output = Path(output_directory)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if _read_json(manifest_path) != manifest:
            raise ValueError("log sweep data/configuration/code/runtime changed; use a new output directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("log sweep output is nonempty without a manifest; choose a new folder")
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest_path, manifest)
    # Verify *all* previously complete trials before updating any scientific
    # result. A corrupt later trial must not allow earlier ones to be retrained.
    for row in rows:
        directory = output / row["run_directory"]
        if (directory / _RECEIPT).is_file():
            row.update(_verify_receipt(directory, row, manifest), status="complete")
        elif (directory / "checkpoint.pt").is_file():
            payload = read_single_checkpoint(directory / "checkpoint.pt")
            _check_checkpoint(payload, row, manifest)
            row.update(status="partial", global_step=payload["global_step"])
        elif directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"partial log trial has no resumable checkpoint: {directory}")
    _check_common_test_stream(rows)
    summary = {
        "schema": LOG_SWEEP_SCHEMA, "manifest_sha256": _manifest_hash(manifest),
        "status": "running", "trials": rows, "selection": None,
        "selection_metric": "validation_log_rmse", "test_used_for_selection": False,
        "total_loss_used_for_selection": False,
        "scope": "single condition and fixed seed; synthetic inputs are not optical validation",
    }
    executed = 0

    def notify(event: dict[str, Any]) -> None:
        if callback is not None:
            callback(event)

    try:
        for row in rows:
            if row["status"] == "complete":
                notify({"event": "trial_reused", "trial_id": row["trial_id"]})
                continue
            if max_trials_this_run is not None and executed >= max_trials_this_run:
                summary["status"] = "paused"
                break
            directory = output / row["run_directory"]
            checkpoint = directory / "checkpoint.pt"
            resume = checkpoint if checkpoint.is_file() else None
            row["status"] = "running"
            _write_summary(output, summary)
            notify({"event": "trial_resumed" if resume else "trial_started",
                    "trial_id": row["trial_id"], "beta": row["beta"],
                    "sampling": row["sampling"], "run_directory": str(directory)})
            attempts_path = output / "attempts" / f"{row['trial_id']}.json"
            attempts = _read_json(attempts_path) if attempts_path.exists() else []
            started = time.perf_counter()
            executed += 1
            succeeded = False
            try:
                result = train_single_condition(
                    reference, model_config, cfg, directory,
                    objective_config=LogObjectiveConfig.from_dict(row["objective_config"]),
                    resume=resume, max_steps_this_run=max_steps_this_trial,
                    make_plots=make_plots, plot_config=plotting,
                    callback=lambda event, name=row["trial_id"]: notify({**event, "trial_id": name}),
                )
                if result.complete:
                    _check_common_test_stream(rows, result.metrics["test"]["uniform_test_points_sha256"])
                if result.complete and diagnostics_enabled:
                    evaluate_angular_diagnostics(
                        result.model, reference, directory / "diagnostics",
                        train_samples=cfg.train_samples, batch_size=cfg.batch_size,
                        config=diagnostic, checkpoint_path=directory / "best.pt",
                        objective_config=LogObjectiveConfig.from_dict(row["objective_config"]),
                    )
                succeeded = result.complete
            finally:
                attempts.append({"elapsed_seconds": time.perf_counter() - started,
                                 "complete": succeeded, "resumed": resume is not None})
                atomic_json(attempts_path, attempts)
            row["global_step"] = result.global_step
            if not result.complete:
                row["status"] = "partial"
                summary["status"] = "paused"
                del result
                break
            elapsed = sum(attempt["elapsed_seconds"] for attempt in attempts)
            row.update(_write_receipt(directory, row, manifest, elapsed), status="complete")
            del result
            notify({"event": "trial_completed", "trial_id": row["trial_id"],
                    "validation_log_rmse": row["validation_log_rmse"],
                    "validation_kl": row["validation_kl"]})
            _write_summary(output, summary)
        if all(row["status"] == "complete" for row in rows):
            summary["status"] = "complete"
            summary["selection"] = _selection(rows, manifest)
            selection_path = output / "selection.json"
            if selection_path.exists() and _read_json(selection_path) != summary["selection"]:
                raise ValueError("saved log sweep selection differs from verified validation results")
            if not selection_path.exists():
                atomic_json(selection_path, summary["selection"])
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


def replot_log_loss_sweep(
    reference: RainbowReference, sweep_directory: str | Path, output_directory: str | Path,
    *, device: str | None = None,
) -> dict[str, Any]:
    """Render verified completed weights in a new tree; never train or edit trials."""
    source, output = Path(sweep_directory).resolve(), Path(output_directory).resolve()
    manifest = _read_json(source / "manifest.json")
    if manifest.get("schema") != LOG_SWEEP_SCHEMA:
        raise ValueError("not a log-loss sweep manifest")
    if reference.fingerprint() != manifest["dataset_fingerprint"]:
        raise ValueError("replot teacher differs from the saved log sweep")
    cfg = SingleTrainingConfig.from_dict(manifest["base_training"])
    rows = _trial_plan(cfg, LogObjectiveConfig.from_dict(manifest["base_objective"]),
                       LogSweepConfig.from_dict(manifest["sweep"]))
    complete = []
    for row in rows:
        directory = source / row["run_directory"]
        if output == directory or output.is_relative_to(directory) or directory.is_relative_to(output):
            raise ValueError("replot output must be disjoint from every saved trial")
        if (directory / _RECEIPT).is_file():
            complete.append({**row, **_verify_receipt(directory, row, manifest)})
    if not complete:
        raise ValueError("no verified completed log trials to replot")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("replot output must be empty; choose a new directory")
    replot_cfg = replace(cfg, device=device or cfg.device)
    if torch.device(replot_cfg.device).type == "cuda" and replot_cfg.deterministic:
        workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if workspace not in (":4096:8", ":16:8"):
            raise ValueError("deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
    target_device = _resolve_device(device or cfg.device)
    _setup_runtime(replot_cfg)
    plot_values = {key: value for key, value in manifest["visualization"].items() if key != "enabled"}
    plotting = RainbowPlotConfig.from_dict(plot_values)
    diagnostic = AngularDiagnosticConfig.from_dict(manifest["diagnostics"]["configuration"])
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": LOG_SWEEP_SCHEMA, "status": "running", "optimizer_updates": 0,
        "source_manifest_sha256": _manifest_hash(manifest), "trials": [],
        "runtime": _runtime(target_device, cfg.dtype),
        "plotting_code_sha256": _file_hash(Path(__file__).with_name("plotting.py")),
        "diagnostic_code_sha256": _file_hash(Path(__file__).with_name("angular_diagnostics.py")),
        "sweep_code_sha256": _file_hash(Path(__file__)),
    }
    atomic_json(output / "replot.json", report)
    for row in complete:
        checkpoint = source / row["run_directory"] / "best.pt"
        model, payload = load_single_checkpoint(checkpoint, device=target_device)
        destination = output / row["trial_id"]
        plot_rainbow_comparison(
            model, reference, destination / "plots", config=plotting,
            checkpoint_path=checkpoint, selected_step=payload["global_step"],
            title=f"{row['trial_id']} | validation log RMSE selected, step {payload['global_step']}",
        )
        if manifest["diagnostics"]["enabled"]:
            evaluate_angular_diagnostics(
                model, reference, destination / "diagnostics", train_samples=cfg.train_samples,
                batch_size=cfg.batch_size, config=diagnostic, checkpoint_path=checkpoint,
                objective_config=LogObjectiveConfig.from_dict(row["objective_config"]),
            )
        report["trials"].append({"trial_id": row["trial_id"], "plot_directory": row["trial_id"],
                                 "checkpoint_sha256": _file_hash(checkpoint),
                                 "training_points_sha256": row["training_points_sha256"]})
        atomic_json(output / "replot.json", report)
        del model
    report["status"] = "complete"
    atomic_json(output / "replot.json", report)
    return report


def _read_base_config(path: Path, device: str | None):
    values = _read_json(path)
    if not isinstance(values, dict) or values.get("family") != FAMILY:
        raise ValueError("base config must describe the Rainbow single-condition family")
    allowed = {"schema_version", "family", "model", "training", "visualization"}
    version = values.get("schema_version")
    if type(version) is not int:
        raise ValueError("base config schema_version must be an integer")
    if version == 3:
        allowed.add("objective")
        if "objective" not in values:
            raise ValueError("schema-3 base config requires an explicit objective")
    elif version != 2:
        raise ValueError("base config requires schema_version=2 or 3")
    if set(values) - allowed or not {"model", "training"} <= values.keys():
        raise ValueError("unknown/missing base config fields; objective requires schema_version=3")
    model = SphereFlowConfig.from_dict(values["model"])
    training_values = dict(values["training"])
    if device is not None:
        training_values["device"] = device
    training = SingleTrainingConfig.from_dict(training_values)
    visualization = dict(values.get("visualization", {}))
    enabled = visualization.pop("enabled", True)
    if type(enabled) is not bool:
        raise ValueError("visualization.enabled must be boolean")
    plotting = RainbowPlotConfig.from_dict(visualization)
    objective = LogObjectiveConfig.from_dict(values["objective"]) if version == 3 else LogObjectiveConfig()
    return model, training, objective, enabled, plotting


def _read_sweep_config(path: Path):
    values = _read_json(path)
    if (
        not isinstance(values, dict) or type(values.get("schema_version")) is not int
        or values.get("schema_version") != 1
        or set(values) != {"schema_version", "sweep", "diagnostics"}
        or not isinstance(values["diagnostics"], dict)
    ):
        raise ValueError("log sweep config requires schema_version=1, sweep and diagnostics")
    sweep = LogSweepConfig.from_dict(values["sweep"])
    diagnostic = values["diagnostics"].copy()
    enabled = diagnostic.pop("enabled", True)
    if type(enabled) is not bool:
        raise ValueError("diagnostics.enabled must be boolean")
    return sweep, enabled, AngularDiagnosticConfig.from_dict(diagnostic)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("run", "plan", "replot"), default="run")
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/rainbow_log_loss.json"))
    parser.add_argument("--sweep-config", type=Path, default=Path("configs/rainbow_log_sweep.json"))
    parser.add_argument("--output", required=True, type=Path, help="new/existing sweep directory")
    parser.add_argument("--device", help="CUDA default from config; cpu is an explicit debug choice")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max-trials-this-run", type=int)
    parser.add_argument("--max-steps-this-trial", type=int)
    parser.add_argument("--plot-output", type=Path, help="replot only: new directory, disjoint from trials")
    args = parser.parse_args(argv)
    if args.mode == "replot":
        destination = args.plot_output or args.output / "replots" / datetime.now(timezone.utc).strftime(
            "%Y%m%d-%H%M%S-%f"
        )
        with RainbowReference(args.record) as reference:
            report = replot_log_loss_sweep(reference, args.output, destination, device=args.device)
        print(json.dumps({**report, "output_directory": str(destination.resolve())}, allow_nan=False))
        return 0
    if args.plot_output is not None:
        parser.error("--plot-output applies only to replot")
    model, training, objective, plot_enabled, plotting = _read_base_config(args.config, args.device)
    sweep, diagnostic_enabled, diagnostic = _read_sweep_config(args.sweep_config)
    if args.mode == "plan":
        rows = _trial_plan(training, objective, sweep)
        print(json.dumps({
            "schema": LOG_SWEEP_SCHEMA, "status": "planned", "optimizer_updates": 0,
            "planned_total_updates": len(rows) * training.steps,
            "record": str(args.record.resolve()), "output": str(args.output.resolve()),
            "model": model.to_dict(), "training": training.to_dict(), "trials": rows,
            "selection": _SELECTION,
            "checks": "configuration only; no CDF/GPU access, output creation or training",
        }, indent=2, allow_nan=False))
        return 0

    def progress(event: dict[str, Any]) -> None:
        if not args.quiet:
            print(json.dumps(event, allow_nan=False), flush=True)

    with RainbowReference(args.record) as reference:
        summary = run_log_loss_sweep(
            reference, model, training, args.output, objective_config=objective,
            sweep_config=sweep, make_plots=plot_enabled and not args.no_plots, plot_config=plotting,
            diagnostic_config=diagnostic, diagnostics_enabled=diagnostic_enabled and not args.no_diagnostics,
            callback=progress, max_trials_this_run=args.max_trials_this_run,
            max_steps_this_trial=args.max_steps_this_trial,
        )
    print(json.dumps({"status": summary["status"], "complete": summary["status"] == "complete",
                      "output_directory": str(args.output.resolve()),
                      "selection": summary["selection"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
