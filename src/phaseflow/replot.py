"""Replot verified completed sweeps without changing their scientific artifacts.

Saved sweep manifests describe the original training implementation. They are
validated as saved records, rather than compared with a newly constructed
training manifest: this is inference and point-pool reconstruction, not resume.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from .plotting import RainbowPlotConfig, plot_rainbow_comparison
from .rainbow import RainbowReference
from .single_condition import (
    SingleTrainingConfig,
    _code_hash,
    _resolve_device,
    _runtime,
    load_single_checkpoint,
)
from .sweep import SWEEP_SCHEMA, SweepConfig, _file_hash, _manifest_hash, _trial, _verify_receipt
from .training import atomic_json

REPLOT_SCHEMA = "phaseflow.sweep_training_point_plots.v1"


def _read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _trial_directory(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Invalid saved trial directory")
    value = PurePosixPath(relative)
    if value.is_absolute() or ".." in value.parts or ":" in relative:
        raise ValueError("Saved trial directory must remain inside the sweep")
    result = (root / relative).resolve()
    if result == root or not result.is_relative_to(root):
        raise ValueError("Saved trial directory must remain inside the sweep")
    return result


def _verify_completed_sweep(reference: RainbowReference, root: Path):
    manifest = _read_object(root / "manifest.json")
    summary = _read_object(root / "summary.json")
    selection = _read_object(root / "selection.json")
    if manifest.get("schema") != SWEEP_SCHEMA:
        raise ValueError("Unsupported saved sweep manifest")
    if manifest.get("dataset_fingerprint") != reference.fingerprint():
        raise ValueError("Replot record differs from the saved sweep teacher")
    manifest_hash = _manifest_hash(manifest)
    if (
        summary.get("schema") != SWEEP_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("manifest_sha256") != manifest_hash
        or summary.get("selection_metric") != "validation_nll"
        or summary.get("test_used_for_selection") is not False
    ):
        raise ValueError("Replot requires a complete, consistent saved sweep summary")
    cfg = SingleTrainingConfig.from_dict(manifest["base_training"])
    sweep = SweepConfig.from_dict(manifest["sweep"])
    rows = summary.get("trials")
    if not isinstance(rows, list) or len(rows) != len(sweep.learning_rates) + len(sweep.train_samples):
        raise ValueError("Saved sweep summary has an unexpected trial count")
    expected_a = [
        _trial("A", f"lr_{float(rate)}", replace(
            cfg, learning_rate=rate, train_samples=sweep.baseline_train_samples,
        )) for rate in sweep.learning_rates
    ]
    physical: dict[str, dict[str, Any]] = {}
    original_hashes = {
        name: _file_hash(root / name)
        for name in (
            "manifest.json", "selection.json", "summary.json", "summary.csv", "summary.md",
            "summary.png",
        ) if (root / name).is_file()
    }

    def verify_row(row: dict[str, Any], expected: dict[str, Any]) -> None:
        if not isinstance(row, dict) or row.get("status") != "complete":
            raise ValueError("Saved sweep includes a non-completed trial")
        for key in (
            "trial_id", "stage", "run_directory", "learning_rate", "train_samples",
            "training_config", "reuse_of",
        ):
            if row.get(key) != expected[key]:
                raise ValueError(f"Saved sweep trial differs in {key}")
        relative = row["run_directory"]
        directory = _trial_directory(root, relative)
        if relative not in physical:
            training = SingleTrainingConfig.from_dict(row["training_config"])
            verified = _verify_receipt(directory, training, manifest)
            receipt_name = f"{relative}/sweep_completed.json"
            original_hashes[receipt_name] = _file_hash(root / receipt_name)
            receipt = _read_object(root / receipt_name)
            for filename, digest in receipt["files_sha256"].items():
                original_hashes[f"{relative}/{filename}"] = digest
            physical[relative] = {
                "run_directory": relative,
                "trial_ids": [],
                "training_config": row["training_config"],
                "results": verified,
                "receipt_sha256": original_hashes[receipt_name],
            }
        saved = physical[relative]
        if saved["training_config"] != row["training_config"]:
            raise ValueError("Reused trial has a different training configuration")
        for key, value in saved["results"].items():
            if row.get(key) != value:
                raise ValueError(f"Saved summary differs from verified trial results in {key}")
        saved["trial_ids"].append(row["trial_id"])

    for row, expected in zip(rows, expected_a, strict=False):
        verify_row(row, expected)
    a_rows = rows[:len(expected_a)]
    winner = min(a_rows, key=lambda row: (row["validation_nll"], row["learning_rate"]))
    expected_selection = {
        "schema": SWEEP_SCHEMA,
        "manifest_sha256": manifest_hash,
        "selected_trial_id": winner["trial_id"],
        "selected_learning_rate": winner["learning_rate"],
        "selected_validation_nll": winner["validation_nll"],
        "rule": manifest["selection"],
        "test_used": False,
        "candidates": [{key: row[key] for key in (
            "trial_id", "learning_rate", "validation_nll", "validation_kl", "selected_step",
            "run_directory",
        )} for row in a_rows],
    }
    if selection != expected_selection or summary.get("selection") != selection:
        raise ValueError("Saved selection differs from verified validation results")
    for row, n in zip(rows[len(expected_a):], sweep.train_samples, strict=True):
        expected = _trial("B", f"n_{n}", replace(
            cfg, train_samples=n, learning_rate=winner["learning_rate"],
        ))
        if expected["training_config"] == winner["training_config"]:
            expected.update(reuse_of=winner["trial_id"], run_directory=winner["run_directory"])
        verify_row(row, expected)
    return manifest, summary, list(physical.values()), original_hashes


def replot_sweep_training_points(
    reference: RainbowReference,
    sweep_directory: str | Path,
    output_directory: str | Path | None = None,
    *,
    device: str = "cuda",
    plot_config: RainbowPlotConfig | None = None,
    callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Plot every saved training pool once per physical trial, without training.

    The default destination is ``<sweep>/training_point_plots``. Existing trial
    plots, completion receipts, metrics, checkpoints and selection are retained.
    An identical replot invocation may rewrite only its own output tree; changed
    inputs/configuration/implementation require a new destination.
    """
    plotting = RainbowPlotConfig() if plot_config is None else plot_config
    if not isinstance(plotting, RainbowPlotConfig):
        raise ValueError("plot_config must be a RainbowPlotConfig")
    root = Path(sweep_directory).resolve()
    manifest, summary, physical, original_hashes = _verify_completed_sweep(reference, root)
    output = (root / "training_point_plots" if output_directory is None
              else Path(output_directory)).resolve()
    if root.is_relative_to(output):
        raise ValueError("Replot output must not overwrite or contain the original sweep")
    for trial in physical:
        original = _trial_directory(root, trial["run_directory"])
        if output.is_relative_to(original) or original.is_relative_to(output):
            raise ValueError("Replot output must not overlap any original trial directory")
    # Resolve CUDA explicitly before writing output; CPU fallback is never used.
    resolved_device = _resolve_device(device)
    sources = Path(__file__).parent
    mappings = [{
        "trial_id": row["trial_id"], "stage": row["stage"],
        "learning_rate": row["learning_rate"], "train_samples": row["train_samples"],
        "selected_step": row["selected_step"], "reuse_of": row["reuse_of"],
        "run_directory": row["run_directory"],
        "plot_directory": row["run_directory"],
    } for row in summary["trials"]]
    inputs = {
        "schema": REPLOT_SCHEMA,
        "scope": "completed_sweep_training_point_visualization_only",
        "sweep_directory": str(root),
        "dataset_fingerprint": reference.fingerprint(),
        "original_manifest_sha256": _manifest_hash(manifest),
        "original_files_sha256": original_hashes,
        "configuration": plotting.to_dict(),
        "scatter_mode": "training",
        "all_training_pool_points": True,
        "original_training_code_fingerprint": manifest["training_code_fingerprint"],
        "current_training_code_fingerprint": _code_hash(),
        "implementation_sha256": {
            name: _file_hash(sources / name)
            for name in ("replot.py", "plotting.py", "training_scatter.py")
        },
        "runtime": _runtime(resolved_device, manifest["base_training"]["dtype"]),
        "logical_trials": mappings,
    }
    input_path = output / "replot_manifest.json"
    if output.exists():
        if not output.is_dir():
            raise FileExistsError("Replot output is not a directory")
        if input_path.exists():
            if _read_object(input_path) != inputs:
                raise ValueError("Replot inputs/configuration/code/runtime changed; use a new output")
        elif any(output.iterdir()):
            raise FileExistsError("Replot output is not empty and has no matching replot manifest")
        for path in output.rglob("*"):
            if path.is_symlink() and not path.resolve().is_relative_to(output):
                raise ValueError("Replot output contains a link outside its own output directory")
            if path.is_file() and path.stat().st_nlink > 1:
                raise ValueError("Replot output contains a hard-linked file; use a new output")
    # Preflight every destination before rendering any trial, including symlinks.
    for trial in physical:
        destination = (output / trial["run_directory"]).resolve()
        if not destination.is_relative_to(output) or destination == output:
            raise ValueError("Replot trial destination escapes the new output directory")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(input_path, inputs)

    def notify(event: str, **values: Any) -> None:
        if callback is not None:
            callback({"event": event, **values})

    result = {
        "schema": REPLOT_SCHEMA,
        "status": "running",
        "scope": inputs["scope"],
        "output_directory": str(output),
        "sweep_directory": str(root),
        "dataset_fingerprint": reference.fingerprint(),
        "replot_manifest_sha256": _file_hash(input_path),
        "logical_trial_count": len(mappings),
        "physical_trial_count": len(physical),
        "optimizer_updates": 0,
        "selection_changed": False,
        "original_artifacts_unchanged": None,
        "logical_trials": mappings,
        "physical_trials": [],
    }
    result_path = output / "replot_summary.json"
    atomic_json(result_path, result)
    notify("replot_started", physical_trial_count=len(physical), logical_trial_count=len(mappings),
           output_directory=str(output))
    try:
        for trial in physical:
            original = root / trial["run_directory"]
            destination = output / trial["run_directory"]
            checkpoint_path = original / "best.pt"
            notify("trial_replot_started", trial_ids=trial["trial_ids"],
                   train_samples=trial["training_config"]["train_samples"],
                   run_directory=str(original))
            model, checkpoint = load_single_checkpoint(checkpoint_path, device=resolved_device)
            try:
                report = plot_rainbow_comparison(
                    model, reference, destination, config=plotting,
                    checkpoint_path=checkpoint_path, selected_step=checkpoint["global_step"],
                    title=f"Validation-selected model (step {checkpoint['global_step']})",
                    scatter_mode="training", training_run_directory=original,
                )
            finally:
                del model  # At most one physical trial's inference model is retained.
            if report["scatter"]["sample_count"] != trial["training_config"]["train_samples"]:
                raise ValueError("Plot did not retain every fixed training-pool entry")
            result["physical_trials"].append({
                "trial_ids": trial["trial_ids"],
                "run_directory": trial["run_directory"],
                "plot_directory": trial["run_directory"],
                "selected_step": checkpoint["global_step"],
                "training_points_sha256": trial["results"]["sample_split"]["training_points_sha256"],
                "sample_count": report["scatter"]["sample_count"],
                "scatter": report["scatter"],
                "files_sha256": {
                    path.relative_to(destination).as_posix(): _file_hash(path)
                    for path in sorted(destination.rglob("*")) if path.is_file()
                },
            })
            del checkpoint, report
            atomic_json(result_path, result)
            notify("trial_replot_completed", trial_ids=trial["trial_ids"],
                   plot_directory=str(destination),
                   sample_count=trial["training_config"]["train_samples"])
        for relative, expected in original_hashes.items():
            if _file_hash(root / relative) != expected:
                raise ValueError(f"Original sweep artifact changed during replot: {relative}")
        result.update(status="complete", original_artifacts_unchanged=True)
        atomic_json(result_path, result)
        notify("replot_completed", output_directory=str(output),
               physical_trial_count=len(physical), logical_trial_count=len(mappings))
        return result
    except BaseException as error:
        result["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        atomic_json(result_path, result)
        raise
