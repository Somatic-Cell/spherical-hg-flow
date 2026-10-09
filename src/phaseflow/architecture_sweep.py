"""Architecture/LR orchestration over unchanged, separately isolated capacity runs.

This module does not implement another trainer. Each candidate invokes the
existing capacity CLI in the same interpreter, checks its import origin, and
verifies the resulting immutable receipts before reporting a completed result.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import angular_diagnostics
from .capacity_sweep import (
    CAPACITY_SCHEMA,
    CapacitySweepConfig,
    _atomic_json,
    _implementation,
    _prepare_runtime,
    _rows,
    _safe_tree,
    _streams,
    _verified_completed_capacity,
    _verify_snapshot,
)
from .cli import _read_rainbow_config
from .rainbow import RainbowReference
from .single_condition import SingleTrainingConfig, _code_hash, _runtime
from .sweep import _atomic_bytes, _file_hash, _manifest_hash, _read_json

SCHEMA = "phaseflow.architecture_sweep.v1"
_LOCK = ".architecture.lock"
_CONFIG_KEYS = {
    "schema_version", "base_config", "num_bins", "train_samples", "architectures",
    "learning_rates", "milestones", "diagnostics",
}


def load_plan(config_path: str | Path, device: str | None = None) -> dict[str, Any]:
    """Validate and expand a JSON plan without writing files or starting CUDA."""
    path = Path(config_path).resolve()
    value = _read_json(path)
    if not isinstance(value, dict) or set(value) != _CONFIG_KEYS or (
        type(value["schema_version"]) is not int or value["schema_version"] != 1
    ):
        raise ValueError("architecture config requires schema_version=1 and exactly the documented keys")
    if not isinstance(value["base_config"], str) or not value["base_config"]:
        raise ValueError("base_config must be a nonempty path relative to the architecture config")
    for key in ("num_bins", "train_samples"):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    model, training, plots, plotting = _read_rainbow_config(path.parent / value["base_config"], device)
    schedule = CapacitySweepConfig(num_bins=(value["num_bins"],), milestones=value["milestones"])
    if training.steps != schedule.milestones[-1]:
        raise ValueError("base training.steps must equal the final milestone from the start")
    if any(step % training.eval_every for step in schedule.milestones):
        raise ValueError("milestones must be multiples of training.eval_every")
    architectures = value["architectures"]
    if not isinstance(architectures, list) or not architectures:
        raise ValueError("architectures must be a nonempty list")
    models, seen = [], set()
    for item in architectures:
        if not isinstance(item, dict) or set(item) != {"num_coupling_layers", "hidden_features"}:
            raise ValueError("each architecture requires num_coupling_layers and hidden_features")
        if not isinstance(item["hidden_features"], list) or not item["hidden_features"]:
            raise ValueError("hidden_features must be a nonempty list of positive integers")
        trial_model = replace(model, num_bins=value["num_bins"], **item)
        identity = (trial_model.num_coupling_layers, trial_model.hidden_features)
        if identity in seen:
            raise ValueError("duplicate architecture")
        seen.add(identity)
        models.append(trial_model)
    rates = value["learning_rates"]
    if not isinstance(rates, list) or not rates or any(
        isinstance(rate, bool) or not isinstance(rate, (int, float))
        or not math.isfinite(rate) or rate <= 0 for rate in rates
    ) or len(set(rates)) != len(rates):
        raise ValueError("learning_rates must be unique finite positive numbers")
    diagnostics = value["diagnostics"]
    if not isinstance(diagnostics, dict):
        raise ValueError("diagnostics must be an object")
    diagnostic_values = dict(diagnostics)
    enabled = diagnostic_values.pop("enabled", True)
    if type(enabled) is not bool:
        raise ValueError("diagnostics.enabled must be boolean")
    diagnostic = angular_diagnostics.AngularDiagnosticConfig.from_dict(diagnostic_values)
    child_sweep = {
        "schema_version": 1, "sweep": schedule.to_dict(),
        "diagnostics": {"enabled": enabled, **diagnostic.to_dict()},
    }
    trials = []
    for trial_model in models:
        for rate in rates:
            cfg = replace(training, train_samples=value["train_samples"], learning_rate=float(rate))
            width = "x".join(map(str, trial_model.hidden_features))
            trial_id = (f"trial_{len(trials) + 1:02d}_l{trial_model.num_coupling_layers}"
                        f"_h{width}_lr{float(rate):.12g}")
            trials.append({
                "trial_id": trial_id,
                "configuration": {
                    "schema_version": 2, "family": "rainbow_single_condition",
                    "model": trial_model.to_dict(), "training": cfg.to_dict(),
                    "visualization": {"enabled": plots, **plotting.to_dict()},
                },
            })
    return {"schema": SCHEMA, "trials": trials, "child_sweep": child_sweep}


def _diagnostic_identity(plan: dict[str, Any]) -> dict[str, Any] | None:
    config = dict(plan["child_sweep"]["diagnostics"])
    if not config.pop("enabled"):
        return None
    return {
        "id": "phaseflow.angular_diagnostics.v1", "configuration": config,
        "implementation_sha256": _file_hash(Path(angular_diagnostics.__file__)),
    }


def _manifest(reference: RainbowReference, plan: dict[str, Any]) -> dict[str, Any]:
    training = SingleTrainingConfig.from_dict(plan["trials"][0]["configuration"]["training"])
    device = _prepare_runtime(training)
    return {
        "schema": SCHEMA, "plan": plan,
        "dataset_fingerprint": reference.fingerprint(), "data_summary": reference.summary(),
        "training_code_fingerprint": _code_hash(), "capacity_implementation": _implementation(),
        "orchestration_implementation": {
            name: _file_hash(Path(__file__).with_name(name))
            for name in ("architecture_sweep.py", "cli.py")
        },
        "runtime": _runtime(device, training.dtype), "fixed_streams": _streams(reference, training),
        "diagnostics": _diagnostic_identity(plan),
        "selection": "minimum best validation NLL after all candidates finish the common final update; "
                     "exact ties use parameter count then declared trial order",
        "scope": "one condition, one seed; fresh independent candidates, no external run reuse",
        "test_policy": "test is report-only and never selects candidates or checkpoints",
        "failure_policy": "one attempt per unfinished trial per invocation; isolated child processes; "
                          "no final selection until every planned trial is complete",
    }


@contextmanager
def _exclusive_output(output: Path):
    """An OS-released advisory lock; its persistent file needs no stale deletion."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / _LOCK).open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("another architecture sweep owns this output directory") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _check_parent(output: Path, manifest: dict[str, Any], *, require: bool = False) -> None:
    _safe_tree(output)
    for ancestor in output.parents:
        if (ancestor / "milestone_completed.json").exists():
            raise ValueError("architecture output must not be inside an immutable milestone")
        if (ancestor / "manifest.json").is_file():
            saved = _read_json(ancestor / "manifest.json")
            if isinstance(saved, dict) and str(saved.get("schema", "")).startswith("phaseflow."):
                raise ValueError("architecture output must not be inside another experiment")
    path = output / "manifest.json"
    if path.exists():
        if _read_json(path) != manifest:
            raise ValueError("architecture sweep data/configuration/code/runtime changed; use a new output directory")
    elif require:
        raise FileNotFoundError("report-only requires an existing architecture manifest")
    elif output.exists() and any(item.name != _LOCK for item in output.iterdir()):
        raise FileExistsError("architecture output is not empty and has no manifest")


def _expected_child(manifest: dict[str, Any], trial: dict[str, Any]) -> dict[str, Any]:
    config = trial["configuration"]
    model, training = config["model"], config["training"]
    visualization = dict(config["visualization"])
    plots = visualization.pop("enabled")
    return {
        "schema": CAPACITY_SCHEMA, "base_model": model, "models": {str(model["num_bins"]): model},
        "base_training": training, "sweep": manifest["plan"]["child_sweep"]["sweep"],
        "visualization_enabled": plots, "plot_configuration": visualization,
        "trainer_automatic_plots": False, "diagnostics": manifest["diagnostics"],
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "data_summary": manifest["data_summary"],
        "training_code_fingerprint": manifest["training_code_fingerprint"],
        "implementation_sha256": manifest["capacity_implementation"],
        "runtime": manifest["runtime"], "fixed_streams": manifest["fixed_streams"],
    }


def _trial_row(trial: dict[str, Any]) -> dict[str, Any]:
    model, cfg = trial["configuration"]["model"], trial["configuration"]["training"]
    widths = model["hidden_features"]
    return {
        "trial_id": trial["trial_id"], "status": "pending",
        "num_coupling_layers": model["num_coupling_layers"], "hidden_features": widths,
        "hidden_depth": len(widths), "hidden_width": widths[0] if len(set(widths)) == 1 else None,
        "learning_rate": cfg["learning_rate"], "num_bins": model["num_bins"],
        "train_samples": cfg["train_samples"], "batch_size": cfg["batch_size"],
        "planned_steps": cfg["steps"], "child_directory": f"trials/{trial['trial_id']}",
        "log_path": f"logs/{trial['trial_id']}.log", "milestones": [],
    }


def _inspect_trial(reference, output: Path, manifest, trial) -> dict[str, Any]:
    row = _inspect_child(reference, output, manifest, trial)
    attempt_path = output / "attempts" / f"{trial['trial_id']}.json"
    if attempt_path.exists():
        attempts = _read_json(attempt_path)
        if not isinstance(attempts, list) or not attempts:
            raise ValueError("invalid saved architecture attempt history")
        last = attempts[-1]
        row.update(last_exit_code=last["exit_code"], last_attempt_seconds=last["elapsed_seconds"])
        if row["status"] != "complete" and last["status"] in ("failed", "interrupted"):
            row.update(status=last["status"], error=last["error"])
    return row


def _inspect_child(reference, output: Path, manifest, trial) -> dict[str, Any]:
    row = _trial_row(trial)
    child = output / row["child_directory"]
    if not child.exists():
        return row
    if not (child / "manifest.json").exists():
        if any(child.iterdir()):
            raise ValueError(f"nonempty child without an experiment manifest: {child}")
        return row
    recorded = _read_json(child / "manifest.json")
    if any(recorded.get(key) != value for key, value in _expected_child(manifest, trial).items()):
        raise ValueError(f"child experiment differs from the frozen architecture plan: {child}")
    report_path = child / "summary.json"
    report = _read_json(report_path) if report_path.exists() else {}
    if report.get("status") == "complete":
        _, verified, final = _verified_completed_capacity(reference, child)
        row.update({key: value for key, value in final.items() if key not in (
            "trial_id", "run_directory", "training_directory", "status",
        )})
        row.update(status="complete", milestones=verified["trials"])
        for key in ("run_directory", "training_directory"):
            row[key] = f"{row['child_directory']}/{final[key]}"
        row["comparison_plot"] = f"{row['run_directory']}/plots/comparison.png"
        return row
    # Inspect all sealed prefixes even when a later update or plot failed.
    for prefix in _rows(recorded):
        archive = child / prefix["run_directory"]
        if (archive / "milestone_completed.json").exists():
            prefix.update(_verify_snapshot(archive, prefix, recorded), status="complete")
            row["milestones"].append(prefix)
    row["status"] = report.get("status", "incomplete")
    if row["status"] not in ("failed", "interrupted", "paused", "incomplete", "running"):
        raise ValueError(f"unexpected incomplete child status: {row['status']}")
    if report.get("error"):
        row["error"] = report["error"]
    if row["milestones"]:
        row["last_completed_milestone"] = row["milestones"][-1]["milestone_step"]
    return row


def _inputs(output: Path, manifest: dict[str, Any], *, write: bool) -> None:
    files = [(output / "inputs/capacity_sweep.json", manifest["plan"]["child_sweep"])]
    files.extend((output / "inputs" / f"{trial['trial_id']}.json", trial["configuration"])
                 for trial in manifest["plan"]["trials"])
    for path, expected in files:
        if path.exists() and _read_json(path) != expected:
            raise ValueError(f"saved child input configuration changed: {path}")
    if write:
        for path, expected in files:
            if not path.exists():
                _atomic_json(path, expected)


def _selection(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any] | None:
    if any(row["status"] != "complete" for row in rows):
        return None
    winner = min(enumerate(rows), key=lambda pair: (
        pair[1]["validation_nll"], pair[1]["parameter_count"], pair[0],
    ))[1]
    return {
        "schema": SCHEMA, "manifest_sha256": _manifest_hash(manifest),
        "selected_trial_id": winner["trial_id"], "selected_run_directory": winner["run_directory"],
        "selected_step": winner["selected_step"], "selected_validation_nll": winner["validation_nll"],
        "num_coupling_layers": winner["num_coupling_layers"],
        "hidden_features": winner["hidden_features"], "learning_rate": winner["learning_rate"],
        "num_bins": winner["num_bins"], "train_samples": winner["train_samples"],
        "common_final_update": winner["planned_steps"], "rule": manifest["selection"],
        "test_used": False,
    }


_COLUMNS = (
    "trial_id", "status", "num_coupling_layers", "hidden_features", "hidden_depth", "hidden_width",
    "learning_rate", "num_bins", "train_samples", "batch_size", "planned_steps", "global_step",
    "selected_step", "processed_examples", "effective_passes", "validation_nll", "validation_kl",
    "validation_kl_standard_error", "latest_validation_nll", "latest_validation_kl",
    "latest_validation_kl_standard_error", "test_nll", "test_kl", "test_kl_standard_error",
    "test_relative_ess", "parameter_count", "parameter_bytes", "checkpoint_bytes", "elapsed_seconds",
    "last_attempt_seconds", "last_exit_code", "run_directory", "training_directory", "log_path", "error",
)


def _plot_summary(output: Path, report: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["trials"]
    figure, axes = plt.subplots(1, 2, figsize=(12, max(4, len(rows) * 0.45)), sharey=True)
    labels = [f"L={row['num_coupling_layers']}, H={row['hidden_features']}, "
              f"lr={row['learning_rate']:g} [{row['status']}]" for row in rows]
    for axis, metric, label in zip(axes, ("validation_kl", "latest_validation_kl"),
                                   ("Validation-selected model", "Last model at planned final step"),
                                   strict=True):
        for index, row in enumerate(rows):
            if row["status"] == "complete":
                error = row.get(f"{metric}_standard_error")
                axis.errorbar(row[metric], index, xerr=None if error is None else 1.96 * error,
                              fmt="o", color="tab:blue", capsize=3)
        axis.set_title(label)
        axis.set_xlabel("Forward KL (lower is better)")
        axis.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(range(len(rows)), labels)
    axes[0].invert_yaxis()
    figure.suptitle(f"Architecture/LR sweep: {report['status']}")
    figure.text(0.5, 0.01, "Bars: 1.96 Monte Carlo SE; not seed or selection uncertainty. "
                "Only completed plans are plotted.", ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))
    try:
        data = io.BytesIO()
        figure.savefig(data, format="png", dpi=140)
        _atomic_bytes(output / "summary.png", data.getvalue())
    finally:
        plt.close(figure)


def _write_summary(output: Path, report: dict[str, Any]) -> None:
    _atomic_json(output / "summary.json", report)
    csv_data = io.StringIO(newline="")
    writer = csv.DictWriter(csv_data, fieldnames=_COLUMNS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in report["trials"]:
        writer.writerow({**row, "hidden_features": json.dumps(row["hidden_features"])})
    _atomic_bytes(output / "summary.csv", csv_data.getvalue().encode("utf-8-sig"))
    lines = ["# Architecture and learning-rate sweep", "", f"Status: **{report['status']}**.", "",
             f"Current trial: {report.get('current_trial_id') or 'none'}.", "",
             "One condition and one seed. All candidates use the same fixed point streams and final "
             "update count. Test results never select a candidate. Milestones are correlated prefixes.", "",
             "| Trial | L | Hidden widths | LR | Status | Best validation KL | Last validation KL | "
             "Test KL | Selected step | Parameters | Comparison |", "|---|---:|---|---:|---|---:|---:|---:|---:|---:|---|"]
    for row in report["trials"]:
        def number(key):
            value = row.get(key)
            return "—" if value is None else f"{value:.7g}"
        plot = (f"[map]({row['comparison_plot']})" if row["status"] == "complete"
                and (output / row.get("comparison_plot", "missing")).is_file() else "—")
        lines.append(f"| [{row['trial_id']}]({row['log_path']}) | {row['num_coupling_layers']} | "
                     f"{row['hidden_features']} | {row['learning_rate']:g} | {row['status']} | "
                     f"{number('validation_kl')} | {number('latest_validation_kl')} | "
                     f"{number('test_kl')} | {number('selected_step')} | {number('parameter_count')} | {plot} |")
    lines += ["", "Elapsed times include checkpointing, evaluation and plotting; they are not inference "
              "benchmarks. Different shapes share seeds, not identical initial weight tensors.", ""]
    if report["selection"] is None:
        lines += ["No final selection: every planned candidate must complete first.", ""]
    else:
        lines += [f"Selected: **{report['selection']['selected_trial_id']}**, by validation NLL.", ""]
    for row in report["trials"]:
        if row.get("error"):
            lines += [f"- {row['trial_id']}: {row['error'].replace(chr(10), ' ')}"]
    _atomic_bytes(output / "summary.md", ("\n".join(lines) + "\n").encode("utf-8"))
    _plot_summary(output, report)


def _pid_running(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() != 87  # Invalid PID; access denied remains conservative.
        try:
            code = wintypes.DWORD()
            kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259
        finally:
            kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _check_active(output: Path) -> None:
    path = output / "active_child.json"
    if path.exists():
        saved = _read_json(path)
        if type(saved.get("pid")) is not int or saved["pid"] <= 0:
            raise ValueError("invalid active child record")
        if _pid_running(saved["pid"]):
            raise RuntimeError(f"recorded child PID {saved['pid']} may still be running; verify its "
                               "process identity before stopping it or restarting the sweep")


def _run_child(command: list[str], log_path: Path, active_path: Path, trial_id: str) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {}
    process = None
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        log.write(f"\n--- Attempt {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ---\n")
        log.flush()
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", env=environment,
                                       **options)
            _atomic_json(active_path, {"pid": process.pid, "trial_id": trial_id})
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return process.wait()
        except KeyboardInterrupt:
            if process is not None and process.poll() is None:
                process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            raise
        finally:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
                if process.stdout is not None:
                    process.stdout.close()
            active_path.unlink(missing_ok=True)


def _command(record: Path, output: Path, trial: dict[str, Any]) -> list[str]:
    # Do not rewrite PYTHONPATH: refuse a mismatched installed checkout instead.
    guard = ("import pathlib, runpy, sys, phaseflow\n"
             "expected = pathlib.Path(sys.argv.pop(1)).resolve()\n"
             "actual = pathlib.Path(phaseflow.__file__).resolve()\n"
             "if actual != expected:\n"
             "    raise RuntimeError(f'Child phaseflow import mismatch: {actual} != {expected}')\n"
             "runpy.run_module('phaseflow', run_name='__main__')\n")
    return [sys.executable, "-u", "-c", guard, str(Path(__file__).with_name("__init__.py").resolve()),
            "sweep-capacity-rainbow", "--record", str(record),
            "--config", str(output / "inputs" / f"{trial['trial_id']}.json"),
            "--sweep-config", str(output / "inputs/capacity_sweep.json"),
            "--output", str(output / "trials" / trial["trial_id"])]


def _record_attempt(output: Path, trial_id: str, *, code: int | None, started: float,
                    status: str, error: str | None = None) -> None:
    path = output / "attempts" / f"{trial_id}.json"
    attempts = _read_json(path) if path.exists() else []
    attempts.append({"exit_code": code, "elapsed_seconds": time.perf_counter() - started,
                     "status": status, "error": error})
    _atomic_json(path, attempts)


def run_architecture_sweep(
    record: str | Path, config_path: str | Path, output_directory: str | Path, *,
    device: str | None = None, dry_run: bool = False, report_only: bool = False,
) -> dict[str, Any]:
    """Run once through the declared candidates; rerun this same call to resume."""
    if type(dry_run) is not bool or type(report_only) is not bool or (dry_run and report_only):
        raise ValueError("dry_run and report_only must be mutually exclusive booleans")
    plan = load_plan(config_path, device)
    _safe_tree(Path(output_directory).absolute())
    record, output = Path(record).resolve(), Path(output_directory).resolve()
    if output.is_relative_to(record) or record.is_relative_to(output):
        raise ValueError("record and architecture output directory trees must be disjoint")
    with RainbowReference(record) as reference:
        manifest = _manifest(reference, plan)
        _check_parent(output, manifest, require=report_only)
        _inputs(output, manifest, write=False)
        # Verify all completed children before any aggregate or training write.
        rows = [_inspect_trial(reference, output, manifest, trial) for trial in plan["trials"]]
        if dry_run:
            return {"schema": SCHEMA, "status": "ready", "trial_count": len(rows),
                    "output_directory": str(output), "plan": plan, "trials": rows,
                    "selection": _selection(rows, manifest), "training_started": False}
        with _exclusive_output(output):
            _check_parent(output, manifest, require=report_only)
            _check_active(output)
            # Recheck after acquiring the lock in case another process just finished.
            rows = [_inspect_trial(reference, output, manifest, trial) for trial in plan["trials"]]
            report = {
                "schema": SCHEMA, "manifest_sha256": _manifest_hash(manifest), "status": "running",
                "trials": rows, "selection": None, "current_trial_id": None,
                "test_used_for_selection": False, "scope": manifest["scope"],
            }
            if not (output / "manifest.json").exists():
                _atomic_json(output / "manifest.json", manifest)
            _inputs(output, manifest, write=not report_only)
            if not report_only:
                _write_summary(output, report)
                for index, trial in enumerate(plan["trials"]):
                    if rows[index]["status"] == "complete":
                        continue
                    row = rows[index]
                    row.update(status="running")
                    row.pop("error", None)
                    report["current_trial_id"] = trial["trial_id"]
                    _write_summary(output, report)
                    print(f"[Architecture {index + 1}/{len(rows)}] {trial['trial_id']}", flush=True)
                    started = time.perf_counter()
                    try:
                        code = _run_child(_command(record, output, trial), output / row["log_path"],
                                          output / "active_child.json", trial["trial_id"])
                    except KeyboardInterrupt:
                        _record_attempt(output, trial["trial_id"], code=None, started=started,
                                        status="interrupted", error="KeyboardInterrupt")
                        row.update(status="interrupted", error="KeyboardInterrupt")
                        report.update(status="interrupted", current_trial_id=None)
                        _write_summary(output, report)
                        raise
                    except Exception as error:
                        _record_attempt(output, trial["trial_id"], code=None, started=started,
                                        status="failed", error=f"{type(error).__name__}: {error}")
                        row.update(status="failed", error=f"{type(error).__name__}: {error}")
                        report.update(status="failed", current_trial_id=None)
                        _write_summary(output, report)
                        raise
                    inspected = _inspect_trial(reference, output, manifest, trial)
                    inspected.update(last_exit_code=code, last_attempt_seconds=time.perf_counter() - started)
                    if code != 0 or inspected["status"] != "complete":
                        inspected.update(status="failed", error=inspected.get("error") or
                                         f"child exit code {code}; inspect {row['log_path']}")
                    _record_attempt(output, trial["trial_id"], code=code, started=started,
                                    status=inspected["status"], error=inspected.get("error"))
                    rows[index] = inspected
                    report["current_trial_id"] = None
                    _write_summary(output, report)
            selection = _selection(rows, manifest)
            report.update(status="complete" if selection else "incomplete", selection=selection,
                          current_trial_id=None)
            if any(row["status"] == "failed" for row in rows):
                report["status"] = "complete_with_failures"
            if selection is not None:
                path = output / "selection.json"
                if path.exists() and _read_json(path) != selection:
                    raise ValueError("saved architecture selection differs from verified candidates")
                _atomic_json(path, selection)
            elif (output / "selection.json").exists():
                raise ValueError("incomplete architecture sweep has a stale selection.json")
            _write_summary(output, report)
            return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", help="CUDA by default through the saved base config; CPU is explicit debug only")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_architecture_sweep(args.record, args.config, args.output, device=args.device,
                                        dry_run=args.dry_run, report_only=args.report_only)
    except KeyboardInterrupt:
        print("Architecture sweep interrupted; repeat the same command to resume.", file=sys.stderr)
        return 130
    if args.dry_run:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print(json.dumps({"status": report["status"], "selection": report["selection"],
                          "summary": str((args.output / "summary.json").resolve())},
                         indent=2, allow_nan=False))
    return 0 if report["status"] in ("ready", "complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
