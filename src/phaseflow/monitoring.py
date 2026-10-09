"""Local scalar monitoring and reproducible learning curves.

Numerical history is checkpoint state. Wall-clock telemetry is deliberately
separate: interrupted and uninterrupted runs can have identical numerical
histories without pretending that their timings are identical.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def check_tensorboard() -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "TensorBoard monitoring was requested but is unavailable in this Python. "
            'Run execute.bat setup, or this interpreter with -m pip install -e ".[monitor]". '
            "Set training.tensorboard=false only if you want JSONL monitoring without TensorBoard."
        ) from error


def _scalar_values(entry: dict[str, Any]) -> dict[str, float]:
    values = {}
    log_objective = entry.get("objective") == "log_density" or "log_mse" in entry
    if log_objective:
        for key, tag in (
            ("loss", "loss/total"),
            ("nll", "nll/train_minibatch"),
            ("log_mse", "log_mse/train_minibatch"),
            ("beta", "objective/beta"),
            ("nll_weight", "objective/nll_weight"),
        ):
            if key in entry:
                values[tag] = entry[key]
    elif "loss" in entry:
        values["nll/train_minibatch"] = entry["loss"]
    for key, tag in (
        ("gradient_norm_before_clip", "optimizer/gradient_norm_before_clip"),
        ("learning_rate", "optimizer/learning_rate"),
        ("examples_seen", "budget/examples_seen"),
        ("effective_passes", "budget/effective_passes"),
    ):
        if key in entry:
            values[tag] = entry[key]
    for key, prefix in (("validation", "validation"), ("train_monitor", "train_fixed_subset")):
        if key not in entry:
            continue
        metric = entry[key]
        for name, tag in (
            ("nll", "nll"),
            ("forward_kl_estimate", "kl"),
            ("forward_kl_standard_error", "kl_standard_error"),
            ("nll_improvement_over_hg", "nll_improvement_over_hg"),
            ("log_mse", "log_mse"),
            ("log_rmse", "log_rmse"),
            ("relative_rmse", "relative_rmse"),
            ("loss", "loss"),
        ):
            if name in metric and metric[name] is not None:
                values[f"{tag}/{prefix}"] = metric[name]
        if key == "validation":
            for name, tag in (
                ("hg_nll", "nll"), ("hg_forward_kl_estimate", "kl"),
                ("hg_log_mse", "log_mse"), ("hg_log_rmse", "log_rmse"),
            ):
                if name in metric:
                    values[f"{tag}/hg_validation"] = metric[name]
    return values


class TrainingMonitor:
    """Publish checked blocks, never synchronizing one GPU scalar per update.

    On resume the JSONL and TensorBoard scalar timeline are reconstructed from
    checkpoint history. A crash can leave logged steps beyond the latest valid
    optimizer checkpoint; those steps must not survive as resumed observations.
    """

    def __init__(
        self, output: Path, history: list[dict[str, Any]], *, tensorboard: bool,
        start_step: int, batch_size: int, metadata: dict[str, Any],
    ) -> None:
        self.writer = None
        self.output = output
        self.start_step = start_step
        self.batch_size = batch_size
        self.started = time.perf_counter()
        self.history_file = (output / "history.jsonl").open("w", encoding="utf-8")
        self.telemetry_file = (output / "monitoring.jsonl").open("a", encoding="utf-8")
        try:
            if tensorboard:
                check_tensorboard()
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(
                    log_dir=str(output / "tensorboard"), purge_step=0, flush_secs=10,
                )
                self.writer.add_text("run/configuration", json.dumps(metadata, indent=2), 0)
            self.publish(history)
        except BaseException:
            self.close()
            raise

    def publish(self, entries: list[dict[str, Any]]) -> None:
        for entry in entries:
            self.history_file.write(json.dumps(entry, allow_nan=False, sort_keys=True) + "\n")
            if self.writer is not None:
                for tag, value in _scalar_values(entry).items():
                    self.writer.add_scalar(tag, value, entry["global_step"])
        self.history_file.flush()
        if self.writer is not None:
            self.writer.flush()

    def telemetry(self, step: int) -> None:
        elapsed = time.perf_counter() - self.started
        examples = (step - self.start_step) * self.batch_size
        event = {
            "global_step": step,
            "session_start_step": self.start_step,
            "session_elapsed_seconds": elapsed,
            "session_examples_seen": examples,
            "session_examples_per_second": examples / elapsed if elapsed > 0 else None,
            "scope": "current invocation; includes logging, validation and checkpoint work",
        }
        self.telemetry_file.write(json.dumps(event, allow_nan=False) + "\n")
        self.telemetry_file.flush()
        if self.writer is not None:
            self.writer.add_scalar("performance/session_elapsed_seconds", elapsed, step)
            if event["session_examples_per_second"] is not None:
                self.writer.add_scalar(
                    "performance/session_examples_per_second",
                    event["session_examples_per_second"], step,
                )
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        self.history_file.close()
        self.telemetry_file.close()

    def __enter__(self) -> TrainingMonitor:
        return self

    def __exit__(self, *args) -> None:
        self.close()


def read_history(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.is_dir():
        path = path / "history.jsonl"
    with path.open(encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            history = []
            lines = handle.readlines()
            for index, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    # A live reader may see the writer's unfinished final line.
                    # Completed malformed lines still fail; data is never repaired.
                    if index == len(lines) - 1 and not line.endswith("\n"):
                        break
                    raise
        else:
            history = json.load(handle)
    if not isinstance(history, list) or not history:
        raise ValueError("history must be a nonempty list of training events")
    steps = [entry.get("global_step") for entry in history]
    if any(type(step) is not int or step < 0 for step in steps):
        raise ValueError("history has an invalid global step")
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("history steps must be strictly increasing")
    return history


def plot_training_history(
    history_path: str | Path, output: str | Path, *, dpi: int = 160,
) -> dict[str, Any]:
    """Use linear NLL/KL axes, retain negative estimates, show MC error bars."""
    if type(dpi) is not int or dpi < 1:
        raise ValueError("dpi must be positive")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    history = read_history(history_path)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    log_objective = any(
        entry.get("objective") == "log_density" or "log_mse" in entry
        or "log_mse" in entry.get("validation", {})
        for entry in history
    )
    fig = Figure(figsize=(11, 12 if log_objective else 8), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(3 if log_objective else 2, 2)
    updates = [entry for entry in history if "loss" in entry]
    if updates:
        axes[0, 0].plot(
            [e["global_step"] for e in updates],
            [e["nll"] if log_objective else e["loss"] for e in updates],
            color="0.65", alpha=0.55, linewidth=0.6, label="Training minibatch (raw)",
        )
    for key, label, color in (
        ("train_monitor", "Fixed training subset", "#1f77b4"),
        ("validation", "Validation", "#d95f02"),
    ):
        entries = [entry for entry in history if key in entry]
        if not entries:
            continue
        steps = [e["global_step"] for e in entries]
        axes[0, 0].plot(steps, [e[key]["nll"] for e in entries], color=color, label=label)
        kl_entries = [
            e for e in entries
            if {"forward_kl_estimate", "forward_kl_standard_error"} <= e[key].keys()
        ]
        if kl_entries:
            axes[0, 1].errorbar(
                [e["global_step"] for e in kl_entries],
                [e[key]["forward_kl_estimate"] for e in kl_entries],
                yerr=[1.96 * e[key]["forward_kl_standard_error"] for e in kl_entries],
                color=color, marker=".", capsize=2, label=label,
            )
    validation = [entry for entry in history if "validation" in entry]
    if validation:
        first = validation[0]["validation"]
        if "hg_nll" in first:
            axes[0, 0].axhline(first["hg_nll"], color="#7570b3", ls="--", label="HG validation")
        if "hg_forward_kl_estimate" in first:
            axes[0, 1].axhline(
                first["hg_forward_kl_estimate"], color="#7570b3", ls="--", label="HG validation",
            )
    axes[0, 0].set_ylabel("NLL (nats / sample, density per sr)")
    axes[0, 1].set_ylabel("Forward KL estimate (nats)")
    axes[0, 1].set_title("Error bars: +/- 1.96 Monte Carlo SE")
    axes[0, 1].axhline(0, color="0.5", linewidth=0.5)
    if log_objective:
        if updates:
            steps = [e["global_step"] for e in updates]
            axes[1, 0].plot(
                steps, [e["loss"] for e in updates], color="0.3", alpha=0.65,
                linewidth=0.6, label="Total minibatch objective",
            )
            if all({"beta", "nll_weight", "nll", "log_mse"} <= e.keys() for e in updates):
                axes[1, 0].plot(
                    steps, [e["nll_weight"] * e["nll"] for e in updates],
                    alpha=0.6, linewidth=0.6, label="Weighted NLL term",
                )
                axes[1, 0].plot(
                    steps, [e["beta"] * e["log_mse"] for e in updates],
                    alpha=0.6, linewidth=0.6, label="Weighted log MSE term",
                )
        axes[1, 0].set_ylabel("Composite objective (not NLL)")
        for key, label, color in (
            ("train_monitor", "Fixed training subset", "#1f77b4"),
            ("validation", "Independent spherical-uniform validation", "#d95f02"),
        ):
            entries = [e for e in history if "log_mse" in e.get(key, {})]
            if entries:
                axes[1, 1].plot(
                    [e["global_step"] for e in entries],
                    [e[key].get("log_rmse", e[key]["log_mse"] ** 0.5) for e in entries],
                    color=color, label=label,
                )
        axes[1, 1].set_ylabel("Solid-angle log PDF RMSE (nats)")
        axes[1, 1].set_title("Uniform-sphere measure; no density floor")
    for key, ax, label in (
        ("gradient_norm_before_clip", axes[-1, 0], "Gradient norm before clipping"),
        ("learning_rate", axes[-1, 1], "Learning rate"),
    ):
        entries = [entry for entry in history if key in entry]
        if entries:
            ax.plot([e["global_step"] for e in entries], [e[key] for e in entries])
        ax.set_ylabel(label)
    for ax in axes.flat:
        ax.set_xlabel("Optimizer updates")
        ax.grid(alpha=0.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.suptitle("Single CDF log-density training history" if log_objective else (
        "Single CDF training history"
    ))
    fig.savefig(output, dpi=dpi)
    fig.clear()
    result = {
        "history": str(Path(history_path).resolve()), "output": str(output.resolve()),
        "last_step": history[-1]["global_step"], "events": len(history),
        "loss_axis": "linear; no clipping or smoothing",
        "error_bars": "1.96 Monte Carlo SE, not variability across training seeds",
    }
    if log_objective:
        result["objective"] = "log_density"
        result["loss_components"] = ["nll", "log_mse", "total_composite_objective"]
        result["log_error_measure"] = "uniform_solid_angle; natural logarithms"
    return result
