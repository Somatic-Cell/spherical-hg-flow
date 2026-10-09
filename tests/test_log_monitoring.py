"""Composite-loss labels and replay use true quantities, not legacy NLL aliases."""

from __future__ import annotations

import json

import pytest

from phaseflow.monitoring import (
    TrainingMonitor,
    _scalar_values,
    plot_training_history,
    read_history,
)


def _history():
    validation = {
        "nll": -0.5, "forward_kl_estimate": 0.1,
        "forward_kl_standard_error": 0.01, "nll_improvement_over_hg": 0.2,
        "hg_nll": -0.3, "hg_forward_kl_estimate": 0.3,
        "log_mse": 0.36, "log_rmse": 0.6, "relative_rmse": 0.8,
    }
    return [
        {"global_step": 0, "validation": validation},
        {
            "global_step": 1, "objective": "log_density", "loss": 1.0,
            "nll": -1.0, "log_mse": 4.0, "beta": 0.5, "nll_weight": 1.0,
            "gradient_norm_before_clip": 2.0, "learning_rate": 0.001,
            "validation": validation,
            "train_monitor": {"nll": -0.8, "log_mse": 0.81, "loss": -0.395},
        },
    ]


def test_composite_total_is_never_published_as_nll():
    values = _scalar_values(_history()[1])
    assert values["loss/total"] == 1.0
    assert values["nll/train_minibatch"] == -1.0
    assert values["log_mse/train_minibatch"] == 4.0
    assert values["nll/train_fixed_subset"] == -0.8
    assert "kl/train_fixed_subset" not in values
    assert values["log_rmse/validation"] == 0.6
    assert values["objective/beta"] == 0.5
    legacy = _scalar_values({"global_step": 1, "loss": -2.0})
    assert legacy == {"nll/train_minibatch": -2.0}


def test_unrepresentable_relative_rmse_does_not_publish_a_false_finite_scalar():
    row = _history()[0]
    row["validation"]["relative_rmse"] = None
    row["validation"]["relative_rmse_status"] = "overflow_float64"
    values = _scalar_values(row)
    assert "relative_rmse/validation" not in values
    assert values["log_rmse/validation"] == 0.6


def test_composite_curves_separate_nll_objective_and_uniform_log_error(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    history = tmp_path / "history.json"
    history.write_text(json.dumps(_history()))
    figures = []
    original = Figure.savefig

    def inspect(fig, *args, **kwargs):
        figures.append([
            {"ylabel": ax.get_ylabel(), "series": {
                line.get_label(): list(line.get_ydata()) for line in ax.lines
            }}
            for ax in fig.axes
        ])
        return original(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect)
    result = plot_training_history(history, tmp_path / "curves.png", dpi=25)
    axes = figures[0]
    assert len(axes) == 6
    assert axes[0]["series"]["Training minibatch (raw)"] == [-1.0]
    assert axes[2]["series"]["Total minibatch objective"] == [1.0]
    assert axes[2]["ylabel"] == "Composite objective (not NLL)"
    assert axes[3]["series"]["Independent spherical-uniform validation"] == [0.6, 0.6]
    assert axes[3]["series"]["Fixed training subset"] == [0.9]
    assert result["objective"] == "log_density"
    assert (tmp_path / "curves.png").is_file()


def test_composite_history_replay_removes_future_jsonl_and_tensorboard_events(tmp_path):
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from torch.utils.tensorboard import SummaryWriter

    history = _history()
    with TrainingMonitor(
        tmp_path, history, tensorboard=True, start_step=0, batch_size=16, metadata={},
    ):
        pass
    future = dict(history[-1], global_step=99)
    with (tmp_path / "history.jsonl").open("a") as stream:
        stream.write(json.dumps(future) + "\n")
    with SummaryWriter(str(tmp_path / "tensorboard")) as writer:
        for name, value in _scalar_values(future).items():
            writer.add_scalar(name, value, 99)
    with TrainingMonitor(
        tmp_path, history, tensorboard=True, start_step=1, batch_size=16, metadata={},
    ) as monitor:
        monitor.publish([dict(history[-1], global_step=2)])
    assert [row["global_step"] for row in read_history(tmp_path)] == [0, 1, 2]
    events = EventAccumulator(str(tmp_path / "tensorboard"), size_guidance={"scalars": 0})
    events.Reload()
    for tag in ("loss/total", "nll/train_minibatch", "log_mse/train_minibatch"):
        assert [event.step for event in events.Scalars(tag)] == [1, 2]
    assert [event.step for event in events.Scalars("log_rmse/validation")] == [0, 1, 2]
