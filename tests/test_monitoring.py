"""Exercise live logging and crash/restart timelines against optimizer state."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from test_single_condition import assert_state_equal, small_config, small_model, smooth_teacher

from phaseflow.cli import main
from phaseflow.monitoring import read_history
from phaseflow.rainbow import RainbowReference
from phaseflow.single_condition import _points, read_single_checkpoint, train_single_condition


def test_live_monitor_resume_removes_future_steps_and_preserves_optimizer(tmp_path, capsys):
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from torch.utils.tensorboard import SummaryWriter

    cfg = small_config(
        dtype="float32", train_samples=256, validation_samples=128, test_samples=128,
        proposal_samples=128, train_monitor_samples=64, batch_size=64, steps=6,
        log_every=1, eval_every=2, checkpoint_every=2, tensorboard=True,
    )
    model = replace(small_model(), geometry_dtype="model", spline_dtype="model")
    run = tmp_path / "resumed"
    observed_steps = []

    def live(entry):
        if entry["global_step"]:
            observed_steps.append(read_history(run / "history.jsonl")[-1]["global_step"])

    with RainbowReference(smooth_teacher(tmp_path / "record")) as record:
        full = train_single_condition(record, model, cfg, tmp_path / "full", make_plots=False)
        partial = train_single_condition(
            record, model, cfg, run, max_steps_this_run=3, callback=live, make_plots=False,
        )
        assert observed_steps == [1, 2, 3]
        assert partial.metrics["test"] is None
        # A process can log later steps and crash before its next checkpoint.
        with (run / "history.jsonl").open("a") as handle:
            handle.write(json.dumps({"global_step": 99, "loss": 999}) + "\n")
        with SummaryWriter(str(run / "tensorboard")) as writer:
            writer.add_scalar("nll/train_minibatch", 999, 99)
        resumed = train_single_condition(
            record, model, cfg, run, resume=partial.checkpoint_path, make_plots=False,
        )
    a = read_single_checkpoint(full.checkpoint_path)
    b = read_single_checkpoint(resumed.checkpoint_path)
    assert_state_equal(a["model_state"], b["model_state"])
    assert_state_equal(a["optimizer_state"], b["optimizer_state"])
    assert a["history"] == b["history"] == read_history(run / "history.jsonl")
    assert full.metrics == resumed.metrics
    assert resumed.metrics["test"]["precision"]["hg_geometry_log_pdf"] == "float32"
    event = EventAccumulator(str(run / "tensorboard"), size_guidance={"scalars": 0})
    event.Reload()
    assert [e.step for e in event.Scalars("nll/train_minibatch")] == list(range(1, 7))
    assert [e.step for e in event.Scalars("nll/validation")] == [0, 2, 4, 6]
    assert [e.step for e in event.Scalars("nll/train_fixed_subset")] == [0, 2, 4, 6]
    assert (run / "learning_curves.png").is_file()
    output = run / "replotted.png"
    assert main([
        "plot-history", "--history", str(run / "history.jsonl"), "--output", str(output),
    ]) == 0
    assert json.loads(capsys.readouterr().out)["last_step"] == 6
    assert output.is_file()


def test_json_monitoring_remains_available_without_tensorboard(tmp_path):
    cfg = small_config(steps=2, log_every=1, tensorboard=False)
    with RainbowReference(smooth_teacher(tmp_path / "record")) as record:
        result = train_single_condition(
            record, small_model(), cfg, tmp_path / "run", make_plots=False,
        )
    history = read_history(result.checkpoint_path.parent / "history.jsonl")
    assert [e["global_step"] for e in history] == [0, 1, 2]
    assert history[-1]["examples_seen"] == cfg.steps * cfg.batch_size
    assert history[-1]["train_monitor"]["sample_count"] == cfg.train_samples
    assert not (result.checkpoint_path.parent / "tensorboard").exists()


def test_point_count_comparisons_share_prefix_and_seed_replicas_share_evaluation(tmp_path):
    with RainbowReference(smooth_teacher(tmp_path / "record")) as record:
        small = _points(record, 64, 2026, "train")
        large = _points(record, 256, 2026, "train")
        for a, b in zip(small, large, strict=True):
            np.testing.assert_array_equal(a, b[:64])
        splits = []
        for seed in (415, 416):
            cfg = small_config(seed=seed, data_seed=2026, steps=0)
            result = train_single_condition(
                record, small_model(), cfg, tmp_path / str(seed), make_plots=False,
            )
            splits.append(read_single_checkpoint(result.checkpoint_path)["sample_split"])
        for key in ("training_points_sha256", "validation_points_sha256", "data_seed"):
            assert splits[0][key] == splits[1][key]


def test_live_reader_ignores_only_unfinished_tail(tmp_path):
    path = tmp_path / "history.jsonl"
    complete = '{"global_step": 0, "loss": -1.5}\n'
    path.write_text(complete + '{"global_step": 1, "loss":')
    assert read_history(path) == [{"global_step": 0, "loss": -1.5}]
    path.write_text(complete + '{"global_step": 1, "loss":\n')
    with pytest.raises(json.JSONDecodeError):
        read_history(path)
