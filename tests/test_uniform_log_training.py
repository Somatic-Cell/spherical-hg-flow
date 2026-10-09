"""Uniform-query learning and workflow checks on a synthetic cell teacher.

These CPU fixtures check integration, not Rainbow optics or CUDA performance.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
from test_log_training import _assert_equal, _model, _training
from test_single_condition import smooth_teacher

from phaseflow.cli import main
from phaseflow.log_objective import LOG_UNIFORM, LogObjectiveConfig, make_training_pool
from phaseflow.rainbow import RainbowReference
from phaseflow.single_condition import FAMILY, read_single_checkpoint, train_single_condition


def _objective():
    return LogObjectiveConfig(
        schema_version=2, sampling="uniform", target_fraction=0.0,
        nll_weight=0.0, beta=1.0,
        validation_uniform_samples=2048, test_uniform_samples=2048,
    )


def test_uniform_log_training_improves_independent_shape_without_cdf_training_queries(
    tmp_path, monkeypatch,
):
    import phaseflow.single_condition as trainer

    original_points = trainer._points

    def no_cdf_training(reference, count, seed, stream):
        assert stream != "train", "uniform training must not sample the target CDF"
        return original_points(reference, count, seed, stream)

    monkeypatch.setattr(trainer, "_points", no_cdf_training)
    config = _training(
        train_samples=4096, validation_samples=2048, test_samples=2048,
        batch_size=256, steps=100, eval_every=20, checkpoint_every=20,
        eval_batch_size=512,
    )
    objective = _objective()
    with RainbowReference(smooth_teacher(tmp_path / "record")) as reference:
        result = train_single_condition(
            reference, _model(), config, tmp_path / "run",
            objective_config=objective, make_plots=False,
        )
        initial, test = result.metrics["initial_validation"], result.metrics["test"]
        assert result.complete
        assert test["log_mse"] < 0.65 * initial["log_mse"]
        assert test["base_g"] == reference.g
        checkpoint = read_single_checkpoint(result.checkpoint_path)
        split = checkpoint["sample_split"]
        assert split["training_pool"]["component_counts"] == {"cdf": 0, "uniform": 4096}
        assert split["training_pool"]["source_streams"] == {"cdf": None, "uniform": 5}
        # Historical stream registry is unchanged; actual sources are recorded
        # in the pool provenance, rather than relabelling the old CDF stream.
        assert split["streams"]["train"] == 0
        assert split["training_points_sha256"] != split["uniform_validation_points_sha256"]
        first = checkpoint["history"][0]["train_monitor"]
        assert first["sample_count"] == config.train_monitor_samples == 65
        assert first["components_measured"]

        # The initial residual is identity. Use the analytic HG density to
        # check both the ordinary uniform log MSE and p/u-weighted NLL monitor.
        pool = make_training_pool(reference, config.train_samples, config.data_seed,
                                  objective, torch.float32)
        mu = pool.directions[:65, 2].astype(np.float64)
        g = reference.g
        log_hg = (math.log1p(-g * g) + LOG_UNIFORM
                  - 1.5 * np.log(1 + g * g - 2 * g * mu))
        labels = pool.log_p[:65]
        expected_nll = np.mean(-np.exp(labels - LOG_UNIFORM) * log_hg)
        expected_log_mse = np.mean(np.square(log_hg - labels))
        assert first["nll"] == pytest.approx(expected_nll, rel=3e-6, abs=3e-6)
        assert first["log_mse"] == pytest.approx(expected_log_mse, rel=3e-6, abs=3e-6)
        assert first["loss"] == first["log_mse"]
        for event in checkpoint["history"]:
            if "loss" in event:
                assert event["components_measured"]
                assert event["sampling"] == "uniform"
                assert event["loss"] == event["log_mse"]
                assert event["nll"] != 0.0


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda:0", marks=[
        pytest.mark.cuda,
        pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
    ]),
])
def test_uniform_query_resume_is_exact_and_uses_resident_labels(tmp_path, monkeypatch, device):
    import phaseflow.single_condition as trainer

    original_terms, original_points = trainer.loss_terms, trainer._points
    calls = []
    queried = []

    def checked_terms(log_q, log_p, objective, **kwargs):
        assert log_q.device == log_p.device
        assert log_q.device.type == torch.device(device).type
        assert log_q.dtype == torch.float32
        assert kwargs["report_components"] is True
        calls.append(len(log_q))
        return original_terms(log_q, log_p, objective, **kwargs)

    def tracked_points(reference, count, seed, stream):
        assert stream != "train"
        queried.append(stream)
        return original_points(reference, count, seed, stream)

    monkeypatch.setattr(trainer, "loss_terms", checked_terms)
    monkeypatch.setattr(trainer, "_points", tracked_points)
    config = _training(device=device, train_samples=257)
    objective = _objective()
    with RainbowReference(smooth_teacher(tmp_path / "record")) as reference:
        full = train_single_condition(reference, _model(), config, tmp_path / "full",
                                      objective_config=objective, make_plots=False)
        queried.clear()
        partial = train_single_condition(
            reference, _model(), config, tmp_path / "resumed",
            objective_config=objective, make_plots=False, max_steps_this_run=3,
        )
        assert not partial.complete and partial.metrics["test"] is None
        assert "test" not in queried
        continued = train_single_condition(
            reference, _model(), config, tmp_path / "resumed",
            objective_config=objective, make_plots=False, resume=partial.checkpoint_path,
        )
        assert "test" in queried and calls
        _assert_equal(read_single_checkpoint(full.checkpoint_path),
                      read_single_checkpoint(continued.checkpoint_path))
        assert full.metrics == continued.metrics
        resumed_history = [json.loads(line) for line in
                           (tmp_path / "resumed/history.jsonl").read_text().splitlines()]
        assert resumed_history == read_single_checkpoint(continued.checkpoint_path)["history"]


def test_uniform_cli_generates_all_three_maps_and_every_actual_training_point(tmp_path, capsys):
    record = smooth_teacher(tmp_path / "record")
    config = _training(steps=2, train_samples=129)
    objective = _objective()
    path, run = tmp_path / "config.json", tmp_path / "run"
    path.write_text(json.dumps({
        "schema_version": 3, "family": FAMILY, "model": _model().to_dict(),
        "training": config.to_dict(), "objective": objective.to_dict(),
        "visualization": {"enabled": True, "dpi": 40, "eval_batch_size": 64},
    }), encoding="utf-8")
    assert main(["train-rainbow", "--record", str(record), "--config", str(path),
                 "--output", str(run), "--quiet"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["complete"]
    assert report["plots"] == str((run / "plots/plots.json").resolve())
    for name in ("reference_pdf.png", "training_samples.png", "nf_pdf.png", "comparison.png"):
        assert (run / "plots" / name).is_file()
    assert (run / "learning_curves.png").is_file()
    checkpoint = read_single_checkpoint(run / "checkpoint.pt")
    with np.load(run / "plots/training_scatter.npz", allow_pickle=False) as scatter:
        assert scatter["directions_nf"].shape == (config.train_samples, 3)
        assert np.all(scatter["components"] == 1)
        with RainbowReference(record) as reference:
            pool = make_training_pool(reference, config.train_samples, config.data_seed,
                                      objective, torch.float32)
            np.testing.assert_array_equal(scatter["directions_nf"], pool.directions)
            np.testing.assert_array_equal(scatter["teacher_log_pdf"], pool.log_p)
    assert checkpoint["sample_split"]["training_points_sha256"] == (
        pool.provenance["runtime_points_sha256"]
    )
