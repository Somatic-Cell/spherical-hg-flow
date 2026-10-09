"""Version-5 training contracts on synthetic teachers; no optical or speed claim."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_rainbow import write_rainbow_fixture
from test_single_condition import smooth_teacher

from phaseflow.cli import main
from phaseflow.log_objective import LogObjectiveConfig
from phaseflow.rainbow import RainbowReference
from phaseflow.single_condition import (
    FAMILY,
    SingleTrainingConfig,
    load_single_checkpoint,
    read_single_checkpoint,
    train_single_condition,
)
from phaseflow.sphere_model import SphereFlowConfig


def _training(**changes):
    return replace(SingleTrainingConfig(
        device="cpu", dtype="float32", seed=41, data_seed=713,
        train_samples=256, validation_samples=128, test_samples=128,
        proposal_samples=0, batch_size=64, steps=6, learning_rate=0.003,
        eval_every=2, checkpoint_every=2, log_every=2,
        eval_batch_size=64, train_monitor_samples=65,
    ), **changes)


def _objective(**changes):
    return replace(LogObjectiveConfig(
        beta=0.3, validation_uniform_samples=128, test_uniform_samples=128,
    ), **changes)


def _model(**changes):
    return replace(SphereFlowConfig(
        num_coupling_layers=2, num_bins=8, hidden_features=(8,),
        geometry_dtype="model", spline_dtype="model",
    ), **changes)


def _train(reference, config, objective, directory, **kwargs):
    return train_single_condition(
        reference, _model(), config, directory, objective_config=objective,
        make_plots=False, **kwargs,
    )


def _assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype
        assert torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_equal(a, b)
    else:
        assert left == right


def _file_hashes(directory):
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*") if path.is_file()
    }


def test_positive_beta_learns_log_shape_on_independent_smooth_teacher_points(tmp_path):
    with RainbowReference(smooth_teacher(tmp_path / "record")) as reference:
        config = _training(
            train_samples=4096, validation_samples=2048, test_samples=2048,
            batch_size=256, steps=100, eval_every=20, checkpoint_every=20,
            eval_batch_size=512,
        )
        objective = _objective(beta=1.0, validation_uniform_samples=2048, test_uniform_samples=2048)
        result = _train(reference, config, objective, tmp_path / "run")
        initial, test = result.metrics["initial_validation"], result.metrics["test"]
        assert result.complete
        assert test["log_mse"] < initial["log_mse"] * 0.65
        assert test["forward_kl_estimate"] < initial["forward_kl_estimate"] - 0.02
        assert test["base_g"] == reference.g
        assert test["scope"] == "final_independent_test_points_same_condition"
        checkpoint = read_single_checkpoint(result.checkpoint_path)
        assert checkpoint["checkpoint_version"] == 5
        assert checkpoint["objective_config"] == objective.to_dict()
        assert checkpoint["selection_metric"] == "log_rmse"
        assert set(checkpoint["selections"]) == {"nll", "log_rmse"}
        assert checkpoint["sample_split"]["training_pool"]["component_counts"] == {
            "cdf": 2048, "uniform": 2048,
        }
        assert checkpoint["history"][0]["train_monitor"]["sample_count"] == 64
        assert len(test["uniform_test_points_sha256"]) == 64


def _resume_check(tmp_path, monkeypatch, device):
    import phaseflow.single_condition as trainer

    real_points, real_uniform = trainer._points, trainer.make_uniform_points
    queried = []

    def tracked_points(reference, count, seed, stream):
        queried.append(("cdf", stream))
        return real_points(reference, count, seed, stream)

    def tracked_uniform(reference, count, seed, stream, geometry_dtype):
        queried.append(("uniform", stream))
        return real_uniform(reference, count, seed, stream, geometry_dtype)

    monkeypatch.setattr(trainer, "_points", tracked_points)
    monkeypatch.setattr(trainer, "make_uniform_points", tracked_uniform)
    with RainbowReference(smooth_teacher(tmp_path / "record")) as reference:
        config, objective = _training(device=device), _objective()
        full = _train(reference, config, objective, tmp_path / "full")
        queried.clear()
        partial = _train(reference, config, objective, tmp_path / "resumed", max_steps_this_run=3)
        assert not partial.complete and partial.metrics["test"] is None
        assert not any(stream == "test" for _, stream in queried)
        checkpoint = read_single_checkpoint(partial.checkpoint_path)
        assert [event["global_step"] for event in checkpoint["history"] if "validation" in event] == [0, 2]
        queried.clear()
        continued = _train(reference, config, objective, tmp_path / "resumed",
                           resume=partial.checkpoint_path)
        assert ("cdf", "test") in queried and ("uniform", "test") in queried
        a = read_single_checkpoint(full.checkpoint_path)
        b = read_single_checkpoint(continued.checkpoint_path)
        # Includes model, both selections, optimizer, RNG, point hashes and the
        # complete numerical history. Wall-clock telemetry is intentionally separate.
        _assert_equal(a, b)
        assert full.metrics == continued.metrics
        for name in ("best.pt", "best_by_nll.pt", "best_by_log.pt"):
            _assert_equal(read_single_checkpoint(tmp_path / "full" / name),
                          read_single_checkpoint(tmp_path / "resumed" / name))
        restored, payload = load_single_checkpoint(continued.best_path, device=device)
        assert restored.device.type == torch.device(device).type
        _assert_equal(restored.state_dict(), payload["model_state"])
        jsonl = [json.loads(line) for line in (tmp_path / "resumed/history.jsonl").read_text().splitlines()]
        assert jsonl == b["history"]


def test_version5_resume_is_exact_and_never_queries_test_before_completion(tmp_path, monkeypatch):
    _resume_check(tmp_path, monkeypatch, "cpu")


@pytest.mark.parametrize("change", [
    {"beta": 0.7}, {"selection_metric": "nll"}, {"sampling": "target", "beta": 0},
])
def test_resume_rejects_changed_objective_before_writing(tmp_path, change):
    with RainbowReference(smooth_teacher(tmp_path / "record")) as reference:
        config, objective = _training(), _objective()
        result = _train(reference, config, objective, tmp_path / "run", max_steps_this_run=3)
        before = _file_hashes(tmp_path / "run")
        with pytest.raises(ValueError, match="original objective"):
            _train(reference, config, replace(objective, **change), tmp_path / "run",
                   resume=result.checkpoint_path)
        assert _file_hashes(tmp_path / "run") == before


@pytest.mark.parametrize("tamper", ["pool_hash", "primary_selection"])
def test_resume_rejects_corrupted_pool_or_selection_without_further_writes(tmp_path, tamper):
    with RainbowReference(smooth_teacher(tmp_path / "record")) as reference:
        config, objective = _training(), _objective()
        result = _train(reference, config, objective, tmp_path / "run", max_steps_this_run=3)
        payload = read_single_checkpoint(result.checkpoint_path)
        if tamper == "pool_hash":
            payload["sample_split"]["training_pool"]["runtime_points_sha256"] = "0" * 64
            expected = "regenerated training/validation"
        else:
            payload["selections"]["log_rmse"]["step"] = config.steps
            expected = "selection"
        torch.save(payload, result.checkpoint_path)
        before = _file_hashes(tmp_path / "run")
        with pytest.raises(ValueError, match=expected):
            _train(reference, config, objective, tmp_path / "run", resume=result.checkpoint_path)
        assert _file_hashes(tmp_path / "run") == before


@pytest.mark.parametrize("primary,expected_step", [("log_rmse", 4), ("nll", 2)])
def test_scheduled_dual_minima_select_corresponding_weights_and_keep_first_ties(
    tmp_path, monkeypatch, primary, expected_step
):
    import phaseflow.single_condition as trainer

    actual_likelihood, actual_shape = trainer._likelihood_metrics, trainer.evaluate_log_shape
    nll_values = iter((3.0, 1.0, 2.0, 1.0))  # step 2 and 6 tie; first is 2.
    shape_values = iter((3.0, 2.0, 1.0, 1.0))  # step 4 and 6 tie; first is 4.
    visited_states = []

    def scheduled_likelihood(model, reference, points, batch_size):
        result = actual_likelihood(model, reference, points, batch_size)
        result["nll"] = next(nll_values)
        return result

    def scheduled_shape(model, reference, points, batch_size):
        result = actual_shape(model, reference, points, batch_size)
        value = next(shape_values)
        result.update(log_rmse=value, log_mse=value**2)
        visited_states.append(copy.deepcopy(model.state_dict()))
        return result

    monkeypatch.setattr(trainer, "_likelihood_metrics", scheduled_likelihood)
    monkeypatch.setattr(trainer, "evaluate_log_shape", scheduled_shape)
    # This test injects only the validation ordering. A separate integration test
    # checks real independent test evaluation and learned accuracy.
    monkeypatch.setattr(trainer, "evaluate_single_condition", lambda *args, **kwargs: {})
    with RainbowReference(smooth_teacher(tmp_path / "record")) as reference:
        objective = _objective(selection_metric=primary)
        result = _train(reference, _training(), objective, tmp_path / "run")
        assert result.metrics["selected_step"] == expected_step
        checkpoint = read_single_checkpoint(result.checkpoint_path)
        assert checkpoint["selections"]["nll"]["step"] == 2
        assert checkpoint["selections"]["log_rmse"]["step"] == 4
        _assert_equal(checkpoint["selections"]["nll"]["model_state"], visited_states[1])
        _assert_equal(checkpoint["selections"]["log_rmse"]["model_state"], visited_states[2])
        _assert_equal(checkpoint["best_state"], visited_states[expected_step // 2])
        for metric, filename, event_index in (("nll", "best_by_nll.pt", 1),
                                            ("log_rmse", "best_by_log.pt", 2)):
            saved = read_single_checkpoint(tmp_path / "run" / filename)
            assert saved["selection_metric"] == metric
            _assert_equal(saved["model_state"], visited_states[event_index])
        _assert_equal(read_single_checkpoint(result.best_path)["model_state"],
                      checkpoint["best_state"])


def test_schema3_cli_train_and_uniform_evaluation_override(tmp_path, capsys):
    record = smooth_teacher(tmp_path / "record")
    config, objective = _training(steps=2), _objective(test_uniform_samples=73)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "schema_version": 3, "family": FAMILY, "model": _model().to_dict(),
        "training": config.to_dict(), "objective": objective.to_dict(),
        "visualization": {"enabled": False},
    }))
    run = tmp_path / "run"
    assert main(["train-rainbow", "--record", str(record), "--config", str(path),
                 "--output", str(run), "--quiet"]) == 0
    trained = json.loads(capsys.readouterr().out)
    assert trained["metrics"]["objective"] == objective.to_dict()
    assert trained["metrics"]["test"]["uniform_sample_count"] == 73
    saved = json.loads((run / "config.json").read_text())
    assert saved["schema_version"] == 3 and saved["objective"] == objective.to_dict()
    for args, count in (([], 73), (["--uniform-samples", "41"], 41),
                        (["--uniform-samples", "0"], 0)):
        output = tmp_path / f"evaluation_{count}.json"
        assert main(["evaluate-rainbow", "--record", str(record), "--checkpoint", str(run / "best.pt"),
                     "--device", "cpu", "--samples", "67", "--proposal-samples", "0",
                     "--batch-size", "32", "--output", str(output), *args]) == 0
        result = json.loads(capsys.readouterr().out)
        assert json.loads(output.read_text()) == result
        assert result["sample_count"] == 67
        if count:
            assert result["uniform_sample_count"] == count
            assert math.isfinite(result["log_rmse"])
            assert result["log_shape_measure"] == "uniform_solid_angle"
        else:
            assert "uniform_sample_count" not in result and "log_rmse" not in result


def test_true_zero_preflight_writes_nothing_and_legacy_nll_still_trains(tmp_path):
    write_rainbow_fixture(tmp_path / "record")
    with RainbowReference(tmp_path / "record") as reference:
        output = tmp_path / "log_run"
        with pytest.raises(ValueError, match="positive teacher"):
            _train(reference, _training(steps=1), _objective(), output)
        assert not output.exists()
        legacy = _train(reference, _training(steps=1), None, tmp_path / "nll_run")
        assert legacy.complete
        checkpoint = read_single_checkpoint(legacy.checkpoint_path)
        assert checkpoint["checkpoint_version"] == 4
        assert "objective_config" not in checkpoint
        assert "log_rmse" not in legacy.metrics["best_validation"]


@pytest.mark.parametrize("objective", [
    _objective(sampling="target", beta=0),
    _objective(beta=0),
    _objective(nll_weight=0, beta=1),
])
def test_reported_components_are_actual_and_target_control_omits_unmeasured_log_error(
    tmp_path, objective
):
    with RainbowReference(smooth_teacher(tmp_path / "record")) as reference:
        result = _train(reference, _training(steps=2), objective, tmp_path / "run")
        history = read_single_checkpoint(result.checkpoint_path)["history"]
        events = [entry for entry in history if "loss" in entry]
        assert events
        for event in events:
            assert event["nll"] != 0
            if objective.sampling == "target":
                assert "log_mse" not in event
                assert "log_mse" not in event["train_monitor"] if "train_monitor" in event else True
            else:
                assert event["log_mse"] > 0
                expected = objective.nll_weight * event["nll"] + objective.beta * event["log_mse"]
                assert event["loss"] == pytest.approx(expected, rel=2e-6, abs=2e-6)
                if "train_monitor" in event:
                    assert event["train_monitor"]["log_mse"] > 0
                    assert event["train_monitor"]["nll"] != 0
        assert history[0]["validation"]["log_mse"] > 0
        assert result.metrics["test"]["log_mse"] > 0


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable; CPU is not GPU validation")
def test_cuda_version5_small_training_and_exact_resume(tmp_path, monkeypatch):
    _resume_check(tmp_path, monkeypatch, "cuda:0")
