"""Synthetic CPU experiment orchestration; these are not Rainbow optics results."""

from __future__ import annotations

import csv
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_single_condition import assert_state_equal, smooth_teacher

import phaseflow.single_condition as single
import phaseflow.sweep as sweep_module
from phaseflow.plotting import RainbowPlotConfig
from phaseflow.rainbow import RainbowReference
from phaseflow.single_condition import SingleTrainingConfig, _array_hash, _points
from phaseflow.sphere_model import SphereFlowConfig
from phaseflow.sweep import SweepConfig, run_single_condition_sweep


def configuration(**changes):
    return replace(SingleTrainingConfig(
        seed=415, data_seed=2026, device="cpu", dtype="float32",
        train_samples=64, validation_samples=128, test_samples=128,
        proposal_samples=16, batch_size=32, steps=4,
        eval_every=2, checkpoint_every=2, eval_batch_size=128,
        tensorboard=False, log_every=1, train_monitor_samples=64,
    ), **changes)


def model_configuration():
    return SphereFlowConfig(
        num_coupling_layers=2, num_bins=8, hidden_features=(8, 8),
        spline_dtype="model", geometry_dtype="model",
    )


def experiment():
    return SweepConfig(learning_rates=(3e-4, 3e-3), train_samples=(32, 64, 128),
                       baseline_train_samples=64)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_sequential_sweep_validation_selection_common_streams_and_real_plots(tmp_path, monkeypatch):
    # Make the reporting-only test NLL prefer the opposite learning rate. Actual
    # training and validation calculations stay untouched. This sentinel guards
    # specifically against accidentally selecting the rate from final test data.
    original_train = sweep_module.train_single_condition
    original_evaluate = single.evaluate_single_condition
    active = {}

    def reporting_sentinel(*args, **kwargs):
        report = original_evaluate(*args, **kwargs)
        report["nll"] = active["learning_rate"] * 1e6
        return report

    def tracked_train(reference, model, cfg, output, **kwargs):
        active["learning_rate"] = cfg.learning_rate
        return original_train(reference, model, cfg, output, **kwargs)

    monkeypatch.setattr(single, "evaluate_single_condition", reporting_sentinel)
    monkeypatch.setattr(sweep_module, "train_single_condition", tracked_train)
    events = []
    cfg, plan, output = configuration(), experiment(), tmp_path / "sweep"
    with RainbowReference(smooth_teacher(tmp_path / "record")) as reference:
        result = run_single_condition_sweep(
            reference, model_configuration(), cfg, output, sweep_config=plan,
            plot_config=RainbowPlotConfig(cdf_samples=32, dpi=50), callback=events.append,
        )
        manifest = read_json(output / "manifest.json")
        largest = _points(reference, 128, cfg.data_seed, "train")
        for n in plan.train_samples:
            independently_sampled = _points(reference, n, cfg.data_seed, "train")
            for array, prefix in zip(independently_sampled, largest, strict=True):
                np.testing.assert_array_equal(array, prefix[:n])
            assert manifest["fixed_streams"]["training_prefix_sha256"][str(n)] == (
                _array_hash(*independently_sampled)
            )
    assert result["status"] == "complete"
    stage_a = [r for r in result["trials"] if r["stage"] == "A"]
    stage_b = [r for r in result["trials"] if r["stage"] == "B"]
    validation_winner = min(stage_a, key=lambda row: row["validation_nll"])
    assert validation_winner["learning_rate"] == 3e-3
    assert min(stage_a, key=lambda row: row["test_nll"])["learning_rate"] == 3e-4
    assert result["selection"]["selected_learning_rate"] == validation_winner["learning_rate"]
    assert result["selection"]["test_used"] is False
    assert all(r["learning_rate"] == validation_winner["learning_rate"] for r in stage_b)
    reused = next(r for r in stage_b if r["train_samples"] == 64)
    assert reused["reuse_of"] == validation_winner["trial_id"]
    assert reused["run_directory"] == validation_winner["run_directory"]
    assert len({r["run_directory"] for r in result["trials"]}) == 4
    split_hashes = {r["sample_split"]["validation_points_sha256"] for r in result["trials"]}
    assert split_hashes == {manifest["fixed_streams"]["validation_points_sha256"]}
    for row in result["trials"]:
        training = row["training_config"]
        assert training["seed"] == cfg.seed and training["data_seed"] == cfg.data_seed
        assert training["steps"] == cfg.steps and training["batch_size"] == cfg.batch_size
        assert row["processed_examples"] == cfg.steps * cfg.batch_size
        assert (output / row["run_directory"] / "plots/comparison.png").is_file()
        assert (output / row["run_directory"] / "sweep_completed.json").is_file()
    for name in ("manifest.json", "selection.json", "summary.json", "summary.csv", "summary.md"):
        assert (output / name).is_file()
    assert (output / "summary.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert sum(e.get("event") == "trial_started" for e in events) == 4
    assert sum(e.get("event") == "stage_b_reuses_a" for e in events) == 1


def test_completed_trials_reuse_and_changed_artifacts_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep_module, "_summary_plot", lambda *args: None)
    cfg, output = configuration(steps=2), tmp_path / "sweep"
    with RainbowReference(smooth_teacher(tmp_path / "record")) as reference:
        first = run_single_condition_sweep(
            reference, model_configuration(), cfg, output,
            sweep_config=experiment(), make_plots=False,
        )
        paths = [output / r["run_directory"] / "checkpoint.pt" for r in first["trials"]]
        original = {path: path.read_bytes() for path in paths}

        def forbidden_training(*args, **kwargs):
            raise AssertionError("completed trials must not train or rewrite checkpoints")

        monkeypatch.setattr(sweep_module, "train_single_condition", forbidden_training)
        again = run_single_condition_sweep(
            reference, model_configuration(), cfg, output,
            sweep_config=experiment(), make_plots=False,
        )
        assert first == again
        assert all(path.read_bytes() == contents for path, contents in original.items())
        damaged = output / first["trials"][0]["run_directory"] / "metrics.json"
        damaged.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="file changed or missing"):
            run_single_condition_sweep(
                reference, model_configuration(), cfg, output,
                sweep_config=experiment(), make_plots=False,
            )
        assert damaged.read_text() == "{}"
        assert all(path.read_bytes() == contents for path, contents in original.items())


def test_interrupted_trial_resumes_exactly_and_keeps_partial_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep_module, "_summary_plot", lambda *args: None)
    original_train = sweep_module.train_single_condition
    interrupted = False

    def interrupt_once(*args, **kwargs):
        nonlocal interrupted
        callback = kwargs["callback"]

        def progress(entry):
            nonlocal interrupted
            callback(entry)
            if not interrupted and entry["global_step"] == 3:
                interrupted = True
                raise KeyboardInterrupt("synthetic controlled interruption")

        kwargs["callback"] = progress
        return original_train(*args, **kwargs)

    monkeypatch.setattr(sweep_module, "train_single_condition", interrupt_once)
    cfg, plan = configuration(steps=6), experiment()
    output = tmp_path / "interrupted"
    with RainbowReference(smooth_teacher(tmp_path / "record")) as reference:
        with pytest.raises(KeyboardInterrupt):
            run_single_condition_sweep(
                reference, model_configuration(), cfg, output,
                sweep_config=plan, make_plots=False,
            )
        summary = read_json(output / "summary.json")
        assert summary["status"] == "interrupted"
        checkpoint_path = output / summary["trials"][0]["run_directory"] / "checkpoint.pt"
        assert single.read_single_checkpoint(checkpoint_path)["global_step"] == 2
        resumed = run_single_condition_sweep(
            reference, model_configuration(), cfg, output,
            sweep_config=plan, make_plots=False,
        )
        uninterrupted = run_single_condition_sweep(
            reference, model_configuration(), cfg, tmp_path / "full",
            sweep_config=plan, make_plots=False,
        )
    assert resumed["status"] == "complete"
    assert resumed["selection"] == uninterrupted["selection"]
    for a, b in zip(resumed["trials"], uninterrupted["trials"], strict=True):
        left = single.read_single_checkpoint(output / a["run_directory"] / "checkpoint.pt")
        right = single.read_single_checkpoint(tmp_path / "full" / b["run_directory"] / "checkpoint.pt")
        assert_state_equal(left["model_state"], right["model_state"])
        assert_state_equal(left["optimizer_state"], right["optimizer_state"])
        assert left["history"] == right["history"]
        assert torch.equal(left["minibatch_rng_state"], right["minibatch_rng_state"])


def test_changed_config_data_or_runtime_refused_before_output_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep_module, "_summary_plot", lambda *args: None)
    cfg, output = configuration(), tmp_path / "sweep"
    with RainbowReference(smooth_teacher(tmp_path / "record")) as reference:
        paused = run_single_condition_sweep(
            reference, model_configuration(), cfg, output, sweep_config=experiment(),
            make_plots=False, max_trials_this_run=1,
        )
        assert paused["status"] == "paused" and paused["selection"] is None
        saved = {name: (output / name).read_bytes()
                 for name in ("manifest.json", "summary.json")}
        for changed in (replace(cfg, steps=8), replace(cfg, seed=416), replace(cfg, data_seed=2027)):
            with pytest.raises(ValueError, match="changed; use a new output"):
                run_single_condition_sweep(
                    reference, model_configuration(), changed, output,
                    sweep_config=experiment(), make_plots=False,
                )
        actual_runtime = sweep_module._runtime
        monkeypatch.setattr(sweep_module, "_runtime", lambda *args: {
            **actual_runtime(*args), "torch_version": "synthetic_changed_runtime",
        })
        with pytest.raises(ValueError, match="changed; use a new output"):
            run_single_condition_sweep(
                reference, model_configuration(), cfg, output,
                sweep_config=experiment(), make_plots=False,
            )
        monkeypatch.setattr(sweep_module, "_runtime", actual_runtime)
    with RainbowReference(smooth_teacher(tmp_path / "other", isotropic=True)) as other:
        with pytest.raises(ValueError, match="changed; use a new output"):
            run_single_condition_sweep(
                other, model_configuration(), cfg, output,
                sweep_config=experiment(), make_plots=False,
            )
    assert all((output / name).read_bytes() == value for name, value in saved.items())


def test_diagnostics_are_reported_hashed_and_not_selection_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep_module, "_summary_plot", lambda *args: None)
    calls = []

    def diagnostic(model, reference, trial_dir):
        calls.append(trial_dir)
        directory = trial_dir / "diagnostics"
        directory.mkdir()
        (directory / "scientific.txt").write_text("synthetic diagnostic", encoding="utf-8")
        return {"reference_band_mass_exact": 0.01, "scope": "synthetic test only",
                "nested": {"not_a_column": [1, 2]}}

    cfg, plan, output = configuration(steps=0), experiment(), tmp_path / "sweep"
    diagnostic_config = {"id": "synthetic", "implementation_sha256": "fixture"}
    with RainbowReference(smooth_teacher(tmp_path / "record")) as reference:
        first = run_single_condition_sweep(
            reference, model_configuration(), cfg, output, sweep_config=plan,
            make_plots=False, diagnostic_callback=diagnostic, diagnostic_config=diagnostic_config,
        )
        assert first["selection"]["selected_learning_rate"] == min(plan.learning_rates)
        assert len(calls) == 4
        with (output / "summary.csv").open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            assert "diag_reference_band_mass_exact" in reader.fieldnames
            assert "diag_nested" not in reader.fieldnames
            assert all(float(row["diag_reference_band_mass_exact"]) == 0.01 for row in reader)
        again = run_single_condition_sweep(
            reference, model_configuration(), cfg, output, sweep_config=plan,
            make_plots=False, diagnostic_callback=diagnostic, diagnostic_config=diagnostic_config,
        )
        assert first == again and len(calls) == 4
        path = calls[0] / "diagnostics/scientific.txt"
        path.write_text("edited", encoding="utf-8")
        with pytest.raises(ValueError, match="file changed or missing"):
            run_single_condition_sweep(
                reference, model_configuration(), cfg, output, sweep_config=plan,
                make_plots=False, diagnostic_callback=diagnostic,
                diagnostic_config=diagnostic_config,
            )


def test_failed_postprocessing_resumes_completed_checkpoint_without_more_updates(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(sweep_module, "_summary_plot", lambda *args: None)
    fail = True

    def diagnostic(model, reference, trial_dir):
        nonlocal fail
        directory = trial_dir / "diagnostics"
        directory.mkdir(exist_ok=True)
        (directory / "example.json").write_text("{}", encoding="utf-8")
        if fail:
            fail = False
            raise OSError("synthetic diagnostic write interruption")
        return {"synthetic_only": True}

    cfg, output = configuration(steps=2), tmp_path / "sweep"
    plan = SweepConfig(learning_rates=(1e-3,), train_samples=(64,), baseline_train_samples=64)
    with RainbowReference(smooth_teacher(tmp_path / "record")) as reference:
        with pytest.raises(OSError, match="synthetic diagnostic"):
            run_single_condition_sweep(
                reference, model_configuration(), cfg, output, sweep_config=plan,
                make_plots=False, diagnostic_callback=diagnostic,
                diagnostic_config={"id": "synthetic", "implementation_sha256": "fixture"},
            )
        stopped = read_json(output / "summary.json")
        trial = output / stopped["trials"][0]["run_directory"]
        assert stopped["status"] == "failed" and not (trial / "sweep_completed.json").exists()
        before = single.read_single_checkpoint(trial / "checkpoint.pt")
        assert before["global_step"] == cfg.steps
        result = run_single_condition_sweep(
            reference, model_configuration(), cfg, output, sweep_config=plan,
            make_plots=False, diagnostic_callback=diagnostic,
            diagnostic_config={"id": "synthetic", "implementation_sha256": "fixture"},
        )
    assert result["status"] == "complete"
    after = single.read_single_checkpoint(trial / "checkpoint.pt")
    assert after["global_step"] == before["global_step"]
    assert_state_equal(after["model_state"], before["model_state"])
    assert_state_equal(after["optimizer_state"], before["optimizer_state"])
    assert after["history"] == before["history"]
    assert torch.equal(after["minibatch_rng_state"], before["minibatch_rng_state"])


def test_default_sweep_requires_cuda_without_cpu_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with RainbowReference(smooth_teacher(tmp_path / "record")) as reference:
        with pytest.raises(RuntimeError, match="CUDA is required by default"):
            run_single_condition_sweep(
                reference, model_configuration(), SingleTrainingConfig(), tmp_path / "sweep",
            )
    assert not (tmp_path / "sweep").exists()


@pytest.mark.parametrize("values", [
    {"learning_rates": []}, {"learning_rates": [1e-3, 1e-3]},
    {"learning_rates": [float("nan")]}, {"learning_rates": [True]},
    {"train_samples": [True]}, {"train_samples": [0]},
    {"train_samples": [64, 64]}, {"baseline_train_samples": 0}, {"unknown": 3},
])
def test_invalid_sweep_config(values):
    with pytest.raises(ValueError):
        SweepConfig.from_dict(values)
