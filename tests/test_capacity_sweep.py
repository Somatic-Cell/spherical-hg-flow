"""Tiny synthetic CPU checks of experiment identity, trajectory and reporting.

These tests do not establish CUDA performance or Rainbow solver accuracy.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from test_single_condition import assert_state_equal, smooth_teacher

import phaseflow.capacity_sweep as capacity
import phaseflow.single_condition as single
from phaseflow.capacity_sweep import (
    CapacitySweepConfig,
    run_capacity_sample_sweep,
    run_capacity_sweep,
)
from phaseflow.plotting import RainbowPlotConfig
from phaseflow.rainbow import RainbowReference
from phaseflow.single_condition import SingleTrainingConfig, read_single_checkpoint
from phaseflow.sphere_model import SphereFlowConfig


def configuration(**changes):
    return replace(SingleTrainingConfig(
        seed=415, data_seed=2026, device="cpu", dtype="float32", learning_rate=0.003,
        train_samples=64, validation_samples=128, test_samples=128, proposal_samples=16,
        batch_size=32, steps=4, eval_every=2, checkpoint_every=2, eval_batch_size=128,
        tensorboard=False, log_every=1, train_monitor_samples=64,
    ), **changes)


def model_configuration():
    return SphereFlowConfig(
        num_coupling_layers=2, num_bins=4, hidden_features=(8, 8),
        geometry_dtype="model", spline_dtype="model",
    )


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fast_summary(monkeypatch):
    monkeypatch.setattr(capacity, "_summary_plot", lambda *args: None)


def test_capacity_seeds_and_validation_selection_not_test_or_intermediate_results(tmp_path, monkeypatch):
    fast_summary(monkeypatch)
    original_train, original_evaluate = capacity.train_single_condition, single.evaluate_single_condition
    active, evaluations, calls = {}, [], []

    def tracked_train(reference, model, cfg, output, **kwargs):
        active["output"] = output
        calls.append((model.num_bins, kwargs["max_steps_this_run"], cfg.steps))
        return original_train(reference, model, cfg, output, **kwargs)

    def inverted_test_ranking(*args, **kwargs):
        result = original_evaluate(*args, **kwargs)
        checkpoint = read_single_checkpoint(active["output"] / "checkpoint.pt")
        evaluations.append(checkpoint["global_step"])
        # Reporting sentinel intentionally reverses the selection ranking while
        # leaving actual training and validation computations untouched.
        result["nll"] = -checkpoint["best_validation"]["nll"]
        return result

    monkeypatch.setattr(capacity, "train_single_condition", tracked_train)
    monkeypatch.setattr(single, "evaluate_single_condition", inverted_test_ranking)
    cfg, output = configuration(), tmp_path / "capacity"
    plan = CapacitySweepConfig(num_bins=(4, 8), milestones=(2, 4))
    with RainbowReference(smooth_teacher(tmp_path / "record")) as reference:
        report = run_capacity_sweep(
            reference, model_configuration(), cfg, output, sweep_config=plan, make_plots=False,
        )
    assert report["status"] == "complete"
    assert calls == [(4, 2, 4), (4, 2, 4), (8, 2, 4), (8, 2, 4)]
    assert evaluations == [4, 4]
    rows = report["trials"]
    final = [row for row in rows if row["milestone_step"] == 4]
    winner = min(final, key=lambda row: (row["validation_nll"], row["num_bins"]))
    assert report["selection"]["selected_num_bins"] == winner["num_bins"]
    assert report["selection"]["test_used"] is False
    assert min(final, key=lambda row: row["test_nll"])["num_bins"] != winner["num_bins"]
    manifest = read_json(output / "manifest.json")
    for row in rows:
        archive = output / row["run_directory"]
        checkpoint = read_single_checkpoint(archive / "checkpoint.pt")
        assert checkpoint["training_config"] == cfg.to_dict()
        assert checkpoint["global_step"] == row["milestone_step"]
        assert checkpoint["model_config"]["num_bins"] == row["num_bins"]
        assert row["selected_step"] <= row["milestone_step"]
        assert row["processed_examples"] == row["milestone_step"] * cfg.batch_size
        assert row["sample_split"]["training_points_sha256"] == (
            manifest["fixed_streams"]["training_prefix_sha256"][str(cfg.train_samples)]
        )
        assert row["sample_split"]["validation_points_sha256"] == (
            manifest["fixed_streams"]["validation_points_sha256"]
        )
        assert (row["test_nll"] is None) == (row["milestone_step"] < cfg.steps)
        assert row["latest_validation_kl"] == checkpoint["latest_validation"]["forward_kl_estimate"]
        assert row["validation_kl"] == checkpoint["best_validation"]["forward_kl_estimate"]
        assert (archive / "milestone_completed.json").is_file()
    assert b"\r" not in (output / "summary.csv").read_bytes()
    assert (output / "summary.csv").read_bytes().startswith(b"\xef\xbb\xbf")


def test_milestone_pause_resume_matches_unsegmented_trajectory_and_freezes_earlier_archive(
    tmp_path, monkeypatch,
):
    fast_summary(monkeypatch)
    cfg = configuration(steps=6)
    plan = CapacitySweepConfig(num_bins=(4,), milestones=(2, 4, 6))
    output = tmp_path / "capacity"
    with RainbowReference(smooth_teacher(tmp_path / "record")) as reference:
        full = single.train_single_condition(
            reference, model_configuration(), cfg, tmp_path / "full", make_plots=False,
        )
        paused = run_capacity_sweep(
            reference, model_configuration(), cfg, output, sweep_config=plan,
            make_plots=False, max_milestones_this_run=1,
        )
        assert paused["status"] == "paused"
        assert paused["selection"] is None
        first = output / paused["trials"][0]["run_directory"]
        original = {path.relative_to(first): path.read_bytes() for path in first.rglob("*")
                    if path.is_file()}
        done = run_capacity_sweep(
            reference, model_configuration(), cfg, output, sweep_config=plan, make_plots=False,
        )
    assert done["status"] == "complete"
    assert all((first / name).read_bytes() == content for name, content in original.items())
    segmented = read_single_checkpoint(output / done["trials"][-1]["run_directory"] / "checkpoint.pt")
    unsegmented = read_single_checkpoint(full.checkpoint_path)
    for key in ("model_state", "optimizer_state", "best_state"):
        assert_state_equal(segmented[key], unsegmented[key])
    for key in ("history", "best_step", "best_validation", "latest_validation", "sample_split"):
        assert segmented[key] == unsegmented[key]
    assert capacity._state_equal(segmented["rng_state"], unsegmented["rng_state"])
    assert capacity._state_equal(segmented["minibatch_rng_state"], unsegmented["minibatch_rng_state"])


def test_full_pool_plots_exist_at_partial_milestone_and_real_summary_png(tmp_path):
    cfg, output = configuration(), tmp_path / "capacity"
    with RainbowReference(smooth_teacher(tmp_path / "record")) as reference:
        report = run_capacity_sweep(
            reference, model_configuration(), cfg, output,
            sweep_config=CapacitySweepConfig(num_bins=(4,), milestones=(2, 4)),
            plot_config=RainbowPlotConfig(dpi=45, cdf_samples=7),
            max_milestones_this_run=1,
        )
    assert report["status"] == "paused"
    row = report["trials"][0]
    snapshot = output / row["run_directory"]
    assert read_json(snapshot / "metrics.json")["test"] is None
    assert read_json(snapshot / "config.json")["visualization"]["enabled"] is False
    plots = read_json(snapshot / "plots/plots.json")
    assert plots["scatter"]["sample_count"] == cfg.train_samples
    assert plots["scatter"]["displayed_sample_count"] == cfg.train_samples
    assert plots["scatter"]["hash_verified"] is True
    with np.load(snapshot / "plots/training_scatter.npz") as points:
        assert points["directions_nf"].shape == (cfg.train_samples, 3)
    for name in ("reference_pdf.png", "cdf_samples.png", "nf_pdf.png", "comparison.png"):
        assert (snapshot / "plots" / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (output / "summary.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    receipt = read_json(snapshot / "milestone_completed.json")
    assert "plots/training_scatter.npz" in receipt["files_sha256"]


def test_completed_sweep_reuses_and_refuses_tampered_archive_before_any_writes(tmp_path, monkeypatch):
    fast_summary(monkeypatch)
    cfg, output = configuration(), tmp_path / "capacity"
    plan = CapacitySweepConfig(num_bins=(4,), milestones=(2, 4))
    with RainbowReference(smooth_teacher(tmp_path / "record")) as reference:
        first = run_capacity_sweep(
            reference, model_configuration(), cfg, output, sweep_config=plan, make_plots=False,
        )

        def forbidden(*args, **kwargs):
            raise AssertionError("completed milestone must not be trained or archived again")

        monkeypatch.setattr(capacity, "train_single_condition", forbidden)
        monkeypatch.setattr(capacity, "_archive_milestone", forbidden)
        again = run_capacity_sweep(
            reference, model_configuration(), cfg, output, sweep_config=plan, make_plots=False,
        )
        assert first == again
        changed = output / first["trials"][-1]["run_directory"] / "history.json"
        changed.write_text("[]\n", encoding="utf-8")
        before = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
        with pytest.raises(ValueError, match="file changed or missing"):
            run_capacity_sweep(
                reference, model_configuration(), cfg, output, sweep_config=plan, make_plots=False,
            )
        assert all(path.read_bytes() == data for path, data in before.items())


@pytest.mark.parametrize("when", ("before_archive", "after_copy"))
def test_archive_interruption_recovers_same_checkpoint_without_retraining(tmp_path, monkeypatch, when):
    fast_summary(monkeypatch)
    cfg, output = configuration(steps=2), tmp_path / "capacity"
    plan = CapacitySweepConfig(num_bins=(4,), milestones=(2,))
    original = capacity._archive_milestone
    checkpoint_bytes = {}

    def interrupt(*args, **kwargs):
        source = args[0]
        checkpoint_bytes["bytes"] = (source / "checkpoint.pt").read_bytes()
        if when == "after_copy":
            original(*args, **kwargs)
        raise KeyboardInterrupt("simulated interruption between optimization and sealing")

    monkeypatch.setattr(capacity, "_archive_milestone", interrupt)
    with RainbowReference(smooth_teacher(tmp_path / "record")) as reference:
        with pytest.raises(KeyboardInterrupt):
            run_capacity_sweep(
                reference, model_configuration(), cfg, output, sweep_config=plan, make_plots=False,
            )
        monkeypatch.setattr(capacity, "_archive_milestone", original)

        def forbidden_training(*args, **kwargs):
            raise AssertionError("a completed trainer return must not reserialize its checkpoint")

        monkeypatch.setattr(capacity, "train_single_condition", forbidden_training)
        report = run_capacity_sweep(
            reference, model_configuration(), cfg, output, sweep_config=plan, make_plots=False,
        )
    row = report["trials"][0]
    assert report["status"] == "complete"
    assert (output / row["run_directory"] / "checkpoint.pt").read_bytes() == checkpoint_bytes["bytes"]


@pytest.mark.parametrize("change", ("steps", "seed", "code", "runtime", "record"))
def test_changed_experiment_is_rejected_without_touching_existing_results(tmp_path, monkeypatch, change):
    fast_summary(monkeypatch)
    cfg, output = configuration(), tmp_path / "capacity"
    plan = CapacitySweepConfig(num_bins=(4,), milestones=(2, 4))
    record_path = smooth_teacher(tmp_path / "record")
    with RainbowReference(record_path) as reference:
        run_capacity_sweep(
            reference, model_configuration(), cfg, output, sweep_config=plan,
            make_plots=False, max_milestones_this_run=1,
        )
    if change == "steps":
        cfg, plan = replace(cfg, steps=6), CapacitySweepConfig(num_bins=(4,), milestones=(2, 4, 6))
    elif change == "seed":
        cfg = replace(cfg, seed=416)
    elif change == "code":
        monkeypatch.setattr(capacity, "_implementation", lambda: {"changed": "implementation"})
    elif change == "runtime":
        original_runtime = capacity._runtime
        monkeypatch.setattr(capacity, "_runtime", lambda *args: {**original_runtime(*args), "changed": 1})
    else:
        record_path = smooth_teacher(tmp_path / "other_record", isotropic=True)
    before = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    with RainbowReference(record_path) as reference, pytest.raises(ValueError, match="changed"):
        run_capacity_sweep(
            reference, model_configuration(), cfg, output, sweep_config=plan, make_plots=False,
        )
    assert all(path.read_bytes() == data for path, data in before.items())


def test_plan_validation_and_tie_breaking(tmp_path):
    for values in ({"num_bins": [8, 4]}, {"milestones": [2, 2]}, {"num_bins": [True]},
                   {"milestones": []}, {"unknown": 1}):
        with pytest.raises(ValueError):
            CapacitySweepConfig.from_dict(values)
    with RainbowReference(smooth_teacher(tmp_path / "record")) as reference:
        with pytest.raises(ValueError, match="final capacity milestone"):
            run_capacity_sweep(
                reference, model_configuration(), configuration(), tmp_path / "wrong_steps",
                sweep_config=CapacitySweepConfig(num_bins=(4,), milestones=(2,)), make_plots=False,
            )
        with pytest.raises(ValueError, match="multiple"):
            run_capacity_sweep(
                reference, model_configuration(), configuration(), tmp_path / "wrong_interval",
                sweep_config=CapacitySweepConfig(num_bins=(4,), milestones=(1, 4)), make_plots=False,
            )
    rows = [{"trial_id": f"bins_{k}", "num_bins": k, "milestone_step": 4, "status": "complete",
             "validation_nll": 1.0, "validation_kl": 0.1, "selected_step": 2,
             "run_directory": f"k_{k}"} for k in (8, 4)]
    manifest = {"base_training": {"steps": 4}, "selection": "test rule"}
    assert capacity._selection(rows, manifest)["selected_num_bins"] == 4


@pytest.mark.parametrize("change", ("future_best", "future_history"))
def test_unsealed_recovery_rejects_selected_or_history_steps_outside_prefix(tmp_path, monkeypatch, change):
    fast_summary(monkeypatch)
    cfg, output = configuration(steps=2), tmp_path / "capacity"
    plan = CapacitySweepConfig(num_bins=(4,), milestones=(2,))

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("stop before sealing")

    original = capacity._archive_milestone
    monkeypatch.setattr(capacity, "_archive_milestone", interrupt)
    with RainbowReference(smooth_teacher(tmp_path / "record")) as reference:
        with pytest.raises(KeyboardInterrupt):
            run_capacity_sweep(
                reference, model_configuration(), cfg, output, sweep_config=plan, make_plots=False,
            )
        live = output / "bins/k_4/training"
        checkpoint = read_single_checkpoint(live / "checkpoint.pt")
        if change == "future_best":
            checkpoint["best_step"] = 3
            expected = "within the archived training prefix"
        else:
            checkpoint["history"][1]["global_step"] = 3
            (live / "history.json").write_text(json.dumps(checkpoint["history"]), encoding="utf-8")
            expected = "one ordered entry per update"
        single._atomic_torch_save(live / "checkpoint.pt", checkpoint)
        monkeypatch.setattr(capacity, "_archive_milestone", original)
        before = (live / "checkpoint.pt").read_bytes()
        with pytest.raises(ValueError, match=expected):
            run_capacity_sweep(
                reference, model_configuration(), cfg, output, sweep_config=plan, make_plots=False,
            )
        assert (live / "checkpoint.pt").read_bytes() == before


def test_sample_sweep_reuses_verified_baseline_and_preserves_parent_and_nested_points(tmp_path, monkeypatch):
    fast_summary(monkeypatch)
    cfg = configuration(steps=2)
    parent, output = tmp_path / "capacity", tmp_path / "samples"
    original_train, trained = capacity.train_single_condition, []
    with RainbowReference(smooth_teacher(tmp_path / "record")) as reference:
        parent_report = run_capacity_sweep(
            reference, model_configuration(), cfg, parent,
            sweep_config=CapacitySweepConfig(num_bins=(4, 8), milestones=(2,)), make_plots=False,
        )
        parent_bytes = {path: path.read_bytes() for path in parent.rglob("*") if path.is_file()}

        def tracked_train(source, model, training, *args, **kwargs):
            trained.append((training.train_samples, model.num_bins, training.steps))
            return original_train(source, model, training, *args, **kwargs)

        monkeypatch.setattr(capacity, "train_single_condition", tracked_train)
        report = run_capacity_sample_sweep(
            reference, parent, output, train_samples=(32, 64, 128), make_plots=False,
        )
        again = run_capacity_sample_sweep(
            reference, parent, output, train_samples=(32, 64, 128), make_plots=False,
        )
    selected_bins = parent_report["selection"]["selected_num_bins"]
    assert report == again
    assert report["status"] == "complete"
    assert trained == [(32, selected_bins, cfg.steps), (128, selected_bins, cfg.steps)]
    assert all(path.read_bytes() == value for path, value in parent_bytes.items())
    manifest = read_json(output / "manifest.json")
    original_row = next(row for row in report["trials"] if row["train_samples"] == 64)
    assert original_row["reuse_of"] == f"capacity:bins_{selected_bins}"
    assert not (output / "n_64").exists()
    for row in report["trials"]:
        assert row["num_bins"] == selected_bins
        assert row["global_step"] == cfg.steps
        assert row["sample_split"]["training_points_sha256"] == (
            manifest["fixed_streams"]["training_prefix_sha256"][str(row["train_samples"])]
        )
        assert row["sample_split"]["validation_points_sha256"] == (
            manifest["fixed_streams"]["validation_points_sha256"]
        )
    winner = min(report["trials"], key=lambda row: (row["validation_nll"], row["train_samples"]))
    assert report["selection"]["selected_train_samples"] == winner["train_samples"]
    assert report["selection"]["test_used"] is False
    assert (output / "summary.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert b"\r" not in (output / "summary.csv").read_bytes()
    assert b"\r" not in (output / "summary.json").read_bytes()


def test_sample_sweep_controlled_pause_and_modified_parent_refusal(tmp_path, monkeypatch):
    fast_summary(monkeypatch)
    cfg, parent, output = configuration(steps=2), tmp_path / "capacity", tmp_path / "samples"
    with RainbowReference(smooth_teacher(tmp_path / "record")) as reference:
        run_capacity_sweep(
            reference, model_configuration(), cfg, parent,
            sweep_config=CapacitySweepConfig(num_bins=(4,), milestones=(2,)), make_plots=False,
        )
        partial = run_capacity_sample_sweep(
            reference, parent, output, train_samples=(32, 64, 128), make_plots=False,
            max_trials_this_run=1,
        )
        assert partial["status"] == "paused"
        assert [row["status"] for row in partial["trials"]] == ["complete", "complete", "pending"]
        assert partial["selection"] is None
        done = run_capacity_sample_sweep(
            reference, parent, output, train_samples=(32, 64, 128), make_plots=False,
        )
        assert done["status"] == "complete"
        before = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
        (parent / "selection.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="selection differs"):
            run_capacity_sample_sweep(
                reference, parent, output, train_samples=(32, 64, 128), make_plots=False,
            )
    assert all(path.read_bytes() == value for path, value in before.items())


def test_sample_sweep_requires_complete_parent_and_separate_output(tmp_path, monkeypatch):
    fast_summary(monkeypatch)
    cfg, parent = configuration(), tmp_path / "capacity"
    with RainbowReference(smooth_teacher(tmp_path / "record")) as reference:
        run_capacity_sweep(
            reference, model_configuration(), cfg, parent,
            sweep_config=CapacitySweepConfig(num_bins=(4,), milestones=(2, 4)),
            make_plots=False, max_milestones_this_run=1,
        )
        with pytest.raises(ValueError, match="complete, consistent summary"):
            run_capacity_sample_sweep(reference, parent, tmp_path / "samples", train_samples=(32, 64))
        with pytest.raises(ValueError, match="disjoint"):
            run_capacity_sample_sweep(reference, parent, parent / "samples", train_samples=(32, 64))
