"""Replot saved synthetic CPU sweeps; these tests do not validate Rainbow optics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from test_single_condition import smooth_teacher
from test_sweep import configuration, model_configuration

import phaseflow.plotting as plotting_module
import phaseflow.replot as replot_module
import phaseflow.single_condition as single_module
import phaseflow.sweep as sweep_module
from phaseflow.plotting import RainbowPlotConfig
from phaseflow.rainbow import RainbowReference
from phaseflow.replot import replot_sweep_training_points
from phaseflow.single_condition import _points
from phaseflow.sweep import SweepConfig, run_single_condition_sweep


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_hashes(directory):
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*") if path.is_file()
    }


def assert_preserved(directory, previous):
    for name, digest in previous.items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest


def forbidden(*args, **kwargs):
    raise AssertionError("Replot must not train or perform additional selection")


def make_small_sweep(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep_module, "_summary_plot", lambda *args: None)
    record = smooth_teacher(tmp_path / "record")
    root = tmp_path / "sweep"
    with RainbowReference(record) as reference:
        summary = run_single_condition_sweep(
            reference, model_configuration(), configuration(steps=0), root,
            sweep_config=SweepConfig(
                learning_rates=(1e-3,), train_samples=(64,), baseline_train_samples=64,
            ),
            make_plots=False,
        )
    return record, root, summary


def test_replot_keeps_old_plots_and_all_scientific_files_without_training(tmp_path, monkeypatch):
    record = smooth_teacher(tmp_path / "record")
    root = tmp_path / "sweep"
    cfg = configuration(steps=1, eval_every=1, checkpoint_every=1)
    plan = SweepConfig(
        learning_rates=(3e-4, 1e-3, 3e-3), train_samples=(16, 32, 64, 128),
        baseline_train_samples=64,
    )
    plot_config = RainbowPlotConfig(cdf_samples=17, seed=19, dpi=35, eval_batch_size=64)
    real_plot = plotting_module.plot_rainbow_comparison
    real_hash = sweep_module._file_hash

    def old_independent_plot(*args, **kwargs):
        kwargs["scatter_mode"] = "independent"
        return real_plot(*args, **kwargs)

    def prior_implementation_hash(path):
        # Model a recorded older plot/sweep implementation. Replot must verify
        # that saved manifest, not reconstruct one with the current code hashes.
        if Path(path).name == "plotting.py":
            return "e" * 64
        if Path(path).name == "sweep.py":
            return "d" * 64
        return real_hash(path)

    with monkeypatch.context() as original_implementation:
        original_implementation.setattr(
            plotting_module, "plot_rainbow_comparison", old_independent_plot,
        )
        original_implementation.setattr(sweep_module, "_file_hash", prior_implementation_hash)
        with RainbowReference(record) as reference:
            summary = run_single_condition_sweep(
                reference, model_configuration(), cfg, root, sweep_config=plan,
                plot_config=plot_config,
            )
    before = file_hashes(root)
    assert len(summary["trials"]) == 7
    assert len({row["run_directory"] for row in summary["trials"]}) == 6
    for row in summary["trials"]:
        old = read_json(root / row["run_directory"] / "plots/plots.json")
        assert old["scatter"]["sample_count"] == 17
    monkeypatch.setattr(sweep_module, "train_single_condition", forbidden)
    monkeypatch.setattr(single_module, "train_single_condition", forbidden)
    monkeypatch.setattr(sweep_module, "run_single_condition_sweep", forbidden)
    events = []
    with RainbowReference(record) as reference:
        result = replot_sweep_training_points(
            reference, root, device="cpu", plot_config=plot_config, callback=events.append,
        )
        assert result["status"] == "complete"
        assert result["logical_trial_count"] == 7
        assert result["physical_trial_count"] == 6
        assert result["optimizer_updates"] == 0
        assert result["selection_changed"] is False
        assert result["original_artifacts_unchanged"] is True
        output = Path(result["output_directory"])
        assert output == root / "training_point_plots"
        assert len(result["physical_trials"]) == 6
        for trial in result["physical_trials"]:
            plot_dir = output / trial["plot_directory"]
            count = trial["sample_count"]
            directions, _ = _points(reference, count, cfg.data_seed, "train")
            with np.load(plot_dir / "training_scatter.npz", allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays["directions_nf"], directions.astype(np.float32))
                assert arrays["directions_nf"].shape == (count, 3)
                assert arrays["source_phi_degrees"].shape == (count,)
                assert arrays["theta_degrees"].shape == (count,)
            split = read_json(root / trial["run_directory"] / "sample_split.json")
            assert trial["training_points_sha256"] == split["training_points_sha256"]
            manifest = read_json(plot_dir / "plots.json")
            assert manifest["scatter"]["sample_count"] == count
            assert count != plot_config.cdf_samples
            assert (plot_dir / "comparison.png").is_file()
        reused = next(row for row in result["logical_trials"] if row["reuse_of"] is not None)
        original = next(row for row in result["logical_trials"]
                        if row["trial_id"] == reused["reuse_of"])
        assert reused["plot_directory"] == original["plot_directory"]
        assert_preserved(root, before)
        again = replot_sweep_training_points(reference, root, device="cpu", plot_config=plot_config)
        assert again["status"] == "complete" and again["physical_trial_count"] == 6
        assert_preserved(root, before)
        with pytest.raises(ValueError, match="Replot inputs/configuration/code/runtime changed"):
            replot_sweep_training_points(
                reference, root, device="cpu", plot_config=RainbowPlotConfig(dpi=36),
            )
        assert_preserved(root, before)
        saved_replot_summary = (output / "replot_summary.json").read_bytes()
        temporary_alias = output / "replot_summary.json.tmp"
        temporary_alias.hardlink_to(root / summary["trials"][0]["run_directory"] / "best.pt")
        with pytest.raises(ValueError, match="hard-linked file"):
            replot_sweep_training_points(reference, root, device="cpu", plot_config=plot_config)
        assert (output / "replot_summary.json").read_bytes() == saved_replot_summary
        assert_preserved(root, before)
        temporary_alias.unlink()
    assert [event["event"] for event in events].count("trial_replot_started") == 6
    assert [event["event"] for event in events].count("trial_replot_completed") == 6
    assert events[0]["event"] == "replot_started"
    assert events[-1]["event"] == "replot_completed"


@pytest.mark.parametrize("damage", ["summary", "selection", "receipt", "teacher", "incomplete"])
def test_replot_refuses_mismatches_before_writing_any_output(tmp_path, monkeypatch, damage):
    record, root, summary = make_small_sweep(tmp_path, monkeypatch)
    if damage == "summary":
        summary["trials"][-1]["validation_nll"] += 1
        (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    elif damage == "selection":
        selection = read_json(root / "selection.json")
        selection["selected_learning_rate"] = 0.003
        (root / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    elif damage == "receipt":
        (root / summary["trials"][0]["run_directory"] / "metrics.json").write_text(
            "{}", encoding="utf-8",
        )
    elif damage == "teacher":
        record = smooth_teacher(tmp_path / "other", isotropic=True)
    else:
        summary["status"] = "paused"
        (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    before = file_hashes(root)
    monkeypatch.setattr(replot_module, "plot_rainbow_comparison", forbidden)
    monkeypatch.setattr(sweep_module, "train_single_condition", forbidden)
    with RainbowReference(record) as reference, pytest.raises(ValueError):
        replot_sweep_training_points(reference, root, device="cpu")
    assert not (root / "training_point_plots").exists()
    assert_preserved(root, before)


def test_replot_refuses_output_overlap_and_unowned_output(tmp_path, monkeypatch):
    record, root, summary = make_small_sweep(tmp_path, monkeypatch)
    original = root / summary["trials"][0]["run_directory"]
    before = file_hashes(root)
    monkeypatch.setattr(replot_module, "plot_rainbow_comparison", forbidden)
    with RainbowReference(record) as reference:
        for output in (root, root.parent, original, original / "plots", original.parent):
            with pytest.raises(ValueError, match="original"):
                replot_sweep_training_points(reference, root, output, device="cpu")
        unknown = tmp_path / "unrelated"
        unknown.mkdir()
        (unknown / "keep.txt").write_text("unrelated artifact", encoding="utf-8")
        with pytest.raises(FileExistsError, match="not empty"):
            replot_sweep_training_points(reference, root, unknown, device="cpu")
        assert (unknown / "keep.txt").read_text(encoding="utf-8") == "unrelated artifact"
    assert_preserved(root, before)
