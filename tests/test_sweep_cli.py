"""CLI wiring with tiny synthetic CPU runs and the real angular diagnostic."""

from __future__ import annotations

import csv
import hashlib
import json

import numpy as np
import pytest
from test_single_condition import smooth_teacher
from test_sweep import configuration, model_configuration

from phaseflow.cli import main


def write_configs(tmp_path):
    training = tmp_path / "training.json"
    training.write_text(json.dumps({
        "schema_version": 2,
        "family": "rainbow_single_condition",
        "model": model_configuration().to_dict(),
        # A CPU override in the CLI must take precedence over this CUDA default.
        "training": configuration(device="cuda", steps=2, train_samples=32).to_dict(),
        "visualization": {"enabled": True, "cdf_samples": 16, "dpi": 50},
    }), encoding="utf-8")
    sweep = tmp_path / "sweep.json"
    sweep.write_text(json.dumps({
        "schema_version": 1,
        "sweep": {
            "learning_rates": [3e-4, 3e-3],
            "train_samples": [16, 32],
            "baseline_train_samples": 32,
        },
        "diagnostics": {
            "enabled": True,
            "theta_band_degrees": [120, 150],
            "source_phi_degrees": [-90, 0, 90],
            "eval_batch_size": 128,
            "dpi": 50,
        },
    }), encoding="utf-8")
    return training, sweep


def test_sweep_cli_pause_resume_and_real_diagnostic_summary(tmp_path, capsys, monkeypatch):
    record = smooth_teacher(tmp_path / "record")
    config, sweep_config = write_configs(tmp_path)
    output = tmp_path / "runs"
    command = [
        "sweep-rainbow", "--record", str(record), "--config", str(config),
        "--sweep-config", str(sweep_config), "--output", str(output),
        "--device", "cpu", "--no-plots", "--quiet",
    ]
    assert main([*command, "--max-trials-this-run", "1"]) == 0
    paused_message = json.loads(capsys.readouterr().out)
    assert paused_message["complete"] is False
    partial = json.loads((output / "summary.json").read_text())
    assert partial["status"] == "paused" and partial["selection"] is None
    assert sum(row["status"] == "complete" for row in partial["trials"]) == 1

    assert main(command) == 0
    completed_message = json.loads(capsys.readouterr().out)
    assert completed_message["complete"] is True
    assert completed_message["output_directory"] == str(output.resolve())
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert len(summary["trials"]) == 4
    assert len({row["run_directory"] for row in summary["trials"]}) == 3
    with (output / "summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    for row in rows:
        assert row["status"] == "complete"
        trial = output / row["run_directory"]
        report = json.loads((trial / "diagnostics/angular_diagnostics.json").read_text())
        actual_config = json.loads((trial / "config.json").read_text())
        assert actual_config["training"]["device"] == "cpu"
        assert actual_config["visualization"]["enabled"] is False
        assert report["source_kind"] == "synthetic_fixture"
        assert report["scope"] == "report_only_same_condition_no_model_selection"
        assert report["checkpoint"]["path"] == str((trial / "best.pt").resolve())
        mass = float(row["diag_reference_band_mass_exact"])
        assert mass == report["reference_band_mass_exact"]
        assert float(row["diag_expected_train_band_samples"]) == float(row["train_samples"]) * mass
        assert report["expected_counts"]["train_samples"] == int(row["train_samples"])
        assert (trial / "diagnostics/angular_profiles.png").read_bytes().startswith(b"\x89PNG")
        assert (trial / "diagnostics/angular_profiles.npz").is_file()
        assert not (trial / "plots/comparison.png").exists()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["diagnostics"]["configuration"]["theta_band_degrees"] == [120.0, 150.0]
    assert len(manifest["diagnostics"]["implementation_sha256"]) == 64

    # The batch replot route needs only the saved completed sweep and record.
    # Remove both original launch configs and forbid additional optimization.
    config.unlink()
    sweep_config.unlink()

    def no_training(*args, **kwargs):
        raise AssertionError("Replot CLI must not invoke training")

    monkeypatch.setattr("phaseflow.sweep.train_single_condition", no_training)
    before = (output / "summary.json").read_bytes()
    assert main([
        "replot-sweep-rainbow", "--record", str(record), "--sweep", str(output),
        "--device", "cpu", "--quiet",
    ]) == 0
    replot = json.loads(capsys.readouterr().out)
    assert replot["status"] == "complete" and replot["optimizer_updates"] == 0
    assert replot["logical_trial_count"] == 4 and replot["physical_trial_count"] == 3
    assert (output / "summary.json").read_bytes() == before
    for row in replot["physical_trials"]:
        plotted = output / "training_point_plots" / row["plot_directory"]
        with np.load(plotted / "training_scatter.npz", allow_pickle=False) as arrays:
            assert arrays["directions_nf"].shape == (row["sample_count"], 3)


def test_standalone_diagnose_uses_selected_checkpoint_and_rejects_other_teacher(tmp_path, capsys):
    record = smooth_teacher(tmp_path / "record")
    config, sweep_config = write_configs(tmp_path)
    output = tmp_path / "sweep"
    assert main([
        "sweep-rainbow", "--record", str(record), "--config", str(config),
        "--sweep-config", str(sweep_config), "--output", str(output),
        "--device", "cpu", "--no-plots", "--quiet", "--max-trials-this-run", "1",
    ]) == 0
    capsys.readouterr()
    summary = json.loads((output / "summary.json").read_text())
    trial = output / summary["trials"][0]["run_directory"]
    # Inference must use best.pt, independently of the resumable last iterate.
    (trial / "checkpoint.pt").rename(trial / "held_latest.pt")
    diagnosis = tmp_path / "standalone"
    assert main([
        "diagnose-rainbow", "--record", str(record), "--run", str(trial),
        "--sweep-config", str(sweep_config), "--output", str(diagnosis), "--device", "cpu",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["checkpoint"]["sha256"] == hashlib.sha256((trial / "best.pt").read_bytes()).hexdigest()
    assert report["checkpoint"]["path"] == str((trial / "best.pt").resolve())
    assert report["model"]["device"] == "cpu"
    assert report["expected_counts"]["train_samples"] == 32
    with np.load(diagnosis / "angular_profiles.npz") as standalone, np.load(
        trial / "diagnostics/angular_profiles.npz"
    ) as during_sweep:
        assert standalone.files == during_sweep.files
        for name in standalone.files:
            np.testing.assert_array_equal(standalone[name], during_sweep[name])
    other_record = smooth_teacher(tmp_path / "other_record", isotropic=True)
    wrong_output = tmp_path / "wrong_teacher_diagnosis"
    with pytest.raises(ValueError, match="diagnostic record differs"):
        main([
            "diagnose-rainbow", "--record", str(other_record), "--run", str(trial),
            "--sweep-config", str(sweep_config), "--output", str(wrong_output), "--device", "cpu",
        ])
    assert not wrong_output.exists()
