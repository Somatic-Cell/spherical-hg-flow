"""Integration of new batch routes, using explicit tiny CPU arithmetic fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from test_single_condition import smooth_teacher
from test_sweep import configuration, model_configuration

from phaseflow.cli import _parser, main


def _configs(path):
    training, sweep = path / "training.json", path / "sweep.json"
    training.write_text(json.dumps({
        "schema_version": 2, "family": "rainbow_single_condition",
        "model": model_configuration().to_dict(),
        "training": configuration(device="cuda", steps=4, train_samples=32).to_dict(),
        "visualization": {"enabled": True, "dpi": 50, "eval_batch_size": 128},
    }), encoding="utf-8")
    sweep.write_text(json.dumps({
        "schema_version": 1,
        "sweep": {"num_bins": [4, 8], "milestones": [2, 4]},
        "diagnostics": {"enabled": True, "eval_batch_size": 128, "dpi": 50},
    }), encoding="utf-8")
    return training, sweep


def test_capacity_cli_milestones_actual_diagnostics_and_audit(tmp_path, capsys):
    record = smooth_teacher(tmp_path / "record")
    config, plan = _configs(tmp_path)
    output = tmp_path / "capacity"
    command = [
        "sweep-capacity-rainbow", "--record", str(record), "--config", str(config),
        "--sweep-config", str(plan), "--output", str(output), "--device", "cpu",
        "--no-plots", "--quiet",
    ]
    assert main([*command, "--max-milestones-this-run", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "paused"
    partial = json.loads((output / "summary.json").read_text())
    rows = [row for row in partial["trials"] if row["status"] == "complete"]
    assert len(rows) == 1
    first = output / rows[0]["run_directory"]
    assert rows[0]["milestone_step"] == 2 and rows[0]["test_nll"] is None
    assert json.loads((first / "metrics.json").read_text())["complete"] is False
    assert (first / "diagnostics/angular_profiles.png").is_file()
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in first.rglob("*") if p.is_file()}
    audit = tmp_path / "audit"
    assert main([
        "audit-sampling-rainbow", "--record", str(record), "--run", str(first),
        "--output", str(audit), "--device", "cpu", "--u-bins", "4", "--phi-bins", "8",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["sample_count"] == 32 and report["downsampling"] is False
    assert report["histogram"]["precast_count_sum"] == 32
    assert (audit / "sampling_histogram.png").is_file()
    with np.load(audit / "sampling_histogram.npz", allow_pickle=False) as arrays:
        assert arrays["precast_fp64_counts"].sum() == 32
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in first.rglob("*") if p.is_file()}
    assert main(command) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in first.rglob("*") if p.is_file()}
    summary = json.loads((output / "summary.json").read_text())
    assert len(summary["trials"]) == 4
    assert summary["test_used_for_selection"] is False
    for row in summary["trials"]:
        snapshot = output / row["run_directory"]
        saved = json.loads((snapshot / "config.json").read_text())
        assert saved["training"]["device"] == "cpu"
        assert saved["training"]["steps"] == 4
        assert row["selected_step"] <= row["milestone_step"]

    # The next stage uses the verified saved experiment, not mutable launch configs.
    config.unlink()
    plan.unlink()
    parent_before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in output.rglob("*") if p.is_file()}
    samples = tmp_path / "samples"
    sample_command = [
        "sweep-samples-rainbow", "--record", str(record), "--capacity", str(output),
        "--output", str(samples), "--device", "cpu", "--train-samples", "16", "32", "64",
        "--quiet",
    ]
    assert main([*sample_command, "--max-trials-this-run", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "paused"
    assert main(sample_command) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    sample_summary = json.loads((samples / "summary.json").read_text())
    assert len(sample_summary["trials"]) == 3
    reused = [row for row in sample_summary["trials"] if row.get("reuse_of")]
    assert len(reused) == 1 and reused[0]["train_samples"] == 32
    assert sample_summary["selected_num_bins"] == summary["selection"]["selected_num_bins"]
    for row in sample_summary["trials"]:
        snapshot = samples / row["run_directory"]
        diagnostics = json.loads((snapshot / "diagnostics/angular_diagnostics.json").read_text())
        assert diagnostics["expected_counts"]["train_samples"] == row["train_samples"]
    assert parent_before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in output.rglob("*") if p.is_file()}


def test_new_command_defaults_and_reject_changed_final_schedule(tmp_path):
    parser = _parser()
    audit = parser.parse_args([
        "audit-sampling-rainbow", "--record", "record", "--run", "run", "--output", "audit",
    ])
    assert audit.device == "cuda"
    record = smooth_teacher(tmp_path / "record")
    config, plan = _configs(tmp_path)
    value = json.loads(plan.read_text())
    value["sweep"]["milestones"] = [2, 6]
    plan.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="training.steps"):
        main([
            "sweep-capacity-rainbow", "--record", str(record), "--config", str(config),
            "--sweep-config", str(plan), "--output", str(tmp_path / "out"),
            "--device", "cpu", "--quiet",
        ])
    assert not (tmp_path / "out").exists()


def test_batch_keeps_shared_python_and_branches_before_old_sweep_configs():
    # Static portability check only: this does not claim a Windows CMD execution.
    root = Path(__file__).resolve().parents[1]
    raw = (root / "sweep.bat").read_bytes()
    assert raw.replace(b"\r\n", b"").find(b"\n") == -1
    script = raw.decode("ascii")
    assert "call environment.bat" in script
    for mode, command in (
        ("capacity", "sweep-capacity-rainbow"),
        ("samples", "sweep-samples-rainbow"),
        ("audit", "audit-sampling-rainbow"),
    ):
        assert script.index(f'if /i "%MODE%"=="{mode}" goto :{mode}') < script.index(
            'if not exist "%SWEEP_CONFIG%"'
        )
        assert f'"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow {command}' in script
    assert 'if "%MODE%"=="" set "MODE=run"' in script
    base = json.loads((root / "configs/rainbow_capacity.json").read_text())
    plan = json.loads((root / "configs/rainbow_capacity_sweep.json").read_text())
    assert base["training"]["steps"] == max(plan["sweep"]["milestones"])
    assert base["training"]["device"] == "cuda"
    assert base["training"]["train_samples"] == 262144
