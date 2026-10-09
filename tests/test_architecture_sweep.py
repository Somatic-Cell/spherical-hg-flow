"""Real tiny CPU subprocess trials; these are not CUDA/Windows performance tests."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from test_capacity_sweep import configuration, model_configuration
from test_single_condition import smooth_teacher

import phaseflow.architecture_sweep as architecture
from phaseflow.capacity_sweep import CapacitySweepConfig, run_capacity_sweep
from phaseflow.rainbow import RainbowReference
from phaseflow.single_condition import read_single_checkpoint
from phaseflow.training_scatter import _state_equal


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def configs(root, *, rates=(0.001, 0.003), plots=False):
    root.mkdir(parents=True, exist_ok=True)
    base = {
        "schema_version": 2, "family": "rainbow_single_condition",
        "model": model_configuration().to_dict(),
        "training": configuration(steps=4, train_samples=32, batch_size=16).to_dict(),
        "visualization": {"enabled": plots, "dpi": 40, "eval_batch_size": 128},
    }
    (root / "base.json").write_text(json.dumps(base), encoding="utf-8")
    plan = {
        "schema_version": 1, "base_config": "base.json", "num_bins": 4, "train_samples": 32,
        "architectures": [{"num_coupling_layers": 2, "hidden_features": [8, 8]}],
        "learning_rates": list(rates), "milestones": [2, 4], "diagnostics": {"enabled": False},
    }
    path = root / "architecture.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def child_imports_this_checkout(monkeypatch):
    # The reusable interpreter may be editable-installed against another checkout.
    # Tests set this explicitly; production orchestration never rewrites PYTHONPATH.
    source = str(Path(__file__).resolve().parents[1] / "src")
    monkeypatch.setenv("PYTHONPATH", source + os.pathsep + os.environ.get("PYTHONPATH", ""))


def test_expansion_strict_validation_and_cpu_is_explicit(tmp_path):
    path = configs(tmp_path)
    plan = architecture.load_plan(path, device="cuda:0")
    assert len(plan["trials"]) == 2
    assert plan["trials"][0]["trial_id"] == "trial_01_l2_h8x8_lr0.001"
    assert plan["trials"][0]["configuration"]["training"]["device"] == "cuda:0"
    bad = read(path)
    bad["architectures"].append(bad["architectures"][0])
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="duplicate"):
        architecture.load_plan(path)


@pytest.mark.parametrize("change", [
    {"learning_rates": [0.001, 0.001]}, {"learning_rates": [True]},
    {"num_bins": True}, {"milestones": [2, 6]},
    {"architectures": [{"num_coupling_layers": 3, "hidden_features": [8, 8]}]},
    {"architectures": [{"num_coupling_layers": 2, "hidden_features": None}]},
    {"unknown": 1},
])
def test_malformed_plan_rejected_before_output(tmp_path, change):
    path = configs(tmp_path)
    value = read(path)
    value.update(change)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        architecture.load_plan(path)


def test_real_subprocess_summary_repeat_report_and_tamper_refusal(tmp_path, monkeypatch):
    record = smooth_teacher(tmp_path / "record")
    config = configs(tmp_path / "config", rates=(0.001,), plots=True)
    value = read(config)
    value["architectures"].append({"num_coupling_layers": 4, "hidden_features": [16, 16]})
    config.write_text(json.dumps(value))
    output = tmp_path / "output"
    ready = architecture.run_architecture_sweep(record, config, output, dry_run=True)
    assert ready["status"] == "ready" and not output.exists()
    report = architecture.run_architecture_sweep(record, config, output)
    assert report["status"] == "complete"
    assert report["selection"]["test_used"] is False
    rows = report["trials"]
    assert len(rows) == 2
    assert len({row["sample_split"]["training_points_sha256"] for row in rows}) == 1
    assert len({row["sample_split"]["validation_points_sha256"] for row in rows}) == 1
    assert len({row["parameter_count"] for row in rows}) == 2
    for row in rows:
        checkpoint = read_single_checkpoint(output / row["run_directory"] / "checkpoint.pt")
        assert checkpoint["training_config"]["learning_rate"] == row["learning_rate"]
        assert checkpoint["model_config"]["num_coupling_layers"] == row["num_coupling_layers"]
        assert checkpoint["model_config"]["hidden_features"] == row["hidden_features"]
        assert checkpoint["global_step"] == 4 and row["selected_step"] <= 4
        assert [item["milestone_step"] for item in row["milestones"]] == [2, 4]
        plot = read(output / row["run_directory"] / "plots/plots.json")
        assert plot["scatter"]["displayed_sample_count"] == 32
        assert plot["scatter"]["hash_verified"] is True
    # Deliberately opposed test rankings cannot change outer validation selection.
    opposed = [dict(row) for row in rows]
    opposed[0].update(validation_nll=2.0, test_nll=-10.0)
    opposed[1].update(validation_nll=1.0, test_nll=10.0)
    selected = architecture._selection(opposed, read(output / "manifest.json"))
    assert selected["selected_trial_id"] == opposed[1]["trial_id"]
    opposed[1]["status"] = "failed"
    assert architecture._selection(opposed, read(output / "manifest.json")) is None
    scientific_before = {p: p.read_bytes() for p in (output / "trials").rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError("verified completed trials must not execute another child")

    monkeypatch.setattr(architecture, "_run_child", forbidden)
    again = architecture.run_architecture_sweep(record, config, output)
    rebuilt = architecture.run_architecture_sweep(record, config, output, report_only=True)
    assert again["selection"] == rebuilt["selection"] == report["selection"]
    assert all(p.read_bytes() == content for p, content in scientific_before.items())
    assert (output / "summary.png").read_bytes().startswith(b"\x89PNG")
    assert b"\r" not in (output / "summary.csv").read_bytes()
    assert "num_coupling_layers" in (output / "summary.csv").read_text(encoding="utf-8-sig")
    # The second child's damage must be discovered before altering parent or first child.
    target = output / rows[-1]["run_directory"] / "history.json"
    target.write_text("[]\n")
    before = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="changed or missing"):
        architecture.run_architecture_sweep(record, config, output)
    assert all(p.read_bytes() == content for p, content in before.items())


def test_failed_child_continues_no_selection_and_retry_resumes_prefix(tmp_path, monkeypatch):
    record = smooth_teacher(tmp_path / "record")
    config, output = configs(tmp_path / "config"), tmp_path / "output"
    original = architecture._run_child
    calls = []

    def first_partial(command, log, active, trial_id):
        calls.append(trial_id)
        if len(calls) == 1:
            # A genuine paused child (exit 0) is still NOT a completed candidate.
            return original([*command, "--max-milestones-this-run", "1"], log, active, trial_id)
        return original(command, log, active, trial_id)

    monkeypatch.setattr(architecture, "_run_child", first_partial)
    failed = architecture.run_architecture_sweep(record, config, output)
    assert len(calls) == 2 and failed["selection"] is None
    assert [row["status"] for row in failed["trials"]] == ["failed", "complete"]
    assert not (output / "selection.json").exists()
    first = failed["trials"][0]
    prefix = output / first["child_directory"] / "bins/k_4/milestones/updates_2"
    frozen = {p: p.read_bytes() for p in prefix.rglob("*") if p.is_file()}
    monkeypatch.setattr(architecture, "_run_child", original)
    completed = architecture.run_architecture_sweep(record, config, output)
    assert completed["status"] == "complete"
    assert all(p.read_bytes() == content for p, content in frozen.items())
    # Match the resumed child to a genuine uninterrupted capacity trajectory.
    with RainbowReference(record) as reference:
        run_capacity_sweep(reference, model_configuration(),
                           replace(configuration(), steps=4, train_samples=32, batch_size=16,
                                   learning_rate=0.001), tmp_path / "direct",
                           sweep_config=CapacitySweepConfig(num_bins=(4,), milestones=(2, 4)),
                           make_plots=False)
    resumed = read_single_checkpoint(output / completed["trials"][0]["run_directory"] / "checkpoint.pt")
    direct = read_single_checkpoint(tmp_path / "direct/bins/k_4/milestones/updates_4/checkpoint.pt")
    for key in ("model_state", "optimizer_state", "best_state", "rng_state", "minibatch_rng_state"):
        assert _state_equal(resumed[key], direct[key])
    assert resumed["history"] == direct["history"]


def test_launch_failure_report_and_ctrl_c_do_not_claim_completion(tmp_path, monkeypatch):
    record = smooth_teacher(tmp_path / "record")
    config = configs(tmp_path / "config", rates=(0.001,))
    output = tmp_path / "failed"
    monkeypatch.setattr(architecture, "_run_child", lambda *args: 23)
    failed = architecture.run_architecture_sweep(record, config, output)
    assert failed["selection"] is None and failed["trials"][0]["last_exit_code"] == 23
    report = architecture.run_architecture_sweep(record, config, output, report_only=True)
    assert report["trials"][0]["status"] == "failed"

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(architecture, "_run_child", interrupt)
    with pytest.raises(KeyboardInterrupt):
        architecture.run_architecture_sweep(record, config, tmp_path / "interrupted")
    saved = read(tmp_path / "interrupted/summary.json")
    assert saved["status"] == "interrupted" and saved["selection"] is None


def test_changed_config_and_unsafe_paths_refused(tmp_path, monkeypatch):
    record = smooth_teacher(tmp_path / "record")
    config = configs(tmp_path / "config", rates=(0.001,))
    output = tmp_path / "output"
    monkeypatch.setattr(architecture, "_run_child", lambda *args: 1)
    architecture.run_architecture_sweep(record, config, output)
    before = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
    value = read(config)
    value["learning_rates"] = [0.002]
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="changed"):
        architecture.run_architecture_sweep(record, config, output)
    assert all(p.read_bytes() == content for p, content in before.items())
    with pytest.raises(ValueError, match="disjoint"):
        architecture.run_architecture_sweep(record, config, record / "new")
    with pytest.raises(ValueError, match="inside another experiment"):
        architecture.run_architecture_sweep(record, config, output / "nested", dry_run=True)


def test_single_output_lock_and_child_import_guard(tmp_path):
    with architecture._exclusive_output(tmp_path):
        with pytest.raises(RuntimeError, match="another architecture sweep"):
            with architecture._exclusive_output(tmp_path):
                raise AssertionError("second lock acquired")
    trial = architecture.load_plan(configs(tmp_path / "config"))["trials"][0]
    command = architecture._command(tmp_path / "record", tmp_path / "output", trial)
    assert command[0] == architecture.sys.executable and command[1] == "-u"
    # A wrong source identity fails before attempting to open any CDF.
    command[4] = str(tmp_path / "wrong/__init__.py")
    code = architecture._run_child(command, tmp_path / "guard.log", tmp_path / "active.json", "guard")
    assert code != 0 and "import mismatch" in (tmp_path / "guard.log").read_text()
    assert not (tmp_path / "active.json").exists()
