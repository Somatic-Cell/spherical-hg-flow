"""Checkpoint/state-machine unit tests; the fake teacher is NOT optical validation.

Tests below the integration marker use the actual trainer when Zuko is installed.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from phaseflow import uniform_branch_sweep as mod


def metric(value=0.3):
    return {"log_rmse": value, "log_mse": value**2, "relative_rmse": value + 0.05,
            "forward_kl_estimate": 0.03, "nll": 1.0 + value}


@pytest.fixture
def parent():
    torch.manual_seed(41)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    batches = torch.Generator().manual_seed(123)
    history = [{"global_step": 0, "validation": metric(1.0)}]
    for step in range(1, 5):
        x = torch.randn(8, 2, generator=batches)
        loss = (model(x) - x.sum(-1, keepdim=True)).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append({"global_step": step, "learning_rate": 0.003,
                        "validation": metric(1.0 / (step + 1))})
    state = copy.deepcopy(model.state_dict())
    validation = history[-1]["validation"]
    return {
        "checkpoint_version": 5, "family": "rainbow_single_condition", "kind": "training",
        "global_step": 4, "model_config": {"geometry_dtype": "model", "spline_dtype": "model"},
        "model_state": state, "training_config": {"steps": 4, "learning_rate": 0.003,
            "train_samples": 32, "dtype": "float32", "eval_every": 2, "device": "cpu"},
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "rng_state": {"torch": torch.get_rng_state()}, "minibatch_rng_state": batches.get_state(),
        "sample_split": {"training_pool": {"component_counts": {"cdf": 0, "uniform": 32}}},
        "history": history, "best_state": copy.deepcopy(state), "best_step": 4,
        "best_validation": validation, "latest_validation": validation,
        "runtime": {"device": "cpu"}, "dtype": "float32", "physics": {"hg_g": 0.2},
        "dataset_fingerprint": "fixture", "code_fingerprint": "fixture",
        "objective_config": {"sampling": "uniform", "target_fraction": 0.0,
                             "nll_weight": 0.0, "beta": 1.0, "selection_metric": "log_rmse"},
        "selection_metric": "log_rmse",
        "selections": {k: {"step": 4, "validation": validation, "model_state": state}
                       for k in ("nll", "log_rmse")},
        "data_provenance": {"record_directory": "fixture"},
    }


def plan_fixture(tmp_path, parent):
    source = tmp_path / "parent"
    source.mkdir()
    torch.save(parent, source / "checkpoint.pt")
    plan = mod.make_plan(source / "checkpoint.pt", tmp_path / "out", parent,
                         total_steps=10, milestones=[6, 10], rates=[0.003, 0.001])
    plan["plot_dpi"] = 40
    return source / "checkpoint.pt", Path(plan["output"]), plan


@pytest.mark.parametrize("lr", [0.003, 0.001])
def test_branch_changes_only_future_rate_and_horizon(parent, lr):
    original = copy.deepcopy(parent)
    child = mod.branch_payload(parent, total_steps=10, learning_rate=lr)
    assert mod.equal_state(parent, original)
    for k in ("model_state", "rng_state", "minibatch_rng_state", "sample_split", "history",
              "selections", "best_state", "best_validation", "runtime", "code_fingerprint"):
        assert mod.equal_state(child[k], parent[k]), k
    assert mod.equal_state(child["optimizer_state"]["state"], parent["optimizer_state"]["state"])
    assert child["training_config"]["steps"] == 10
    assert all(g["lr"] == lr for g in child["optimizer_state"]["param_groups"])
    child["model_state"]["weight"].add_(1)
    assert mod.equal_state(parent, original), "Child states must not alias parent tensors"


def advance(p, stop):
    q = copy.deepcopy(p)
    model = torch.nn.Linear(2, 1)
    model.load_state_dict(q["model_state"])
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    optimizer.load_state_dict(q["optimizer_state"])
    batches = torch.Generator()
    batches.set_state(q["minibatch_rng_state"])
    torch.set_rng_state(q["rng_state"]["torch"])
    for step in range(q["global_step"] + 1, stop + 1):
        x = torch.randn(8, 2, generator=batches)
        target = x.sum(-1, keepdim=True) + torch.rand(8, 1) * 0.01
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x) - target).square().mean()
        loss.backward()
        optimizer.step()
        q["history"].append({"global_step": step, "loss": float(loss.detach()),
                              "learning_rate": optimizer.param_groups[0]["lr"]})
    q.update(model_state=copy.deepcopy(model.state_dict()),
             optimizer_state=copy.deepcopy(optimizer.state_dict()),
             global_step=stop, minibatch_rng_state=batches.get_state(),
             rng_state={"torch": torch.get_rng_state()})
    return q


@pytest.mark.parametrize("lr", [0.003, 0.001])
def test_actual_adam_update_continuation_and_interruption_are_equal(parent, lr, tmp_path):
    child = mod.branch_payload(parent, total_steps=10, learning_rate=lr)
    full = advance(child, 10)
    intermediate = advance(child, 6)
    torch.save(intermediate, tmp_path / "checkpoint.pt")
    torch.rand(1000)  # Any intervening rendering/other work must not affect future updates.
    resumed = advance(mod.load_payload(tmp_path / "checkpoint.pt"), 10)
    assert mod.equal_state(full, resumed)
    manual = copy.deepcopy(parent)
    manual["training_config"]["steps"] = 10
    manual["training_config"]["learning_rate"] = lr
    for group in manual["optimizer_state"]["param_groups"]:
        group["lr"] = lr
    assert mod.equal_state(full, advance(manual, 10))


@pytest.mark.parametrize("key,value", [
    ("kind", "inference"), ("checkpoint_version", 4), ("dtype", "float64"),
    ("global_step", 3), ("selection_metric", "nll"),
])
def test_reject_wrong_parent_kind_version_or_step(parent, key, value):
    parent[key] = value
    with pytest.raises(ValueError):
        mod.validate_parent(parent, 4)


@pytest.mark.parametrize("key,value", [
    ("sampling", "target_uniform"), ("nll_weight", 1.0), ("beta", 0.5),
    ("target_fraction", 0.5), ("selection_metric", "nll"),
])
def test_reject_changed_objective(parent, key, value):
    parent["objective_config"][key] = value
    with pytest.raises(ValueError):
        mod.validate_parent(parent, 4)


@pytest.mark.parametrize("rate", [0.0, -1.0, float("nan"), float("inf"), True])
def test_reject_invalid_rate(parent, rate):
    with pytest.raises(ValueError):
        mod.branch_payload(parent, total_steps=10, learning_rate=rate)


def test_optimizer_counters_must_be_preserved(parent):
    first = next(iter(parent["optimizer_state"]["state"].values()))
    first["step"].zero_()
    with pytest.raises(ValueError, match="counters"):
        mod.validate_parent(parent, 4)


def test_all_samples_are_uniform(parent):
    parent["sample_split"]["training_pool"]["component_counts"] = {"cdf": 16, "uniform": 16}
    with pytest.raises(ValueError, match="all-uniform"):
        mod.validate_parent(parent, 4)


@pytest.mark.parametrize("suffix", ["", "nested"])
def test_output_must_not_be_inside_parent(tmp_path, parent, suffix):
    source = tmp_path / "parent"
    source.mkdir()
    torch.save(parent, source / "checkpoint.pt")
    with pytest.raises(ValueError, match="disjoint"):
        mod.make_plan(source / "checkpoint.pt", source / suffix, parent,
                      total_steps=10, milestones=[6, 10], rates=[0.003])


@pytest.mark.parametrize("steps", [[6, 8], [6, 6, 10], [3, 10], [7, 10]])
def test_milestones_have_common_fixed_horizon(tmp_path, parent, steps):
    source, out, _ = plan_fixture(tmp_path, parent)
    with pytest.raises(ValueError):
        mod.make_plan(source, out, parent, total_steps=10, milestones=steps, rates=[0.003])


def test_parent_snapshot_is_exact_and_two_branches_share_parent(tmp_path, parent):
    source, root, plan = plan_fixture(tmp_path, parent)
    original = source.read_bytes()
    with mod.output_lock(root):
        mod.init_root(plan, root, source)
        a = mod.init_branch(root, plan, parent, 0.003)
        b = mod.init_branch(root, plan, parent, 0.001)
        for run in (a, b):
            mod.validate_child(mod.load_payload(run / "checkpoint.pt"), parent, plan,
                               mod.read_json(run / "branch.json")["learning_rate_after"])
        assert mod.equal_state(mod.load_payload(a / "checkpoint.pt")["minibatch_rng_state"],
                               mod.load_payload(b / "checkpoint.pt")["minibatch_rng_state"])
    assert source.read_bytes() == original == (root / "parent_checkpoint.pt").read_bytes()
    assert set(source.parent.iterdir()) == {source}


def test_rerun_does_not_rewrite_branch_and_rejects_manifest_changes(tmp_path, parent):
    source, root, plan = plan_fixture(tmp_path, parent)
    with mod.output_lock(root):
        mod.init_root(plan, root, source)
        run = mod.init_branch(root, plan, parent, 0.003)
        original = (run / "checkpoint.pt").read_bytes()
        assert mod.init_branch(root, plan, parent, 0.003) == run
        assert (run / "checkpoint.pt").read_bytes() == original
        changed = copy.deepcopy(plan)
        changed["total_steps"] = 20
        with pytest.raises(ValueError):
            mod.init_root(changed, root, source)


def complete_files(run, payload, step, last=0.25, best=0.2, test=0.1):
    p = copy.deepcopy(payload)
    p.update(global_step=step, best_step=4, best_validation=metric(best),
             latest_validation=metric(last))
    p["history"] = p["history"] + [{"global_step": step, "validation": metric(last)}]
    for name in ("checkpoint.pt", "best.pt", "best_by_log.pt", "best_by_nll.pt"):
        torch.save(p, run / name)
    mod.atomic_json(run / "config.json", {"training": p["training_config"]})
    mod.atomic_json(run / "sample_split.json", p["sample_split"])
    mod.atomic_json(run / "history.json", p["history"])
    mod.atomic_json(run / "metrics.json", {"global_step": step, "selected_step": p["best_step"],
                    "test": metric(test) if step == p["training_config"]["steps"] else None})


def test_snapshot_is_immutable_and_never_uses_future_model(tmp_path, parent):
    source, root, plan = plan_fixture(tmp_path, parent)
    with mod.output_lock(root):
        mod.init_root(plan, root, source)
        run = mod.init_branch(root, plan, parent, 0.003)
        p = mod.load_payload(run / "checkpoint.pt")
        complete_files(run, p, 6)
        dest = mod.snapshot(run, 6)
        hashes = {n: mod.file_hash(dest / n) for n in mod.STATE_FILES}
        complete_files(run, p, 10)
        mod.snapshot(run, 10)
        assert hashes == {n: mod.file_hash(dest / n) for n in mod.STATE_FILES}
        with pytest.raises(ValueError, match="relabel"):
            mod.snapshot(run, 8)
        (dest / "history.json").write_text("[]")
        with pytest.raises(ValueError, match="changed"):
            mod.validate_snapshot(dest)


def test_snapshot_rejects_stale_metrics_after_crash(tmp_path, parent):
    source, root, plan = plan_fixture(tmp_path, parent)
    with mod.output_lock(root):
        mod.init_root(plan, root, source)
        run = mod.init_branch(root, plan, parent, 0.003)
        p = mod.load_payload(run / "checkpoint.pt")
        complete_files(run, p, 6)
        mod.atomic_json(run / "metrics.json", {"global_step": 4, "selected_step": 4})
        with pytest.raises(ValueError, match="stale"):
            mod.snapshot(run, 6)


def test_validation_not_test_selects_and_partial_has_no_selection(tmp_path, parent):
    source, root, plan = plan_fixture(tmp_path, parent)
    with mod.output_lock(root):
        mod.init_root(plan, root, source)
        a = mod.init_branch(root, plan, parent, 0.003)
        complete_files(a, mod.load_payload(a / "checkpoint.pt"), 10, best=0.15, test=100.0)
        mod.snapshot(a, 10)
        result = mod.write_report(root, plan)
        assert not result["training_complete"]
        assert not (root / "selection.json").exists()
        b = mod.init_branch(root, plan, parent, 0.001)
        complete_files(b, mod.load_payload(b / "checkpoint.pt"), 10, best=0.2, test=0.00001)
        mod.snapshot(b, 10)
        result = mod.write_report(root, plan)
        assert result["selection"]["trial_id"] == "lr_0.003"
        assert result["test_used_for_selection"] is False
        assert result["selection"]["selected_step"] == 4  # Includes the parent best.
        assert (root / "summary.png").is_file()


def test_cli_plan_does_not_create_output(tmp_path, parent, capsys):
    source, root, plan = plan_fixture(tmp_path, parent)
    rc = mod.main(["plan", "--parent", str(source), "--output", str(root),
                   "--parent-step", "4", "--steps", "10", "--milestones", "6", "10"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["additional_updates_total"] == 12
    assert not root.exists()


def test_extra_zoom_uses_full_arrays_without_smoothing(tmp_path):
    import numpy as np
    theta = np.arange(0.5, 180.0, 1.0)
    pdf = np.ones((3, len(theta))) / (4 * np.pi)
    np.savez(tmp_path / "angular_profiles.npz", theta_midpoints_degrees=theta,
             theta_edges_degrees=np.arange(181.), reference_log_pdf=np.log(pdf),
             nf_log_pdf=np.log(pdf * 0.9), hg_log_pdf=np.log(pdf),
             actual_source_phi_degrees=np.array([-90., 0., 90.]))
    original = mod.file_hash(tmp_path / "angular_profiles.npz")
    mod.extra_profile_zoom(tmp_path / "angular_profiles.npz", 40)
    assert len(list(tmp_path.glob("profile_120_160_*.png"))) == 3
    assert mod.file_hash(tmp_path / "angular_profiles.npz") == original


def test_batch_keeps_environment_and_exact_two_branches():
    batch = (Path(__file__).resolve().parents[1] / "sweep_log_uniform_continue.bat").read_text()
    assert "call environment.bat" in batch
    assert "--parent-step 2000 --steps 5000 --milestones 3000 5000" in batch
    assert "--learning-rates 0.003 0.001" in batch
    assert "--allow-cpu-test" not in batch and "pause" not in batch.lower()
    assert "--no-plots" not in batch


@pytest.mark.integration
@pytest.mark.skipif(importlib.util.find_spec("zuko") is None, reason="Zuko unavailable")
def test_real_flow_fork_resume_and_plot(tmp_path):
    # Runs only with the actual project dependencies; no surrogate model is substituted.
    from test_log_training import _model, _training
    from test_single_condition import smooth_teacher

    from phaseflow.log_objective import LogObjectiveConfig
    from phaseflow.rainbow import RainbowReference
    from phaseflow.single_condition import train_single_condition

    config = _training(steps=4, train_samples=256, train_monitor_samples=64)
    objective = LogObjectiveConfig(schema_version=2, sampling="uniform", target_fraction=0,
        nll_weight=0, beta=1, validation_uniform_samples=128, test_uniform_samples=128)
    with RainbowReference(smooth_teacher(tmp_path / "record")) as reference:
        original = train_single_condition(reference, _model(), config, tmp_path / "parent",
                                           objective_config=objective, make_plots=False)
        source = original.checkpoint_path
        p = mod.load_payload(source)
        source_hash = mod.file_hash(source)
        mod.preflight(p, reference, "cpu", cpu_test=True)
        plan = mod.make_plan(source, tmp_path / "sweep", p, total_steps=10,
                             milestones=[6, 10], rates=[0.003, 0.001])
        plan["plot_dpi"] = 40
        result = mod.run_sweep(plan, reference, p, device="cpu", dpi=40, plots=True, mode="run")
        assert result["training_complete"]
        assert all(row["plots_complete"] for row in result["trials"])
        assert mod.file_hash(source) == source_hash
        # Each branch's split/charts must retain every original uniform point.
        for rate in plan["learning_rates"]:
            branch = Path(plan["output"]) / mod.branch_name(rate)
            child = mod.load_payload(branch / "checkpoint.pt")
            mod.validate_child(child, p, plan, rate)
            assert child["history"][:len(p["history"])] == p["history"]
        result2 = mod.run_sweep(plan, reference, p, device="cpu", dpi=40, plots=True, mode="run")
        assert result2 == result
