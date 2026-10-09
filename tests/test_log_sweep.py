"""Small synthetic CPU experiments; no Rainbow optics/CUDA claims."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from test_single_condition import assert_state_equal, smooth_teacher
from test_sweep import configuration, model_configuration

import phaseflow.log_sweep as sweep_module
from phaseflow.log_objective import LogObjectiveConfig
from phaseflow.log_sweep import LogSweepConfig, main, run_log_loss_sweep
from phaseflow.plotting import RainbowPlotConfig
from phaseflow.rainbow import RainbowReference
from phaseflow.single_condition import read_single_checkpoint


def objective():
    return LogObjectiveConfig(validation_uniform_samples=64, test_uniform_samples=64)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_beta_selection_never_uses_total_loss_test_or_nll():
    common = {"status": "complete", "kind": "beta", "nll_weight": 1,
              "sampling": "target_uniform", "selected_step": 4}
    rows = [
        {**common, "trial_id": "a", "beta": 0.0, "validation_log_rmse": 0.4,
         "validation_kl": 0.1, "total_loss": -1000, "test_log_rmse": 0.001,
         "run_directory": "trials/a"},
        {**common, "trial_id": "b", "beta": 0.1, "validation_log_rmse": 0.2,
         "validation_kl": 0.3, "total_loss": 1000, "test_log_rmse": 100,
         "run_directory": "trials/b"},
    ]
    chosen = sweep_module._selection(rows, {})
    assert chosen["selected_trial_id"] == "b"
    assert chosen["test_used"] is False and chosen["total_loss_used"] is False
    assert chosen["pareto_trial_ids"] == ["a", "b"]
    rows[0]["validation_log_rmse"] = 0.2
    assert sweep_module._selection(rows, {})["selected_trial_id"] == "a"


@pytest.mark.parametrize("changes", [
    {"betas": []}, {"betas": [0, 0.0]}, {"betas": [float("nan")]},
    {"betas": [-1]}, {"betas": [True]}, {"include_target_nll_control": 1},
    {"include_pure_log_control": "false"}, {"unknown": True},
])
def test_invalid_sweep_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        LogSweepConfig.from_dict(changes)


def test_distinct_close_betas_cannot_share_a_trial_directory():
    rows = sweep_module._trial_plan(
        configuration(), objective(),
        LogSweepConfig(betas=(0.10000001, 0.10000002), include_target_nll_control=False),
    )
    assert len({row["run_directory"] for row in rows}) == 2


def test_plan_preserves_saved_model_and_training_without_data_gpu_or_writes(tmp_path, capsys):
    config = tmp_path / "base.json"
    values = {
        "schema_version": 2, "family": "rainbow_single_condition",
        "model": model_configuration().to_dict(),
        "training": configuration(device="cuda", steps=6).to_dict(),
    }
    config.write_text(json.dumps(values), encoding="utf-8")
    sweep = tmp_path / "sweep.json"
    sweep.write_text(json.dumps({"schema_version": 1, "sweep": {"betas": [0, 0.1],
                                                                "include_pure_log_control": True},
                                 "diagnostics": {"enabled": False}}), encoding="utf-8")
    output = tmp_path / "not_created"
    assert main(["plan", "--record", str(tmp_path / "missing_record"), "--config", str(config),
                 "--sweep-config", str(sweep), "--output", str(output)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["status"] == "planned" and plan["optimizer_updates"] == 0
    assert plan["planned_total_updates"] == 24
    assert plan["model"] == values["model"] and plan["training"] == values["training"]
    assert not output.exists()
    assert all(row["selection_metric"] == "log_rmse" for row in plan["trials"])
    assert plan["trials"][0]["sampling"] == "target"
    assert plan["trials"][-1]["nll_weight"] == 0
    values["objective"] = objective().to_dict()
    config.write_text(json.dumps(values), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version=3"):
        sweep_module._read_base_config(config, None)


def test_actual_beta_runs_share_pool_holdouts_and_preserve_completed_artifacts(tmp_path, monkeypatch):
    cfg, output = configuration(steps=2), tmp_path / "sweep"
    plan = LogSweepConfig(betas=(0.0, 0.1), include_pure_log_control=True)
    with RainbowReference(smooth_teacher(tmp_path / "record")) as reference:
        result = run_log_loss_sweep(
            reference, model_configuration(), cfg, output, objective_config=objective(),
            sweep_config=plan, make_plots=False, diagnostics_enabled=False,
        )
        assert result["status"] == "complete"
        rows = result["trials"]
        assert len(rows) == 4
        mix = [row for row in rows if row["sampling"] == "target_uniform"]
        assert len({row["training_points_sha256"] for row in mix}) == 1
        assert rows[0]["training_points_sha256"] != mix[0]["training_points_sha256"]
        assert len({row["validation_points_sha256"] for row in rows}) == 1
        assert len({row["validation_uniform_points_sha256"] for row in rows}) == 1
        assert len({row["test_uniform_points_sha256"] for row in rows}) == 1
        winner = min(rows, key=lambda r: (r["validation_log_rmse"], r["beta"], r["trial_id"]))
        assert result["selection"]["selected_trial_id"] == winner["trial_id"]
        checkpoints = {output / row["run_directory"] / name for row in rows
                       for name in ("checkpoint.pt", "best.pt", "best_by_nll.pt", "best_by_log.pt")}
        original = {path: path.read_bytes() for path in checkpoints}

        def forbidden(*args, **kwargs):
            raise AssertionError("completed trial must not train again")

        monkeypatch.setattr(sweep_module, "train_single_condition", forbidden)
        again = run_log_loss_sweep(
            reference, model_configuration(), cfg, output, objective_config=objective(),
            sweep_config=plan, make_plots=False, diagnostics_enabled=False,
        )
        assert again == result
        assert all(path.read_bytes() == value for path, value in original.items())
        assert b"\r\n" not in (output / "summary.csv").read_bytes()
        assert (output / "summary.png").read_bytes().startswith(b"\x89PNG")
        before = (output / "summary.json").read_bytes()
        damaged = output / rows[-1]["run_directory"] / "best_by_nll.pt"
        damaged.write_bytes(b"damaged completion artifact")
        with pytest.raises(ValueError, match="file changed or missing"):
            run_log_loss_sweep(
                reference, model_configuration(), cfg, output, objective_config=objective(),
                sweep_config=plan, make_plots=False, diagnostics_enabled=False,
            )
        assert (output / "summary.json").read_bytes() == before


def test_partial_trial_resumes_exactly_and_never_declares_early_winner(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep_module, "_summary_plot", lambda *args: None)
    cfg = configuration(steps=4)
    plan = LogSweepConfig(betas=(0.1,), include_target_nll_control=False)
    output = tmp_path / "paused"
    with RainbowReference(smooth_teacher(tmp_path / "record")) as reference:
        partial = run_log_loss_sweep(
            reference, model_configuration(), cfg, output, objective_config=objective(),
            sweep_config=plan, make_plots=False, diagnostics_enabled=False,
            max_steps_this_trial=2,
        )
        assert partial["status"] == "paused" and partial["selection"] is None
        assert partial["trials"][0]["status"] == "partial"
        assert partial["trials"][0]["global_step"] == 2
        assert not (output / "selection.json").exists()
        saved = {name: (output / name).read_bytes() for name in ("manifest.json", "summary.json")}
        with pytest.raises(ValueError, match="changed; use a new output"):
            run_log_loss_sweep(
                reference, model_configuration(), replace(cfg, learning_rate=0.01), output,
                objective_config=objective(), sweep_config=plan, make_plots=False,
                diagnostics_enabled=False,
            )
        assert all((output / name).read_bytes() == contents for name, contents in saved.items())
        resumed = run_log_loss_sweep(
            reference, model_configuration(), cfg, output, objective_config=objective(),
            sweep_config=plan, make_plots=False, diagnostics_enabled=False,
        )
        full_path = tmp_path / "full"
        full = run_log_loss_sweep(
            reference, model_configuration(), cfg, full_path, objective_config=objective(),
            sweep_config=plan, make_plots=False, diagnostics_enabled=False,
        )
    assert resumed["status"] == full["status"] == "complete"
    assert resumed["selection"] == full["selection"]
    relative = resumed["trials"][0]["run_directory"]
    left = read_single_checkpoint(output / relative / "checkpoint.pt")
    right = read_single_checkpoint(full_path / relative / "checkpoint.pt")
    for name in ("model_state", "best_state", "optimizer_state"):
        assert_state_equal(left[name], right[name])
    assert sweep_module._state_equal(left["minibatch_rng_state"], right["minibatch_rng_state"])
    assert left["history"] == right["history"]


def test_real_mixture_plots_and_replot_do_not_change_completed_run(tmp_path):
    cfg = configuration(steps=2, train_samples=32, batch_size=16)
    plan = LogSweepConfig(betas=(0.1,), include_target_nll_control=False)
    output = tmp_path / "sweep"
    with RainbowReference(smooth_teacher(tmp_path / "record")) as reference:
        result = run_log_loss_sweep(
            reference, model_configuration(), cfg, output, objective_config=objective(),
            sweep_config=plan, plot_config=RainbowPlotConfig(dpi=40),
            diagnostic_config=sweep_module.AngularDiagnosticConfig(dpi=40),
        )
        trial = output / result["trials"][0]["run_directory"]
        assert (trial / "plots/comparison.png").is_file()
        diagnosis = read_json(trial / "diagnostics/angular_diagnostics.json")
        assert diagnosis["schema"] == "phaseflow.angular_diagnostics.v2"
        files = {path: path.read_bytes() for path in trial.rglob("*") if path.is_file()}
        report = sweep_module.replot_log_loss_sweep(
            reference, output, output / "replots" / "test", device="cpu",
        )
        assert report["status"] == "complete" and report["optimizer_updates"] == 0
        assert all(path.read_bytes() == contents for path, contents in files.items())
        with pytest.raises(ValueError, match="disjoint"):
            sweep_module.replot_log_loss_sweep(reference, output, trial / "plots2", device="cpu")
