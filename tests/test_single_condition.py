"""Single-record end-to-end tests using an analytic synthetic cell teacher.

The fixtures verify the learning pipeline, not rainbow optics or solver accuracy.
"""

from __future__ import annotations

import json
from dataclasses import replace
from functools import partial

import numpy as np
import pytest
import torch
from test_rainbow import write_rainbow_fixture

from phaseflow.cli import main
from phaseflow.export import export_model
from phaseflow.rainbow import RainbowReference
from phaseflow.single_condition import (
    FAMILY,
    SingleTrainingConfig,
    evaluate_single_condition,
    load_single_checkpoint,
    read_single_checkpoint,
    train_single_condition,
)
from phaseflow.sphere_model import SingleConditionSphereFlow, SphereFlowConfig
from phaseflow.training import load_model_checkpoint

# Most numerical tests do not need four raster figures for every short run.
# The CLI integration below exercises automatic plotting with the real default.
train_single_condition = partial(train_single_condition, make_plots=False)


def smooth_teacher(path, *, isotropic=False):
    """Exact cell integrals of (1+b mu)(1-a cos(2 phi_source))/(4 pi)."""
    edges = (1 - np.cos(np.linspace(0, np.pi, 17))) / 2
    phi = np.linspace(-np.pi, np.pi, 25)
    a, b = (0.0, 0.0) if isotropic else (0.65, 0.6)
    mu_mass = np.diff(edges) * (1 + b * (1 - edges[:-1] - edges[1:]))
    phi_mass = (np.diff(phi) - a * np.diff(np.sin(2 * phi)) / 2) / (2 * np.pi)
    mass = phi_mass[:, None] * mu_mass[None, :]
    write_rainbow_fixture(path, masses=mass, u_edges=edges)
    return path


def small_config(**changes):
    config = SingleTrainingConfig(
        device="cpu",
        dtype="float64",
        train_samples=2048,
        validation_samples=2048,
        test_samples=2048,
        proposal_samples=512,
        batch_size=256,
        steps=8,
        learning_rate=0.003,
        eval_every=4,
        checkpoint_every=4,
        eval_batch_size=512,
    )
    return replace(config, **changes)


def small_model():
    return SphereFlowConfig(num_coupling_layers=2, num_bins=8, hidden_features=(16, 16))


def assert_state_equal(left, right):
    assert left.keys() == right.keys()
    for name in left:
        if isinstance(left[name], torch.Tensor):
            torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
        elif isinstance(left[name], dict):
            assert_state_equal(left[name], right[name])
        else:
            assert left[name] == right[name]


def test_training_improves_independent_kl_and_preserves_external_g(tmp_path):
    with RainbowReference(smooth_teacher(tmp_path / "record")) as record:
        result = train_single_condition(
            record, small_model(), small_config(steps=100, eval_every=20), tmp_path / "run"
        )
        initial, test = result.metrics["initial_validation"], result.metrics["test"]
        assert result.complete and result.global_step == 100
        assert test["scope"] == "final_independent_test_points_same_condition"
        assert test["forward_kl_estimate"] < initial["forward_kl_estimate"] - 0.07
        assert test["nll_improvement_over_hg"] > 0.07
        assert test["base_g"] == record.g
        assert 0 < test["proposal"]["relative_ess"] <= 1.00000000001
        assert test["proposal"]["sample_eval_log_pdf_max_abs_error"] < 1e-9
        restored, checkpoint = load_single_checkpoint(result.best_path, device="cpu")
        assert_state_equal(restored.state_dict(), result.model.state_dict())
        assert checkpoint["kind"] == "inference"
        assert checkpoint["global_step"] == result.metrics["selected_step"]
        split = json.loads((tmp_path / "run/sample_split.json").read_text())
        assert split["training_points_sha256"] != split["validation_points_sha256"]
        assert "g_head" not in " ".join(restored.state_dict().keys())


def test_controlled_resume_exact_and_does_not_use_test_early(tmp_path):
    with RainbowReference(smooth_teacher(tmp_path / "record")) as record:
        cfg = small_config(steps=12)
        full = train_single_condition(record, small_model(), cfg, tmp_path / "full")
        partial = train_single_condition(
            record, small_model(), cfg, tmp_path / "resumed", max_steps_this_run=5
        )
        assert not partial.complete and partial.global_step == 5
        assert partial.metrics["test"] is None
        payload = read_single_checkpoint(partial.checkpoint_path)
        assert [e["global_step"] for e in payload["history"] if "validation" in e] == [0, 4]
        continued = train_single_condition(
            record, small_model(), cfg, tmp_path / "resumed", resume=partial.checkpoint_path
        )
        a, b = (
            read_single_checkpoint(full.checkpoint_path),
            read_single_checkpoint(continued.checkpoint_path),
        )
        assert_state_equal(a["model_state"], b["model_state"])
        assert_state_equal(a["best_state"], b["best_state"])
        assert a["history"] == b["history"]
        assert full.metrics == continued.metrics
        assert torch.equal(a["minibatch_rng_state"], b["minibatch_rng_state"])
        with pytest.raises(ValueError, match="inference-only"):
            train_single_condition(
                record, small_model(), cfg, tmp_path / "invalid", resume=continued.best_path
            )


def test_resume_rejects_changed_plan_and_record_without_overwriting(tmp_path):
    with RainbowReference(smooth_teacher(tmp_path / "record")) as record:
        cfg = small_config(steps=2)
        result = train_single_condition(record, small_model(), cfg, tmp_path / "run")
        before = result.checkpoint_path.read_bytes()
        with pytest.raises(ValueError, match="original model and training"):
            train_single_condition(
                record,
                small_model(),
                replace(cfg, train_samples=100),
                tmp_path / "run",
                resume=result.checkpoint_path,
            )
        with pytest.raises(FileExistsError):
            train_single_condition(record, small_model(), cfg, tmp_path / "run")
        with RainbowReference(smooth_teacher(tmp_path / "other", isotropic=True)) as other:
            with pytest.raises(ValueError, match="record differs"):
                train_single_condition(
                    other, small_model(), cfg, tmp_path / "run", resume=result.checkpoint_path
                )
        assert result.checkpoint_path.read_bytes() == before


def test_uniform_reference_standard_metrics_and_rng_independence(tmp_path):
    torch.set_num_threads(1)
    with RainbowReference(smooth_teacher(tmp_path / "record", isotropic=True)) as record:
        model = SingleConditionSphereFlow(record.g, record.condition[1], small_model())
        before = torch.get_rng_state().clone()
        report = evaluate_single_condition(model, record, samples=1024, batch_size=113, seed=19)
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        assert report["nll"] == pytest.approx(np.log(4 * np.pi), abs=1e-13)
        assert abs(report["forward_kl_estimate"]) < 1e-13
        assert report["proposal"]["relative_ess"] == pytest.approx(1, abs=1e-13)
        assert report["proposal"]["importance_weight_mean"] == pytest.approx(1, abs=1e-13)
        again = evaluate_single_condition(model, record, samples=1024, batch_size=113, seed=19)
        assert report == again


def test_float32_proposal_uniforms_are_representable_and_g_is_preserved(tmp_path):
    with RainbowReference(smooth_teacher(tmp_path / "record")) as record:
        model = SingleConditionSphereFlow(
            record.g, record.condition[1], small_model(), dtype=torch.float32
        )
        report = evaluate_single_condition(model, record, samples=128, batch_size=64)
        assert report["base_g"] == record.g
        assert report["proposal"]["uniform_midpoint_bits"] == 23


def test_cli_inspect_train_evaluate_and_native_export_guard(tmp_path, capsys):
    record_path = smooth_teacher(tmp_path / "record")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "family": FAMILY,
                "model": small_model().to_dict(),
                "training": small_config(steps=2).to_dict(),
                "visualization": {"cdf_samples": 128, "dpi": 80},
            }
        )
    )
    assert main(["inspect-rainbow", "--record", str(record_path)]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["condition"][0] == 550.0
    run = tmp_path / "run"
    assert (
        main(
            [
                "train-rainbow",
                "--record",
                str(record_path),
                "--config",
                str(config_path),
                "--output",
                str(run),
                "--quiet",
            ]
        )
        == 0
    )
    trained = json.loads(capsys.readouterr().out)
    assert trained["complete"] is True
    plot_manifest = json.loads((run / "plots/plots.json").read_text())
    assert plot_manifest["selected_step"] == trained["metrics"]["selected_step"]
    for name in ("reference_pdf.png", "cdf_samples.png", "nf_pdf.png", "comparison.png"):
        assert (run / "plots" / name).is_file()
    report_path = tmp_path / "evaluation.json"
    assert (
        main(
            [
                "evaluate-rainbow",
                "--record",
                str(record_path),
                "--checkpoint",
                str(run / "best.pt"),
                "--samples",
                "128",
                "--device",
                "cpu",
                "--seed",
                "19",
                "--output",
                str(report_path),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report == json.loads(report_path.read_text())
    assert report["sample_count"] == 128 and report["scope"] == "independent_points_same_condition"
    with pytest.raises(ValueError, match="Native/OptiX export"):
        load_model_checkpoint(run / "best.pt")
    model, _ = load_single_checkpoint(run / "best.pt", device="cpu")
    with pytest.raises(ValueError, match="new native format"):
        export_model(model, tmp_path / "unsupported_export")
    assert not (tmp_path / "unsupported_export").exists()
    assert main([
        "plot-rainbow", "--record", str(record_path), "--checkpoint", str(run / "best.pt"),
        "--output", str(tmp_path / "replot"), "--device", "cpu",
        "--dpi", "80",
    ]) == 0
    replotted = json.loads(capsys.readouterr().out)
    assert replotted["selected_step"] == trained["metrics"]["selected_step"]
    assert replotted["scatter"]["sample_count"] == small_config().train_samples
    # A display count must never silently replace the actual training pool.
    rejected = tmp_path / "wrong_plot_count"
    with pytest.raises(ValueError, match="--samples and --seed require --scatter independent"):
        main([
            "plot-rainbow", "--record", str(record_path), "--checkpoint", str(run / "best.pt"),
            "--output", str(rejected), "--device", "cpu", "--samples", "128",
        ])
    assert not rejected.exists()


def test_nonfinite_update_does_not_overwrite_valid_checkpoint(tmp_path, monkeypatch):
    """A GPU-style deferred finite check must keep the previous saved boundary."""
    with RainbowReference(smooth_teacher(tmp_path / "record")) as record:
        cfg = small_config(steps=8)
        output = tmp_path / "run"
        partial_run = train_single_condition(
            record, small_model(), cfg, output, max_steps_this_run=4
        )
        before = partial_run.checkpoint_path.read_bytes()
        original_step = torch.optim.Adam.step

        def invalid_step(optimizer, *args, **kwargs):
            result = original_step(optimizer, *args, **kwargs)
            with torch.no_grad():
                optimizer.param_groups[0]["params"][0].fill_(float("nan"))
            return result

        monkeypatch.setattr(torch.optim.Adam, "step", invalid_step)
        with pytest.raises(FloatingPointError, match="nonfinite.*update"):
            train_single_condition(
                record, small_model(), cfg, output, resume=partial_run.checkpoint_path
            )
        assert before == partial_run.checkpoint_path.read_bytes()


def test_incomplete_training_does_not_create_final_plots(tmp_path):
    with RainbowReference(smooth_teacher(tmp_path / "record")) as record:
        output = tmp_path / "run"
        result = train_single_condition(
            record, small_model(), small_config(), output, max_steps_this_run=1, make_plots=True
        )
        assert not result.complete and not (output / "plots").exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_samples": 0},
        {"validation_samples": 1},
        {"test_samples": 0},
        {"proposal_samples": 1},
        {"steps": -1},
        {"seed": True},
        {"dtype": "float16"},
        {"learning_rate": 0},
        {"grad_clip_norm": -1},
        {"eval_every": 0},
    ],
)
def test_training_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        small_config(**kwargs)
