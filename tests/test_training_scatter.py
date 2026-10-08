"""Recorded-pool identity and faithful plots; synthetic CPU fixtures only."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_rainbow import write_rainbow_fixture

from phaseflow.plotting import RainbowPlotConfig, plot_rainbow_comparison
from phaseflow.rainbow import RainbowReference
from phaseflow.single_condition import (
    SingleTrainingConfig,
    _array_hash,
    _points,
    load_single_checkpoint,
    read_single_checkpoint,
    train_single_condition,
)
from phaseflow.sphere_model import SingleConditionSphereFlow, SphereFlowConfig
from phaseflow.training_scatter import resolve_training_scatter


def _configuration(**changes):
    return replace(
        SingleTrainingConfig(
            device="cpu", dtype="float32", seed=11, data_seed=37,
            train_samples=137, validation_samples=32, test_samples=32,
            proposal_samples=0, steps=2, batch_size=16, eval_every=1,
            checkpoint_every=1, eval_batch_size=64, log_every=1,
        ),
        **changes,
    )


def _model_config(*, geometry_dtype="model"):
    return SphereFlowConfig(
        num_coupling_layers=2, num_bins=4, hidden_features=(4,),
        geometry_dtype=geometry_dtype,
    )


@pytest.fixture
def trained(tmp_path):
    write_rainbow_fixture(tmp_path / "record", frame_rotation=0.37)
    with RainbowReference(tmp_path / "record") as record:
        cfg = _configuration()
        run = tmp_path / "run"
        result = train_single_condition(record, _model_config(), cfg, run, make_plots=False)
        yield record, cfg, run, result


def test_automatic_training_plot_uses_complete_actual_pool_and_data_seed(tmp_path, monkeypatch):
    from matplotlib.axes import Axes

    write_rainbow_fixture(tmp_path / "record", frame_rotation=-0.29)
    observed = []
    original = Axes.scatter

    def record_scatter(ax, x, y, *args, **kwargs):
        observed.append((np.asarray(x).copy(), np.asarray(y).copy()))
        return original(ax, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", record_scatter)
    with RainbowReference(tmp_path / "record") as record:
        cfg = _configuration()
        run = tmp_path / "run"
        result = train_single_condition(
            record, _model_config(), cfg, run,
            plot_config=RainbowPlotConfig(cdf_samples=7, seed=99, dpi=30),
        )
        manifest = json.loads((run / "plots/plots.json").read_text())
        scatter = manifest["scatter"]
        assert scatter["sample_count"] == scatter["displayed_sample_count"] == 137
        assert scatter["seed"] == 11 and scatter["data_seed"] == 37
        assert scatter["stream_id"] == 0 and scatter["hash_verified"]
        assert scatter["downsampling"] is False
        assert manifest["configuration_usage"]["cdf_samples"] == "unused"
        assert manifest["configuration_usage"]["seed"] == "unused"
        points = _points(record, 137, 37, "train")
        assert scatter["training_points_sha256"] == _array_hash(*points)
        assert scatter["training_points_sha256"] != _array_hash(*_points(record, 137, 11, "train"))
        with np.load(run / "plots/training_scatter.npz", allow_pickle=False) as saved:
            expected = torch.as_tensor(points[0], dtype=torch.float32).numpy()
            np.testing.assert_array_equal(saved["directions_nf"], expected)
            assert saved["directions_nf"].dtype == np.float32
            assert saved["directions_nf"].shape == (137, 3)
            source = expected.astype(np.float64) @ record.source_to_nf
            phi = (np.arctan2(source[:, 1], source[:, 0]) + np.pi) % (2 * np.pi) - np.pi
            theta = np.arctan2(np.hypot(source[:, 0], source[:, 1]), source[:, 2])
            np.testing.assert_array_equal(saved["source_phi_degrees"], np.rad2deg(phi))
            np.testing.assert_array_equal(saved["theta_degrees"], np.rad2deg(theta))
            assert json.loads(saved["provenance_json"].item()) == scatter
            assert len(observed) == 2
            for x, y in observed:
                np.testing.assert_array_equal(x, saved["source_phi_degrees"])
                np.testing.assert_array_equal(y, saved["theta_degrees"])
            expected_band = np.count_nonzero(
                (saved["theta_degrees"] >= 120) & (saved["theta_degrees"] <= 150)
            )
            assert scatter["band"]["observed_train_samples"] == expected_band
        assert manifest["selected_step"] == result.metrics["selected_step"]


@pytest.mark.parametrize("geometry_dtype,expected_dtype", [("model", np.float32), ("float64", np.float64)])
def test_geometry_precision_matches_saved_training_mode(tmp_path, geometry_dtype, expected_dtype):
    write_rainbow_fixture(tmp_path / "record", frame_rotation=0.13)
    with RainbowReference(tmp_path / "record") as record:
        cfg = _configuration(data_seed=None, steps=0)
        result = train_single_condition(
            record, _model_config(geometry_dtype=geometry_dtype), cfg,
            tmp_path / "run", make_plots=False,
        )
        pool = resolve_training_scatter(result.model, record, result.best_path)
        assert pool.directions_nf.dtype == expected_dtype
        expected = _points(record, cfg.train_samples, cfg.seed, "train")[0].astype(expected_dtype)
        np.testing.assert_array_equal(pool.directions_nf, expected)
        assert pool.provenance["data_seed"] == cfg.seed


def test_largest_requested_pool_has_no_display_cap(tmp_path, monkeypatch):
    from matplotlib.axes import Axes

    write_rainbow_fixture(tmp_path / "record")
    counts = []
    original = Axes.scatter

    def count_scatter(ax, x, y, *args, **kwargs):
        counts.append((len(x), len(y)))
        return original(ax, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", count_scatter)
    with RainbowReference(tmp_path / "record") as record:
        cfg = _configuration(train_samples=262144, steps=0)
        result = train_single_condition(
            record, _model_config(), cfg, tmp_path / "run", make_plots=False
        )
        manifest = plot_rainbow_comparison(
            result.model, record, tmp_path / "plots", checkpoint_path=result.best_path,
            config=RainbowPlotConfig(cdf_samples=3, dpi=20),
        )
        assert counts == [(262144, 262144), (262144, 262144)]
        assert manifest["scatter"]["sample_count"] == 262144
        with np.load(tmp_path / "plots/training_scatter.npz", allow_pickle=False) as arrays:
            assert arrays["directions_nf"].shape == (262144, 3)


def test_fp32_polar_angles_retain_the_nonzero_transverse_components(tmp_path):
    masses = np.tile(np.array([0.5, 0.2, 0.3]) / 4, (4, 1))
    write_rainbow_fixture(
        tmp_path / "record", masses=masses, u_edges=np.array([0, 1e-12, 0.3, 1.0]),
        frame_rotation=0.4,
    )
    with RainbowReference(tmp_path / "record") as record:
        result = train_single_condition(
            record, _model_config(), _configuration(steps=0), tmp_path / "run", make_plots=False
        )
        pool = resolve_training_scatter(result.model, record, result.best_path)
        rounded_z = pool.directions_nf[:, 2] == 1
        assert rounded_z.any()
        assert np.all(np.hypot(pool.directions_nf[rounded_z, 0], pool.directions_nf[rounded_z, 1]) > 0)
        assert np.all(pool.theta_degrees[rounded_z] > 0)
        assert np.all(pool.theta_degrees[rounded_z] < 0.001)


@pytest.mark.parametrize("damage", ["config", "split", "hash", "seed", "best_step", "best_state", "validation"])
def test_inconsistent_saved_provenance_is_rejected_before_output(trained, tmp_path, damage):
    record, _, run, result = trained
    checkpoint = read_single_checkpoint(result.checkpoint_path)
    best = read_single_checkpoint(result.best_path)
    config = json.loads((run / "config.json").read_text())
    split = json.loads((run / "sample_split.json").read_text())
    if damage == "config":
        config["training"]["train_samples"] += 1
    elif damage == "split":
        split["train_samples"] += 1
    elif damage == "hash":
        split["training_points_sha256"] = "0" * 64
        checkpoint["sample_split"] = split
    elif damage == "seed":
        config["training"]["data_seed"] += 1
        checkpoint["training_config"] = config["training"]
        split["data_seed"] = config["training"]["data_seed"]
        checkpoint["sample_split"] = split
    elif damage == "best_step":
        best["global_step"] += 1
    elif damage == "best_state":
        key = next(key for key, value in best["model_state"].items() if isinstance(value, torch.Tensor))
        best["model_state"][key] = best["model_state"][key] + 0.1
    else:
        best["validation"]["nll"] += 1
    (run / "config.json").write_text(json.dumps(config))
    (run / "sample_split.json").write_text(json.dumps(split))
    torch.save(checkpoint, result.checkpoint_path)
    torch.save(best, result.best_path)
    output = tmp_path / "invalid_plots"
    with pytest.raises(ValueError):
        plot_rainbow_comparison(result.model, record, output, checkpoint_path=result.best_path)
    assert not output.exists()


def test_wrong_model_reference_dtype_and_missing_run_rejected(trained, tmp_path):
    record, _, _, result = trained
    with pytest.raises(ValueError, match="requires a checkpoint"):
        plot_rainbow_comparison(result.model, record, tmp_path / "no_context")
    assert not (tmp_path / "no_context").exists()
    with pytest.raises(ValueError, match="selected_step"):
        plot_rainbow_comparison(
            result.model, record, tmp_path / "wrong_step", checkpoint_path=result.best_path,
            selected_step=999,
        )
    changed_model, _ = load_single_checkpoint(result.best_path, device="cpu")
    with torch.no_grad():
        next(changed_model.parameters()).add_(0.1)
    with pytest.raises(ValueError, match="model state"):
        resolve_training_scatter(changed_model, record, result.best_path)
    changed_model, _ = load_single_checkpoint(result.best_path, device="cpu")
    changed_model.double()
    with pytest.raises(ValueError, match="dtype"):
        resolve_training_scatter(changed_model, record, result.best_path)
    write_rainbow_fixture(tmp_path / "different_record", frame_rotation=0.38)
    with RainbowReference(tmp_path / "different_record") as different:
        with pytest.raises(ValueError, match="reference record"):
            resolve_training_scatter(result.model, different, result.best_path)
    copied = tmp_path / "copied.pt"
    copied.write_bytes(result.best_path.read_bytes())
    with pytest.raises(ValueError, match="saved checkpoint"):
        resolve_training_scatter(result.model, record, copied)
    verified = resolve_training_scatter(result.model, record, copied, run_directory=result.best_path.parent)
    assert verified.provenance["model_state_verified"]


def test_read_only_regeneration_allows_changed_runtime_and_preserves_global_rng(trained):
    record, _, run, result = trained
    for path in (result.best_path, result.checkpoint_path):
        payload = read_single_checkpoint(path)
        payload["code_fingerprint"] = "old_training_implementation"
        payload["runtime"] = {"device": "unavailable_original_gpu"}
        torch.save(payload, path)
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    pool = resolve_training_scatter(result.model, record, result.best_path)
    assert pool.provenance["hash_verified"]
    torch.testing.assert_close(torch_state, torch.get_rng_state(), rtol=0, atol=0)
    now = np.random.get_state()
    assert numpy_state[0] == now[0] and numpy_state[2:] == now[2:]
    np.testing.assert_array_equal(numpy_state[1], now[1])
    current, _ = load_single_checkpoint(run / "checkpoint.pt", device="cpu")
    current_pool = resolve_training_scatter(current, record, run / "checkpoint.pt")
    assert current_pool.provenance["plotted_model_scope"].startswith("training_checkpoint")
    np.testing.assert_array_equal(current_pool.directions_nf, pool.directions_nf)


def test_bad_mode_has_no_implicit_diagnostic_fallback(tmp_path):
    write_rainbow_fixture(tmp_path / "record")
    with RainbowReference(tmp_path / "record") as record:
        model = SingleConditionSphereFlow(record.g, record.condition[1], _model_config())
        with pytest.raises(ValueError, match="scatter_mode"):
            plot_rainbow_comparison(model, record, tmp_path / "bad", scatter_mode="automatic")
        with pytest.raises(ValueError, match="only valid"):
            plot_rainbow_comparison(
                model, record, tmp_path / "bad", scatter_mode="independent",
                training_run_directory=tmp_path,
            )
