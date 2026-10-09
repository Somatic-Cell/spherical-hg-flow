"""Actual mixed-pool coverage and independently selected v5 inference identity."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_rainbow import write_rainbow_fixture
from test_training_scatter import _configuration, _model_config

from phaseflow.log_objective import LogObjectiveConfig, make_training_pool
from phaseflow.plotting import RainbowPlotConfig, plot_rainbow_comparison
from phaseflow.rainbow import RainbowReference
from phaseflow.single_condition import (
    load_single_checkpoint,
    read_single_checkpoint,
    train_single_condition,
)
from phaseflow.training_scatter import resolve_training_scatter


def _positive_record(path):
    masses = np.array([[1, 2, 3], [2, 1, 1], [3, 1, 2], [1, 2, 4]], dtype=np.float64)
    masses /= masses.sum()
    write_rainbow_fixture(path, masses=masses, frame_rotation=0.37)


def _objective(**changes):
    return replace(
        LogObjectiveConfig(validation_uniform_samples=32, test_uniform_samples=32),
        **changes,
    )


@pytest.fixture
def log_trained(tmp_path):
    _positive_record(tmp_path / "record")
    with RainbowReference(tmp_path / "record") as reference:
        cfg = _configuration(train_samples=138)
        objective = _objective()
        run = tmp_path / "run"
        result = train_single_condition(
            reference, _model_config(), cfg, run,
            objective_config=objective, make_plots=False,
        )
        yield reference, cfg, objective, run, result


@pytest.mark.parametrize("count", [138, 262144])
def test_mixed_plot_displays_every_component_point_and_archives_actual_labels(
    tmp_path, monkeypatch, count,
):
    from matplotlib.axes import Axes

    _positive_record(tmp_path / "record")
    seen = []
    original = Axes.scatter

    def observe(ax, x, y, *args, **kwargs):
        seen.append((np.asarray(x).copy(), np.asarray(y).copy(), kwargs["label"]))
        return original(ax, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", observe)
    with RainbowReference(tmp_path / "record") as reference:
        cfg = _configuration(train_samples=count, steps=0)
        objective = _objective()
        result = train_single_condition(
            reference, _model_config(), cfg, tmp_path / "run",
            objective_config=objective, make_plots=False,
        )
        manifest = plot_rainbow_comparison(
            result.model, reference, tmp_path / "plots", checkpoint_path=result.best_path,
            config=RainbowPlotConfig(cdf_samples=1, dpi=20),
        )
        pool = make_training_pool(
            reference, count, cfg.data_seed, objective, result.model.geometry_dtype,
        )
        assert manifest["schema"] == "phaseflow.rainbow_plots.v3"
        assert manifest["scatter"]["component_counts"] == {"cdf": count // 2, "uniform": count // 2}
        assert manifest["scatter"]["sample_count"] == count
        assert manifest["scatter"]["displayed_sample_count"] == count
        assert manifest["scatter"]["downsampling"] is False
        assert manifest["scatter"]["training_pool"] == pool.provenance
        assert "mixed training queries" in manifest["coordinates"]["scatter_chart_density"]
        assert manifest["selection_metric"] == "log_rmse"
        assert len(seen) == 4
        with np.load(tmp_path / "plots/training_scatter.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["directions_nf"], pool.directions)
            np.testing.assert_array_equal(archive["components"], pool.components)
            np.testing.assert_array_equal(archive["teacher_log_pdf"], pool.log_p)
            np.testing.assert_array_equal(archive["teacher_log_pdf"], reference.log_prob(pool.directions))
            assert archive["directions_nf"].dtype == np.float32
            assert archive["components"].dtype == np.uint8
            for index, (x, y, label) in enumerate(seen):
                code = index % 2
                mask = archive["components"] == code
                np.testing.assert_array_equal(x, archive["source_phi_degrees"][mask])
                np.testing.assert_array_equal(y, archive["theta_degrees"][mask])
                assert len(x) == count // 2
                assert ("CDF" if code == 0 else "Spherical uniform") in label


def test_dual_selection_is_verified_against_its_own_saved_state(log_trained):
    reference, _, _, run, _ = log_trained
    training = read_single_checkpoint(run / "checkpoint.pt")
    for filename, metric in (("best.pt", "log_rmse"), ("best_by_log.pt", "log_rmse"),
                             ("best_by_nll.pt", "nll")):
        path = run / filename
        model, payload = load_single_checkpoint(path, device="cpu")
        points = resolve_training_scatter(model, reference, path)
        assert points.provenance["plotted_checkpoint"]["selection_metric"] == metric
        assert payload["global_step"] == training["selections"][metric]["step"]

    # A damaged secondary selection must be caught even if the primary remains
    # valid and the passed inference model matches its own file exactly.
    secondary = training["selections"]["nll"]
    state_key = next(
        key for key, value in secondary["model_state"].items() if isinstance(value, torch.Tensor)
    )
    secondary["model_state"][state_key] = secondary["model_state"][state_key] + 1
    torch.save(training, run / "checkpoint.pt")
    model, _ = load_single_checkpoint(run / "best_by_nll.pt", device="cpu")
    with pytest.raises(ValueError, match="best state"):
        resolve_training_scatter(model, reference, run / "best_by_nll.pt")


@pytest.mark.parametrize("damage", ["raw_hash", "runtime_hash", "components", "objective", "selection"])
def test_mixed_identity_damage_is_rejected_before_creating_plots(log_trained, tmp_path, damage):
    reference, _, _, run, result = log_trained
    training = read_single_checkpoint(run / "checkpoint.pt")
    inference = read_single_checkpoint(result.best_path)
    split = json.loads((run / "sample_split.json").read_text())
    if damage in ("raw_hash", "runtime_hash"):
        split["training_pool"][f"{damage.removesuffix('_hash')}_points_sha256"] = "0" * 64
    elif damage == "components":
        split["training_pool"]["component_counts"]["uniform"] += 1
    elif damage == "objective":
        inference["objective_config"]["beta"] += 1
    else:
        metric = inference["selection_metric"]
        training["selections"][metric]["validation"]["log_rmse"] += 1
    training["sample_split"] = split
    (run / "sample_split.json").write_text(json.dumps(split))
    torch.save(training, run / "checkpoint.pt")
    torch.save(inference, result.best_path)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        plot_rainbow_comparison(result.model, reference, output, checkpoint_path=result.best_path)
    assert not output.exists()
