"""All-uniform query provenance and plots; synthetic CPU fixtures only."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from test_rainbow import write_rainbow_fixture
from test_training_scatter import _configuration, _model_config

from phaseflow.angular_diagnostics import AngularDiagnosticConfig, evaluate_angular_diagnostics
from phaseflow.log_objective import LogObjectiveConfig, make_training_pool
from phaseflow.plotting import RainbowPlotConfig, plot_rainbow_comparison
from phaseflow.rainbow import RainbowReference
from phaseflow.single_condition import train_single_condition


def _positive_record(path):
    masses = np.array([[1, 2, 3], [2, 1, 1], [3, 1, 2], [1, 2, 4]], dtype=np.float64)
    masses /= masses.sum()
    write_rainbow_fixture(
        path, masses=masses, u_edges=np.array([0, 0.01, 0.78, 1]), frame_rotation=0.37,
    )


def _objective():
    return LogObjectiveConfig(
        schema_version=2, nll_weight=0, beta=1, sampling="uniform", target_fraction=0,
        validation_uniform_samples=32, test_uniform_samples=32,
    )


@pytest.mark.parametrize("count", [137, 262144])
def test_uniform_plot_uses_every_actual_query_without_cdf_sampling(
    tmp_path, monkeypatch, count,
):
    from matplotlib.axes import Axes
    from matplotlib.image import imread

    _positive_record(tmp_path / "record")
    seen = []
    original = Axes.scatter

    def observe(ax, x, y, *args, **kwargs):
        seen.append((np.asarray(x).copy(), np.asarray(y).copy(), kwargs["label"]))
        return original(ax, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", observe)
    with RainbowReference(tmp_path / "record") as reference:
        cfg, objective = _configuration(train_samples=count, steps=0), _objective()
        run, plots = tmp_path / "run", tmp_path / "plots"
        result = train_single_condition(
            reference, _model_config(), cfg, run, objective_config=objective, make_plots=False,
        )
        original_run_hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in run.iterdir() if path.is_file()
        }

        def reject_target_sampling(*args, **kwargs):
            raise AssertionError("All-uniform training-scatter regeneration must not sample the CDF")

        # Target-distributed validation/test points are allowed during training.
        # Regenerating this training pool and its maps must need no CDF samples.
        monkeypatch.setattr(reference, "sample", reject_target_sampling)
        manifest = plot_rainbow_comparison(
            result.model, reference, plots, checkpoint_path=result.best_path,
            config=RainbowPlotConfig(cdf_samples=1, dpi=30),
        )
        pool = make_training_pool(
            reference, count, cfg.data_seed, objective, result.model.geometry_dtype,
        )
        scatter = manifest["scatter"]
        assert scatter["sampling_distribution"] == "uniform"
        assert scatter["component_counts"] == {"cdf": 0, "uniform": count}
        assert scatter["source_streams"] == {"cdf": None, "uniform": 5}
        assert scatter["sample_count"] == scatter["displayed_sample_count"] == count
        assert scatter["downsampling"] is False and scatter["hash_verified"] is True
        assert scatter["training_pool"] == pool.provenance
        assert scatter["band"]["observed_component_samples"]["cdf"] == 0
        assert scatter["band"]["observed_component_samples"]["uniform"] == (
            scatter["band"]["observed_train_samples"]
        )
        assert manifest["objective"] == objective.to_dict()
        assert manifest["configuration_usage"]["cdf_samples"] == "unused"
        assert manifest["figures"]["panel_titles"][1] == (
            f"Spherical-uniform training pool (N = {count:,})"
        )
        assert "Every shown point is an actual spherical-uniform" in (
            manifest["figures"]["scatter_chart_note"]
        )
        assert "u_per_sr=1/(4*pi)" in manifest["coordinates"]["scatter_chart_density"]
        assert manifest["coordinates"]["frame"] == "recorded_solver_source"
        assert manifest["coordinates"]["axes_width_over_height"] == 2
        assert manifest["coordinates"]["equal_area"] is False
        assert manifest["grid"]["theta_count"] == reference.n_theta
        assert manifest["grid"]["phi_count"] == reference.n_phi
        assert manifest["grid"]["resolution"] == "full_native_cdf_cells_no_angular_downsampling"
        assert manifest["color_scale"]["shared_between"] == ["reference_pdf", "nf_pdf"]
        assert manifest["color_scale"]["normalization"] == "matplotlib.colors.LogNorm"
        assert manifest["color_scale"]["density_floor"] is None
        with np.load(plots / "training_scatter.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["directions_nf"], pool.directions)
            np.testing.assert_array_equal(archive["teacher_log_pdf"], reference.log_prob(pool.directions))
            assert archive["directions_nf"].dtype == np.float32
            assert archive["components"].dtype == np.uint8
            assert np.all(archive["components"] == 1)
            assert json.loads(archive["provenance_json"].item()) == scatter
            source = pool.directions.astype(np.float64) @ reference.source_to_nf
            phi = (np.arctan2(source[:, 1], source[:, 0]) + np.pi) % (2 * np.pi) - np.pi
            theta = np.arctan2(np.hypot(source[:, 0], source[:, 1]), source[:, 2])
            np.testing.assert_array_equal(archive["source_phi_degrees"], np.rad2deg(phi))
            np.testing.assert_array_equal(archive["theta_degrees"], np.rad2deg(theta))
            assert len(seen) == 2  # Individual and combined figure, one collection per figure.
            for x, y, label in seen:
                assert len(x) == len(y) == count
                np.testing.assert_array_equal(x, archive["source_phi_degrees"])
                np.testing.assert_array_equal(y, archive["theta_degrees"])
                assert label == f"Spherical uniform (N = {count:,})"
        assert original_run_hashes == {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in run.iterdir() if path.is_file()
        }
    assert set(manifest["files"]) == {"reference_pdf", "training_samples", "nf_pdf", "comparison"}
    assert not (plots / "cdf_samples.png").exists()
    shapes = []
    for stem in ("reference_pdf", "training_samples", "nf_pdf"):
        pixels = imread(plots / f"{stem}.png")
        assert np.isfinite(pixels).all() and np.ptp(pixels[..., :3]) > 0.5
        shapes.append(pixels.shape)
    assert shapes[0] == shapes[1] == shapes[2]


def test_uniform_angular_counts_use_solid_angle_instead_of_teacher_mass(tmp_path):
    from matplotlib.image import imread

    _positive_record(tmp_path / "record")
    with RainbowReference(tmp_path / "record") as reference:
        cfg, objective = _configuration(train_samples=137, steps=0), _objective()
        result = train_single_condition(
            reference, _model_config(), cfg, tmp_path / "run",
            objective_config=objective, make_plots=False,
        )
        report = evaluate_angular_diagnostics(
            result.model, reference, tmp_path / "diagnostics", train_samples=cfg.train_samples,
            batch_size=cfg.batch_size, objective_config=objective,
            checkpoint_path=result.best_path, config=AngularDiagnosticConfig(dpi=30),
        )
    # A full-azimuth spherical band's area fraction is (cos(low)-cos(high))/2.
    expected_mass = (np.cos(np.deg2rad(120)) - np.cos(np.deg2rad(150))) / 2
    assert abs(report["reference_band_mass_exact"] - expected_mass) > 0.01
    assert report["uniform_band_mass_exact"] == pytest.approx(expected_mass)
    assert report["training_query_band_mass"] == pytest.approx(expected_mass)
    assert report["expected_train_band_samples"] == pytest.approx(cfg.train_samples * expected_mass)
    assert report["expected_uniform_train_band_samples"] == pytest.approx(
        cfg.train_samples * expected_mass
    )
    assert report["expected_minibatch_band_samples"] == pytest.approx(cfg.batch_size * expected_mass)
    assert report["expected_cdf_train_band_samples"] == 0
    counts = report["expected_counts"]
    assert counts["sampling"] == "uniform" and counts["target_fraction"] == 0
    assert counts["cdf_samples"] == 0 and counts["uniform_samples"] == cfg.train_samples
    assert counts["observed_train_band_samples"] is None
    assert counts["method"].startswith("N*u(R) and B*u(R)")
    pixels = imread(tmp_path / "diagnostics/angular_profiles.png")
    assert np.isfinite(pixels).all() and np.ptp(pixels[..., :3]) > 0.5
