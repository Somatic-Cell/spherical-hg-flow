"""Coordinate, measure and rendering checks using explicitly synthetic records."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
from test_rainbow import write_rainbow_fixture

from phaseflow.plotting import (
    RainbowPlotConfig,
    cdf_scatter_coordinates,
    evaluate_rainbow_grid,
    plot_rainbow_comparison,
)
from phaseflow.rainbow import RainbowReference
from phaseflow.sphere_model import SingleConditionSphereFlow, SphereFlowConfig


class DirectionalDensity(torch.nn.Module):
    """Normalized analytic q=(1+0.6*x_nf)/(4*pi) for an independent frame check."""

    def __init__(self, reference, *, device="cpu", invalid_log_pdf=None):
        super().__init__()
        self.register_buffer("anchor", torch.zeros((), dtype=torch.float64, device=device))
        self.hg_g = reference.g
        self.incident_cosine = float(reference.condition[1])
        self.invalid_log_pdf = invalid_log_pdf
        self.calls = []

    @property
    def device(self):
        return self.anchor.device

    @property
    def dtype(self):
        return self.anchor.dtype

    def log_prob(self, outgoing):
        self.calls.append((outgoing.shape, outgoing.device, outgoing.dtype, torch.is_grad_enabled()))
        assert not self.training
        if self.invalid_log_pdf is not None:
            return torch.full_like(outgoing[..., 0], self.invalid_log_pdf)
        return torch.log1p(0.6 * outgoing[..., 0]) - math.log(4 * math.pi)


def test_native_grid_preserves_measure_zeros_orientation_and_model_mode(tmp_path):
    angle = 0.37
    masses, edges, _ = write_rainbow_fixture(tmp_path / "record", frame_rotation=angle)
    with RainbowReference(tmp_path / "record") as record:
        model = DirectionalDensity(record)
        assert model.training
        rng_before = torch.get_rng_state().clone()
        grid = evaluate_rainbow_grid(model, record, batch_size=5)
        assert model.training
        torch.testing.assert_close(rng_before, torch.get_rng_state(), rtol=0, atol=0)
        solid_angles = (4 * np.pi / masses.shape[0]) * np.diff(edges)
        np.testing.assert_allclose(grid.reference_pdf, masses.T / solid_angles[:, None], rtol=1e-14)
        np.testing.assert_array_equal(grid.reference_pdf == 0, masses.T == 0)
        np.testing.assert_allclose(grid.cell_solid_angles, solid_angles, rtol=1e-15)
        # Independent canonical-frame derivation: x_nf = r*sin(phi_source+angle).
        source_phi = -np.pi + (np.arange(4) + 0.5) * np.pi / 2
        u_mid = (edges[:-1] + edges[1:]) / 2
        expected = (
            1 + 0.6 * (2 * np.sqrt(u_mid * (1 - u_mid)))[:, None] * np.sin(source_phi + angle)
        ) / (4 * np.pi)
        np.testing.assert_allclose(grid.nf_pdf, expected, rtol=1e-14)
        np.testing.assert_allclose(grid.theta_edges_degrees, np.rad2deg(np.arccos(1 - 2 * edges)))
        np.testing.assert_array_equal(grid.phi_edges_degrees, [-180, -90, 0, 90, 180])
        assert grid.reference_mass_exact == pytest.approx(1, abs=1e-14)
        assert grid.nf_mass_midpoint_estimate == pytest.approx(1, abs=1e-14)
        assert [call[0] for call in model.calls] == [(5, 3), (5, 3), (2, 3)]
        assert all(call[1].type == "cpu" and call[2] == torch.float64 for call in model.calls)
        assert all(not call[3] for call in model.calls)


def test_cdf_scatter_matches_saved_cell_masses_and_keeps_global_rng(tmp_path):
    masses, edges, _ = write_rainbow_fixture(tmp_path / "record", frame_rotation=-0.29)
    with RainbowReference(tmp_path / "record") as record:
        rng_before = np.random.get_state()
        phi, theta = cdf_scatter_coordinates(record, samples=50000, seed=29)
        rng_after = np.random.get_state()
        assert rng_before[0] == rng_after[0]
        np.testing.assert_array_equal(rng_before[1], rng_after[1])
        assert rng_before[2:] == rng_after[2:]
        again = cdf_scatter_coordinates(record, samples=50000, seed=29)
        np.testing.assert_array_equal(phi, again[0])
        np.testing.assert_array_equal(theta, again[1])
        assert np.all((-180 <= phi) & (phi < 180))
        assert np.all((0 < theta) & (theta < 180))
        # Bin in actual u, not in uniform theta. This also detects a missing or
        # reversed source/NF rotation in the plotted points.
        u = np.sin(np.deg2rad(theta) / 2) ** 2
        observed, _, _ = np.histogram2d(phi, u, bins=[np.linspace(-180, 180, 5), edges])
        np.testing.assert_allclose(observed / len(phi), masses, atol=0.005, rtol=0)
        assert not np.any(observed[masses == 0])


def test_aligned_figures_manifest_and_log_color_range(tmp_path):
    from matplotlib.image import imread

    write_rainbow_fixture(tmp_path / "record")
    with RainbowReference(tmp_path / "record") as record:
        model = DirectionalDensity(record)
        grid = evaluate_rainbow_grid(model, record)
        manifest = plot_rainbow_comparison(
            model,
            record,
            tmp_path / "plots",
            config=RainbowPlotConfig(cdf_samples=1000, dpi=60, write_pdf=True),
            selected_step=17,
        )
        assert model.training
        positive = grid.reference_pdf[grid.reference_pdf > 0]
        assert manifest["color_scale"]["vmin"] == min(positive.min(), grid.nf_pdf.min())
        assert manifest["color_scale"]["vmax"] == max(positive.max(), grid.nf_pdf.max())
        assert manifest["color_scale"]["density_floor"] is None
        assert manifest["color_scale"]["shared_between"] == ["reference_pdf", "nf_pdf"]
        assert manifest["grid"]["zero_reference_cells"] == 6
        assert manifest["grid"]["theta_count"] == 3 and manifest["grid"]["phi_count"] == 4
        assert manifest["selected_step"] == 17
        assert manifest["source_kind"] == "synthetic_fixture"
        assert "Synthetic fixture" in manifest["title"]
        assert "550 nm" in manifest["title"]
        assert manifest["dataset_fingerprint"] == record.fingerprint()
        assert manifest["coordinates"]["y_limits"] == [180.0, 0.0]
        assert manifest["coordinates"]["axes_width_over_height"] == 2
        assert manifest["coordinates"]["equal_area"] is False
        assert manifest["scatter"]["source"] == "stored_cdf_independent_diagnostic_stream"
        assert manifest["scatter"]["stream_id"] >= 100
        np.testing.assert_array_equal(manifest["coordinates"]["source_to_nf"], record.source_to_nf)
        images = [imread(tmp_path / "plots" / f"{name}.png") for name in manifest["files"]]
        assert all(image.shape == (444, 720, 4) for image in images[:3])
        assert images[3].shape == (372, 1260, 4)
        assert all(np.count_nonzero(np.any(image[..., :3] < 0.8, axis=-1)) > 200 for image in images)
        for paths in manifest["files"].values():
            assert len(paths) == 2
            for filename in paths:
                assert (tmp_path / "plots" / filename).stat().st_size > 1000
        loaded = json.loads((tmp_path / "plots/plots.json").read_text())
        assert loaded == manifest
        width, height = manifest["figures"]["individual_size_inches"]
        _, _, ax_width, ax_height = manifest["figures"]["individual_axes_rectangle"]
        assert ax_width * width / (ax_height * height) == pytest.approx(2)


def test_isotropic_constant_plot_has_finite_display_range_without_changing_density(tmp_path):
    write_rainbow_fixture(
        tmp_path / "record", masses=np.ones((1, 1)), u_edges=np.array([0.0, 1.0])
    )
    with RainbowReference(tmp_path / "record") as record:
        model = SingleConditionSphereFlow(
            record.g,
            record.condition[1],
            SphereFlowConfig(num_coupling_layers=2, num_bins=4, hidden_features=(4,)),
            device="cpu",
        )
        model.eval()
        grid = evaluate_rainbow_grid(model, record)
        assert grid.reference_pdf[0, 0] == pytest.approx(1 / (4 * np.pi), abs=1e-15)
        assert grid.nf_pdf[0, 0] == pytest.approx(1 / (4 * np.pi), abs=1e-15)
        result = plot_rainbow_comparison(
            model,
            record,
            tmp_path / "plots",
            config=RainbowPlotConfig(cdf_samples=10, dpi=30),
        )
        assert not model.training
        assert 0 < result["color_scale"]["vmin"] < 1 / (4 * np.pi)
        assert result["color_scale"]["vmax"] > 1 / (4 * np.pi)
        assert result["grid"]["reference_mass_exact"] == pytest.approx(1, abs=1e-14)
        assert result["grid"]["nf_mass_midpoint_estimate"] == pytest.approx(1, abs=1e-14)


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf, -800.0, 800.0])
def test_invalid_nf_grid_fails_without_floor_or_mode_change(tmp_path, bad_value):
    write_rainbow_fixture(tmp_path / "record")
    with RainbowReference(tmp_path / "record") as record:
        model = DirectionalDensity(record, invalid_log_pdf=bad_value)
        with pytest.raises(FloatingPointError, match="NF"):
            evaluate_rainbow_grid(model, record, batch_size=5)
        assert model.training


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cdf_samples": 0},
        {"cdf_samples": True},
        {"seed": -1},
        {"eval_batch_size": 0},
        {"dpi": 0},
        {"write_pdf": 1},
    ],
)
def test_plot_config_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        RainbowPlotConfig(**kwargs)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device is not available")
def test_grid_uses_cuda_for_inference_and_matches_cpu(tmp_path):
    write_rainbow_fixture(tmp_path / "record")
    with RainbowReference(tmp_path / "record") as record:
        cpu = DirectionalDensity(record)
        gpu = DirectionalDensity(record, device="cuda")
        on_cpu = evaluate_rainbow_grid(cpu, record, batch_size=5)
        on_gpu = evaluate_rainbow_grid(gpu, record, batch_size=5)
        assert all(call[1].type == "cuda" for call in gpu.calls)
        np.testing.assert_allclose(on_cpu.nf_pdf, on_gpu.nf_pdf, rtol=2e-14)
