"""Saved-cell arithmetic and source-frame diagnostics on synthetic fixtures."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest
import torch
from test_rainbow import write_rainbow_fixture

from phaseflow.angular_diagnostics import (
    AngularDiagnosticConfig,
    _hg_band_probability,
    _reference_band_and_zeros,
    evaluate_angular_diagnostics,
)
from phaseflow.rainbow import RainbowReference


class AnalyticDirectionalDensity(torch.nn.Module):
    """Independent normalized q=(1+0.6*x_nf)/(4*pi), not a solver result."""

    def __init__(self, reference, *, device="cpu", invalid=False):
        super().__init__()
        self.register_buffer("anchor", torch.zeros((), dtype=torch.float64, device=device))
        self.child = torch.nn.Identity()
        self.child.eval()
        self.hg_g = reference.g
        self.incident_cosine = float(reference.condition[1])
        self.calls = []
        self.invalid = invalid

    @property
    def device(self):
        return self.anchor.device

    @property
    def dtype(self):
        return self.anchor.dtype

    def log_prob(self, outgoing):
        assert not self.training and not torch.is_grad_enabled()
        self.calls.append((outgoing.shape, outgoing.device, outgoing.dtype))
        if self.invalid:
            return torch.full_like(outgoing[..., 0], torch.nan)
        return torch.log1p(0.6 * outgoing[..., 0]) - math.log(4 * math.pi)


def _degrees(u):
    return math.degrees(2 * math.asin(math.sqrt(u)))


def test_exact_nonuniform_partial_cell_mass_zero_area_and_expected_counts(tmp_path, monkeypatch):
    monkeypatch.setattr("phaseflow.angular_diagnostics._ZERO_SCAN_ELEMENTS", 2)
    masses = np.array([[0.1, 0, 0], [0.05, 0.1, 0.15], [0.2, 0.05, 0.35], [0, 0, 0]])
    edges = np.array([0.0, 0.1, 0.7, 1.0])
    write_rainbow_fixture(tmp_path / "record", masses=masses, u_edges=edges)
    config = AngularDiagnosticConfig(theta_band_degrees=(_degrees(0.05), _degrees(0.8)), dpi=45)
    with RainbowReference(tmp_path / "record") as reference:
        report = evaluate_angular_diagnostics(
            AnalyticDirectionalDensity(reference),
            reference,
            tmp_path / "diagnostics",
            train_samples=4096,
            batch_size=1024,
            config=config,
        )
    fractions = np.array([0.5, 1.0, 1 / 3])
    expected_mass = float(np.sum(masses * fractions))
    expected_zero_fraction = (0.1 + 2 * 0.6 + 2 * 0.3) / 4
    expected_band_zero_fraction = (0.05 + 2 * 0.6 + 2 * 0.1) / (4 * 0.75)
    assert report["reference_band_mass_exact"] == pytest.approx(expected_mass, abs=5e-16)
    assert report["expected_train_band_samples"] == pytest.approx(4096 * expected_mass)
    assert report["expected_minibatch_band_samples"] == pytest.approx(1024 * expected_mass)
    assert report["reference_zero_solid_angle_fraction"] == pytest.approx(expected_zero_fraction)
    assert report["reference_band_zero_solid_angle_fraction"] == pytest.approx(
        expected_band_zero_fraction
    )
    assert report["band"]["solid_angle_sr"] == pytest.approx(4 * math.pi * 0.75)
    assert report["expected_counts"]["observed_train_band_samples"] is None
    assert report["band"]["nf_band_mass"] is None
    assert report["band"]["conditional_kl"] is None


def test_native_source_cuts_preserve_zero_cells_frame_device_modes_and_rng(tmp_path):
    rotation = 0.37
    masses, edges, _ = write_rainbow_fixture(tmp_path / "record", frame_rotation=rotation)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"synthetic checkpoint identity only; model supplied by test")
    with RainbowReference(tmp_path / "record") as reference:
        model = AnalyticDirectionalDensity(reference)
        torch_before = torch.get_rng_state().clone()
        numpy_before = np.random.get_state()
        report = evaluate_angular_diagnostics(
            model,
            reference,
            tmp_path / "diagnostics",
            train_samples=16384,
            batch_size=1024,
            config=AngularDiagnosticConfig(eval_batch_size=2, dpi=45),
            checkpoint_path=checkpoint,
        )
        assert model.training and not model.child.training
        torch.testing.assert_close(torch_before, torch.get_rng_state(), rtol=0, atol=0)
        numpy_after = np.random.get_state()
        assert numpy_before[0] == numpy_after[0]
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        assert numpy_before[2:] == numpy_after[2:]
        assert all(call[0][0] <= 2 and call[1].type == "cpu" for call in model.calls)
        assert all(call[2] == torch.float64 for call in model.calls)
        np.testing.assert_allclose(report["coordinates"]["source_to_nf"], reference.source_to_nf)
    with np.load(tmp_path / "diagnostics" / "angular_profiles.npz", allow_pickle=False) as profiles:
        np.testing.assert_array_equal(profiles["phi_cell_indices"], [0, 1, 2])
        np.testing.assert_array_equal(profiles["actual_source_phi_degrees"], [-135, -45, 45])
        np.testing.assert_array_equal(profiles["requested_source_phi_degrees"], [-90, 0, 90])
        np.testing.assert_array_equal(np.isneginf(profiles["reference_log_pdf"]), masses[:3] == 0)
        expected_teacher = masses[:3] / (math.pi * np.diff(edges))[None, :]
        np.testing.assert_allclose(
            np.exp(profiles["reference_log_pdf"]), expected_teacher, rtol=1e-14, atol=0
        )
        phi = np.deg2rad([-135, -45, 45])
        u = (edges[:-1] + edges[1:]) / 2
        # Independent canonical frame derivation: x_nf=r*sin(phi_source+rotation).
        expected_nf = (
            1 + 0.6 * np.sin(phi[:, None] + rotation) * (2 * np.sqrt(u * (1 - u)))[None, :]
        ) / (4 * math.pi)
        np.testing.assert_allclose(np.exp(profiles["nf_log_pdf"]), expected_nf, rtol=1e-14)
        np.testing.assert_allclose(
            profiles["theta_edges_degrees"], np.rad2deg(np.arccos(1 - 2 * edges)), atol=3e-14
        )
    assert report["source_kind"] == "synthetic_fixture"
    assert report["checkpoint"]["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert report["profiles"]["density_floor"] is None
    assert report["profiles"]["zero_reference_cells_per_profile"] == [1, 3, 1]
    assert report["scope"] == "report_only_same_condition_no_model_selection"


def test_profile_artifacts_roundtrip_and_have_finite_image_pixels(tmp_path):
    from matplotlib.image import imread

    write_rainbow_fixture(tmp_path / "record", masses=np.ones((1, 1)), u_edges=np.array([0, 1]))
    config = AngularDiagnosticConfig(theta_band_degrees=(0, 180), source_phi_degrees=(180,), dpi=55)
    with RainbowReference(tmp_path / "record") as reference:
        report = evaluate_angular_diagnostics(
            AnalyticDirectionalDensity(reference),
            reference,
            tmp_path / "diagnostics",
            train_samples=65536,
            batch_size=1024,
            config=config,
        )
    assert report["reference_band_mass_exact"] == 1
    assert report["hg_band_probability_width"] == 1
    assert report["reference_zero_solid_angle_fraction"] == 0
    assert report["reference_band_zero_solid_angle_fraction"] == 0
    assert report["profiles"]["actual_source_phi_degrees"] == [0]
    directory = tmp_path / "diagnostics"
    saved = json.loads((directory / "angular_diagnostics.json").read_text(encoding="utf-8"))
    assert saved == report
    for key in ("figure", "arrays"):
        artifact = report["artifacts"][key]
        assert (
            hashlib.sha256((directory / artifact["path"]).read_bytes()).hexdigest()
            == artifact["sha256"]
        )
    picture = imread(directory / "angular_profiles.png")
    assert picture.ndim == 3 and picture.shape[0] > 200 and picture.shape[1] > 200
    assert np.isfinite(picture).all() and np.ptp(picture[..., :3]) > 0.5
    assert not tuple(directory.glob("*.tmp"))


@pytest.mark.parametrize(
    "g", [0.0, 1e-12, -1e-12, 0.8, -0.8, np.nextafter(1.0, 0), np.nextafter(-1.0, 0)]
)
def test_hg_band_probability_against_independent_density_quadrature(g):
    low, high = 0.75, 0.94
    nodes, weights = np.polynomial.legendre.leggauss(64)
    u = low + (nodes + 1) * (high - low) / 2
    denominator_squared = (1 - g) ** 2 + 4 * g * u if g >= 0 else (1 + g) ** 2 - 4 * g * (1 - u)
    density_u = (1 - g) * (1 + g) / denominator_squared**1.5
    integrated = float(np.dot(weights, density_u) * (high - low) / 2)
    assert _hg_band_probability(float(g), low, high) == pytest.approx(integrated, rel=2e-14, abs=0)
    assert _hg_band_probability(float(g), 0, 1) == pytest.approx(1, abs=3e-16)


@pytest.mark.parametrize(
    "arguments",
    [
        {"theta_band_degrees": (150, 120)},
        {"theta_band_degrees": (0, 181)},
        {"theta_band_degrees": (0, float("nan"))},
        {"theta_band_degrees": (0, True)},
        {"source_phi_degrees": ()},
        {"source_phi_degrees": (181,)},
        {"source_phi_degrees": (float("inf"),)},
        {"eval_batch_size": 0},
        {"dpi": True},
    ],
)
def test_invalid_configuration_is_rejected(arguments):
    with pytest.raises(ValueError):
        AngularDiagnosticConfig(**arguments)


def test_configuration_json_roundtrip_and_unknown_key_rejection():
    config = AngularDiagnosticConfig()
    assert AngularDiagnosticConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    with pytest.raises(ValueError, match="configuration"):
        AngularDiagnosticConfig.from_dict({"loss": "logarithmic"})


def test_tiny_partial_band_mass_near_cdf_one_is_not_lost(tmp_path):
    tiny_mass = 2.0**-51
    masses = np.array([[0.5, 0.5 - tiny_mass, tiny_mass]])
    edges = np.array([0.0, 0.1, 0.7, 1.0])
    write_rainbow_fixture(tmp_path / "record", masses=masses, u_edges=edges)
    low, high = 0.8, 0.8000001
    with RainbowReference(tmp_path / "record") as reference:
        report = _reference_band_and_zeros(reference, low, high)
    expected = tiny_mass * (high - low) / (edges[-1] - edges[-2])
    assert expected > 0
    assert report["reference_band_mass_exact"] == pytest.approx(expected, rel=3e-16, abs=0)


def test_falsey_invalid_configuration_is_not_replaced_by_defaults(tmp_path):
    write_rainbow_fixture(tmp_path / "record")
    with RainbowReference(tmp_path / "record") as reference:
        with pytest.raises(ValueError, match="AngularDiagnosticConfig"):
            evaluate_angular_diagnostics(
                AnalyticDirectionalDensity(reference),
                reference,
                tmp_path / "diagnostics",
                train_samples=4096,
                batch_size=1024,
                config={},
            )


def test_nonfinite_model_fails_without_artifacts_and_restores_mode(tmp_path):
    write_rainbow_fixture(tmp_path / "record")
    with RainbowReference(tmp_path / "record") as reference:
        model = AnalyticDirectionalDensity(reference, invalid=True)
        with pytest.raises(FloatingPointError, match="invalid log PDF"):
            evaluate_angular_diagnostics(
                model, reference, tmp_path / "diagnostics", train_samples=4096, batch_size=1024
            )
        assert model.training and not model.child.training
    assert not (tmp_path / "diagnostics").exists()


def test_reference_mismatch_rejected_before_evaluation(tmp_path):
    write_rainbow_fixture(tmp_path / "record")
    with RainbowReference(tmp_path / "record") as reference:
        model = AnalyticDirectionalDensity(reference)
        model.hg_g = reference.g + 0.01
        with pytest.raises(ValueError, match="does not match"):
            evaluate_angular_diagnostics(
                model, reference, tmp_path / "diagnostics", train_samples=4096, batch_size=1024
            )
        assert model.calls == []


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_model_profiles_use_selected_cuda_device(tmp_path):
    write_rainbow_fixture(tmp_path / "record")
    with RainbowReference(tmp_path / "record") as reference:
        model = AnalyticDirectionalDensity(reference, device="cuda")
        report = evaluate_angular_diagnostics(
            model,
            reference,
            tmp_path / "diagnostics",
            train_samples=4096,
            batch_size=1024,
            config=AngularDiagnosticConfig(eval_batch_size=2, dpi=40),
        )
        assert all(call[1].type == "cuda" for call in model.calls)
        assert report["model"]["device"].startswith("cuda")
