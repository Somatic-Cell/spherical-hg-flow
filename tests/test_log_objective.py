"""Analytic arithmetic and synthetic-cell checks, not Rainbow optical validation."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_rainbow import write_rainbow_fixture

from phaseflow.log_objective import (
    LOG_UNIFORM,
    LogObjectiveConfig,
    evaluate_log_shape,
    loss_terms,
    make_training_pool,
    make_uniform_points,
    verify_positive_teacher,
)
from phaseflow.rainbow import RainbowReference
from phaseflow.single_condition import _array_hash


def positive_teacher(path, *, rotation=0.37):
    edges = np.array([0.0, 0.005, 0.2, 0.9, 1.0])
    masses = np.arange(1, 25, dtype=np.float64).reshape(6, 4)
    masses /= masses.sum()
    write_rainbow_fixture(path, masses=masses, u_edges=edges, frame_rotation=rotation)
    return path, masses, edges


@pytest.mark.parametrize("changes", [
    {"schema_version": True}, {"schema_version": 2}, {"nll_weight": -1},
    {"beta": float("nan")}, {"beta": float("inf")}, {"beta": True}, {"beta": "0.1"},
    {"nll_weight": 0, "beta": 0}, {"sampling": "uniform"},
    {"sampling": "target", "beta": 0.1}, {"target_fraction": 0.4},
    {"target_fraction": True}, {"selection_metric": "test_log_rmse"},
    {"validation_uniform_samples": 1}, {"test_uniform_samples": True},
    {"zero_policy": "floor"},
])
def test_config_rejects_unsupported_or_ambiguous_experiments(changes):
    with pytest.raises(ValueError):
        LogObjectiveConfig.from_dict(changes)


def test_config_roundtrip_and_exact_half_pool_count():
    config = LogObjectiveConfig()
    assert LogObjectiveConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="unknown"):
        LogObjectiveConfig.from_dict({"epsilon": 1e-8})
    with pytest.raises(ValueError, match="JSON object"):
        LogObjectiveConfig.from_dict([])
    config.validate_training_count(262144)
    for count in (1, 137, 262145):
        with pytest.raises(ValueError, match="even"):
            config.validate_training_count(count)
    for count in (0, True, 4.0):
        with pytest.raises(ValueError, match="positive integer"):
            config.validate_training_count(count)
    replace(config, sampling="target", beta=0).validate_training_count(137)


def _analytic_two_region_check(device, dtype, tolerance):
    # Region solid-angle fractions are unequal. Their p/u half mixture assigns
    # mass exactly 1/2 to each region, so one point per region is exact quadrature.
    area = torch.tensor([0.1, 0.9], device=device, dtype=dtype)
    target_mass = torch.tensor([0.9, 0.1], device=device, dtype=dtype)
    logits = torch.tensor([math.log(0.8), math.log(0.2)], device=device, dtype=dtype,
                          requires_grad=True)
    log_q = logits.log_softmax(0) - area.log() + LOG_UNIFORM
    log_p = (target_mass.log() - area.log() + LOG_UNIFORM).detach().requires_grad_()
    config = LogObjectiveConfig(beta=0.37)
    actual = loss_terms(log_q, log_p, config)
    direct_nll = -(target_mass * log_q).sum()
    direct_log = (area * (logits.log_softmax(0) - target_mass.log()).square()).sum()
    direct = direct_nll + config.beta * direct_log
    torch.testing.assert_close(actual["nll"], direct_nll, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(actual["log_mse"], direct_log, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(actual["loss"], direct, rtol=tolerance, atol=tolerance)
    gradient, label_gradient = torch.autograd.grad(
        actual["loss"], (logits, log_p), allow_unused=True, retain_graph=True
    )
    expected_gradient, = torch.autograd.grad(direct, logits)
    torch.testing.assert_close(gradient, expected_gradient, rtol=tolerance, atol=tolerance)
    assert label_gradient is None
    assert actual["loss"].dtype == dtype
    # An unweighted average over r would instead fit the mixture distribution.
    assert abs(float((actual["nll"] + log_q.mean()).detach())) > 0.5


def test_weighted_loss_and_gradient_match_analytic_solid_angle_integrals():
    _analytic_two_region_check("cpu", torch.float64, 2e-14)


def test_training_arithmetic_stays_fp32_with_fp64_teacher_labels():
    q = torch.tensor([-4.0, -1.0], dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor([-3.0, -2.0], dtype=torch.float64, requires_grad=True)
    terms = loss_terms(q, teacher, LogObjectiveConfig())
    assert all(value.dtype == torch.float32 for value in terms.values())
    terms["loss"].backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert teacher.grad is None


def test_extreme_log_labels_and_disabled_terms_are_finite_without_density_clipping():
    q = torch.tensor([-2.0, -3.0], dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor([-1000.0, 1000.0], dtype=torch.float64)
    terms = loss_terms(q, teacher, LogObjectiveConfig(beta=0.1))
    # For these extremes each ratio is saturated at either 0 or 2.
    assert float(terms["nll"].detach()) == pytest.approx(3.0, rel=1e-4)
    assert float(terms["log_mse"].detach()) == pytest.approx(998.0**2, rel=1e-5)
    terms["loss"].backward()
    assert torch.isfinite(q.grad).all()

    q2 = torch.tensor([-2.0, -3.0], dtype=torch.float32, requires_grad=True)
    zero_teacher = torch.full((2,), -math.inf, dtype=torch.float64)
    inactive = loss_terms(q2, zero_teacher, LogObjectiveConfig(beta=0))
    assert float(inactive["loss"].detach()) == 0
    assert float(inactive["log_mse"]) == 0
    inactive["loss"].backward()
    assert torch.isfinite(q2.grad).all()
    assert torch.equal(q2.grad, torch.zeros_like(q2))
    target = loss_terms(q2, zero_teacher, LogObjectiveConfig(sampling="target", beta=0))
    assert float(target["nll"].detach()) == 2.5


def test_normalized_exact_teacher_is_stationary_for_both_losses():
    area = torch.tensor([0.1, 0.9], dtype=torch.float64)
    target = torch.tensor([0.9, 0.1], dtype=torch.float64)
    logits = target.log().requires_grad_()
    q = logits.log_softmax(0) - area.log() + LOG_UNIFORM
    p = target.log() - area.log() + LOG_UNIFORM
    result = loss_terms(q, p, LogObjectiveConfig(beta=3.0))
    assert float(result["log_mse"].detach()) < 1e-28
    result["loss"].backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), rtol=0, atol=2e-14)


def test_optional_reporting_measures_inactive_components_without_adding_them_to_loss():
    q = torch.tensor([-2.0, -3.0], dtype=torch.float32, requires_grad=True)
    p = torch.tensor([-4.0, -2.0], dtype=torch.float64)
    both = loss_terms(q, p, LogObjectiveConfig(beta=1.0))
    nll_only = loss_terms(q, p, LogObjectiveConfig(beta=0), report_components=True)
    log_only = loss_terms(q, p, LogObjectiveConfig(nll_weight=0, beta=1), report_components=True)
    torch.testing.assert_close(nll_only["log_mse"], both["log_mse"])
    torch.testing.assert_close(nll_only["loss"], both["nll"])
    torch.testing.assert_close(log_only["nll"], both["nll"])
    torch.testing.assert_close(log_only["loss"], both["log_mse"])
    assert float(nll_only["log_mse"].detach()) > 0
    # Even if a caller requests diagnostics outside the positive-teacher
    # contract, an inactive infinite log term must not corrupt the NLL loss.
    zero = loss_terms(q, torch.full_like(p, -math.inf), LogObjectiveConfig(beta=0),
                      report_components=True)
    assert torch.isinf(zero["log_mse"])
    assert float(zero["loss"].detach()) == 0


def test_global_zero_scan_finds_unsampled_cells_and_zero_marginal_rows(tmp_path, monkeypatch):
    import phaseflow.log_objective as module

    masses, edges, _ = write_rainbow_fixture(tmp_path / "zero")
    area = np.broadcast_to(np.diff(edges) / len(masses), masses.shape)
    expected_zero_fraction = area[masses == 0].sum()
    # Deliberately small chunks force both axes to cross scan boundaries.
    monkeypatch.setattr(module, "_ZERO_SCAN_ELEMENTS", 2)
    with RainbowReference(tmp_path / "zero") as reference:
        with pytest.raises(ValueError, match="whole-sphere log-PDF") as caught:
            verify_positive_teacher(reference)
        assert f"zero_solid_angle_fraction={expected_zero_fraction:.17g}" in str(caught.value)
        assert f"zero_cell_count={np.count_nonzero(masses == 0)}" in str(caught.value)
        # Existing target sampling remains valid; the new workflow does not alter p.
        from phaseflow.single_condition import _points

        _, labels = _points(reference, 128, 18, "train")
        assert np.isfinite(labels).all()

    path, masses, _ = positive_teacher(tmp_path / "positive")
    with RainbowReference(path) as reference:
        report = verify_positive_teacher(reference)
        assert report["reference_zero_solid_angle_fraction"] == 0
        assert report["zero_cell_count"] == 0
        assert report["scanned_cell_count"] == masses.size


def test_spherical_uniform_points_follow_area_on_rotated_nonuniform_table(tmp_path):
    path, masses, edges = positive_teacher(tmp_path / "record", rotation=0.41)
    with RainbowReference(path) as reference:
        directions, labels = make_uniform_points(reference, 80000, 182, "validation", torch.float64)
        source = directions @ reference.source_to_nf
        np.testing.assert_allclose(np.linalg.norm(source, axis=1), 1, atol=5e-16, rtol=0)
        mu = source[:, 2]
        assert abs(mu.mean()) < 0.008
        assert abs(np.square(mu).mean() - 1 / 3) < 0.005
        u = (1 - mu) / 2
        phi = np.arctan2(source[:, 1], source[:, 0])
        histogram, _, _ = np.histogram2d(
            phi, u, bins=(np.linspace(-np.pi, np.pi, len(masses) + 1), edges)
        )
        expected = np.broadcast_to(len(directions) * np.diff(edges) / len(masses), masses.shape)
        chi_squared = np.sum(np.square(histogram - expected) / expected)
        assert chi_squared < 65
        # The same samples cannot plausibly be target-distributed on this fixture.
        assert np.max(np.abs(histogram / len(directions) - masses)) > 0.04
        np.testing.assert_array_equal(labels, reference.log_prob(directions))


def test_uniform_queries_respect_recorded_frame_and_independent_prefix_streams(tmp_path):
    path_a, _, _ = positive_teacher(tmp_path / "a", rotation=-0.23)
    path_b, _, _ = positive_teacher(tmp_path / "b", rotation=1.17)
    before_numpy = np.random.get_state()
    before_torch = torch.get_rng_state().clone()
    with RainbowReference(path_a) as a, RainbowReference(path_b) as b:
        first = make_uniform_points(a, 96, 93, "validation", torch.float64)
        rotated = make_uniform_points(b, 96, 93, 6, torch.float64)
        np.testing.assert_allclose(first[0] @ a.source_to_nf, rotated[0] @ b.source_to_nf,
                                   rtol=0, atol=6e-16)
        np.testing.assert_array_equal(first[1], rotated[1])
        prefix = make_uniform_points(a, 24, 93, "validation", torch.float64)
        np.testing.assert_array_equal(prefix[0], first[0][:24])
        np.testing.assert_array_equal(prefix[1], first[1][:24])
        hashes = {
            _array_hash(*make_uniform_points(a, 96, 93, stream, torch.float64))
            for stream in ("train", "validation", "test")
        }
        assert len(hashes) == 3
    after_numpy = np.random.get_state()
    assert before_numpy[0] == after_numpy[0]
    np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
    assert before_numpy[2:] == after_numpy[2:]
    assert torch.equal(before_torch, torch.get_rng_state())


def test_pool_identity_is_shared_across_beta_and_tracks_actual_runtime_values(tmp_path):
    path, _, _ = positive_teacher(tmp_path / "record")
    with RainbowReference(path) as reference:
        base = LogObjectiveConfig(beta=0)
        pool = make_training_pool(reference, 128, 527, base, torch.float32)
        other = make_training_pool(reference, 128, 527, replace(base, beta=3), torch.float32)
        assert pool.directions.dtype == np.float32 and pool.log_p.dtype == np.float64
        assert pool.components.dtype == np.uint8
        assert pool.provenance == other.provenance
        np.testing.assert_array_equal(pool.directions, other.directions)
        np.testing.assert_array_equal(pool.log_p, reference.log_prob(pool.directions))
        np.testing.assert_array_equal(pool.components, np.tile([0, 1], 64))
        assert pool.provenance["component_counts"] == {"cdf": 64, "uniform": 64}
        assert pool.provenance["runtime_points_sha256"] == _array_hash(
            pool.directions, pool.log_p, pool.components
        )
        raw_precision = make_training_pool(reference, 128, 527, base, torch.float64)
        assert raw_precision.provenance["raw_points_sha256"] == pool.provenance["raw_points_sha256"]
        assert raw_precision.provenance["runtime_points_sha256"] != pool.provenance["runtime_points_sha256"]
        prefix = make_training_pool(reference, 32, 527, base, torch.float32)
        np.testing.assert_array_equal(prefix.directions, pool.directions[:32])
        np.testing.assert_array_equal(prefix.log_p, pool.log_p[:32])
        target = make_training_pool(reference, 137, 527, replace(base, sampling="target"), torch.float32)
        assert target.provenance["component_counts"] == {"cdf": 137, "uniform": 0}
        assert not target.components.any()


def test_teacher_labels_are_requeried_after_rounding_across_cell_edge(tmp_path, monkeypatch):
    import phaseflow.single_condition as trainer

    write_rainbow_fixture(tmp_path / "record", masses=np.array([[0.8, 0.2]]),
                          u_edges=np.array([0.0, 0.1, 1.0]))
    with RainbowReference(tmp_path / "record") as reference:
        u = 0.1 - 1e-10
        raw = np.tile([2 * math.sqrt(u * (1 - u)), 0.0, 1 - 2 * u], (2, 1))
        original_labels = reference.log_prob(raw)
        runtime_labels = reference.log_prob(raw.astype(np.float32))
        assert np.all(original_labels != runtime_labels)
        monkeypatch.setattr(trainer, "_points", lambda *args: (raw, original_labels))
        pool = make_training_pool(reference, 2, 12,
                                  LogObjectiveConfig(sampling="target", beta=0), torch.float32)
        np.testing.assert_array_equal(pool.log_p, runtime_labels)


class _ConstantOffsetModel:
    """Analytic evaluation probe; it is not an NF or an optical reference."""

    def __init__(self, reference, offset, dtype=torch.float64):
        self.hg_g = reference.g
        self.incident_cosine = float(reference.condition[1])
        self.device = torch.device("cpu")
        self.geometry_dtype = dtype
        self.reference = reference
        self.offset = offset
        self.batch_sizes = []

    def log_prob(self, directions):
        assert directions.dtype == self.geometry_dtype
        self.batch_sizes.append(len(directions))
        return torch.as_tensor(self.reference.log_prob(directions.numpy()), dtype=self.geometry_dtype) + self.offset


@pytest.mark.parametrize("offset", [0.0, math.log(2.0), -math.log(2.0), 400.0, 900.0])
def test_shape_metrics_have_known_log_error_and_stable_relative_rms(tmp_path, offset):
    path, _, _ = positive_teacher(tmp_path / "record")
    with RainbowReference(path) as reference:
        points = make_uniform_points(reference, 101, 54, "test", torch.float64)
        model = _ConstantOffsetModel(reference, offset)
        result = evaluate_log_shape(model, reference, points, 29)
        assert result["log_mse"] == pytest.approx(offset**2, rel=2e-14, abs=1e-25)
        assert result["log_rmse"] == pytest.approx(abs(offset), abs=1e-14)
        assert result["log_mse_standard_error"] < 1e-9
        assert result["uniform_sample_count"] == 101
        assert model.batch_sizes == [29, 29, 29, 14]
        if offset == 900:
            assert result["relative_rmse"] is None
            assert result["relative_rmse_status"] == "overflow_float64"
        else:
            assert result["relative_rmse"] == pytest.approx(abs(math.expm1(offset)), abs=1e-15)
            assert result["relative_rmse_status"] == "finite"


def test_shape_metrics_use_float64_reductions_and_reject_bad_values(tmp_path):
    path, _, _ = positive_teacher(tmp_path / "record")
    with RainbowReference(path) as reference:
        points = make_uniform_points(reference, 31, 81, "validation", torch.float32)
        model = _ConstantOffsetModel(reference, 0.5, torch.float32)
        actual = evaluate_log_shape(model, reference, points, 7)
        with torch.no_grad():
            raw = model.log_prob(torch.as_tensor(points[0])).double().numpy() - points[1]
        assert actual["log_mse"] == np.square(raw).mean()
        assert actual["log_mse_standard_error"] == pytest.approx(np.square(raw).std(ddof=1) / math.sqrt(31))
        with pytest.raises(FloatingPointError, match="nonfinite"):
            evaluate_log_shape(_ConstantOffsetModel(reference, math.nan), reference, points, 7)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable; CPU is not GPU validation")
def test_cuda_fp32_analytic_loss_and_gradient():
    _analytic_two_region_check("cuda", torch.float32, 3e-6)
