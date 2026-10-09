"""Uniform-query arithmetic checks on synthetic cells, not optical validation."""

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
    loss_terms,
    make_training_pool,
    make_uniform_points,
)
from phaseflow.rainbow import RainbowReference
from phaseflow.single_condition import _array_hash


def uniform_config(**changes):
    return replace(
        LogObjectiveConfig(
            schema_version=2, sampling="uniform", target_fraction=0.0,
            nll_weight=0.0, beta=1.0,
        ),
        **changes,
    )


@pytest.mark.parametrize("changes", [
    {"schema_version": 1}, {"schema_version": True}, {"schema_version": 3},
    {"sampling": "target"}, {"sampling": "target_uniform"},
    {"target_fraction": 0.5}, {"target_fraction": True},
])
def test_uniform_pool_requires_explicit_new_schema_and_zero_target_fraction(changes):
    with pytest.raises(ValueError):
        uniform_config(**changes)
    # Old configs never acquire a new sampling interpretation by omission.
    with pytest.raises(ValueError):
        LogObjectiveConfig.from_dict({"sampling": "uniform", "target_fraction": 0.0})


def test_uniform_schema_roundtrip_and_odd_point_counts():
    config = uniform_config()
    assert LogObjectiveConfig.from_dict(config.to_dict()) == config
    assert config.to_dict()["target_fraction"] == 0.0
    for count in (1, 137, 262145):
        config.validate_training_count(count)
    for count in (0, True, 4.0):
        with pytest.raises(ValueError, match="positive integer"):
            config.validate_training_count(count)
    # Weighted NLL is available explicitly, including NLL-only controls.
    uniform_config(nll_weight=1.0)
    uniform_config(nll_weight=1.0, beta=0.0)
    with pytest.raises(ValueError, match="coefficient"):
        uniform_config(beta=0.0)


def test_entire_training_pool_is_uniform_solid_angle_and_never_samples_cdf(tmp_path, monkeypatch):
    import phaseflow.single_condition as trainer

    edges = np.array([0.0, 0.005, 0.2, 0.9, 1.0])
    masses = np.arange(1, 25, dtype=np.float64).reshape(6, 4)
    masses /= masses.sum()
    write_rainbow_fixture(tmp_path / "record", masses=masses, u_edges=edges, frame_rotation=0.41)

    def forbidden_target_sampling(*args, **kwargs):
        raise AssertionError("uniform training must not invoke the CDF sampler")

    monkeypatch.setattr(trainer, "_points", forbidden_target_sampling)
    with RainbowReference(tmp_path / "record") as reference:
        pool = make_training_pool(reference, 80000, 182, uniform_config(), torch.float32)
        source = pool.directions.astype(np.float64) @ reference.source_to_nf
        np.testing.assert_allclose(np.linalg.norm(source, axis=1), 1, atol=5e-8, rtol=0)
        mu = source[:, 2]
        assert abs(mu.mean()) < 0.008
        assert abs(np.square(mu).mean() - 1 / 3) < 0.005
        histogram, _, _ = np.histogram2d(
            np.arctan2(source[:, 1], source[:, 0]), (1 - mu) / 2,
            bins=(np.linspace(-np.pi, np.pi, len(masses) + 1), edges),
        )
        expected = np.broadcast_to(len(source) * np.diff(edges) / len(masses), masses.shape)
        assert np.sum(np.square(histogram - expected) / expected) < 65
        assert np.max(np.abs(histogram / len(source) - masses)) > 0.04
        np.testing.assert_array_equal(pool.log_p, reference.log_prob(pool.directions))
        np.testing.assert_array_equal(pool.components, np.ones(len(source), dtype=np.uint8))
        assert pool.directions.dtype == np.float32 and pool.log_p.dtype == np.float64
        provenance = pool.provenance
        assert provenance["sampling"] == "uniform"
        assert provenance["target_fraction"] == 0.0
        assert provenance["component_counts"] == {"cdf": 0, "uniform": len(source)}
        assert provenance["order"] == "uniform"
        assert provenance["source_streams"] == {"cdf": None, "uniform": 5}
        assert provenance["runtime_points_sha256"] == _array_hash(
            pool.directions, pool.log_p, pool.components
        )


def test_uniform_training_reuses_existing_uniform_stream_and_deterministic_prefix(tmp_path):
    masses = np.array([[0.7, 0.1], [0.1, 0.1]])
    write_rainbow_fixture(tmp_path / "record", masses=masses, u_edges=np.array([0, 0.3, 1]))
    with RainbowReference(tmp_path / "record") as reference:
        mixed = make_training_pool(reference, 256, 527, LogObjectiveConfig(), torch.float32)
        uniform = make_training_pool(reference, 256, 527, uniform_config(), torch.float32)
        prefix = make_training_pool(reference, 73, 527, uniform_config(), torch.float32)
        weighted = make_training_pool(
            reference, 256, 527, uniform_config(nll_weight=1.0, beta=0.1), torch.float32,
        )
        old_stream = make_uniform_points(reference, 256, 527, "train", torch.float32)
        np.testing.assert_array_equal(uniform.directions, old_stream[0])
        np.testing.assert_array_equal(uniform.log_p, old_stream[1])
        np.testing.assert_array_equal(uniform.directions[:128], mixed.directions[1::2])
        np.testing.assert_array_equal(uniform.log_p[:128], mixed.log_p[1::2])
        np.testing.assert_array_equal(prefix.directions, uniform.directions[:73])
        np.testing.assert_array_equal(prefix.log_p, uniform.log_p[:73])
        assert uniform.provenance == weighted.provenance
        raw = make_uniform_points(reference, 256, 527, "train", torch.float64)
        assert uniform.provenance["raw_points_sha256"] == _array_hash(*raw, uniform.components)
        # Validation/test retain their independent existing streams.
        for stream in ("validation", "test"):
            other = make_uniform_points(reference, 256, 527, stream, torch.float32)
            assert not np.array_equal(other[0], uniform.directions)


def test_uniform_teacher_requeries_actual_fp32_points_across_cell_boundary(tmp_path, monkeypatch):
    import phaseflow.log_objective as module

    write_rainbow_fixture(
        tmp_path / "record", masses=np.array([[0.8, 0.2]]), u_edges=np.array([0.0, 0.1, 1.0]),
    )
    with RainbowReference(tmp_path / "record") as reference:
        u = 0.1 - 1e-10
        raw = np.tile([2 * math.sqrt(u * (1 - u)), 0.0, 1 - 2 * u], (3, 1))
        raw_labels = reference.log_prob(raw)
        actual_labels = reference.log_prob(raw.astype(np.float32))
        assert np.all(raw_labels != actual_labels)
        monkeypatch.setattr(module, "_uniform_directions", lambda *args: raw)
        pool = make_training_pool(reference, 3, 12, uniform_config(), torch.float32)
        np.testing.assert_array_equal(pool.log_p, actual_labels)
        assert pool.provenance["raw_points_sha256"] == _array_hash(raw, raw_labels, pool.components)
        assert pool.provenance["runtime_points_sha256"] == _array_hash(
            raw.astype(np.float32), actual_labels, pool.components,
        )


@pytest.mark.parametrize("dtype,tolerance", [(torch.float32, 2e-6), (torch.float64, 2e-14)])
def test_uniform_loss_and_gradients_match_exact_two_region_integrals(dtype, tolerance):
    # Four equally weighted solid-angle queries: one in an area-1/4 region,
    # three in the area-3/4 region. This quadrature is exact for this teacher.
    area = torch.tensor([0.25, 0.75], dtype=dtype)
    mass = torch.tensor([0.9, 0.1], dtype=dtype)
    logits = torch.tensor([math.log(0.7), math.log(0.3)], dtype=dtype, requires_grad=True)
    region_q = logits.log_softmax(0) - area.log() + LOG_UNIFORM
    index = torch.tensor([0, 1, 1, 1])
    log_q = region_q[index]
    log_p = (mass.log() - area.log() + LOG_UNIFORM)[index].to(torch.float64).requires_grad_()
    config = uniform_config(nll_weight=0.6, beta=0.35)
    result = loss_terms(log_q, log_p, config)
    expected_nll = -(mass * region_q).sum()
    expected_log = (area * (logits.log_softmax(0) - mass.log()).square()).sum()
    expected = config.nll_weight * expected_nll + config.beta * expected_log
    for name, value in (("nll", expected_nll), ("log_mse", expected_log), ("loss", expected)):
        torch.testing.assert_close(result[name], value, rtol=tolerance, atol=tolerance)
        assert result[name].dtype == dtype
    gradient, label_gradient = torch.autograd.grad(
        result["loss"], (logits, log_p), retain_graph=True, allow_unused=True,
    )
    expected_gradient, = torch.autograd.grad(expected, logits)
    torch.testing.assert_close(gradient, expected_gradient, rtol=tolerance, atol=tolerance)
    assert label_gradient is None
    # Plain averaging of NLL would incorrectly learn uniform sphere density.
    assert abs(float((result["nll"] + log_q.mean()).detach())) > 0.5


def test_pure_uniform_log_has_no_density_weight_and_skips_inactive_overflow():
    log_q = torch.tensor([-2.0, -3.0], dtype=torch.float32, requires_grad=True)
    # p/u would overflow FP32 at the first point. It is inactive for pure log.
    log_p = torch.tensor([1000.0, -1000.0], dtype=torch.float64, requires_grad=True)
    config = uniform_config(beta=0.3)
    plain = loss_terms(log_q, log_p, config)
    reported = loss_terms(log_q, log_p, config, report_components=True)
    error = log_q - log_p.detach().float()
    expected = config.beta * error.square().mean()
    torch.testing.assert_close(plain["loss"], expected, rtol=0, atol=0)
    torch.testing.assert_close(reported["loss"], expected, rtol=0, atol=0)
    assert plain["nll"] == 0
    assert torch.isinf(reported["nll"]) and reported["nll"] > 0
    reported["loss"].backward()
    torch.testing.assert_close(log_q.grad, 2 * config.beta * error / len(error))
    assert torch.isfinite(log_q.grad).all() and log_q.grad.dtype == torch.float32
    assert log_p.grad is None


@pytest.mark.parametrize("nll_weight,beta", [(0.0, 1.0), (1.0, 0.0)])
def test_reporting_inactive_uniform_components_measures_raw_terms(nll_weight, beta):
    q = torch.tensor([-2.0, -3.0], dtype=torch.float32, requires_grad=True)
    p = torch.tensor([-4.0, -2.0], dtype=torch.float64, requires_grad=True)
    config = uniform_config(nll_weight=nll_weight, beta=beta)
    report = loss_terms(q, p, config, report_components=True)
    expected_nll = -(torch.exp(p.detach().float() - LOG_UNIFORM) * q).mean()
    expected_log = (q - p.detach().float()).square().mean()
    torch.testing.assert_close(report["nll"], expected_nll)
    torch.testing.assert_close(report["log_mse"], expected_log)
    torch.testing.assert_close(report["loss"], nll_weight * expected_nll + beta * expected_log)


def test_uniform_nll_does_not_evaluate_or_multiply_an_inactive_infinite_log_error():
    q = torch.tensor([-2.0, -3.0], dtype=torch.float32, requires_grad=True)
    p = torch.full((2,), -1e30, dtype=torch.float64)
    config = uniform_config(nll_weight=1.0, beta=0.0)
    silent = loss_terms(q, p, config)
    report = loss_terms(q, p, config, report_components=True)
    assert silent["log_mse"] == 0
    assert torch.isinf(report["log_mse"])
    assert report["loss"] == 0
    report["loss"].backward()
    torch.testing.assert_close(q.grad, torch.zeros_like(q))
