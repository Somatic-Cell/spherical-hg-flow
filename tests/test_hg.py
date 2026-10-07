"""Distribution, derivative, and endpoint tests for the analytic HG core."""

import math

import numpy as np
import pytest
import torch

from phaseflow.hg import hg_cdf, hg_cdf_and_log_prob, hg_icdf, hg_log_prob, hg_prob


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_isotropic_value_and_no_small_g_dead_zone(dtype):
    mu = torch.linspace(-1, 1, 301, dtype=dtype)
    g = torch.tensor(0.0, dtype=dtype, requires_grad=True)
    torch.testing.assert_close(hg_cdf(mu, g), (mu + 1) / 2)
    torch.testing.assert_close(hg_prob(mu, g), torch.full_like(mu, 1 / (4 * math.pi)))
    u = torch.linspace(0, 1, 301, dtype=dtype)
    torch.testing.assert_close(hg_icdf(u, g), 2 * u - 1)
    # An isotropic threshold would incorrectly make this derivative zero.
    derivative = torch.autograd.grad(hg_icdf(torch.tensor(0.3, dtype=dtype), g), g)[0]
    torch.testing.assert_close(derivative, torch.tensor(1.5 * (1 - 0.4**2), dtype=dtype))


def test_cdf_quantile_round_trip_float64_and_endpoints():
    g = torch.tensor([-0.999, -0.8, -1e-12, 0.0, 1e-12, 0.8, 0.999], dtype=torch.float64)
    u = torch.linspace(0, 1, 2001, dtype=torch.float64)[:, None]
    mu = hg_icdf(u, g)
    assert bool(torch.isfinite(mu).all())
    assert bool(((mu >= -1) & (mu <= 1)).all())
    assert bool((mu.diff(dim=0) >= 0).all())
    torch.testing.assert_close(hg_cdf(mu, g), u.expand_as(mu), atol=3e-10, rtol=0)
    torch.testing.assert_close(mu[0], -torch.ones_like(g), atol=0, rtol=0)
    torch.testing.assert_close(mu[-1], torch.ones_like(g), atol=0, rtol=0)


def test_float32_moderate_g_and_finite_extreme_g():
    g = torch.tensor([-0.9, -1e-7, 0.0, 1e-7, 0.9], dtype=torch.float32)
    u = torch.linspace(0, 1, 1001, dtype=torch.float32)[:, None]
    mu = hg_icdf(u, g)
    torch.testing.assert_close(hg_cdf(mu, g), u.expand_as(mu), atol=1.5e-5, rtol=1e-6)
    extreme = torch.nextafter(torch.tensor(1.0), torch.tensor(0.0))
    mu = hg_icdf(u, torch.stack((-extreme, extreme)))
    assert bool(torch.isfinite(mu).all())
    assert bool(((mu >= -1) & (mu <= 1)).all())
    assert bool(torch.isfinite(hg_log_prob(mu, torch.stack((-extreme, extreme)))).all())
    # Deliberately do not demand uniform-coordinate round trips at extreme g:
    # a float32 cosine cannot resolve a peak narrower than its ulp near +/-1.


def test_solid_angle_normalization_and_first_moment():
    nodes, weights = np.polynomial.legendre.leggauss(256)
    mu = torch.from_numpy(nodes)[:, None]
    w = torch.from_numpy(weights)[:, None]
    g = torch.tensor([-0.9, -0.3, 0.0, 0.3, 0.9], dtype=torch.float64)
    marginal = 2 * math.pi * hg_prob(mu, g)
    torch.testing.assert_close((w * marginal).sum(0), torch.ones_like(g), atol=8e-12, rtol=0)
    torch.testing.assert_close((w * marginal * mu).sum(0), g, atol=8e-12, rtol=0)


def test_density_is_cdf_derivative_and_inverse_reciprocal():
    mu = torch.tensor([-0.9, -0.2, 0.3, 0.99], dtype=torch.float64, requires_grad=True)
    g = torch.tensor([-0.8, 0.0, 1e-12, 0.9], dtype=torch.float64)
    derivative = torch.autograd.grad(hg_cdf(mu, g).sum(), mu)[0]
    torch.testing.assert_close(derivative, 2 * math.pi * hg_prob(mu, g), atol=3e-12, rtol=3e-12)
    u = torch.tensor([0.1, 0.3, 0.6, 0.9], dtype=torch.float64, requires_grad=True)
    sampled = hg_icdf(u, g)
    derivative = torch.autograd.grad(sampled.sum(), u)[0]
    torch.testing.assert_close(
        derivative, 1 / (2 * math.pi * hg_prob(sampled, g)), atol=3e-12, rtol=3e-12
    )


@pytest.mark.parametrize("function", [hg_cdf, hg_icdf, hg_log_prob])
def test_gradcheck_through_g_zero(function):
    x = torch.tensor([0.12, 0.38, 0.71], dtype=torch.float64, requires_grad=True)
    g = torch.tensor([-0.3, 0.0, 0.6], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(function, (x, g), eps=1e-6, atol=2e-6, rtol=2e-5)


def test_support_invalid_values_and_fused_evaluation():
    mu = torch.tensor([-2.0, -1.0, 0.2, 1.0, 2.0], dtype=torch.float64)
    cdf, log_prob = hg_cdf_and_log_prob(mu, 0.7)
    torch.testing.assert_close(cdf, hg_cdf(mu, 0.7))
    torch.testing.assert_close(log_prob, hg_log_prob(mu, 0.7))
    assert cdf[0] == 0 and cdf[-1] == 1
    assert log_prob[0] == -torch.inf and log_prob[-1] == -torch.inf
    assert torch.isnan(hg_icdf(1.1, 0.0))
    assert torch.isnan(hg_log_prob(0.0, 1.0))
    with pytest.raises(ValueError):
        hg_icdf(-0.1, 0.0, validate_args=True)
    with pytest.raises(ValueError):
        hg_log_prob(0.0, -1.0, validate_args=True)
    with pytest.raises((TypeError, ValueError)):
        hg_cdf(torch.tensor(0.2, dtype=torch.float32), torch.tensor(0.3, dtype=torch.float64))
