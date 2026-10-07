"""Inverse, Jacobian, probability, endpoint, and Zuko integration checks."""

import numpy as np
import pytest
import torch

from phaseflow.splines import BoundedRQSTransform, identity_derivative_logit


def _random_transform(dtype=torch.float64, bins=12, batch=()):
    generator = torch.Generator().manual_seed(1906)
    widths = torch.randn(batch + (bins,), dtype=dtype, generator=generator) * 0.6
    heights = torch.randn(batch + (bins,), dtype=dtype, generator=generator) * 0.6
    derivatives = torch.randn(batch + (bins + 1,), dtype=dtype, generator=generator) * 0.6
    return BoundedRQSTransform.from_logits(widths, heights, derivatives, validate_args=True)


@pytest.mark.parametrize("dtype,tolerance", [(torch.float32, 7e-7), (torch.float64, 2e-15)])
def test_identity_and_trainable_endpoint_slopes(dtype, tolerance):
    bins = 16
    zero = torch.zeros(bins, dtype=dtype)
    derivative_logits = torch.full(
        (bins + 1,), identity_derivative_logit(), dtype=dtype, requires_grad=True
    )
    transform = BoundedRQSTransform.from_logits(zero, zero, derivative_logits)
    x = torch.linspace(0, 1, 2001, dtype=dtype)
    y, ladj = transform.call_and_ladj(x)
    recovered, inverse_ladj = transform.inv.call_and_ladj(y)
    torch.testing.assert_close(y, x, atol=tolerance, rtol=0)
    torch.testing.assert_close(recovered, x, atol=tolerance, rtol=0)
    torch.testing.assert_close(ladj, torch.zeros_like(ladj), atol=tolerance, rtol=0)
    torch.testing.assert_close(inverse_ladj, -ladj, atol=3 * tolerance, rtol=0)
    assert y[0] == 0 and y[-1] == 1
    assert transform.inv.inv is transform
    # The end slopes participate in training; unlike a linear-tail NSF, they
    # are not forced back to 1 after initialization.
    endpoint_ladj = transform.call_and_ladj(torch.tensor([0.0, 1.0], dtype=dtype))[1]
    gradient = torch.autograd.grad(endpoint_ladj.sum(), derivative_logits)[0]
    assert gradient[0] > 0 and gradient[-1] > 0


@pytest.mark.parametrize("dtype,tolerance", [(torch.float32, 3e-6), (torch.float64, 5e-14)])
def test_round_trip_and_log_jacobian_cancellation(dtype, tolerance):
    transform = _random_transform(dtype)
    x = torch.linspace(0, 1, 4001, dtype=dtype)
    y, forward_ladj = transform.call_and_ladj(x)
    actual, inverse_ladj = transform.inv.call_and_ladj(y)
    torch.testing.assert_close(actual, x, atol=tolerance, rtol=0)
    torch.testing.assert_close(inverse_ladj, -forward_ladj, atol=12 * tolerance, rtol=0)
    assert bool((y.diff() > 0).all())
    torch.testing.assert_close(transform.inv(transform(x)), x, atol=tolerance, rtol=0)


def test_reported_jacobians_match_autograd_both_directions():
    transform = _random_transform()
    x = torch.tensor([0.07, 0.28, 0.51, 0.77, 0.96], dtype=torch.float64, requires_grad=True)
    y, ladj = transform.call_and_ladj(x)
    derivative = torch.autograd.grad(y.sum(), x)[0]
    torch.testing.assert_close(derivative.log(), ladj, atol=2e-13, rtol=1e-13)
    y = y.detach().requires_grad_()
    x, ladj = transform.inv.call_and_ladj(y)
    derivative = torch.autograd.grad(x.sum(), y)[0]
    torch.testing.assert_close(derivative.log(), ladj, atol=2e-13, rtol=1e-13)


@pytest.mark.parametrize("inverse", [False, True])
def test_gradcheck_all_logits_and_input(inverse):
    generator = torch.Generator().manual_seed(75)
    widths = (torch.randn(4, dtype=torch.float64, generator=generator) * 0.3).requires_grad_()
    heights = (torch.randn(4, dtype=torch.float64, generator=generator) * 0.3).requires_grad_()
    derivatives = (torch.randn(5, dtype=torch.float64, generator=generator) * 0.3).requires_grad_()
    value = torch.tensor([0.09, 0.37, 0.84], dtype=torch.float64, requires_grad=True)

    def evaluate(w, h, d, x):
        transform = BoundedRQSTransform.from_logits(w, h, d)
        return (transform.inv if inverse else transform).call_and_ladj(x)

    assert torch.autograd.gradcheck(
        evaluate, (widths, heights, derivatives, value), eps=1e-6, atol=3e-6, rtol=3e-5
    )


def test_distribution_normalization_with_independent_quadrature():
    transform = _random_transform()
    # A data->uniform transform has density T'(x).  Integrate on each bin so
    # this check does not depend on sampled points or the inverse function.
    nodes, weights = np.polynomial.legendre.leggauss(80)
    nodes = torch.from_numpy(nodes)
    weights = torch.from_numpy(weights)
    lo, hi = transform.horizontal[:-1], transform.horizontal[1:]
    locations = lo[:, None] + (hi - lo)[:, None] * (nodes + 1) / 2
    density = transform.call_and_ladj(locations)[1].exp()
    mass = (density * weights * (hi - lo)[:, None] / 2).sum()
    torch.testing.assert_close(mass, torch.tensor(1.0, dtype=torch.float64), atol=2e-12, rtol=0)


def test_broadcasting_and_knots_including_boundaries():
    transform = _random_transform(batch=(3, 1))
    values = torch.linspace(0, 1, 19, dtype=torch.float64)[None, :]
    y, ladj = transform.call_and_ladj(values)
    assert y.shape == (3, 19) and ladj.shape == (3, 19)
    torch.testing.assert_close(transform.inv(y), values.expand(3, 19), atol=5e-14, rtol=0)
    transform = _random_transform()
    y, ladj = transform.call_and_ladj(transform.horizontal)
    torch.testing.assert_close(y, transform.vertical, atol=3e-16, rtol=0)
    torch.testing.assert_close(ladj, transform.derivatives.log(), atol=2e-14, rtol=0)
    torch.testing.assert_close(transform.inv(y), transform.horizontal, atol=3e-15, rtol=0)


def test_no_identity_tails_and_invalid_parameters():
    transform = _random_transform()
    with pytest.raises(ValueError):
        transform(torch.tensor([-0.1, 1.1], dtype=torch.float64))
    transform._validate_args = False
    assert bool(torch.isnan(transform(torch.tensor([-0.1, 1.1], dtype=torch.float64))).all())
    with pytest.raises(ValueError):
        BoundedRQSTransform(
            torch.tensor([0.0, 1.0]), torch.ones(2), torch.ones(3), validate_args=True
        )
    with pytest.raises(ValueError):
        BoundedRQSTransform.from_logits(
            torch.zeros(4), torch.zeros(4), torch.zeros(5), min_bin_width=0.25
        )
    with pytest.raises(ValueError):
        BoundedRQSTransform.from_logits(torch.zeros(4), torch.zeros(4), torch.zeros(3))


def test_torch_cache_protocol_preserves_parameters_and_invalid_lanes():
    transform = _random_transform()
    cached = transform.with_cache(1)
    x = torch.tensor([0.13, 0.41, 0.87], dtype=torch.float64)
    torch.testing.assert_close(cached(x), transform(x), atol=0, rtol=0)
    torch.testing.assert_close(cached.inv(cached(x)), x, atol=0, rtol=0)
    torch.testing.assert_close(transform.inv.with_cache(1)(transform(x)), x, atol=5e-14, rtol=0)
    invalid = BoundedRQSTransform(torch.tensor([0.0, 1.0]), torch.ones(2), torch.ones(3))
    assert torch.isnan(invalid.with_cache(1)(torch.tensor(0.2)))


def test_zuko_coupling_and_composition_protocol():
    from zuko.transforms import ComposedTransform, CouplingTransform, DependentTransform

    def meta(x):
        logits = torch.stack((x, -x, 0.5 * x, -0.5 * x), dim=-1)
        slopes = (
            torch.zeros(x.shape + (5,), dtype=x.dtype, device=x.device)
            + identity_derivative_logit()
        )
        return DependentTransform(BoundedRQSTransform.from_logits(logits, -logits, slopes), 1)

    first = CouplingTransform(meta, torch.tensor([True, False]))
    second = CouplingTransform(meta, torch.tensor([False, True]))
    transform = ComposedTransform(first, second)
    x = torch.tensor([[0.21, 0.65], [0.73, 0.17]], dtype=torch.float64)
    y, ladj = transform.call_and_ladj(x)
    restored, inverse_ladj = transform.inv.call_and_ladj(y)
    torch.testing.assert_close(restored, x, atol=3e-14, rtol=0)
    torch.testing.assert_close(
        ladj + inverse_ladj, torch.zeros(2, dtype=torch.float64), atol=3e-13, rtol=0
    )
