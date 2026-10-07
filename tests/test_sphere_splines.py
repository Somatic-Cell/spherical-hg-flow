"""Circle boundary, reflection, analytic inverse, and parameter-gradient checks."""

import pytest
import torch

from phaseflow.sphere_splines import CircularRQSTransform


def random_circle(*, reflected=False, dtype=torch.float64):
    generator = torch.Generator().manual_seed(8704)
    bins = 5 if reflected else 10
    widths = 0.6 * torch.randn(bins, generator=generator, dtype=dtype)
    heights = 0.6 * torch.randn(bins, generator=generator, dtype=dtype)
    slopes = 0.6 * torch.randn(bins + int(reflected), generator=generator, dtype=dtype)
    factory = (
        CircularRQSTransform.from_reflection_logits
        if reflected
        else CircularRQSTransform.from_logits
    )
    return factory(widths, heights, slopes, validate_args=True)


@pytest.mark.parametrize("reflected", [False, True])
def test_circle_seam_has_equal_positive_one_sided_derivatives(reflected):
    transform = random_circle(reflected=reflected)
    endpoints = torch.tensor([0.0, 1.0], dtype=torch.float64)
    values, log_derivative = transform.call_and_ladj(endpoints)
    torch.testing.assert_close(values, endpoints, rtol=0, atol=0)
    torch.testing.assert_close(log_derivative[0], log_derivative[1], rtol=0, atol=5e-15)
    assert bool(torch.isfinite(log_derivative).all())
    # Compare actual one-sided differences of the lifted map, independently
    # of the reported Jacobian.  Both approach the same circle derivative.
    epsilon = 1e-7
    limits = torch.tensor([epsilon, 1 - epsilon], dtype=torch.float64)
    inner = transform(limits)
    left = inner[0] / epsilon
    right = (1 - inner[1]) / epsilon
    torch.testing.assert_close(left, right, rtol=0, atol=3e-5)
    torch.testing.assert_close(left, log_derivative[0].exp(), rtol=0, atol=3e-5)


@pytest.mark.parametrize("reflected", [False, True])
@pytest.mark.parametrize("dtype,tolerance", [(torch.float32, 4e-6), (torch.float64, 6e-14)])
def test_circular_round_trip_and_fused_inverse_logdet(reflected, dtype, tolerance):
    transform = random_circle(reflected=reflected, dtype=dtype)
    x = torch.linspace(0, 1, 2001, dtype=dtype)
    y, forward_logdet = transform.call_and_ladj(x)
    recovered, inverse_logdet = transform.inv.call_and_ladj(y)
    torch.testing.assert_close(recovered, x, rtol=0, atol=tolerance)
    torch.testing.assert_close(inverse_logdet, -forward_logdet, rtol=0, atol=12 * tolerance)
    assert bool((y.diff() > 0).all())


def test_reflection_equivariance_without_identifying_opposite_meridians():
    transform = random_circle(reflected=True)
    x = torch.linspace(0, 1, 2001, dtype=torch.float64)
    y, logdet = transform.call_and_ladj(x)
    reflected_y, reflected_logdet = transform.call_and_ladj(1 - x)
    torch.testing.assert_close(reflected_y, 1 - y, rtol=0, atol=2e-15)
    torch.testing.assert_close(reflected_logdet, logdet, rtol=0, atol=8e-14)
    fixed = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    values, derivatives = transform.call_and_ladj(fixed)
    torch.testing.assert_close(values, fixed, rtol=0, atol=4e-16)
    # phi=0 and phi=pi are distinct physical directions and are not required
    # to have the same local density, despite both being symmetry meridians.
    assert abs(float(derivatives[0] - derivatives[1])) > 0.01


@pytest.mark.parametrize("inverse", [False, True])
def test_reflection_constraints_keep_gradients_through_all_independent_logits(inverse):
    generator = torch.Generator().manual_seed(59)
    values = [
        (0.2 * torch.randn(n, dtype=torch.float64, generator=generator)).requires_grad_()
        for n in (3, 3, 4)
    ]
    x = torch.tensor([0.07, 0.31, 0.64, 0.93], dtype=torch.float64, requires_grad=True)

    def evaluate(w, h, d, coordinate):
        transform = CircularRQSTransform.from_reflection_logits(w, h, d)
        return (transform.inv if inverse else transform).call_and_ladj(coordinate)

    assert torch.autograd.gradcheck(
        evaluate, (*values, x), eps=1e-6, atol=3e-6, rtol=3e-5
    )


def test_circle_constructor_cannot_accept_independent_seam_slopes():
    with pytest.raises(ValueError, match="unique derivatives"):
        CircularRQSTransform(torch.ones(4), torch.ones(4), torch.ones(5))


def test_axial_identity_blend_preserves_exact_identity_and_circle_boundary():
    generator = torch.Generator().manual_seed(72)
    arguments = [torch.randn(n, dtype=torch.float64, generator=generator) for n in (4, 4, 5)]
    transform = CircularRQSTransform.from_reflection_logits(*arguments, identity_strength=0)
    x = torch.linspace(0, 1, 233, dtype=torch.float64)
    y, logdet = transform.call_and_ladj(x)
    torch.testing.assert_close(y, x, rtol=0, atol=2e-16)
    torch.testing.assert_close(logdet, torch.zeros_like(x), rtol=0, atol=5e-16)
