"""Coordinate, symmetry, and conditioning-frame tests."""

import math

import pytest
import torch

from phaseflow.geometry import (
    direction_to_folded,
    direction_to_mu_phi,
    folded_to_direction,
    mu_phi_to_direction,
    normalize_direction,
    scattering_frame,
)


def test_scattering_frame_definition_and_orthonormality():
    incident = torch.tensor(
        [[0.0, 0.0, 1.0], [0.4, -0.2, 0.7], [1.0, 0.0, 0.0]], dtype=torch.float64
    )
    axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    x, y, z = scattering_frame(incident, axis, validate_args=True)
    basis = torch.stack((x, y, z), dim=-1)
    torch.testing.assert_close(
        basis.transpose(-1, -2) @ basis,
        torch.eye(3, dtype=torch.float64).expand(3, 3, 3),
        atol=1e-14,
        rtol=0,
    )
    torch.testing.assert_close(torch.linalg.cross(x, y, dim=-1), z, atol=1e-14, rtol=0)
    torch.testing.assert_close(basis[0], torch.eye(3, dtype=torch.float64), atol=0, rtol=0)
    # x points toward the projected particle axis, not its negative.
    assert bool(((x[:2] * axis).sum(-1) > 0).all())


def test_no_finite_axial_cone_is_replaced_by_the_fallback():
    incident = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    tiny = torch.tensor([1e-250, 0.0, 1.0], dtype=torch.float64)
    x, y, _ = scattering_frame(incident, tiny)
    torch.testing.assert_close(
        x, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), atol=0, rtol=0
    )
    torch.testing.assert_close(
        y, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), atol=0, rtol=0
    )
    # Exact parallelism still has a finite, deterministic frame.
    basis = torch.stack(scattering_frame(incident, incident))
    assert bool(torch.isfinite(basis).all())
    torch.testing.assert_close(
        basis @ basis.T, torch.eye(3, dtype=torch.float64), atol=1e-14, rtol=0
    )


@pytest.mark.parametrize("dtype,tolerance", [(torch.float32, 2e-6), (torch.float64, 2e-14)])
def test_round_trip_in_oblique_world_frame(dtype, tolerance):
    generator = torch.Generator().manual_seed(3401)
    incident = torch.tensor([0.4, -0.2, 0.7], dtype=dtype)
    axis = torch.tensor([0.1, 1.0, 0.2], dtype=dtype)
    mu = torch.rand(1000, generator=generator, dtype=dtype) * 1.98 - 0.99
    phi = (torch.rand(1000, generator=generator, dtype=dtype) * 2 - 1) * math.pi
    direction = mu_phi_to_direction(mu, phi, incident, axis)
    actual_mu, actual_phi = direction_to_mu_phi(direction, incident, axis)
    torch.testing.assert_close(actual_mu, mu, atol=tolerance, rtol=0)
    torch.testing.assert_close(actual_phi.sin(), phi.sin(), atol=tolerance, rtol=0)
    torch.testing.assert_close(actual_phi.cos(), phi.cos(), atol=tolerance, rtol=0)
    torch.testing.assert_close(
        direction.square().sum(-1), torch.ones_like(mu), atol=tolerance, rtol=0
    )


def test_folded_mirror_pair_and_pole_convention():
    incident = torch.tensor([0.4, -0.2, 0.7], dtype=torch.float64)
    axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    mu = torch.tensor([-0.8, 0.0, 0.9], dtype=torch.float64)
    v = torch.tensor([0.0, 0.27, 1.0], dtype=torch.float64)
    minus = folded_to_direction(mu, v, -1.0, incident, axis)
    plus = folded_to_direction(mu, v, 1.0, incident, axis)
    _, y, _ = scattering_frame(incident, axis)
    reflected = plus - 2 * (plus * y).sum(-1, keepdim=True) * y
    torch.testing.assert_close(reflected, minus, atol=2e-15, rtol=0)
    for direction in (minus, plus):
        actual_mu, actual_v = direction_to_folded(direction, incident, axis)
        torch.testing.assert_close(actual_mu, mu, atol=2e-15, rtol=0)
        torch.testing.assert_close(actual_v, v, atol=2e-15, rtol=0)
    poles = mu_phi_to_direction(torch.tensor([-1.0, 1.0], dtype=torch.float64), 1.7, incident, axis)
    actual_mu, phi = direction_to_mu_phi(poles, incident, axis)
    torch.testing.assert_close(
        actual_mu, torch.tensor([-1.0, 1.0], dtype=torch.float64), atol=0, rtol=0
    )
    torch.testing.assert_close(phi, torch.zeros_like(phi), atol=0, rtol=0)


def test_rotation_covariance_away_from_degenerate_axis():
    rotation, _ = torch.linalg.qr(
        torch.tensor([[0.2, 0.3, 0.9], [-0.8, 0.4, 0.1], [0.5, 0.1, 0.3]], dtype=torch.float64)
    )
    if torch.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    incident = torch.tensor([0.4, -0.2, 0.7], dtype=torch.float64)
    axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    outgoing = torch.tensor([[0.1, 0.3, 0.8], [0.3, -0.6, 0.2]], dtype=torch.float64)
    before = direction_to_mu_phi(outgoing, incident, axis)
    after = direction_to_mu_phi(outgoing @ rotation.T, incident @ rotation.T, axis @ rotation.T)
    for a, b in zip(before, after):
        torch.testing.assert_close(a, b, atol=2e-14, rtol=0)


def test_scaled_normalization_and_invalid_inputs():
    v = torch.tensor([[1e300, 2e300, -3e300], [1e-300, 2e-300, -3e-300]], dtype=torch.float64)
    actual = normalize_direction(v, validate_args=True)
    torch.testing.assert_close(actual[0], actual[1], atol=3e-16, rtol=0)
    torch.testing.assert_close(actual.square().sum(-1), torch.ones(2, dtype=torch.float64))
    assert bool(torch.isnan(normalize_direction(torch.zeros(3))).all())
    with pytest.raises(ValueError):
        normalize_direction(torch.zeros(3), validate_args=True)
    incident = torch.tensor([0.0, 0.0, 1.0])
    assert bool(torch.isnan(folded_to_direction(0.0, 0.5, 0.0, incident)).all())
    with pytest.raises(ValueError):
        folded_to_direction(0.0, 1.2, 1.0, incident, validate_args=True)
