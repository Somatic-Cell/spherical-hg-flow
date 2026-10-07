"""Sphere/HG measure, genuine coupling, exact symmetry, and precision checks."""

import copy
import math

import numpy as np
import pytest
import torch

from phaseflow.hg import hg_icdf, hg_log_prob
from phaseflow.sphere_model import SingleConditionSphereFlow, SphereFlowConfig


def make_model(*, g=0.35, eta=0.2, reflected=True, dtype=torch.float64, perturb=True):
    torch.manual_seed(802)
    model = SingleConditionSphereFlow(
        g,
        eta,
        SphereFlowConfig(
            hidden_features=(12,), num_bins=8, num_coupling_layers=4, mirror_symmetry=reflected
        ),
        dtype=dtype,
    )
    if perturb:
        with torch.no_grad():
            for layer in model.couplings:
                layer.hyper[-1].weight.normal_(0, 0.09)
                layer.hyper[-1].bias.add_(0.09 * torch.randn_like(layer.hyper[-1].bias))
    return model


def directions_from_angles(mu, phi):
    mu, phi = torch.broadcast_tensors(mu, phi)
    radius = ((1 - mu) * (1 + mu)).sqrt()
    return torch.stack((radius * phi.cos(), radius * phi.sin(), mu), -1)


@pytest.mark.parametrize("g", [-0.9991, 0.0, 0.9991])
@pytest.mark.parametrize("dtype,tolerance", [(torch.float32, 4e-6), (torch.float64, 7e-15)])
def test_identity_residual_is_the_supplied_hg_without_g_regression(g, dtype, tolerance):
    model = make_model(g=g, dtype=dtype, perturb=False)
    mu = torch.linspace(-0.999, 0.999, 131, dtype=torch.float64)
    phi = torch.linspace(-math.pi, math.pi, 131, dtype=torch.float64)
    actual = model.log_prob(directions_from_angles(mu, phi))
    torch.testing.assert_close(actual, hg_log_prob(mu, g), rtol=0, atol=tolerance)
    assert model.hg_g == g
    assert all("hg" not in name for name, _ in model.named_parameters())
    u = torch.tensor([[0.2, 0.1], [0.4, 0.4], [0.7, 0.8]], dtype=dtype)
    actual_direction, _ = model.sample_from_uniform(u)
    expected_mu = hg_icdf(u[:, 0].double(), g)
    expected_phi = 2 * math.pi * (u[:, 1].double() - 0.5)
    expected = directions_from_angles(expected_mu, expected_phi)
    torch.testing.assert_close(actual_direction, expected, rtol=0, atol=tolerance)


@pytest.mark.parametrize("reflected", [False, True])
@pytest.mark.parametrize("dtype,tolerance", [(torch.float32, 8e-5), (torch.float64, 8e-9)])
@pytest.mark.parametrize("g", [-0.995, 0.995])
def test_sampling_and_arbitrary_direction_eval_agree(reflected, dtype, tolerance, g):
    model = make_model(g=g, reflected=reflected, dtype=dtype)
    u = 0.98 * torch.rand(251, 2, dtype=dtype) + 0.01
    directions, sample_logp = model.sample_from_uniform(u)
    assert directions.dtype == torch.float64 and sample_logp.dtype == torch.float64
    torch.testing.assert_close(
        torch.linalg.vector_norm(directions, dim=-1),
        torch.ones_like(sample_logp),
        rtol=0,
        atol=3e-16,
    )
    torch.testing.assert_close(sample_logp, model.log_prob(directions), rtol=0, atol=tolerance)


def test_both_coordinates_are_coupled_and_joint_logdet_matches_autograd():
    model = make_model(reflected=False)
    point = torch.tensor([0.31, 0.27], dtype=torch.float64, requires_grad=True)
    _, reported = model._run_couplings(point, inverse=False)
    jacobian = torch.autograd.functional.jacobian(
        lambda x: model._run_couplings(x, inverse=False)[0], point
    )
    sign, expected = torch.linalg.slogdet(jacobian)
    assert sign == 1
    assert float(jacobian[0, 1].abs()) > 1e-4
    assert float(jacobian[1, 0].abs()) > 1e-4
    torch.testing.assert_close(reported, expected, rtol=0, atol=3e-14)
    transformed, forward = model._run_couplings(point, inverse=False)
    restored, reverse = model._run_couplings(transformed, inverse=True)
    torch.testing.assert_close(restored, point, rtol=0, atol=2e-14)
    torch.testing.assert_close(forward + reverse, torch.zeros_like(forward), rtol=0, atol=3e-13)


@pytest.mark.parametrize("reflected", [False, True])
def test_full_composition_preserves_the_periodic_seam(reflected):
    model = make_model(reflected=reflected)
    seam = torch.tensor([[0.27, 0.0], [0.27, 1.0]], dtype=torch.float64)
    result, logdet = model._run_couplings(seam, inverse=False)
    torch.testing.assert_close(result[0, 0], result[1, 0], rtol=0, atol=2e-15)
    torch.testing.assert_close(result[:, 1], seam[:, 1], rtol=0, atol=0)
    torch.testing.assert_close(logdet[0], logdet[1], rtol=0, atol=6e-15)


@pytest.mark.parametrize("reflected", [False, True])
def test_surface_normalization_with_independent_solid_angle_quadrature(reflected):
    model = make_model(reflected=reflected)
    nodes, weights = np.polynomial.legendre.leggauss(128)
    phi = (torch.arange(256, dtype=torch.float64) + 0.5) * (2 * math.pi / 256) - math.pi
    directions = directions_from_angles(torch.from_numpy(nodes)[:, None], phi[None, :])
    with torch.no_grad():
        pdf = model.pdf(directions)
    integral = (pdf.numpy() * weights[:, None]).sum() * (2 * math.pi / 256)
    assert abs(integral - 1) < 5e-4
    assert float(pdf.std(dim=1).max()) > 1e-4


def test_reflection_is_enforced_on_the_full_circle_and_optional_for_generic_model():
    model = make_model()
    u = torch.tensor([[0.19, 0.16], [0.51, 0.36], [0.87, 0.71]], dtype=torch.float64)
    reflected_u = u * torch.tensor([1.0, -1.0]) + torch.tensor([0.0, 1.0])
    directions, logp = model.sample_from_uniform(u)
    reflected_directions, reflected_logp = model.sample_from_uniform(reflected_u)
    expected = directions * torch.tensor([1.0, -1.0, 1.0])
    torch.testing.assert_close(reflected_directions, expected, rtol=0, atol=3e-14)
    torch.testing.assert_close(reflected_logp, logp, rtol=0, atol=3e-13)
    torch.testing.assert_close(model.log_prob(expected), logp, rtol=0, atol=3e-13)
    generic = make_model(reflected=False)
    difference = (generic.log_prob(expected) - generic.log_prob(directions)).abs().max()
    assert float(difference.detach()) > 1e-4


@pytest.mark.parametrize("eta", [-1.0, 1.0])
@pytest.mark.parametrize("reflected", [False, True])
def test_axial_incidence_eliminates_azimuth_dependence(eta, reflected):
    model = make_model(eta=eta, reflected=reflected)
    phi = torch.linspace(-math.pi, math.pi, 401, dtype=torch.float64)
    directions = directions_from_angles(torch.tensor(0.37, dtype=torch.float64), phi)
    logp = model.log_prob(directions)
    torch.testing.assert_close(logp, logp[0].expand_as(logp), rtol=0, atol=3e-15)
    u = torch.tensor([[0.2, 0.07], [0.2, 0.54]], dtype=torch.float64)
    output, _ = model._run_couplings(u, inverse=True)
    torch.testing.assert_close(output[:, 1], u[:, 1], rtol=0, atol=3e-16)
    torch.testing.assert_close(output[0, 0], output[1, 0], rtol=0, atol=2e-16)


def test_each_sample_or_eval_runs_each_conditioner_once():
    model = make_model()
    calls = [0] * len(model.couplings)

    def record(index):
        def hook(_module, _inputs, _output):
            calls[index] += 1

        return hook

    handles = [
        layer.hyper.register_forward_hook(record(i)) for i, layer in enumerate(model.couplings)
    ]
    try:
        directions, _ = model.sample_and_log_prob(13)
        assert calls == [1] * len(calls)
        calls[:] = [0] * len(calls)
        model.log_prob(directions)
        assert calls == [1] * len(calls)
    finally:
        for handle in handles:
            handle.remove()


def test_external_physics_round_trips_exactly_and_never_downcasts_with_weights():
    g = 0.999_999_971_283_746
    eta = -0.138_276_463_281_792
    model = make_model(g=g, eta=eta)
    model.float()
    assert model.dtype == torch.float32
    assert model.hg_g == g and model.incident_cosine == eta
    state = copy.deepcopy(model.state_dict())
    restored = make_model(g=0, eta=0).float()
    restored.load_state_dict(state)
    assert restored.hg_g == g and restored.incident_cosine == eta
    restored.double()
    assert restored.hg_g == g and restored.incident_cosine == eta
    assert SphereFlowConfig.from_dict(restored.config.to_dict()) == restored.config


def test_exact_poles_use_canonical_azimuth_and_rounded_samples_are_reported():
    model = make_model()
    directions = torch.tensor(
        [[0.0, 0.0, 1.0], [-0.0, 0.0, 1.0], [-0.0, -0.0, 1.0]], dtype=torch.float64
    )
    logp = model.log_prob(directions)
    assert torch.equal(logp, logp[0].expand_as(logp))
    too_narrow = make_model(g=1 - 2**-40, perturb=False)
    u = torch.tensor([[0.5, 0.4]], dtype=torch.float64)
    with pytest.raises(FloatingPointError, match="rounded spherical pole"):
        too_narrow.sample_from_uniform(u)
    too_narrow.validate_args = False
    output, logp = too_narrow.sample_from_uniform(u)
    assert bool(torch.isnan(output).all()) and bool(torch.isnan(logp).all())


@pytest.mark.parametrize("g", [-1.0, 1.0, float("nan"), float("inf")])
def test_invalid_external_g_is_rejected_without_clipping(g):
    with pytest.raises(ValueError, match="external HG g"):
        SingleConditionSphereFlow(g, 0.2)


def test_invalid_inputs_and_incompatible_checkpoint_fail_explicitly():
    model = make_model()
    with pytest.raises(ValueError, match="unit vectors"):
        model.log_prob(torch.tensor([1.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="uniforms"):
        model.sample_from_uniform(torch.tensor([0.0, 0.4]))
    with pytest.raises(ValueError, match="uniforms"):
        model.sample_from_uniform(torch.tensor([0.4, 1.0]))
    with pytest.raises(ValueError, match="incident_cosine"):
        SingleConditionSphereFlow(0.1, 1.1)
    with pytest.raises(ValueError, match="configuration differs"):
        model.set_extra_state({})


def test_float32_uniform_conversion_reports_representation_failure_separately():
    model = make_model(dtype=torch.float32)
    precise = torch.tensor([[1 - 2**-40, 0.4]], dtype=torch.float64)
    with pytest.raises(FloatingPointError, match="uniforms rounded"):
        model.sample_from_uniform(precise)
    model.validate_args = False
    directions, logp = model.sample_from_uniform(precise)
    assert bool(torch.isnan(directions).all()) and bool(torch.isnan(logp).all())
