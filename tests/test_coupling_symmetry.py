"""Physical axial symmetry, boundary continuity and analytic invertibility."""

import math

import pytest
import torch
from zuko.transforms import ComposedTransform

from phaseflow.coupling import AxialSymmetricCouplingTransform
from phaseflow.model import ModelConfig, PhaseFlow


def _nonidentity_model():
    generator = torch.Generator().manual_seed(2107)
    model = PhaseFlow(
        ModelConfig(
            one_blob_bins=4,
            num_bins=6,
            num_coupling_layers=4,
            hidden_features=(12, 12),
            g_hidden_features=(8,),
        )
    ).double()
    with torch.no_grad():
        model.g_head.mlp[-1].bias.fill_(0.65)
        for layer in model.couplings:
            for parameter in layer.hyper.parameters():
                parameter.copy_(
                    torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype) * 0.35
                )
    return model


def _directions(mu, phi):
    radius = ((1 - mu) * (1 + mu)).sqrt()
    return torch.stack((radius * phi.cos(), radius * phi.sin(), mu), dim=-1)


@pytest.mark.parametrize("incident_cosine", [-1.0, 1.0])
def test_physical_pdf_is_independent_of_arbitrary_axial_frame(incident_cosine):
    model = _nonidentity_model()
    conditions = torch.tensor([547.0, incident_cosine], dtype=torch.float64)
    mu, phi = torch.meshgrid(
        torch.tensor([-0.71, 0.13, 0.89], dtype=torch.float64),
        torch.linspace(-math.pi, math.pi, 39, dtype=torch.float64),
        indexing="ij",
    )
    log_pdf = model.log_prob(_directions(mu, phi), conditions)
    torch.testing.assert_close(log_pdf, log_pdf[:, :1].expand_as(log_pdf), atol=3e-13, rtol=0)
    # A nontrivial trained residual is tested, not only the initial HG model.
    assert float((log_pdf[1] - log_pdf[0]).abs().max().detach()) > 0.1


@pytest.mark.parametrize("incident_cosine", [-1.0, 1.0])
def test_axial_sampler_leaves_azimuth_uniform_and_radial_sample_independent(incident_cosine):
    model = _nonidentity_model()
    conditions = torch.tensor([532.0, incident_cosine], dtype=torch.float64)
    second = torch.linspace(0.013, 0.987, 73, dtype=torch.float64)
    uniforms = torch.stack((torch.full_like(second, 0.63), second), -1)
    directions, log_pdf = model.sample_from_uniform(uniforms, conditions)
    torch.testing.assert_close(directions[:, 2], directions[:1, 2].expand(73), atol=2e-15, rtol=0)
    folded_second = torch.where(second < 0.5, 2 * second, 2 * second - 1)
    recovered_azimuth = torch.atan2(directions[:, 1], directions[:, 0]).abs() / math.pi
    torch.testing.assert_close(recovered_azimuth, folded_second, atol=3e-15, rtol=0)
    torch.testing.assert_close(log_pdf, model.log_prob(directions, conditions), atol=3e-12, rtol=0)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_approach_to_axial_incidence_is_continuous(sign):
    model = _nonidentity_model()
    phi = torch.linspace(-2.9, 2.9, 31, dtype=torch.float64)
    directions = _directions(torch.full_like(phi, 0.37), phi)
    axial = model.log_prob(directions, torch.tensor([593.0, sign], dtype=torch.float64))
    errors = []
    for inclination in (1e-3, 1e-6):
        eta = sign * math.cos(inclination)
        actual = model.log_prob(directions, torch.tensor([593.0, eta], dtype=torch.float64))
        errors.append((actual - axial).abs().max())
    assert errors[1] < 0.02 * errors[0]
    assert errors[1] < 5e-5


def test_near_axial_azimuth_modulation_can_be_first_order_in_inclination():
    model = PhaseFlow(
        ModelConfig(
            one_blob_bins=0,
            num_bins=4,
            num_coupling_layers=2,
            hidden_features=(4,),
            g_hidden_features=(4,),
        )
    ).double()
    with torch.no_grad():
        model.couplings[1].hyper[-1].bias[4:8] = torch.tensor(
            [0.7, -0.2, 0.4, -0.9], dtype=torch.float64
        )
    phi = math.pi * torch.tensor([0.17, 0.61], dtype=torch.float64)
    directions = _directions(torch.full_like(phi, 0.3), phi)
    contrasts = []
    for inclination in (1e-3, 5e-4):
        conditions = torch.tensor([550.0, math.cos(inclination)], dtype=torch.float64)
        log_pdf = model.log_prob(directions, conditions)
        contrasts.append(log_pdf[0] - log_pdf[1])
    assert abs(float(contrasts[0].detach())) > 1e-5
    # Halving inclination halves the leading azimuth variation. A sin^2
    # gate instead gives a ratio near 1/4 and excludes this allowed behavior.
    torch.testing.assert_close(
        contrasts[1] / contrasts[0],
        torch.tensor(0.5, dtype=torch.float64),
        atol=2e-3,
        rtol=0,
    )


def test_axial_parameter_gradients_are_finite_with_trainable_g_context():
    model = _nonidentity_model()
    assert model.config.include_g_context
    conditions = torch.tensor([[531.0, -1.0], [587.0, 1.0]], dtype=torch.float64)
    directions = _directions(
        torch.tensor([-0.24, 0.71], dtype=torch.float64),
        torch.tensor([0.63, -1.78], dtype=torch.float64),
    )
    loss = -model.log_prob(directions, conditions).sum()
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
    # Conditions themselves do not require gradients. The eta endpoint's
    # singular coordinate derivative is not part of the parameter contract.
    assert float(model.g_head.mlp[-1].bias.grad.abs().max()) > 1e-5


def test_gated_couplings_remain_bijective_and_have_correct_batched_jacobians():
    torch.manual_seed(112)
    layers = [
        AxialSymmetricCouplingTransform(
            features=2,
            context=3,
            mask=[index % 2 == 1, index % 2 == 0],
            num_bins=7,
            hidden_features=(12,),
        ).double()
        for index in range(4)
    ]
    context = torch.tensor(
        [[0.2, 0.13, 0.2], [0.4, 0.5, -0.3], [0.7, 0.91, 0.7]], dtype=torch.float64
    )
    transform = ComposedTransform(*(layer(context) for layer in layers))
    values = 0.02 + 0.96 * torch.rand(113, 3, 2, dtype=torch.float64)
    actual, ladj = transform.call_and_ladj(values)
    restored, inverse_ladj = transform.inv.call_and_ladj(actual)
    assert actual.shape == values.shape and ladj.shape == (113, 3)
    torch.testing.assert_close(restored, values, atol=3e-12, rtol=0)
    torch.testing.assert_close(ladj + inverse_ladj, torch.zeros_like(ladj), atol=3e-11, rtol=0)
    assert bool(torch.isfinite(ladj).all())
    assert float((actual - values).abs().max().detach()) > 0.01
    # The full two-dimensional Jacobian includes conditioner dependence.
    point = values[0, 0].detach().requires_grad_()
    one_transform = ComposedTransform(*(layer(context[0]) for layer in layers))
    jacobian = torch.autograd.functional.jacobian(one_transform, point)
    expected = torch.linalg.slogdet(jacobian).logabsdet
    torch.testing.assert_close(one_transform.call_and_ladj(point)[1], expected, atol=2e-12, rtol=0)


def test_invalid_feature_count_cannot_bypass_axial_coupling_class():
    with pytest.raises(ValueError, match="exactly two"):
        AxialSymmetricCouplingTransform(features=1, context=2)
