"""Surface-measure, bidirectional inference, and conditioning regression tests."""

import math

import numpy as np
import pytest
import torch

from phaseflow.encoding import ConditionEncoder
from phaseflow.hg import hg_log_prob
from phaseflow.model import ModelConfig, PhaseFlow


def make_model(dtype=torch.float64, g=0.55, perturb=True):
    torch.manual_seed(709)
    model = PhaseFlow(
        ModelConfig(
            g_hidden_features=(8,),
            hidden_features=(12,),
            num_coupling_layers=4,
            num_bins=8,
            one_blob_bins=5,
        )
    ).to(dtype=dtype)
    with torch.no_grad():
        model.g_head.mlp[-1].bias.fill_(math.atanh(g / model.config.g_limit))
        if perturb:
            for layer in model.couplings:
                layer.hyper[-1].weight.normal_(0, 0.07)
                layer.hyper[-1].bias.add_(torch.randn_like(layer.hyper[-1].bias) * 0.07)
    return model


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_identity_residual_has_hg_surface_density(dtype):
    model = make_model(dtype, g=0.91, perturb=False)
    c = torch.tensor([550.0, -0.35], dtype=dtype)
    z = torch.linspace(-0.999, 0.999, 97, dtype=torch.float64)
    phi = torch.linspace(-math.pi, math.pi, 97, dtype=torch.float64)
    radius = torch.sqrt(1 - z.square())
    directions = torch.stack((radius * phi.cos(), radius * phi.sin(), z), -1)
    actual = model.log_prob(directions, c)
    expected = hg_log_prob(z, model.hg_g(c).double())
    # The model is initialized in torch's default float32 before conversion;
    # its identity slope biases retain that initialization roundoff in float64.
    tolerance = 3e-6 if dtype == torch.float32 else 1e-7
    torch.testing.assert_close(actual, expected, rtol=0, atol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("g", [-0.995, 0.0, 0.995])
def test_sample_pdf_matches_arbitrary_direction_eval(dtype, g):
    model = make_model(dtype, g=g)
    c = torch.tensor([[400.0, -0.8], [560.0, 0.1], [700.0, 0.75]], dtype=dtype)
    uniforms = torch.rand(121, 3, 2, dtype=dtype) * 0.98 + 0.01
    directions, sample_logp = model.sample_from_uniform(uniforms, c)
    assert directions.dtype == torch.float64
    torch.testing.assert_close(
        torch.linalg.vector_norm(directions, dim=-1),
        torch.ones_like(sample_logp),
        rtol=0,
        atol=3e-16,
    )
    tolerance = 3e-5 if dtype == torch.float32 else 8e-9
    torch.testing.assert_close(sample_logp, model.log_prob(directions, c), rtol=0, atol=tolerance)


def test_solid_angle_normalization_and_exact_reflection():
    model = make_model()
    c = torch.tensor([580.0, -0.3], dtype=torch.float64)
    mu, weights = np.polynomial.legendre.leggauss(128)
    phi = (np.arange(256) + 0.5) * (2 * math.pi / 256) - math.pi
    mu, phi = np.meshgrid(mu, phi, indexing="ij")
    transverse = np.sqrt(1 - mu * mu)
    directions = torch.tensor(
        np.stack((transverse * np.cos(phi), transverse * np.sin(phi), mu), -1)
    )
    with torch.no_grad():
        density = model.pdf(directions, c)
        reflected = directions * torch.tensor([1.0, -1.0, 1.0])
        assert torch.equal(density, model.pdf(reflected, c))
    integral = (density.numpy() * weights[:, None]).sum() * (2 * math.pi / 256)
    assert abs(integral - 1.0) < 5e-4
    # The learned family must retain two-dimensional directional variation.
    assert float(density.std(dim=1).max()) > 1e-4


def test_each_direction_runs_each_conditioner_once():
    model = make_model()
    c = torch.tensor([550.0, 0.3], dtype=torch.float64)
    calls = [0] * len(model.couplings)

    def record(index):
        def hook(_module, _inputs, _output):
            calls[index] += 1

        return hook

    handles = [
        layer.hyper.register_forward_hook(record(i)) for i, layer in enumerate(model.couplings)
    ]
    try:
        directions, _ = model.sample_and_log_prob(c, 7)
        assert calls == [1] * len(model.couplings)
        calls[:] = [0] * len(model.couplings)
        model.log_prob(directions, c)
        assert calls == [1] * len(model.couplings)
    finally:
        for handle in handles:
            handle.remove()


def test_distribution_shapes_and_context_broadcasting():
    model = make_model()
    c = torch.tensor([[500.0, -0.6], [650.0, 0.5]], dtype=torch.float64)
    distribution = model(c)
    assert distribution.event_shape == (3,)
    assert distribution.batch_shape == (2,)
    directions, logp = distribution.rsample_and_log_prob((3, 4))
    assert directions.shape == (3, 4, 2, 3)
    assert logp.shape == (3, 4, 2)
    torch.testing.assert_close(logp, distribution.log_prob(directions), atol=1e-10, rtol=0)
    assert distribution.rsample().shape == (2, 3)
    assert model(c[0]).rsample().shape == (3,)
    assert model.log_prob(torch.tensor([1.0, 0.0, 0.0]), c).shape == (2,)
    assert distribution.expand((5, 2)).rsample().shape == (5, 2, 3)


def test_joint_gradient_includes_hg_and_probability_coordinates():
    model = make_model()
    c = torch.tensor([570.0, -0.35], dtype=torch.float64)
    directions = torch.tensor([[0.6, 0.0, 0.8], [0.0, -0.8, -0.6]], dtype=torch.float64)
    parameter = model.g_head.mlp[-1].bias
    loss = -model.log_prob(directions, c).mean()
    gradient = torch.autograd.grad(loss, parameter)[0].item()
    epsilon = 1e-6
    original = parameter.detach().clone()
    with torch.no_grad():
        parameter.copy_(original + epsilon)
        plus = -model.log_prob(directions, c).mean().item()
        parameter.copy_(original - epsilon)
        minus = -model.log_prob(directions, c).mean().item()
        parameter.copy_(original)
    assert math.isfinite(gradient)
    assert abs(gradient - (plus - minus) / (2 * epsilon)) < 1e-7


def test_pole_density_uses_canonical_azimuth_for_signed_zeros():
    model = make_model()
    c = torch.tensor([570.0, 0.1], dtype=torch.float64)
    directions = torch.tensor(
        [[0.0, 0.0, 1.0], [-0.0, 0.0, 1.0], [-0.0, -0.0, 1.0]], dtype=torch.float64
    )
    logp = model.log_prob(directions, c)
    assert torch.equal(logp, logp[0].expand_as(logp))


def test_one_blob_layout_and_dtype_conversion():
    encoder = ConditionEncoder(380.0, 720.0, bins=5).float().double()
    c = torch.tensor([[380.0, -1.0], [550.0, 0.0], [720.0, 1.0]], dtype=torch.float64)
    encoded = encoder(c)
    raw = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=torch.float64)
    edges = torch.arange(6, dtype=torch.float64) / 5
    standard = torch.distributions.Normal(0.0, 1.0)
    cdf = standard.cdf((edges - raw[..., None]) * 5)
    expected = torch.cat((raw, (cdf[..., 1:] - cdf[..., :-1]).flatten(-2)), -1)
    torch.testing.assert_close(encoded, expected, rtol=0, atol=2e-16)
    assert encoded[0, 2:7].sum() < 0.51  # Boundary mass is not renormalized.
    assert torch.equal(ConditionEncoder(380.0, 720.0, bins=0)(c), raw)


@pytest.mark.parametrize(
    "change",
    [
        {"num_bins": 3.5},
        {"num_coupling_layers": 3},
        {"hidden_features": [True]},
        {"one_blob_bins": -1},
        {"include_g_context": 1},
        {"g_limit": 1.0},
    ],
)
def test_configuration_rejects_ambiguous_or_invalid_fields(change):
    with pytest.raises(ValueError):
        ModelConfig(**change)


def test_invalid_physical_inputs_fail_before_inference():
    model = make_model()
    with pytest.raises(ValueError, match="outside"):
        model.hg_g(torch.tensor([300.0, 0.0]))
    with pytest.raises(ValueError, match="incident_cosine"):
        model.hg_g(torch.tensor([500.0, 30.0]))
    with pytest.raises(ValueError, match="unit"):
        model.log_prob(torch.tensor([1.0, 1.0, 1.0]), torch.tensor([500.0, 0.0]))
    with pytest.raises(ValueError, match="uniforms"):
        model.sample_from_uniform(torch.tensor([0.0, 0.2]), torch.tensor([500.0, 0.0]))
