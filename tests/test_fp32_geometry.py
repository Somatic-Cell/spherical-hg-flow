"""FP32 sphere arithmetic against independent HG and FP64 references."""

import copy
import math

import numpy as np
import pytest
import torch

from phaseflow.hg import _hg_cdf_and_log_prob_from_distances, _hg_icdf_distances
from phaseflow.sphere_model import SingleConditionSphereFlow, SphereFlowConfig


def make_fp32_model(g=0.8, *, perturb=False, eta=0.2, reflected=True):
    torch.manual_seed(821)
    model = SingleConditionSphereFlow(
        g,
        eta,
        SphereFlowConfig(
            hidden_features=(12,),
            num_bins=8,
            num_coupling_layers=4,
            mirror_symmetry=reflected,
            geometry_dtype="model",
            spline_dtype="model",
        ),
        dtype=torch.float32,
    )
    if perturb:
        with torch.no_grad():
            for layer in model.couplings:
                layer.hyper[-1].weight.normal_(0, 0.07)
                layer.hyper[-1].bias.add_(0.07 * torch.randn_like(layer.hyper[-1].bias))
    return model


def analytic_log_hg_from_directions(directions, g):
    """Independent scalar FP64 reference using the half scattering angle."""
    result = []
    for x, y, z in directions.detach().double().cpu().tolist():
        theta = math.atan2(math.hypot(x, y), z if g >= 0 else -z)
        magnitude = abs(g)
        squared = (1 - magnitude) ** 2 + 4 * magnitude * math.sin(theta / 2) ** 2
        result.append(
            math.log1p(-magnitude) + math.log1p(magnitude)
            - math.log(4 * math.pi) - 1.5 * math.log(squared)
        )
    return torch.tensor(result, dtype=torch.float64)


@pytest.mark.parametrize("g", [-1 + 2**-40, -0.999, 0.0, 0.999, 1 - 2**-40])
def test_fp32_identity_hg_resolves_lobes_even_when_g_and_z_round_to_one(g):
    model = make_fp32_model(g)
    uniforms = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.7, 0.6], [0.9, 0.8]])
    directions, sample_logp = model.sample_from_uniform(uniforms)
    assert model.dtype == model.geometry_dtype == model.spline_dtype == torch.float32
    assert directions.dtype == sample_logp.dtype == torch.float32
    assert all(buffer.dtype == torch.float32 for buffer in model.buffers())
    assert model.hg_g == g
    assert bool((torch.linalg.vector_norm(directions[:, :2], dim=-1) > 0).all())
    expected = analytic_log_hg_from_directions(directions, g)
    torch.testing.assert_close(sample_logp.double(), expected, atol=1.5e-5, rtol=0)
    torch.testing.assert_close(model.log_prob(directions).double(), expected, atol=1.5e-5, rtol=0)
    torch.testing.assert_close(
        model.hg_base_direction_log_prob(directions).double(), expected, atol=1.5e-5, rtol=0
    )
    if abs(g) > 1 - 2**-30:
        # These values are representable directions, not exact poles.
        assert bool((directions[:, 2].abs() == 1).all())
        assert abs(float(model._hg_coefficient)) == 1


@pytest.mark.parametrize("sign", [-1, 1])
def test_tiny_transverse_directions_have_distinct_finite_fp32_density(sign):
    model = make_fp32_model(sign * (1 - 2**-30))
    radius = torch.tensor([2**-33, 2**-30, 2**-27], dtype=torch.float32)
    directions = torch.stack((radius, torch.zeros_like(radius), sign * torch.ones_like(radius)), -1)
    actual = model.log_prob(directions)
    assert bool((actual.diff() < -0.1).all())
    torch.testing.assert_close(
        actual.double(), analytic_log_hg_from_directions(directions, model.hg_g),
        atol=1.5e-5, rtol=0,
    )


@pytest.mark.parametrize("reflected", [False, True])
@pytest.mark.parametrize("g", [-0.999, 0.999])
def test_fp32_nonidentity_sample_eval_and_cylinder_inverse_match(reflected, g):
    model = make_fp32_model(g, perturb=True, reflected=reflected)
    uniforms = 0.98 * torch.rand(1024, 2) + 0.01
    directions, logp = model.sample_from_uniform(uniforms)
    torch.testing.assert_close(logp, model.log_prob(directions), atol=1e-5, rtol=0)
    transformed, forward = model._run_couplings(uniforms, inverse=False)
    restored, inverse = model._run_couplings(transformed, inverse=True)
    torch.testing.assert_close(restored, uniforms, atol=5e-7, rtol=0)
    torch.testing.assert_close(forward + inverse, torch.zeros_like(forward), atol=6e-6, rtol=0)
    torch.testing.assert_close(
        torch.linalg.vector_norm(directions, dim=-1), torch.ones_like(logp), atol=3e-7, rtol=0
    )


def test_fp32_likelihood_and_weight_gradients_match_same_formula_in_fp64():
    model = make_fp32_model(0.97, perturb=True)
    reference = copy.deepcopy(model).double()
    directions, _ = model.sample_from_uniform(0.9 * torch.rand(128, 2) + 0.05)
    directions = directions.detach()
    actual = model.log_prob(directions)
    expected = reference.log_prob(directions.double())
    torch.testing.assert_close(actual.double(), expected, atol=8e-6, rtol=0)
    (-actual.mean()).backward()
    (-expected.mean()).backward()
    for learned, precise in zip(model.parameters(), reference.parameters(), strict=True):
        assert learned.grad is not None and bool(torch.isfinite(learned.grad).all())
        torch.testing.assert_close(learned.grad.double(), precise.grad, atol=2e-6, rtol=8e-4)


def test_fp32_mirror_seam_and_axial_invariance():
    model = make_fp32_model(0.7, perturb=True)
    u = torch.tensor([[0.19, 0.16], [0.51, 0.36], [0.87, 0.71]])
    directions, logp = model.sample_from_uniform(u)
    reflected = directions * torch.tensor([1.0, -1.0, 1.0])
    torch.testing.assert_close(model.log_prob(reflected), logp, atol=8e-6, rtol=0)
    seam = torch.tensor([[0.27, 0.0], [0.27, 1.0]])
    result, logdet = model._run_couplings(seam, inverse=False)
    torch.testing.assert_close(result[0, 0], result[1, 0], atol=3e-7, rtol=0)
    torch.testing.assert_close(logdet[0], logdet[1], atol=2e-6, rtol=0)
    for eta in (-1.0, 1.0):
        axial = make_fp32_model(0.7, eta=eta, perturb=True)
        phi = torch.linspace(-math.pi, math.pi, 71)
        d = torch.stack((math.sqrt(0.75) * phi.cos(), math.sqrt(0.75) * phi.sin(),
                         torch.full_like(phi, 0.5)), -1)
        p = axial.log_prob(d)
        torch.testing.assert_close(p, p[0].expand_as(p), atol=3e-6, rtol=0)


@pytest.mark.parametrize("g", [-0.8, 0.0, 0.8])
def test_fp32_hg_normalization_first_moment_and_cdf_jacobian(g):
    model = make_fp32_model(g)
    nodes, weights = np.polynomial.legendre.leggauss(128)
    mu = torch.tensor(nodes, dtype=torch.float32)
    directions = torch.stack(((1 - mu.square()).sqrt(), torch.zeros_like(mu), mu), -1)
    density = model.pdf(directions).detach().double().numpy()
    assert abs(float(np.sum(weights * density) * 2 * math.pi) - 1) < 2e-6
    assert abs(float(np.sum(weights * density * nodes) * 2 * math.pi) - g) < 2e-6
    x = torch.tensor([-0.9, -0.2, 0.3, 0.9], requires_grad=True)
    a, b = model._hg_endpoint_distances
    cdf, logp = _hg_cdf_and_log_prob_from_distances(1 - x, 1 + x, a, b)
    derivative = torch.autograd.grad(cdf.sum(), x)[0]
    torch.testing.assert_close(derivative, 2 * math.pi * logp.exp(), atol=2e-6, rtol=2e-6)


def test_fp32_hg_quantile_distances_round_trip_and_exact_endpoints():
    u = torch.linspace(0, 1, 2001)[:, None]
    g = [-1 + 2**-40, -0.9, 0.0, 0.9, 1 - 2**-40]
    a = torch.tensor([1 - value for value in g])
    b = torch.tensor([1 + value for value in g])
    dm, dp = _hg_icdf_distances(u, a, b)
    cdf, logp = _hg_cdf_and_log_prob_from_distances(dm, dp, a, b)
    torch.testing.assert_close(cdf, u.expand_as(cdf), atol=3e-7, rtol=0)
    assert bool(torch.isfinite(logp).all())
    assert bool((dp[0] == 0).all()) and bool((dm[-1] == 0).all())


def test_old_checkpoint_missing_geometry_precision_keeps_fp64_behavior():
    old_config = SphereFlowConfig(hidden_features=(12,), num_bins=8, num_coupling_layers=4)
    original = SingleConditionSphereFlow(0.9, 0.2, old_config, dtype=torch.float32)
    state = copy.deepcopy(original.state_dict())
    state["_extra_state"]["config"].pop("geometry_dtype")
    restored_config = SphereFlowConfig.from_dict(state["_extra_state"]["config"])
    assert restored_config.geometry_dtype == "float64"
    restored = SingleConditionSphereFlow(0.0, 0.0, restored_config, dtype=torch.float32)
    restored.load_state_dict(state)
    u = torch.tensor([[0.2, 0.3], [0.6, 0.7]], dtype=torch.float64)
    actual = restored.sample_from_uniform(u)
    expected = original.sample_from_uniform(u)
    assert restored.geometry_dtype == torch.float64
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match="geometry_dtype"):
        SphereFlowConfig(geometry_dtype="float16")


def test_fp32_external_g_and_endpoint_cache_survive_moves_and_state_restore():
    g = 1 - 2**-40
    source = make_fp32_model(g)
    source.double().float()
    assert source.hg_g == g
    assert float(source._hg_endpoint_distances[0]) == 1 - g
    target = make_fp32_model(0.0)
    target.load_state_dict(copy.deepcopy(source.state_dict()))
    assert target.hg_g == g
    torch.testing.assert_close(target._hg_endpoint_distances, source._hg_endpoint_distances)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_cuda_fp32_geometry_sampling_density_and_gradients():
    cpu = make_fp32_model(0.999, perturb=True)
    cuda = copy.deepcopy(cpu).cuda()
    u = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.7, 0.6], [0.9, 0.8]])
    expected_d, expected_p = cpu.sample_from_uniform(u)
    actual_d, actual_p = cuda.sample_from_uniform(u.cuda())
    assert actual_d.is_cuda and actual_d.dtype == torch.float32
    torch.testing.assert_close(actual_d.cpu(), expected_d, atol=4e-7, rtol=1e-5)
    torch.testing.assert_close(actual_p.cpu(), expected_p, atol=2e-5, rtol=0)
    p = cuda.log_prob(actual_d.detach())
    torch.testing.assert_close(actual_p, p, atol=2e-5, rtol=0)
    (-p.mean()).backward()
    assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
               for parameter in cuda.parameters())
