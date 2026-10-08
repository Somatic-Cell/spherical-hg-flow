"""Sphere/HG measure, genuine coupling, exact symmetry, and precision checks."""

import copy
import math

import numpy as np
import pytest
import torch

from phaseflow.hg import hg_icdf, hg_log_prob
from phaseflow.sphere_model import SingleConditionSphereFlow, SphereFlowConfig


def make_model(
    *, g=0.35, eta=0.2, reflected=True, dtype=torch.float64, spline_dtype="model", perturb=True
):
    torch.manual_seed(802)
    model = SingleConditionSphereFlow(
        g,
        eta,
        SphereFlowConfig(
            hidden_features=(12,),
            num_bins=8,
            num_coupling_layers=4,
            mirror_symmetry=reflected,
            spline_dtype=spline_dtype,
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


def test_hg_device_cache_preserves_binary64_physics_across_dtype_moves_and_deepcopy():
    g = 0.999_999_971_283_746
    model = make_model(g=g, spline_dtype="float64")
    expected_bits = torch.tensor(g, dtype=torch.float64).view(torch.int64)

    def check_cached_physics(candidate):
        assert candidate.hg_g == g
        assert candidate._hg_coefficient.dtype == torch.float64
        assert candidate._hg_coefficient.device == candidate.device
        assert torch.equal(candidate._hg_coefficient.cpu().view(torch.int64), expected_bits)
        assert "_hg_coefficient" not in candidate.state_dict()
        mu = torch.tensor([-0.31, 0.72, 1 - 2**-32], dtype=torch.float64)
        torch.testing.assert_close(
            candidate.hg_base_log_prob(mu).cpu(), hg_log_prob(mu, g), rtol=0, atol=0
        )

    check_cached_physics(model)
    check_cached_physics(model.float())
    check_cached_physics(model.double())
    check_cached_physics(model.to("cpu", dtype=torch.float32))
    copied = copy.deepcopy(model)
    check_cached_physics(copied)
    assert copied._hg_coefficient.data_ptr() != model._hg_coefficient.data_ptr()


def test_loading_physics_refreshes_the_nonpersistent_hg_cache():
    source = make_model(g=-0.918_372_645_091_827, spline_dtype="float64")
    target = make_model(g=0.0, spline_dtype="float64", dtype=torch.float32)
    target.set_extra_state(copy.deepcopy(source.get_extra_state()))
    assert target.hg_g == source.hg_g
    assert target._hg_coefficient.item() == source.hg_g
    assert target._hg_coefficient.dtype == torch.float64
    replacement = make_model(g=0.731_982_734_651_092, spline_dtype="float64")
    target.load_state_dict(replacement.state_dict())
    assert target.hg_g == replacement.hg_g
    assert target._hg_coefficient.item() == replacement.hg_g
    assert target._hg_coefficient.dtype == torch.float64
    assert target.dtype == torch.float32
    assert "_hg_coefficient" not in target.state_dict()


def test_hg_calls_reuse_cached_device_scalar_without_per_call_scalar_construction(monkeypatch):
    model = make_model(dtype=torch.float32, spline_dtype="float64")
    model.validate_args = False
    uniforms = torch.tensor([[0.19, 0.24], [0.67, 0.83]], dtype=torch.float64)
    original_as_tensor = torch.as_tensor
    cached_pointer = model._hg_coefficient.data_ptr()

    def tensor_input_only(value, *args, **kwargs):
        assert isinstance(value, torch.Tensor), "per-call scalar construction can copy CPU -> CUDA"
        return original_as_tensor(value, *args, **kwargs)

    def no_tensor_construction(*_args, **_kwargs):
        pytest.fail("the fixed HG coefficient must not be recreated during model arithmetic")

    with monkeypatch.context() as context:
        context.setattr(torch, "as_tensor", tensor_input_only)
        context.setattr(torch, "tensor", no_tensor_construction)
        directions, logp = model.sample_from_uniform(uniforms)
        evaluated = model.log_prob(directions)
        baseline = model.hg_base_log_prob(directions[:, 2])
    assert model._hg_coefficient.data_ptr() == cached_pointer
    assert bool(torch.isfinite(baseline).all())
    torch.testing.assert_close(logp, evaluated, rtol=0, atol=3e-12)


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


@pytest.mark.parametrize("reflected", [False, True])
@pytest.mark.parametrize("g", [-0.995, 0.995])
def test_float32_conditioners_keep_double_precision_cylinder_round_trips(reflected, g):
    model = make_model(
        g=g, reflected=reflected, dtype=torch.float32, spline_dtype="float64"
    )
    u = 0.98 * torch.rand(251, 2, dtype=torch.float64) + 0.01
    transformed, logdet = model._run_couplings(u, inverse=False)
    restored, reverse = model._run_couplings(transformed, inverse=True)
    assert model.dtype == torch.float32 and model.spline_dtype == torch.float64
    assert transformed.dtype == logdet.dtype == torch.float64
    torch.testing.assert_close(restored, u, rtol=0, atol=3e-13)
    torch.testing.assert_close(logdet + reverse, torch.zeros_like(logdet), rtol=0, atol=3e-12)
    directions, logp = model.sample_from_uniform(u)
    torch.testing.assert_close(logp, model.log_prob(directions), rtol=0, atol=8e-9)


def test_double_precision_splines_accept_uniforms_unrepresentable_in_float32():
    model = make_model(g=0.0, dtype=torch.float32, spline_dtype="float64", perturb=False)
    u = torch.tensor([[1 - 2**-26, 1 - 2**-40]], dtype=torch.float64)
    assert bool((u.float() == 1).all())
    directions, logp = model.sample_from_uniform(u)
    assert bool(torch.isfinite(directions).all() & torch.isfinite(logp).all())
    assert bool((directions[:, 2].abs() < 1).all())
    torch.testing.assert_close(logp, model.log_prob(directions), rtol=0, atol=8e-13)


def test_mixed_precision_random_sampling_keeps_the_double_precision_uniform_stream():
    # Equal weights, with only their dtype changed, must retain the same
    # externally reproducible base stream when both use float64 coordinates.
    mixed = make_model(dtype=torch.float32, spline_dtype="float64")
    full_double = copy.deepcopy(mixed).double()
    mixed_samples, _ = mixed.sample_and_log_prob(
        37, generator=torch.Generator().manual_seed(591)
    )
    double_samples, _ = full_double.sample_and_log_prob(
        37, generator=torch.Generator().manual_seed(591)
    )
    torch.testing.assert_close(mixed_samples, double_samples, rtol=0, atol=2e-7)


def test_mixed_precision_preserves_reflection_and_periodic_seam():
    model = make_model(dtype=torch.float32, spline_dtype="float64")
    u = torch.tensor([[0.19, 0.16], [0.51, 0.36], [0.87, 0.71]], dtype=torch.float64)
    reflected_u = torch.stack((u[:, 0], 1 - u[:, 1]), -1)
    directions, logp = model.sample_from_uniform(u)
    reflected_directions, reflected_logp = model.sample_from_uniform(reflected_u)
    torch.testing.assert_close(
        reflected_directions, directions * torch.tensor([1.0, -1.0, 1.0]), rtol=0, atol=3e-13
    )
    torch.testing.assert_close(reflected_logp, logp, rtol=0, atol=3e-12)
    seam = torch.tensor([[0.27, 0.0], [0.27, 1.0]], dtype=torch.float64)
    transformed, logdet = model._run_couplings(seam, inverse=False)
    torch.testing.assert_close(transformed[0, 0], transformed[1, 0], rtol=0, atol=0)
    torch.testing.assert_close(transformed[:, 1], seam[:, 1], rtol=0, atol=0)
    torch.testing.assert_close(logdet[0], logdet[1], rtol=0, atol=3e-14)


@pytest.mark.parametrize("eta", [-1.0, 1.0])
def test_mixed_precision_preserves_axial_incidence_invariance(eta):
    model = make_model(eta=eta, dtype=torch.float32, spline_dtype="float64")
    phi = torch.linspace(-math.pi, math.pi, 41, dtype=torch.float64)
    directions = directions_from_angles(torch.tensor(0.37, dtype=torch.float64), phi)
    logp = model.log_prob(directions)
    torch.testing.assert_close(logp, logp[0].expand_as(logp), rtol=0, atol=3e-15)


def test_mixed_precision_training_gradients_agree_with_double_precision_reference():
    mixed = make_model(dtype=torch.float32, spline_dtype="float64")
    full_double = copy.deepcopy(mixed).double()
    mu = torch.linspace(-0.9, 0.9, 97, dtype=torch.float64)
    phi = torch.linspace(-2.9, 2.9, 97, dtype=torch.float64)
    directions = directions_from_angles(mu, phi)
    mixed_loss = -mixed.log_prob(directions).mean()
    reference_loss = -full_double.log_prob(directions).mean()
    torch.testing.assert_close(mixed_loss, reference_loss, rtol=0, atol=2e-7)
    mixed_loss.backward()
    reference_loss.backward()
    for parameter, reference_parameter in zip(mixed.parameters(), full_double.parameters()):
        assert parameter.grad is not None and reference_parameter.grad is not None
        assert parameter.grad.dtype == torch.float32
        assert bool(torch.isfinite(parameter.grad).all())
        torch.testing.assert_close(
            parameter.grad.double(), reference_parameter.grad, rtol=5e-4, atol=3e-7
        )


def test_old_checkpoint_missing_spline_dtype_retains_its_original_precision():
    original = make_model(dtype=torch.float32)
    old_state = copy.deepcopy(original.state_dict())
    old_state["_extra_state"]["config"].pop("spline_dtype")
    old_config = SphereFlowConfig.from_dict(old_state["_extra_state"]["config"])
    assert old_config.spline_dtype == "model"
    restored = SingleConditionSphereFlow(0.0, 0.0, old_config, dtype=torch.float32)
    restored.load_state_dict(old_state)
    assert restored.spline_dtype == torch.float32
    assert restored.hg_g == original.hg_g
    u = torch.tensor([[0.2, 0.3], [0.6, 0.7]], dtype=torch.float32)
    for actual, expected in zip(restored.sample_from_uniform(u), original.sample_from_uniform(u)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    mixed = make_model(dtype=torch.float32, spline_dtype="float64")
    with pytest.raises(ValueError, match="configuration differs"):
        mixed.load_state_dict(old_state)
    with pytest.raises(ValueError, match="spline_dtype"):
        SphereFlowConfig(spline_dtype="float16")


def test_trusted_sampling_evaluation_and_backward_do_not_read_host_scalars(monkeypatch):
    model = make_model(dtype=torch.float32, spline_dtype="float64")
    model.validate_args = False
    u = torch.tensor([[0.2, 0.3], [0.6, 0.7]], dtype=torch.float64)

    def scalar_read_forbidden(*_args, **_kwargs):
        raise AssertionError("trusted flow arithmetic must not synchronize tensor scalars")

    with monkeypatch.context() as context:
        for name in ("__bool__", "__float__", "__int__", "item"):
            context.setattr(torch.Tensor, name, scalar_read_forbidden)
        directions, logp = model.sample_from_uniform(u)
        evaluated = model.log_prob(directions.detach())
        (-evaluated.mean()).backward()
    torch.testing.assert_close(logp, evaluated, rtol=0, atol=3e-12)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device is unavailable")
def test_cuda_mixed_precision_sampling_evaluation_and_gradient_parity():
    cpu = make_model(dtype=torch.float32, spline_dtype="float64")
    cuda = copy.deepcopy(cpu).cuda()
    assert cuda._hg_coefficient.device == cuda.device
    assert cuda._hg_coefficient.dtype == torch.float64
    assert cuda._hg_coefficient.item() == cpu.hg_g
    cpu.validate_args = cuda.validate_args = False
    u = torch.tensor([[0.12, 0.08], [0.31, 0.4], [0.68, 0.83]], dtype=torch.float64)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        cpu_directions, cpu_logp = cpu.sample_from_uniform(u)
        cuda_directions, cuda_logp = cuda.sample_from_uniform(u.cuda())
        torch.testing.assert_close(cuda_directions.cpu(), cpu_directions, rtol=0, atol=3e-7)
        torch.testing.assert_close(cuda_logp.cpu(), cpu_logp, rtol=0, atol=3e-6)
        cpu_loss = -cpu.log_prob(cpu_directions.detach()).mean()
        # Evaluate identical physical directions when comparing training gradients.
        cuda_loss = -cuda.log_prob(cpu_directions.detach().cuda()).mean()
        cpu_loss.backward()
        cuda_loss.backward()
        for cpu_parameter, cuda_parameter in zip(cpu.parameters(), cuda.parameters()):
            torch.testing.assert_close(
                cuda_parameter.grad.cpu(), cpu_parameter.grad, rtol=5e-4, atol=3e-6
            )
        torch.testing.assert_close(
            cuda_logp, cuda.log_prob(cuda_directions), rtol=0, atol=3e-6
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
