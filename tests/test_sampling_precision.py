"""Report rounded poles instead of assigning a surface PDF to a collapsed atom."""

import pytest
import torch

from phaseflow.model import ModelConfig, PhaseFlow


def _endpoint_compressing_model(*, validate_args=True):
    model = PhaseFlow(
        ModelConfig(
            num_bins=4,
            num_coupling_layers=2,
            one_blob_bins=0,
            hidden_features=(4,),
            g_hidden_features=(4,),
        ),
        validate_args=validate_args,
    ).float()
    with torch.no_grad():
        # A steep data->uniform endpoint slope compresses inverse samples
        # within less than a float32 ULP of the interval endpoint.
        model.couplings[0].hyper[-1].bias[-1] = 100.0
        # Nonuniform azimuth reveals that canonical pole PDF substitution
        # would merely hide the inconsistent preimages of the same direction.
        model.couplings[1].hyper[-1].bias[4:8] = torch.tensor([1.0, -1.0, 1.0, -1.0])
    return model


def _inputs():
    uniforms = torch.tensor(
        [
            [1 - 2**-24, 0.20],
            [1 - 2**-24, 0.35],
            [0.37, 0.20],
        ],
        dtype=torch.float32,
    )
    conditions = torch.tensor([550.0, 0.0], dtype=torch.float32)
    return uniforms, conditions


def test_rounded_pole_in_open_uniform_domain_raises_a_precision_error():
    model = _endpoint_compressing_model()
    uniforms, conditions = _inputs()
    with pytest.raises(FloatingPointError):
        model.sample_from_uniform(uniforms, conditions)


def test_validation_off_marks_only_rounded_pole_lanes_invalid():
    model = _endpoint_compressing_model(validate_args=False)
    uniforms, conditions = _inputs()
    directions, log_pdf = model.sample_from_uniform(uniforms, conditions)
    assert bool(torch.isnan(directions[:2]).all())
    assert bool(torch.isnan(log_pdf[:2]).all())
    assert bool(torch.isfinite(directions[2]).all()) and bool(torch.isfinite(log_pdf[2]))
    torch.testing.assert_close(
        log_pdf[2], model.log_prob(directions[2], conditions), atol=3e-5, rtol=0
    )


def test_double_precision_resolves_this_specific_endpoint_compression():
    # This is a regression example, not a claim that arbitrary learned flows
    # and arbitrarily extreme uniforms cannot round to a pole in float64.
    model = _endpoint_compressing_model().double()
    uniforms, conditions = _inputs()
    directions, log_pdf = model.sample_from_uniform(uniforms, conditions)
    assert directions.dtype == torch.float64
    assert bool(torch.isfinite(directions).all()) and bool(torch.isfinite(log_pdf).all())
    assert bool((directions[:, 2].abs() < 1).all())
    torch.testing.assert_close(log_pdf, model.log_prob(directions, conditions), atol=3e-8, rtol=0)
