from __future__ import annotations

import numpy as np
import pytest
import torch

from phaseflow.data import PhasePointCloud, PointSampler, split_conditions
from phaseflow.synthetic import demo_moment, demo_pdf, make_demo


def _weighted_cloud() -> PhasePointCloud:
    z = np.array([-0.8, 0.6, -0.3, 0.9])
    outgoing = np.stack((np.sqrt(1 - z * z), np.zeros_like(z), z), axis=-1)
    return PhasePointCloud(
        conditions=np.array([[500.0, -0.4], [600.0, 0.7]]),
        outgoing=outgoing,
        condition_index=np.array([0, 0, 1, 1]),
        mode="quadrature",
        weights=np.array([1.0, 3.0, 8.0, 2.0]),
    )


def test_explicit_weight_semantics_and_balanced_sampling() -> None:
    cloud = _weighted_cloud()
    expected = np.array([0.25 * -0.8 + 0.75 * 0.6, 0.8 * -0.3 + 0.2 * 0.9])
    np.testing.assert_allclose(cloud.moment_targets(), expected, rtol=0, atol=1e-14)
    for method in ("mass", "uniform"):
        sampler = PointSampler(cloud, [0, 1], seed=71, point_sampling=method)
        batch = sampler.sample(60000, dtype=torch.float64)
        estimate = (batch.loss_weight * batch.outgoing[:, 2]).mean().item()
        assert abs(estimate - expected.mean()) < 0.012
        assert abs((batch.condition_index == 0).double().mean().item() - 0.5) < 0.01
        if method == "mass":
            assert torch.equal(batch.loss_weight, torch.ones_like(batch.loss_weight))


def test_invalid_or_ambiguous_data_is_rejected() -> None:
    cloud = _weighted_cloud()
    arguments = dict(
        conditions=cloud.conditions, outgoing=cloud.outgoing, condition_index=cloud.condition_index
    )
    with pytest.raises(ValueError, match="weights are forbidden"):
        PhasePointCloud(**arguments, mode="target_samples", weights=np.ones(4))
    with pytest.raises(ValueError, match="requires explicit"):
        PhasePointCloud(**arguments, mode="quadrature")
    with pytest.raises(ValueError, match="zero total"):
        PhasePointCloud(**arguments, mode="quadrature", weights=np.array([0.0, 0.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="cropped"):
        PhasePointCloud(**arguments, coverage="rainbow_only")
    with pytest.raises(ValueError, match="unit vectors"):
        PhasePointCloud(**{**arguments, "outgoing": cloud.outgoing * 2})
    with pytest.raises(ValueError, match="must be unique"):
        PhasePointCloud(**{**arguments, "conditions": np.array([[500.0, 0.0], [500.0, 0.0]])})


def test_pickle_free_npz_roundtrip_and_sampler_resume(tmp_path) -> None:
    cloud = _weighted_cloud()
    path = tmp_path / "points.npz"
    cloud.save_npz(path)
    restored = PhasePointCloud.load_npz(path)
    assert cloud.fingerprint() == restored.fingerprint()
    np.testing.assert_array_equal(cloud.outgoing, restored.outgoing)
    first = PointSampler(cloud, [0, 1], seed=5)
    first.sample(7)
    state = first.state_dict()
    expected = first.sample(40)
    second = PointSampler(restored, [0, 1], seed=999)
    second.load_state_dict(state)
    actual = second.sample(40)
    assert torch.equal(expected.outgoing, actual.outgoing)
    assert torch.equal(expected.condition_index, actual.condition_index)
    malformed = tmp_path / "unlabelled.npz"
    np.savez(malformed, outgoing=cloud.outgoing)
    with pytest.raises(ValueError, match="missing required"):
        PhasePointCloud.load_npz(malformed)


def test_float32_network_batches_retain_float64_polar_geometry() -> None:
    z = np.array([1.0 - 1e-9, 1.0 - 2e-9], dtype=np.float64)
    outgoing = np.stack((np.sqrt((1 - z) * (1 + z)), np.zeros_like(z), z), axis=-1)
    cloud = PhasePointCloud(
        conditions=np.array([[550.0, 0.5]]),
        outgoing=outgoing,
        condition_index=np.zeros(2, dtype=np.int64),
    )
    batch = PointSampler(cloud, [0], seed=8).sample(100, dtype=torch.float32)
    assert batch.conditions.dtype == torch.float32
    assert batch.loss_weight.dtype == torch.float32
    assert batch.outgoing.dtype == torch.float64
    assert torch.unique(batch.outgoing[:, 2]).numel() == 2
    assert torch.all(batch.outgoing[:, 2] < 1)


def test_moment_target_matches_inference_under_tolerated_vector_roundoff() -> None:
    outgoing = np.array([[0.6, 0.0, 0.8]], dtype=np.float64) * (1 + 1e-5)
    cloud = PhasePointCloud(
        conditions=np.array([[550.0, 0.5]]),
        outgoing=outgoing,
        condition_index=np.array([0]),
    )
    np.testing.assert_allclose(cloud.moment_targets(), [0.8], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(cloud.outgoing, outgoing)


def test_split_holds_out_complete_conditions_and_single_condition_is_explicit() -> None:
    split = split_conditions(12, 0.25, 53)
    assert len(split["validation"]) == 3
    assert not set(split["train"]).intersection(split["validation"])
    assert sorted(split["train"] + split["validation"]) == list(range(12))
    assert split == split_conditions(12, 0.25, 53)
    assert split_conditions(1, 0, 2) == {"train": [0], "validation": []}
    with pytest.raises(ValueError, match="single condition"):
        split_conditions(1, 0.2, 2)


def test_analytic_demo_normalization_moments_and_genuine_azimuth_dependence() -> None:
    cloud = make_demo(
        mode="quadrature",
        wavelengths_nm=(550.0,),
        incident_cosines=(-0.5, 0.0, 0.5),
        n_mu=48,
        n_phi=64,
    )
    for ci in range(cloud.num_conditions):
        assert abs(cloud.weights[cloud.group_indices(ci)].sum() - 1.0) < 1e-10
    np.testing.assert_allclose(
        cloud.moment_targets(), demo_moment(cloud.conditions), rtol=0, atol=1e-10
    )
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.6, 0.8, 0.0], [0.6, -0.8, 0.0]])
    pdf = demo_pdf(directions, np.array([550.0, 0.0]))
    assert abs(pdf[0] - pdf[1]) > 0.01
    assert pdf[2] == pdf[3]
    assert "not_a_rainbow" in cloud.metadata["purpose"]


@pytest.mark.parametrize("incident_cosine", [-1.0, 1.0])
def test_analytic_demo_is_azimuth_invariant_at_axial_incidence(incident_cosine) -> None:
    phi = np.linspace(0.0, 2.0 * np.pi, 17, endpoint=False)
    for scattering_cosine in (-0.8, 0.0, 0.7):
        radius = np.sqrt(1.0 - scattering_cosine**2)
        outgoing = np.stack(
            (radius * np.cos(phi), radius * np.sin(phi), np.full_like(phi, scattering_cosine)),
            axis=-1,
        )
        density = demo_pdf(outgoing, np.array([550.0, incident_cosine]))
        np.testing.assert_array_equal(density, np.full_like(density, density[0]))
