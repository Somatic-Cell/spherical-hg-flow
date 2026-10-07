"""Analytic integration-test data. These functions do NOT simulate raindrops.

The demo combines two HG lobes and a smooth azimuth-dependent polynomial lobe.
It is normalized on the full sphere, respects reflection across the xz plane,
and becomes axisymmetric at axial incidence. It exercises a genuinely 2D flow
without claiming validation of any rainbow, wave-optics, or droplet model.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .data import PhasePointCloud


def _parameters(conditions: NDArray[np.float64]) -> tuple[NDArray[np.float64], ...]:
    wavelength, incident_cosine = conditions[..., 0], conditions[..., 1]
    spectral = (wavelength - 380.0) / 340.0
    if np.any((spectral < 0) | (spectral > 1)):
        raise ValueError("the analytic demo is defined for wavelengths in [380, 720] nm")
    if np.any(np.abs(incident_cosine) > 1):
        raise ValueError("incident_cosine must lie in [-1, 1]")
    g1 = 0.35 + 0.25 * spectral + 0.10 * incident_cosine
    g2 = -0.20 - 0.25 * spectral - 0.05 * incident_cosine
    w1 = 0.55 + 0.10 * incident_cosine
    w2 = np.full_like(w1, 0.20)
    w3 = 1.0 - w1 - w2
    a = 0.35 * incident_cosine
    # The azimuthal modulation must vanish when the incident direction is on
    # the particle symmetry axis: the projected +x axis is then arbitrary.
    b = 0.70 * (1.0 - incident_cosine**2)
    return g1, g2, w1, w2, w3, a, b


def hg_pdf(cosine: NDArray[np.float64], g: NDArray[np.float64] | float) -> NDArray[np.float64]:
    """HG per steradian, with +z pointing along incident propagation."""
    return (1.0 - np.asarray(g) ** 2) / (
        4.0 * np.pi * (1.0 + np.asarray(g) ** 2 - 2.0 * np.asarray(g) * cosine) ** 1.5
    )


def demo_pdf(outgoing: NDArray[np.float64], conditions: NDArray[np.float64]) -> NDArray[np.float64]:
    outgoing = np.asarray(outgoing, dtype=np.float64)
    conditions = np.asarray(conditions, dtype=np.float64)
    g1, g2, w1, w2, w3, a, b = _parameters(conditions)
    x, y, z = outgoing[..., 0], outgoing[..., 1], outgoing[..., 2]
    polynomial = (1.0 + a * z) * (1.0 + b * (x * x - y * y)) / (4.0 * np.pi)
    return w1 * hg_pdf(z, g1) + w2 * hg_pdf(z, g2) + w3 * polynomial


def demo_moment(conditions: NDArray[np.float64]) -> NDArray[np.float64]:
    g1, g2, w1, w2, w3, a, _ = _parameters(np.asarray(conditions, dtype=np.float64))
    return w1 * g1 + w2 * g2 + w3 * a / 3.0


def _spherical(cosine: NDArray[np.float64], phi: NDArray[np.float64]) -> NDArray[np.float64]:
    radius = np.sqrt(np.maximum(0.0, (1.0 - cosine) * (1.0 + cosine)))
    return np.stack((radius * np.cos(phi), radius * np.sin(phi), cosine), axis=-1)


def _sample_hg(count: int, g: float, rng: np.random.Generator) -> NDArray[np.float64]:
    u = rng.random(count)
    if abs(g) < 1e-10:
        cosine = 2.0 * u - 1.0
    else:
        cosine = (1.0 + g * g - ((1.0 - g * g) / (1.0 - g + 2.0 * g * u)) ** 2) / (2.0 * g)
    return _spherical(np.clip(cosine, -1.0, 1.0), 2.0 * np.pi * rng.random(count))


def _sample_polynomial(
    count: int, a: float, b: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    result = np.empty((count, 3), dtype=np.float64)
    filled = 0
    bound = (1.0 + abs(a)) * (1.0 + abs(b))
    while filled < count:
        candidate_count = max(32, int(1.5 * (count - filled) * bound))
        directions = _spherical(
            2.0 * rng.random(candidate_count) - 1.0, 2.0 * np.pi * rng.random(candidate_count)
        )
        x, y, z = directions.T
        numerator = (1.0 + a * z) * (1.0 + b * (x * x - y * y))
        accepted = directions[rng.random(candidate_count) * bound < numerator]
        take = min(len(accepted), count - filled)
        result[filled : filled + take] = accepted[:take]
        filled += take
    return result


def sample_demo_condition(
    condition: NDArray[np.float64], count: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    if count < 1:
        raise ValueError("count must be positive")
    parameters = _parameters(np.asarray(condition, dtype=np.float64))
    g1, g2, w1, w2, _, a, b = (float(x) for x in parameters)
    choices = rng.random(count)
    mask1, mask2 = choices < w1, (choices >= w1) & (choices < w1 + w2)
    mask3 = ~(mask1 | mask2)
    result = np.empty((count, 3), dtype=np.float64)
    result[mask1] = _sample_hg(int(mask1.sum()), g1, rng)
    result[mask2] = _sample_hg(int(mask2.sum()), g2, rng)
    result[mask3] = _sample_polynomial(int(mask3.sum()), a, b, rng)
    return result


def make_demo(
    *,
    mode: Literal["target_samples", "quadrature"] = "target_samples",
    points_per_condition: int = 4096,
    seed: int = 17,
    wavelengths_nm: tuple[float, ...] = (400.0, 550.0, 700.0),
    incident_cosines: tuple[float, ...] = (-0.75, -0.25, 0.25, 0.75),
    n_mu: int = 48,
    n_phi: int = 64,
) -> PhasePointCloud:
    """Create a labelled synthetic dataset with deterministic condition groups.

    ``quadrature`` uses Gauss-Legendre integration in cos(theta) and uniform
    midpoint azimuths. The weights already contain the solid-angle measure.
    ``target_samples`` uses exact mixture sampling with rejection for the smooth
    polynomial lobe. No per-point PDF weights are attached to those samples.
    """
    if mode not in ("target_samples", "quadrature"):
        raise ValueError("mode must be target_samples or quadrature")
    conditions = np.asarray(
        [(wavelength, cosine) for wavelength in wavelengths_nm for cosine in incident_cosines],
        dtype=np.float64,
    )
    if conditions.size == 0:
        raise ValueError("at least one wavelength and incident cosine are required")
    rng = np.random.default_rng(seed)
    direction_groups, index_groups, weight_groups = [], [], []
    if mode == "quadrature":
        if n_mu < 2 or n_phi < 4:
            raise ValueError("quadrature requires n_mu >= 2 and n_phi >= 4")
        mu, mu_weight = np.polynomial.legendre.leggauss(n_mu)
        phi = 2.0 * np.pi * (np.arange(n_phi) + 0.5) / n_phi
        mu_grid, phi_grid = np.meshgrid(mu, phi, indexing="ij")
        quadrature_outgoing = _spherical(mu_grid.ravel(), phi_grid.ravel())
        solid_angle_weight = np.repeat(mu_weight, n_phi) * (2.0 * np.pi / n_phi)
    for ci, condition in enumerate(conditions):
        if mode == "quadrature":
            outgoing = quadrature_outgoing.copy()
            weights = demo_pdf(outgoing, condition) * solid_angle_weight
            weight_groups.append(weights)
        else:
            outgoing = sample_demo_condition(condition, points_per_condition, rng)
        direction_groups.append(outgoing)
        index_groups.append(np.full(len(outgoing), ci, dtype=np.int64))
    metadata = {
        "source": "analytic_hg_mixture_plus_polynomial_lobe",
        "purpose": "synthetic_implementation_test_only_not_a_rainbow_simulation",
        "seed": seed,
        "analytic_moment_g": demo_moment(conditions).tolist(),
        "target_sample_weights": "none",
        "quadrature_weights": "density_times_solid_angle_measure",
        "jitter_or_smoothing": "none",
    }
    return PhasePointCloud(
        conditions=conditions,
        outgoing=np.concatenate(direction_groups),
        condition_index=np.concatenate(index_groups),
        mode=mode,
        weights=np.concatenate(weight_groups) if weight_groups else None,
        metadata=metadata,
    )
