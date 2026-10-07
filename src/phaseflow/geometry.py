"""Scattering-frame and mirror-folding coordinates for the HG residual model.

The frame's z axis is the incident propagation direction.  Its x axis points
toward the particle axis projected onto the plane perpendicular to z; y=z*x.
The xz plane is therefore the physical mirror plane.  At exactly axial
incidence this plane is undefined and a deterministic orthonormal frame is
chosen.  No finite cone is replaced by this arbitrary frame.

Angular coordinates use mu=cos(scattering angle) and phi in [-pi, pi].  The
folded coordinate is v=abs(phi)/pi in [0, 1].  Mirror restoration requires a
separate sign, with the two signs each having probability 1/2.  This is an
almost-everywhere chart plus a discrete branch, not a global diffeomorphism
of S^2.  At a pole phi=0 is the canonical coordinate.  A density expressed
in these coordinates is not automatically continuous or C1 at the poles.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = [
    "normalize_direction",
    "scattering_frame",
    "direction_to_mu_phi",
    "mu_phi_to_direction",
    "direction_to_folded",
    "folded_to_direction",
]


def _direction(
    value: Tensor | tuple[float, float, float], reference: Tensor | None = None
) -> Tensor:
    if not isinstance(value, Tensor):
        value = torch.as_tensor(
            value,
            dtype=reference.dtype if reference is not None else torch.get_default_dtype(),
            device=reference.device if reference is not None else None,
        )
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError("Directions must have float32 or float64 dtype")
    if value.ndim < 1 or value.shape[-1] != 3:
        raise ValueError("A direction must have shape (..., 3)")
    if reference is not None and (
        value.dtype != reference.dtype or value.device != reference.device
    ):
        raise ValueError("Direction tensors must have the same dtype and device")
    return value


def normalize_direction(direction: Tensor, *, validate_args: bool = False) -> Tensor:
    """Normalize without overflow/underflow from squaring the original scale.

    Zero and nonfinite vectors return NaNs, or raise when validation is on.
    No epsilon-length clamping changes the orientation of short vectors.
    """
    direction = _direction(direction)
    scale = direction.abs().amax(dim=-1, keepdim=True)
    valid = torch.isfinite(direction).all(dim=-1, keepdim=True) & (scale > 0)
    if validate_args and not bool(valid.all()):
        raise ValueError("Directions must be finite and nonzero")
    safe_scale = torch.where(valid, scale, torch.ones_like(scale))
    scaled = torch.where(valid, direction, torch.zeros_like(direction)) / safe_scale
    norm_squared = scaled.square().sum(dim=-1, keepdim=True)
    safe_norm = torch.where(valid, norm_squared, torch.ones_like(norm_squared)).sqrt()
    unit = scaled / safe_norm
    return torch.where(valid, unit, torch.full_like(unit, torch.nan))


def scattering_frame(
    incident: Tensor,
    axis: Tensor | tuple[float, float, float] = (0.0, 1.0, 0.0),
    *,
    validate_args: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return the orthonormal (x, y, z) scattering frame, broadcast in batch.

    Only *exactly* parallel/antiparallel axes use the deterministic fallback.
    Normalizing the cross product with a scale first retains very small
    nonzero transverse components instead of applying an angular threshold.
    """
    incident = _direction(incident)
    axis = _direction(axis, incident)
    incident, axis = torch.broadcast_tensors(incident, axis)
    z = normalize_direction(incident, validate_args=validate_args)
    a = normalize_direction(axis, validate_args=validate_args)
    normal = torch.linalg.cross(z, a, dim=-1)
    parallel = normal.abs().amax(dim=-1, keepdim=True) == 0
    parallel = parallel | (z == a).all(dim=-1, keepdim=True) | (z == -a).all(dim=-1, keepdim=True)
    # Crossing with the least-aligned coordinate axis is uniformly safe.
    reference_index = z.abs().argmin(dim=-1)
    reference = torch.nn.functional.one_hot(reference_index, num_classes=3).to(dtype=z.dtype)
    fallback = torch.linalg.cross(z, reference, dim=-1)
    y = normalize_direction(torch.where(parallel, fallback, normal), validate_args=validate_args)
    x = normalize_direction(torch.linalg.cross(y, z, dim=-1), validate_args=validate_args)
    return x, y, z


def direction_to_mu_phi(
    outgoing: Tensor,
    incident: Tensor,
    axis: Tensor | tuple[float, float, float] = (0.0, 1.0, 0.0),
    *,
    validate_args: bool = False,
) -> tuple[Tensor, Tensor]:
    """Convert a world direction to (mu, phi) in the physical scattering frame.

    Nonunit vectors are normalized.  Exact poles have canonical phi=0.  Dot
    products are clamped only to remove unit-vector roundoff outside [-1, 1].
    """
    incident = _direction(incident)
    outgoing = _direction(outgoing, incident)
    x, y, z = scattering_frame(incident, axis, validate_args=validate_args)
    outgoing, x, y, z = torch.broadcast_tensors(outgoing, x, y, z)
    w = normalize_direction(outgoing, validate_args=validate_args)
    mu = (w * z).sum(dim=-1).clamp(-1, 1)
    local_x, local_y = (w * x).sum(dim=-1), (w * y).sum(dim=-1)
    # Detect exact poles before another normalization can perturb their scale.
    polar = torch.linalg.cross(outgoing, z, dim=-1).abs().amax(dim=-1) == 0
    polar = polar | (outgoing == z).all(dim=-1) | (outgoing == -z).all(dim=-1)
    polar = polar & torch.isfinite(w).all(dim=-1)
    mu = torch.where(polar, torch.where(mu >= 0, torch.ones_like(mu), -torch.ones_like(mu)), mu)
    phi = torch.atan2(local_y, local_x)
    phi = torch.where(polar, torch.zeros_like(phi), phi)
    return mu, phi


def _scalar(value: Tensor | float, reference: Tensor) -> Tensor:
    if not isinstance(value, Tensor):
        value = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if value.dtype != reference.dtype or value.device != reference.device:
        raise ValueError("Coordinates and direction tensors must have the same dtype and device")
    return value


def mu_phi_to_direction(
    mu: Tensor | float,
    phi: Tensor | float,
    incident: Tensor,
    axis: Tensor | tuple[float, float, float] = (0.0, 1.0, 0.0),
    *,
    validate_args: bool = False,
) -> Tensor:
    """Map scattering cosine/azimuth to a world direction.

    The closed mu endpoints are accepted and give exactly +/- the frame z
    vector independently of phi.  Finite phi values outside [-pi, pi] are
    equivalent by periodicity.  Invalid mu or nonfinite phi produce NaNs.
    """
    incident = _direction(incident)
    mu, phi = torch.broadcast_tensors(_scalar(mu, incident), _scalar(phi, incident))
    valid = torch.isfinite(mu) & (mu.abs() <= 1) & torch.isfinite(phi)
    if validate_args and not bool(valid.all()):
        raise ValueError("mu must be in [-1, 1] and both angular coordinates must be finite")
    m = torch.where(valid, mu, torch.zeros_like(mu))
    p = torch.where(valid, phi, torch.zeros_like(phi))
    radius = ((1 - m) * (1 + m)).sqrt()
    x, y, z = scattering_frame(incident, axis, validate_args=validate_args)
    direction = (
        (radius * p.cos())[..., None] * x + (radius * p.sin())[..., None] * y + m[..., None] * z
    )
    return torch.where(valid[..., None], direction, torch.full_like(direction, torch.nan))


def direction_to_folded(
    outgoing: Tensor,
    incident: Tensor,
    axis: Tensor | tuple[float, float, float] = (0.0, 1.0, 0.0),
    *,
    validate_args: bool = False,
) -> tuple[Tensor, Tensor]:
    """Return (mu, abs(phi)/pi); the physical mirror sides are identified."""
    mu, phi = direction_to_mu_phi(outgoing, incident, axis, validate_args=validate_args)
    return mu, phi.abs() / math.pi


def folded_to_direction(
    mu: Tensor | float,
    v: Tensor | float,
    sign: Tensor | float,
    incident: Tensor,
    axis: Tensor | tuple[float, float, float] = (0.0, 1.0, 0.0),
    *,
    validate_args: bool = False,
) -> Tensor:
    """Restore a mirror branch from v in [0,1] and sign equal to -1 or +1.

    The two signs must be equiprobable when sampling a mirror-symmetric HG
    residual density.  Their factor 1/2 cancels the folded angular Jacobian's
    factor two, leaving physical density ``hg_pdf * residual_square_pdf``.
    """
    incident = _direction(incident)
    mu, v, sign = torch.broadcast_tensors(
        _scalar(mu, incident), _scalar(v, incident), _scalar(sign, incident)
    )
    valid = torch.isfinite(v) & (v >= 0) & (v <= 1) & ((sign == -1) | (sign == 1))
    if validate_args and not bool(valid.all()):
        raise ValueError("Folded azimuth v must be in [0, 1] and sign must be -1 or +1")
    phi = torch.where(valid, sign * (math.pi * v), torch.full_like(v, torch.nan))
    return mu_phi_to_direction(mu, phi, incident, axis, validate_args=validate_args)
