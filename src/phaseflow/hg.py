"""Analytic Henyey--Greenstein probabilities and probability coordinates.

``mu`` is the cosine between *propagation* directions.  Thus positive ``g``
means forward scattering.  ``hg_log_prob`` is a density per unit solid angle;
the density of ``mu`` alone is ``2*pi*exp(hg_log_prob(mu, g))``.

All formulas below are algebraic rearrangements of the HG distribution.  In
particular, there is no small-|g| approximation or isotropic threshold.  The
functions preserve float32/float64 dtype and device, and broadcast their
arguments.  Callers requiring accurate probability coordinates for a narrow
forward peak should use float64: converting a cosine close to one to float32
irreversibly loses angular information, regardless of the evaluation formula.

The closed endpoint conventions are CDF(-1)=0, CDF(1)=1, ICDF(0)=-1 and
ICDF(1)=1.  CDF extends as 0/1 outside the cosine interval, and log probability
is -inf there.  Invalid g or invalid ICDF probabilities give NaN, or a
ValueError when validate_args=True.  Tensor range checking is optional so that
the regular GPU path does not synchronize with the host.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["hg_log_prob", "hg_prob", "hg_cdf", "hg_icdf", "hg_cdf_and_log_prob"]


def _pair(a: Tensor | float, b: Tensor | float) -> tuple[Tensor, Tensor]:
    """Coerce scalars without moving tensors or silently changing precision."""
    reference = a if isinstance(a, Tensor) else b if isinstance(b, Tensor) else None
    if reference is None:
        a = torch.as_tensor(a, dtype=torch.get_default_dtype())
        b = torch.as_tensor(b, dtype=a.dtype, device=a.device)
    else:
        if not isinstance(a, Tensor):
            a = torch.as_tensor(a, dtype=reference.dtype, device=reference.device)
        if not isinstance(b, Tensor):
            b = torch.as_tensor(b, dtype=reference.dtype, device=reference.device)
    if a.dtype not in (torch.float32, torch.float64) or b.dtype != a.dtype:
        raise TypeError("HG arguments must have the same float32 or float64 dtype")
    if a.device != b.device:
        raise ValueError("HG arguments must be on the same device")
    return torch.broadcast_tensors(a, b)


def _check_g(g: Tensor, validate_args: bool) -> Tensor:
    valid = torch.isfinite(g) & (g.abs() < 1)
    if validate_args and not bool(valid.all()):
        raise ValueError("HG g must be finite and strictly between -1 and 1")
    return valid


def _common(mu: Tensor, g: Tensor, valid_g: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # A bad g is replaced only on invalid lanes, whose final result is NaN.
    # This keeps invalid intermediate arithmetic out of the autograd graph.
    g = torch.where(valid_g, g, torch.zeros_like(g))
    m = mu.clamp(-1, 1)
    a, b = 1 - g, 1 + g
    # Both branches are identical in exact arithmetic.  The selected branch
    # writes the squared denominator as a sum of nonnegative terms.
    denominator_squared = torch.where(
        g >= 0,
        a.square() + 2 * g * (1 - m),
        b.square() - 2 * g * (1 + m),
    )
    return m, a, b, denominator_squared.sqrt()


def hg_cdf_and_log_prob(
    mu: Tensor | float, g: Tensor | float, *, validate_args: bool = False
) -> tuple[Tensor, Tensor]:
    """Return HG's cosine CDF and solid-angle log PDF, sharing the square root.

    ``g`` is held fixed when taking the angular change-of-variables Jacobian.
    Ordinary autograd through g remains available, including at g=0.
    """
    mu, g = _pair(mu, g)
    valid_g = _check_g(g, validate_args)
    if validate_args and not bool(torch.isfinite(mu).all()):
        raise ValueError("HG cosine must be finite")
    m, a, b, s = _common(mu, g, valid_g)

    # Rationalizing the familiar 1/g CDF expression removes its removable
    # singularity at g=0.  Compute the smaller tail to avoid subtracting two
    # nearly equal positive quantities in the tails.
    lower = a * (1 + m) / (s * (b + s))
    upper = b * (1 - m) / (s * (a + s))
    cdf = torch.where(lower <= 0.5, lower, 1 - upper)
    # Besides documenting the distribution's extension, these select exact
    # endpoint values despite accumulated floating-point roundoff.
    cdf = torch.where(mu <= -1, torch.zeros_like(cdf), cdf)
    cdf = torch.where(mu >= 1, torch.ones_like(cdf), cdf)

    safe_g = torch.where(valid_g, g, torch.zeros_like(g))
    log_prob = torch.log1p(-safe_g) + torch.log1p(safe_g) - math.log(4 * math.pi) - 3 * s.log()
    log_prob = torch.where(mu.abs() <= 1, log_prob, torch.full_like(log_prob, -torch.inf))
    valid = valid_g & ~torch.isnan(mu)
    nan = torch.full_like(cdf, torch.nan)
    return torch.where(valid, cdf, nan), torch.where(valid, log_prob, nan)


def hg_cdf(mu: Tensor | float, g: Tensor | float, *, validate_args: bool = False) -> Tensor:
    """HG cosine CDF on [-1, 1], including g=0 without approximation."""
    return hg_cdf_and_log_prob(mu, g, validate_args=validate_args)[0]


def hg_log_prob(mu: Tensor | float, g: Tensor | float, *, validate_args: bool = False) -> Tensor:
    """HG log density with respect to solid angle, in sr^-1 before log."""
    mu, g = _pair(mu, g)
    valid_g = _check_g(g, validate_args)
    if validate_args and not bool(torch.isfinite(mu).all()):
        raise ValueError("HG cosine must be finite")
    _, _, _, s = _common(mu, g, valid_g)
    safe_g = torch.where(valid_g, g, torch.zeros_like(g))
    result = torch.log1p(-safe_g) + torch.log1p(safe_g) - math.log(4 * math.pi) - 3 * s.log()
    result = torch.where(mu.abs() <= 1, result, torch.full_like(result, -torch.inf))
    return torch.where(valid_g & ~torch.isnan(mu), result, torch.full_like(result, torch.nan))


def hg_prob(mu: Tensor | float, g: Tensor | float, *, validate_args: bool = False) -> Tensor:
    """HG probability density with respect to solid angle."""
    return hg_log_prob(mu, g, validate_args=validate_args).exp()


def hg_icdf(u: Tensor | float, g: Tensor | float, *, validate_args: bool = False) -> Tensor:
    """Analytic inverse of :func:`hg_cdf`, valid also at exactly g=0.

    The usual inverse contains subtraction followed by division by g.  Here
    its two endpoint distances 1+mu and 1-mu are evaluated as products of
    nonnegative terms.  Selecting the smaller distance avoids cancellation
    at both angular endpoints; the selected formulas are exactly equivalent.
    """
    u, g = _pair(u, g)
    valid_g = _check_g(g, validate_args)
    valid_u = torch.isfinite(u) & (u >= 0) & (u <= 1)
    if validate_args and not bool(valid_u.all()):
        raise ValueError("HG inverse-CDF probability must be finite and in [0, 1]")
    safe_g = torch.where(valid_g, g, torch.zeros_like(g))
    p = torch.where(valid_u, u, torch.full_like(u, 0.5))
    a, b = 1 - safe_g, 1 + safe_g
    complement = 1 - p
    denominator = a * complement + b * p
    one_plus_mu = 2 * p * (b / denominator).square() * (a * complement + p)
    one_minus_mu = 2 * complement * (a / denominator).square() * (complement + b * p)
    result = torch.where(one_plus_mu <= 1, one_plus_mu - 1, 1 - one_minus_mu)
    return torch.where(valid_g & valid_u, result, torch.full_like(result, torch.nan))


def _hg_cdf_and_log_prob_from_distances(
    one_minus_mu: Tensor, one_plus_mu: Tensor, one_minus_g: Tensor, one_plus_g: Tensor
) -> tuple[Tensor, Tensor]:
    """HG using separately retained endpoint distances, without forming mu/g.

    Internal sphere-flow kernel. Inputs are already checked, have one dtype
    and device, and represent 1-mu, 1+mu, 1-g and 1+g, respectively. Keeping
    the small distances prevents a narrow lobe disappearing when mu or g
    rounds to +/-1 in FP32. This is the same analytic HG as the cosine API;
    it adds no threshold, density floor, clipping, or asymptotic approximation.
    """
    a, b = one_minus_g, one_plus_g
    dm, dp = one_minus_mu, one_plus_mu
    s = torch.where(
        a <= b, a.square() + (b - a) * dm, b.square() + (a - b) * dp
    ).sqrt()
    lower = a * dp / (s * (b + s))
    upper = b * dm / (s * (a + s))
    cdf = torch.where(lower <= 0.5, lower, 1 - upper)
    cdf = torch.where(dp == 0, torch.zeros_like(cdf), cdf)
    cdf = torch.where(dm == 0, torch.ones_like(cdf), cdf)
    log_prob = a.log() + b.log() - math.log(4 * math.pi) - 3 * s.log()
    return cdf, log_prob


def _hg_icdf_distances(
    u: Tensor, one_minus_g: Tensor, one_plus_g: Tensor
) -> tuple[Tensor, Tensor]:
    """Return (1-mu, 1+mu) without subtracting a tiny distance from one.

    Inputs follow the validated internal-kernel contract of
    ``_hg_cdf_and_log_prob_from_distances``; u is in [0,1]. The products below
    are the same rationalized quantile used in ``hg_icdf``. The sphere flow
    retains both distances to construct its transverse direction and PDF.
    """
    a, b = one_minus_g, one_plus_g
    complement = 1 - u
    denominator = a * complement + b * u
    one_plus_mu = 2 * u * (b / denominator).square() * (a * complement + u)
    one_minus_mu = 2 * complement * (a / denominator).square() * (complement + b * u)
    return one_minus_mu, one_plus_mu
