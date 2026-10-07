"""Circular RQS for the cylindrical sphere construction of Rezende et al. (2020).

The circle is represented by [0, 1] with its two endpoints identified.  The
returned interval lift fixes 0 and 1, and its positive endpoint derivatives are
tied.  These are the boundary conditions in Sec. 2.1.2 of
https://proceedings.mlr.press/v119/rezende20a/rezende20a.pdf .

``from_reflection_logits`` is an explicit physical-symmetry extension: reflected
bins and slopes enforce F(1-v)=1-F(v) on the full circle.  It does not fold the
random variable or introduce a random branch.  The ordinary constructor and
``from_logits`` impose only the paper's circular boundary conditions.

Both directions reuse the analytic RQS algebra in ``BoundedRQSTransform``.
There are no iterative inverses or identity tails.  The torch Transform and
Zuko ``call_and_ladj`` protocols apply to the closed interval lift.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .splines import BoundedRQSTransform, _floors

__all__ = ["CircularRQSTransform"]


def _identity_blend(
    widths: Tensor, heights: Tensor, derivatives: Tensor, strength: float
) -> tuple[Tensor, Tensor, Tensor]:
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("identity_strength must be a finite scalar in [0, 1]")
    bins = widths.shape[-1]
    return (
        strength * widths + (1 - strength) / bins,
        strength * heights + (1 - strength) / bins,
        strength * derivatives + (1 - strength),
    )


class CircularRQSTransform(BoundedRQSTransform):
    """Monotone circular RQS with K widths, K heights, and K unique slopes.

    The K slopes are at knots 0, ..., K-1.  The constructor appends slope 0
    at knot K, so the seam can never acquire two independent derivatives.
    Values 0 and 1 denote the same point on the circle, but their interval-lift
    values remain 0 and 1 respectively.  Other values must lie in [0, 1].
    """

    def __init__(
        self,
        widths: Tensor,
        heights: Tensor,
        derivatives: Tensor,
        *,
        validate_args: bool = False,
        cache_size: int = 0,
    ) -> None:
        if not isinstance(widths, Tensor) or widths.ndim < 1:
            raise TypeError("circular RQS widths require a final bin axis")
        if not isinstance(derivatives, Tensor) or derivatives.ndim < 1:
            raise TypeError("circular RQS derivatives require a final knot axis")
        if derivatives.shape[-1] != widths.shape[-1]:
            raise ValueError("circular RQS requires K unique derivatives for K bins")
        closed_derivatives = torch.cat((derivatives, derivatives[..., :1]), dim=-1)
        super().__init__(
            widths,
            heights,
            closed_derivatives,
            validate_args=validate_args,
            cache_size=cache_size,
        )

    @classmethod
    def from_logits(
        cls,
        widths: Tensor,
        heights: Tensor,
        derivatives: Tensor,
        *,
        min_bin_width: float = 1e-5,
        min_bin_height: float = 1e-5,
        min_derivative: float = 1e-4,
        identity_strength: float = 1.0,
        validate_args: bool = False,
        cache_size: int = 0,
    ) -> "CircularRQSTransform":
        """Constrain K+K+K logits, tying the final slope to the first.

        ``identity_strength`` blends the *constrained* masses and slopes with
        identity values.  This preserves positivity and circularity and is
        used to impose azimuthal independence at exactly axial incidence.
        """
        if not isinstance(widths, Tensor) or widths.ndim < 1:
            raise TypeError("circular RQS width logits require a final bin axis")
        bins = widths.shape[-1]
        _floors(bins, min_bin_width, min_bin_height, min_derivative)
        widths = min_bin_width + (1 - bins * min_bin_width) * widths.softmax(-1)
        heights = min_bin_height + (1 - bins * min_bin_height) * heights.softmax(-1)
        derivatives = min_derivative + F.softplus(derivatives)
        widths, heights, derivatives = _identity_blend(
            widths, heights, derivatives, identity_strength
        )
        return cls(
            widths, heights, derivatives, validate_args=validate_args, cache_size=cache_size
        )

    @classmethod
    def from_reflection_logits(
        cls,
        half_widths: Tensor,
        half_heights: Tensor,
        half_derivatives: Tensor,
        *,
        min_bin_width: float = 1e-5,
        min_bin_height: float = 1e-5,
        min_derivative: float = 1e-4,
        identity_strength: float = 1.0,
        validate_args: bool = False,
        cache_size: int = 0,
    ) -> "CircularRQSTransform":
        """Build a full 2H-bin circle from H+H+(H+1) independent logits.

        Each half has probability-coordinate length 1/2.  Bin masses are
        reflected, and knot derivatives have the sequence
        (d0, ..., dH, dH-1, ..., d1, d0).  Thus both symmetry meridians are
        fixed and their positive slopes are independent of each other.
        In particular, d0 need not equal dH: they are different circle points.

        The bin floors refer to each bin on the *full* unit-length circle.
        """
        if not isinstance(half_widths, Tensor) or half_widths.ndim < 1:
            raise TypeError("reflected circular RQS logits require a final bin axis")
        half_bins = half_widths.shape[-1]
        if (
            not isinstance(half_heights, Tensor)
            or half_heights.ndim < 1
            or half_heights.shape[-1] != half_bins
            or not isinstance(half_derivatives, Tensor)
            or half_derivatives.ndim < 1
            or half_derivatives.shape[-1] != half_bins + 1
        ):
            raise ValueError("reflection logits require H widths, H heights, H+1 derivatives")
        _floors(2 * half_bins, min_bin_width, min_bin_height, min_derivative)
        w = min_bin_width + (0.5 - half_bins * min_bin_width) * half_widths.softmax(-1)
        h = min_bin_height + (0.5 - half_bins * min_bin_height) * half_heights.softmax(-1)
        d = min_derivative + F.softplus(half_derivatives)
        widths = torch.cat((w, w.flip(-1)), dim=-1)
        heights = torch.cat((h, h.flip(-1)), dim=-1)
        derivatives = torch.cat((d, d[..., 1:-1].flip(-1)), dim=-1)
        widths, heights, derivatives = _identity_blend(
            widths, heights, derivatives, identity_strength
        )
        return cls(
            widths, heights, derivatives, validate_args=validate_args, cache_size=cache_size
        )
