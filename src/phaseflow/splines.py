"""Compact-support rational-quadratic spline transforms for Zuko coupling.

This is an independent implementation of the rational-quadratic formulas in
Durkan et al., *Neural Spline Flows*, NeurIPS 2019, equations (4)--(8):
https://papers.neurips.cc/paper_files/paper/2019/file/7ac71d433f282034e088473244df8c02-Paper.pdf

Unlike an unbounded spline with identity tails, this transform maps [0,1]
onto [0,1] and has K+1 independently trainable positive knot derivatives,
including both endpoints.  It is an interval transform, not a circular
transform.  In the phase model it operates on the mirror-folded square.

The class uses the torch Transform protocol and Zuko's call_and_ladj
protocol.  Its inverse also has a fused call_and_ladj implementation.  Both
directions use bin lookup and analytic algebra only; no iterative inversion
or numerically fitted inverse is used.  Parameter tensors remain attached to
their autograd graph.
"""

from __future__ import annotations

import copy
import math
import weakref

import torch
import torch.nn.functional as F
from torch import Size, Tensor
from torch.distributions import Transform, constraints

__all__ = ["BoundedRQSTransform", "identity_derivative_logit"]


def identity_derivative_logit(min_derivative: float = 1e-4) -> float:
    """Logit yielding slope one under ``min_derivative + softplus(logit)``.

    Equal width/height logits and this value at all K+1 derivative logits
    parameterize the identity exactly in real arithmetic (up to rounding in
    the selected floating-point dtype).  Endpoints are not fixed to slope 1
    after initialization; they remain trainable.
    """
    if not math.isfinite(min_derivative) or not 0 < min_derivative < 1:
        raise ValueError("Identity initialization requires 0 < min_derivative < 1")
    return math.log(math.expm1(1 - min_derivative))


def _floors(bins: int, min_bin_width: float, min_bin_height: float, min_derivative: float) -> None:
    if not math.isfinite(min_bin_width) or not 0 < bins * min_bin_width < 1:
        raise ValueError("min_bin_width must be positive and bins*min_bin_width must be < 1")
    if not math.isfinite(min_bin_height) or not 0 < bins * min_bin_height < 1:
        raise ValueError("min_bin_height must be positive and bins*min_bin_height must be < 1")
    if not math.isfinite(min_derivative) or min_derivative <= 0:
        raise ValueError("min_derivative must be finite and positive")


class BoundedRQSTransform(Transform):
    """Monotone rational-quadratic map from the closed unit interval to itself.

    Args:
        widths: Positive bin masses of shape ``(..., K)``.  The last axis is
            normalized to sum one; no sorting or clipping changes the bins.
        heights: Positive output-bin masses of shape ``(..., K)``, normalized
            independently in the same way.
        derivatives: Positive slopes of shape ``(..., K+1)`` at all knots.
        validate_args: Check tensor values and raise on invalid data.  Off by
            default to avoid host synchronization on every conditioned batch.
            Invalid lanes then return NaN rather than an identity-tail value.
        cache_size: The standard torch Transform cache size; zero by default.

    Batch axes and input values follow torch broadcasting.  The transform is
    scalar-valued (event_dim=0).  Wrap with Zuko's DependentTransform when
    treating several scalar transforms as a vector event.

    A value exactly on an interior knot uses the bin on its right.  The upper
    endpoint uses the last bin.  Value and first derivative agree on both
    sides of an interior knot.  The endpoints map exactly to 0 and 1.

    Positive masses too small to form distinct cumulative knots in the chosen
    dtype are invalid, not silently merged.  Practical concentration limits
    from floor parameters and floating-point precision must be assessed for
    the target phase functions.
    """

    domain = constraints.unit_interval
    codomain = constraints.unit_interval
    bijective = True
    sign = 1

    def __init__(
        self,
        widths: Tensor,
        heights: Tensor,
        derivatives: Tensor,
        *,
        validate_args: bool = False,
        cache_size: int = 0,
    ) -> None:
        super().__init__(cache_size=cache_size)
        for name, value in (("widths", widths), ("heights", heights), ("derivatives", derivatives)):
            if not isinstance(value, Tensor) or value.ndim < 1:
                raise TypeError(f"{name} must be a floating-point tensor with a knot/bin axis")
            if value.dtype not in (torch.float32, torch.float64):
                raise TypeError("RQS parameters must use float32 or float64")
            if value.dtype != widths.dtype or value.device != widths.device:
                raise ValueError("All RQS parameters must have the same dtype and device")
        bins = widths.shape[-1]
        if bins < 1 or heights.shape[-1] != bins or derivatives.shape[-1] != bins + 1:
            raise ValueError(
                "RQS parameter lengths must be K widths, K heights, and K+1 derivatives"
            )
        self._validate_args = bool(validate_args)
        self._batch_shape = Size(
            torch.broadcast_shapes(widths.shape[:-1], heights.shape[:-1], derivatives.shape[:-1])
        )
        widths = widths.expand(self._batch_shape + (bins,))
        heights = heights.expand(self._batch_shape + (bins,))
        derivatives = derivatives.expand(self._batch_shape + (bins + 1,))
        valid = (
            (torch.isfinite(widths) & (widths > 0)).all(dim=-1)
            & (torch.isfinite(heights) & (heights > 0)).all(dim=-1)
            & (torch.isfinite(derivatives) & (derivatives > 0)).all(dim=-1)
        )
        if self._validate_args and not bool(valid.all()):
            raise ValueError("RQS bin masses and derivatives must be finite and positive")

        # Normalize positive masses with scaling first, so their sum cannot
        # overflow.  Invalid lanes use a harmless identity until masked out.
        widths = torch.where(valid[..., None], widths, torch.ones_like(widths))
        heights = torch.where(valid[..., None], heights, torch.ones_like(heights))
        widths = widths / widths.amax(dim=-1, keepdim=True)
        heights = heights / heights.amax(dim=-1, keepdim=True)
        widths = widths / widths.sum(dim=-1, keepdim=True)
        heights = heights / heights.sum(dim=-1, keepdim=True)
        zeros = torch.zeros_like(widths[..., :1])
        ones = torch.ones_like(zeros)
        horizontal = torch.cat((zeros, widths.cumsum(dim=-1)[..., :-1], ones), dim=-1)
        vertical = torch.cat((zeros, heights.cumsum(dim=-1)[..., :-1], ones), dim=-1)
        valid = (
            valid
            & (horizontal.diff(dim=-1) > 0).all(dim=-1)
            & (vertical.diff(dim=-1) > 0).all(dim=-1)
        )
        if self._validate_args and not bool(valid.all()):
            raise ValueError("RQS cumulative knots collapse in the selected floating-point dtype")
        fallback = torch.arange(bins + 1, dtype=widths.dtype, device=widths.device) / bins
        self.horizontal = torch.where(valid[..., None], horizontal, fallback)
        self.vertical = torch.where(valid[..., None], vertical, fallback)
        self.derivatives = torch.where(valid[..., None], derivatives, torch.ones_like(derivatives))
        self.widths = self.horizontal.diff(dim=-1)
        self.heights = self.vertical.diff(dim=-1)
        self._valid_parameters = valid

    @classmethod
    def from_logits(
        cls,
        widths: Tensor,
        heights: Tensor,
        derivatives: Tensor,
        *,
        min_bin_width: float = 1e-4,
        min_bin_height: float = 1e-4,
        min_derivative: float = 1e-4,
        validate_args: bool = False,
        cache_size: int = 0,
    ) -> "BoundedRQSTransform":
        """Constrain logits with explicit positive floors, including endpoints.

        Widths/heights use ``floor + (1-K*floor)*softmax(logits)``;
        derivatives use ``min_derivative + softplus(logits)``.  The constructor
        then normalizes positive bin masses and pins cumulative endpoints.
        Floors are modeling constraints, not just implementation tolerances.
        """
        if not isinstance(widths, Tensor) or widths.ndim < 1:
            raise TypeError("RQS width logits must have a final bin axis")
        bins = widths.shape[-1]
        _floors(bins, min_bin_width, min_bin_height, min_derivative)
        widths = min_bin_width + (1 - bins * min_bin_width) * widths.softmax(dim=-1)
        heights = min_bin_height + (1 - bins * min_bin_height) * heights.softmax(dim=-1)
        derivatives = min_derivative + F.softplus(derivatives)
        return cls(widths, heights, derivatives, validate_args=validate_args, cache_size=cache_size)

    @property
    def bins(self) -> int:
        return self.horizontal.shape[-1] - 1

    def __repr__(self) -> str:
        return f"{type(self).__name__}(bins={self.bins}, batch_shape={tuple(self._batch_shape)})"

    def forward_shape(self, shape: Size) -> Size:
        return Size(torch.broadcast_shapes(shape, self._batch_shape))

    def inverse_shape(self, shape: Size) -> Size:
        return self.forward_shape(shape)

    def _prepare(self, value: Tensor) -> tuple[Tensor, Tensor]:
        if not isinstance(value, Tensor):
            value = torch.as_tensor(
                value, dtype=self.horizontal.dtype, device=self.horizontal.device
            )
        if value.dtype != self.horizontal.dtype or value.device != self.horizontal.device:
            raise ValueError("RQS values and parameters must have the same dtype and device")
        value = value.expand(self.forward_shape(value.shape))
        valid = torch.isfinite(value) & (value >= 0) & (value <= 1) & self._valid_parameters
        if self._validate_args and not bool(valid.all()):
            raise ValueError("RQS inputs must be finite and in [0, 1], with valid parameters")
        return torch.where(valid, value, torch.full_like(value, 0.5)), valid

    def _bin(
        self, value: Tensor, *, inverse: bool
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        shape = value.shape
        lookup = self.vertical if inverse else self.horizontal
        lookup = lookup.expand(shape + (self.bins + 1,)).contiguous()
        index = (
            torch.searchsorted(lookup, value[..., None].contiguous(), right=True).squeeze(-1) - 1
        )
        index = index.clamp(0, self.bins - 1)

        def select(array: Tensor, offset: int = 0) -> Tensor:
            expanded = array.expand(shape + (array.shape[-1],))
            return expanded.gather(-1, (index + offset)[..., None]).squeeze(-1)

        return (
            select(self.horizontal),
            select(self.vertical),
            select(self.widths),
            select(self.heights),
            select(self.derivatives),
            select(self.derivatives, 1),
        )

    @staticmethod
    def _scaled_slopes(
        width: Tensor, height: Tensor, d0: Tensor, d1: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        secant = height / width
        scale = torch.maximum(secant, torch.maximum(d0, d1))
        return secant / scale, d0 / scale, d1 / scale, scale

    @staticmethod
    def _log_derivative(t: Tensor, s: Tensor, d0: Tensor, d1: Tensor, scale: Tensor) -> Tensor:
        complement = 1 - t
        cross = t * complement
        # The denominator cannot cancel catastrophically: the negative part
        # is at most half of s.  This form also gives D=s exactly at identity.
        denominator = s + ((d0 - s) + (d1 - s)) * cross
        numerator = d0 * complement.square() + 2 * s * cross + d1 * t.square()
        return scale.log() + 2 * s.log() + numerator.log() - 2 * denominator.log()

    def _forward(self, value: Tensor) -> tuple[Tensor, Tensor]:
        x, valid = self._prepare(value)
        x0, y0, width, height, left, right = self._bin(x, inverse=False)
        s, d0, d1, scale = self._scaled_slopes(width, height, left, right)
        t = (x - x0) / width
        complement = 1 - t
        denominator = s + ((d0 - s) + (d1 - s)) * t * complement
        fraction = t * (s * t + d0 * complement) / denominator
        remaining = complement * (s * complement + d1 * t) / denominator
        # Evaluate from the nearer output endpoint.  The two expressions are
        # the same rational-quadratic function, not approximations.
        y = torch.where(fraction <= 0.5, y0 + height * fraction, y0 + height - height * remaining)
        y = torch.where(x == 0, torch.zeros_like(y), y)
        y = torch.where(x == 1, torch.ones_like(y), y)
        ladj = self._log_derivative(t, s, d0, d1, scale)
        nan = torch.full_like(y, torch.nan)
        return torch.where(valid, y, nan), torch.where(valid, ladj, nan)

    def inverse_and_ladj(self, value: Tensor) -> tuple[Tensor, Tensor]:
        """Inverse value and inverse log Jacobian, with a single bin lookup."""
        y, valid = self._prepare(value)
        x0, y0, width, height, left, right = self._bin(y, inverse=True)
        s, d0, d1, scale = self._scaled_slopes(width, height, left, right)
        p = (y - y0) / height
        complement = 1 - p
        # These are the quadratic coefficients after normalizing by height.
        # sqrt(discriminant) is rewritten as a positive sum; this avoids
        # subtracting nearly equal terms in b*b - 4*a*c.
        b = d0 * complement + p * (2 * s - d1)
        a = s - b
        difference = d0 * complement - d1 * p
        root = (difference.square() + 4 * s.square() * p * complement).sqrt()
        positive_b = b >= 0
        denominator_a = torch.where(positive_b, torch.ones_like(a), 2 * a)
        denominator_b = torch.where(positive_b, b + root, torch.ones_like(b))
        t_from_b = 2 * s * p / denominator_b
        t_from_a = (root - b) / denominator_a
        t = torch.where(positive_b, t_from_b, t_from_a)
        # a=0 is not a special approximation: its linear equation is already
        # solved exactly by t_from_b.  Endpoints share the same formula.
        x = x0 + width * t
        x = torch.where(y == 0, torch.zeros_like(x), x)
        x = torch.where(y == 1, torch.ones_like(x), x)
        ladj = -self._log_derivative(t, s, d0, d1, scale)
        nan = torch.full_like(x, torch.nan)
        return torch.where(valid, x, nan), torch.where(valid, ladj, nan)

    def _call(self, x: Tensor) -> Tensor:
        return self._forward(x)[0]

    def _inverse(self, y: Tensor) -> Tensor:
        return self.inverse_and_ladj(y)[0]

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return self._forward(x)[1]

    def call_and_ladj(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self._forward(x)

    @property
    def inv(self) -> Transform:
        inverse = self._inv() if self._inv is not None else None
        if inverse is None:
            inverse = _InverseBoundedRQS(self)
            self._inv = weakref.ref(inverse)
        return inverse

    def with_cache(self, cache_size: int = 1) -> "BoundedRQSTransform":
        if cache_size == self._cache_size:
            return self
        result = copy.copy(self)
        Transform.__init__(result, cache_size=cache_size)
        return result


class _InverseBoundedRQS(Transform):
    """Fused inverse wrapper so a Zuko inverse does not repeat forward work."""

    domain = constraints.unit_interval
    codomain = constraints.unit_interval
    bijective = True
    sign = 1

    def __init__(self, base: BoundedRQSTransform) -> None:
        super().__init__(cache_size=base._cache_size)
        self.base = base

    @property
    def inv(self) -> BoundedRQSTransform:
        return self.base

    def _call(self, value: Tensor) -> Tensor:
        return self.base._inv_call(value)

    def _inverse(self, value: Tensor) -> Tensor:
        return self.base(value)

    def call_and_ladj(self, value: Tensor) -> tuple[Tensor, Tensor]:
        return self.base.inverse_and_ladj(value)

    def log_abs_det_jacobian(self, y: Tensor, x: Tensor) -> Tensor:
        return -self.base.log_abs_det_jacobian(x, y)

    def forward_shape(self, shape: Size) -> Size:
        return self.base.inverse_shape(shape)

    def inverse_shape(self, shape: Size) -> Size:
        return self.base.forward_shape(shape)

    def with_cache(self, cache_size: int = 1) -> Transform:
        return self.base.with_cache(cache_size).inv
