"""Zuko coupling with the axial-incidence rotational symmetry built in.

The residual lives on the mirror-folded HG probability square (u,v).  At
incident_cosine=+/-1 an axisymmetric particle has no distinguished azimuth.
Consequently every radial update must be independent of v and every azimuth
update must be the identity.  Both requirements are enforced continuously
using a=sqrt((1-incident_cosine)*(1+incident_cosine))=sin(inclination);
setting only the azimuth update to identity would not suffice.

This is a physical modeling constraint in addition to reflection folding.
It does not guarantee a globally smooth spherical diffeomorphism, nor does
it assume symmetry between the two signs of incident_cosine.
"""

from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Transform
from zuko.flows.coupling import GeneralCouplingTransform
from zuko.transforms import DependentTransform
from zuko.utils import broadcast, unpack

from .splines import BoundedRQSTransform, _floors

__all__ = ["AxialSymmetricCouplingTransform"]


class AxialSymmetricCouplingTransform(GeneralCouplingTransform):
    """A two-dimensional bounded RQS coupling with axial azimuth symmetry.

    The constructor follows Zuko's coupling arguments for features, context,
    mask and MLP keywords.  The univariate family and parameter shapes are
    fixed by ``num_bins``.  A true mask entry denotes the retained coordinate.
    The context begins with normalized wavelength and normalized signed
    incident cosine, exactly as in :class:`phaseflow.encoding.ConditionEncoder`.

    The radial conditioner sees ``0.5+a*(v-0.5)`` while the actual retained v
    stays unchanged.  For an azimuth update, positive spline bin masses and
    derivatives are blended with their identity values before constructing
    the spline.  The positive floors therefore remain valid for 0<=a<=1.

    The gate is always enabled.  It permits azimuth dependence of first order
    in inclination near axial incidence, as allowed for a smooth axisymmetric
    scatterer.  For example, axis dot outgoing equals
    eta*mu + sin(inclination)*sqrt(1-mu**2)*cos(phi).  A sin(inclination)**2
    gate would unnecessarily remove this allowed first-order term.

    The derivative of the gate with respect to eta is singular at eta=+/-1.
    Conditions are fixed for the directional Jacobian and parameter training;
    endpoint gradients with respect to eta itself are not guaranteed.  The
    caller validates conditions in their configured domain before ``meta``.
    """

    def __new__(cls, features: int = 2, *args, **kwargs):
        # Zuko's parent __new__ otherwise returns an ElementWiseTransform for
        # features=1, bypassing this class's two-coordinate invariant.
        if features != 2:
            raise ValueError("AxialSymmetricCouplingTransform requires exactly two features")
        return super().__new__(cls, features=features)

    def __init__(
        self,
        features: int = 2,
        context: int = 0,
        mask: Tensor | None = None,
        *,
        num_bins: int = 16,
        min_bin_width: float = 1e-4,
        min_bin_height: float = 1e-4,
        min_derivative: float = 1e-4,
        **kwargs,
    ) -> None:
        if isinstance(context, bool) or not isinstance(context, int) or context < 2:
            raise ValueError("context must include normalized wavelength and incident cosine")
        if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins < 1:
            raise ValueError("num_bins must be a positive integer")
        _floors(num_bins, min_bin_width, min_bin_height, min_derivative)
        retained_mask = (
            torch.tensor([False, True]) if mask is None else torch.as_tensor(mask, dtype=torch.bool)
        )
        if retained_mask.shape != (2,) or int(retained_mask.sum()) != 1:
            raise ValueError("mask must retain exactly one of the two coordinates")
        factory = partial(
            BoundedRQSTransform.from_logits,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
        )
        super().__init__(
            features=features,
            context=context,
            mask=retained_mask,
            univariate=factory,
            shapes=((num_bins,), (num_bins,), (num_bins + 1,)),
            **kwargs,
        )
        self.retained_index = 0 if bool(retained_mask[0]) else 1
        self.context_features = context
        self.num_bins = num_bins
        self.min_bin_width = float(min_bin_width)
        self.min_bin_height = float(min_bin_height)
        self.min_derivative = float(min_derivative)

    def meta(self, c: Tensor, x: Tensor) -> Transform:
        if c is None or c.ndim < 1 or c.shape[-1] != self.context_features:
            raise ValueError("the complete encoded condition context is required")
        if x.ndim < 1 or x.shape[-1] != 1:
            raise ValueError("the retained coordinate must have final dimension one")
        x, c = broadcast(x, c, ignore=1)
        eta = 2 * c[..., 1:2] - 1
        strength = ((1 - eta) * (1 + eta)).sqrt()
        conditioned_x = 0.5 + strength * (x - 0.5) if self.retained_index == 1 else x
        parameters = self.hyper(torch.cat((conditioned_x, c), dim=-1))
        parameters = parameters.unflatten(-1, (-1, self.total))
        width_logits, height_logits, derivative_logits = unpack(parameters, self.shapes)
        widths = self.min_bin_width + (
            1 - self.num_bins * self.min_bin_width
        ) * width_logits.softmax(-1)
        heights = self.min_bin_height + (
            1 - self.num_bins * self.min_bin_height
        ) * height_logits.softmax(-1)
        derivatives = self.min_derivative + F.softplus(derivative_logits)
        if self.retained_index == 0:
            strength = strength[..., None]
            complement = 1 - strength
            widths = strength * widths + complement / self.num_bins
            heights = strength * heights + complement / self.num_bins
            derivatives = strength * derivatives + complement
        return DependentTransform(BoundedRQSTransform(widths, heights, derivatives), 1)
