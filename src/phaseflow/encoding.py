"""The exact condition feature layout shared with native inference.

Physical input order: wavelength in nm, cosine of the angle between the
particle axis and the incident propagation direction. This is not a latitude
in radians. There is no north/south folding.
"""

import math

import torch
from torch import Tensor, nn


class ConditionEncoder(nn.Module):
    """Raw normalized scalars followed by Gaussian one-blob integrals.

    Output: [s_lambda, s_eta, blob_lambda[0:K], blob_eta[0:K]].
    The Gaussian has sigma=1/K. Mass outside [0,1] is not renormalized; these
    are features, not a probability density. K=0 disables one-blob features.
    """

    def __init__(self, wavelength_min_nm: float, wavelength_max_nm: float, bins: int = 16):
        super().__init__()
        self.wavelength_min_nm = float(wavelength_min_nm)
        self.wavelength_max_nm = float(wavelength_max_nm)
        if isinstance(bins, bool) or not isinstance(bins, int) or bins < 0:
            raise ValueError("one_blob_bins must be a nonnegative integer")
        self.bins = bins
        if not 0 < self.wavelength_min_nm < self.wavelength_max_nm:
            raise ValueError("wavelength bounds must be positive and increasing")
        if not math.isfinite(self.wavelength_max_nm):
            raise ValueError("wavelength bounds must be finite")
        # Integer knots survive model.float()/double() without retaining a
        # previously rounded grid. Division occurs in the current input dtype.
        self.register_buffer("edges", torch.arange(self.bins + 1, dtype=torch.int64))

    @property
    def out_features(self) -> int:
        return 2 + 2 * self.bins

    def normalized(self, conditions: Tensor) -> Tensor:
        wavelength, cosine = conditions.unbind(-1)
        return torch.stack(
            (
                (wavelength - self.wavelength_min_nm)
                / (self.wavelength_max_nm - self.wavelength_min_nm),
                (cosine + 1) * 0.5,
            ),
            dim=-1,
        )

    def forward(self, conditions: Tensor) -> Tensor:
        raw = self.normalized(conditions)
        if self.bins == 0:
            return raw
        edges = self.edges.to(dtype=raw.dtype) / self.bins
        z = (edges - raw[..., :, None]) * (self.bins / math.sqrt(2))
        cumulative = 0.5 * torch.erf(z)
        blobs = cumulative[..., 1:] - cumulative[..., :-1]
        return torch.cat((raw, blobs.flatten(-2)), dim=-1)
