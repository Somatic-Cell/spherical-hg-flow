"""Single-condition HG flow using circular/interval RQS on the sphere.

This implements the circular-spline branch of Rezende et al. (2020),
Sec. 2.1.2, periodic coupling of Sec. 2.2, and the S^2 cylinder construction
of Sec. 2.3.1.  It does not implement the paper's Mobius or exponential-map
alternatives.  Source: https://proceedings.mlr.press/v119/rezende20a/rezende20a.pdf

In the local frame, +z is the incident propagation direction and the particle
axis lies in the xz plane.  Let mu=z and phi=atan2(y,x).  With external fixed
HG coefficient g, the cylinder coordinates are

    t = H_g(mu),  v = (phi + pi)/(2*pi),  with v periodic.

Let N map these data coordinates to a uniform cylinder.  Then
log q_omega = log h_g(mu) + log|det DN|.  Sampling runs N^{-1}, followed by
the analytic HG quantile.  Both directions call one conditioner per layer.
All returned densities are per steradian and the stochastic dimension is two.

Two explicit physical extensions are enabled by default: reflected spline
parameters and an even periodic conditioner enforce y -> -y symmetry; an
incidence gate enforces exact azimuthal independence for axial incidence.
Reflection uses one full circle, with no abs-folded chart or discrete branch.
The generic untied circular-spline baseline is available with mirror_symmetry=False.

As in the paper's supplement App. A.1, the cylinder chart is valid almost
everywhere.  Positive endpoint slopes keep density finite, but do not guarantee
a direction-independent limiting density at the two spherical poles.  Exactly
polar PDF queries use canonical phi=0.  A sample rounded onto a pole is a
numerical error, never clipped or silently resampled.

This Python model is distinct from the legacy v1 model/native export contract.
It has a fixed external g, no HG MLP, and no conditional interpolation model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields

import torch
from torch import Tensor, nn
from zuko.nn import MLP

from .hg import hg_cdf_and_log_prob, hg_icdf, hg_log_prob
from .sphere_splines import CircularRQSTransform
from .splines import BoundedRQSTransform, identity_derivative_logit

__all__ = ["SphereFlowConfig", "SingleConditionSphereFlow"]


@dataclass(frozen=True)
class SphereFlowConfig:
    """Architecture of a single-condition full-circle HG residual flow.

    ``num_bins`` is the number of bins on each full interval/circle.  With
    reflection symmetry, half of the circular bins are independently learned.
    Smooth SiLU conditioners preserve periodic differentiability at the seam.
    """

    hidden_features: tuple[int, ...] = (64, 64)
    num_coupling_layers: int = 4
    num_bins: int = 16
    mirror_symmetry: bool = True
    min_bin_width: float = 1e-5
    min_bin_height: float = 1e-5
    min_derivative: float = 1e-4
    activation: str = "silu"

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden_features", tuple(self.hidden_features))
        if not self.hidden_features or any(
            isinstance(n, bool) or not isinstance(n, int) or n < 1
            for n in self.hidden_features
        ):
            raise ValueError("hidden_features must be a nonempty sequence of positive integers")
        for name in ("num_coupling_layers", "num_bins"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.num_coupling_layers < 2 or self.num_coupling_layers % 2:
            raise ValueError("num_coupling_layers must be even and at least two")
        if self.num_bins < 2:
            raise ValueError("num_bins must be at least two")
        if not isinstance(self.mirror_symmetry, bool):
            raise ValueError("mirror_symmetry must be a boolean")
        if self.mirror_symmetry and self.num_bins % 2:
            raise ValueError("mirror-symmetric circle requires an even num_bins")
        for name in ("min_bin_width", "min_bin_height"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value < 1 / self.num_bins:
                raise ValueError(f"{name} must be finite and in (0, 1/num_bins)")
        if not math.isfinite(self.min_derivative) or not 0 < self.min_derivative < 1:
            raise ValueError("min_derivative must be in (0,1) for identity initialization")
        if self.activation != "silu":
            raise ValueError("this model uses the smooth silu conditioner")

    def to_dict(self) -> dict:
        result = asdict(self)
        result["hidden_features"] = list(result["hidden_features"])
        return result

    @classmethod
    def from_dict(cls, value: Mapping) -> "SphereFlowConfig":
        unknown = set(value) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown sphere model configuration fields: {sorted(unknown)}")
        return cls(**dict(value))


class _CylinderCoupling(nn.Module):
    """One scalar update conditioned only on the retained scalar coordinate."""

    def __init__(
        self, config: SphereFlowConfig, *, update_circle: bool, dtype: torch.dtype, device
    ) -> None:
        super().__init__()
        self.config = config
        self.update_circle = update_circle
        bins = config.num_bins
        if update_circle and config.mirror_symmetry:
            self.parameter_sizes = (bins // 2, bins // 2, bins // 2 + 1)
        else:
            self.parameter_sizes = (bins, bins, bins if update_circle else bins + 1)
        inputs = 1 if update_circle or config.mirror_symmetry else 2
        self.hyper = MLP(
            inputs,
            sum(self.parameter_sizes),
            hidden_features=config.hidden_features,
            activation=nn.SiLU,
        ).to(device=device, dtype=dtype)
        nn.init.zeros_(self.hyper[-1].weight)
        nn.init.zeros_(self.hyper[-1].bias)
        # Initializing after dtype conversion preserves a double-precision
        # identity start when the constructor is asked for float64.
        with torch.no_grad():
            self.hyper[-1].bias[sum(self.parameter_sizes[:2]) :] = identity_derivative_logit(
                config.min_derivative
            )

    def scalar_transform(self, retained: Tensor, strength: float):
        cfg = self.config
        if self.update_circle:
            features = retained[..., None]
        else:
            # Periodic embedding, even under v -> 1-v in the physical model.
            phi = 2 * math.pi * (retained.remainder(1) - 0.5)
            features = (strength * phi.cos())[..., None]
            if not cfg.mirror_symmetry:
                features = torch.cat((features, (strength * phi.sin())[..., None]), -1)
        logits = self.hyper(features)
        widths, heights, derivatives = logits.split(self.parameter_sizes, dim=-1)
        kwargs = {
            "min_bin_width": cfg.min_bin_width,
            "min_bin_height": cfg.min_bin_height,
            "min_derivative": cfg.min_derivative,
        }
        if not self.update_circle:
            return BoundedRQSTransform.from_logits(widths, heights, derivatives, **kwargs)
        factory = (
            CircularRQSTransform.from_reflection_logits
            if cfg.mirror_symmetry
            else CircularRQSTransform.from_logits
        )
        return factory(widths, heights, derivatives, identity_strength=strength, **kwargs)


class SingleConditionSphereFlow(nn.Module):
    """Two-dimensional surface density for one condition and external HG g.

    Conditions are fixed, so no condition encoder or g regression is trained.
    ``hg_g`` and ``incident_cosine`` are exact Python binary64 constants and
    are included in state_dict's versioned extra state.  Moving neural weights
    to float32 therefore never rounds or clips the physical g.

    HG calculations and returned directions/log PDFs use float64; conditioner
    and spline calculations follow the model dtype.  Default float64 is
    intended for the initial correctness/accuracy experiments.
    """

    model_family = "single_condition_circular_hg"
    schema_version = 1

    def __init__(
        self,
        hg_g: float,
        incident_cosine: float,
        config: SphereFlowConfig | None = None,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
        validate_args: bool = True,
    ) -> None:
        super().__init__()
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("sphere flow supports only float32 and float64 model weights")
        self.config = config or SphereFlowConfig()
        self.validate_args = bool(validate_args)
        self._set_physics(hg_g, incident_cosine)
        self.couplings = nn.ModuleList(
            _CylinderCoupling(self.config, update_circle=bool(i % 2), dtype=dtype, device=device)
            for i in range(self.config.num_coupling_layers)
        )

    def _set_physics(self, hg_g: float, incident_cosine: float) -> None:
        if isinstance(hg_g, bool) or isinstance(incident_cosine, bool):
            raise ValueError("HG g and incident_cosine must be real-valued physical constants")
        g, eta = float(hg_g), float(incident_cosine)
        if not math.isfinite(g) or not -1 < g < 1:
            raise ValueError("external HG g must be finite and strictly between -1 and 1")
        if not math.isfinite(eta) or not -1 <= eta <= 1:
            raise ValueError("incident_cosine must be finite and in [-1, 1]")
        self._hg_g = g
        self._incident_cosine = eta
        self._azimuth_strength = math.sqrt((1 - eta) * (1 + eta))

    @property
    def hg_g(self) -> float:
        return self._hg_g

    @property
    def incident_cosine(self) -> float:
        return self._incident_cosine

    @property
    def dtype(self) -> torch.dtype:
        return self.couplings[0].hyper[0].weight.dtype

    @property
    def device(self) -> torch.device:
        return self.couplings[0].hyper[0].weight.device

    def get_extra_state(self) -> dict:
        return {
            "model_family": self.model_family,
            "schema_version": self.schema_version,
            "config": self.config.to_dict(),
            "hg_g": self.hg_g,
            "incident_cosine": self.incident_cosine,
        }

    def set_extra_state(self, state: dict) -> None:
        if (
            state.get("model_family") != self.model_family
            or state.get("schema_version") != self.schema_version
            or state.get("config") != self.config.to_dict()
        ):
            raise ValueError("checkpoint sphere model family, schema, or configuration differs")
        self._set_physics(state["hg_g"], state["incident_cosine"])

    def _run_couplings(self, value: Tensor, *, inverse: bool) -> tuple[Tensor, Tensor]:
        if self.dtype not in (torch.float32, torch.float64):
            raise TypeError("sphere flow supports only float32 and float64 model weights")
        value = value.to(device=self.device, dtype=self.dtype)
        log_det = torch.zeros(value.shape[:-1], device=self.device, dtype=self.dtype)
        layers = reversed(self.couplings) if inverse else self.couplings
        for layer in layers:
            changed = 1 if layer.update_circle else 0
            retained = 1 - changed
            scalar = layer.scalar_transform(value[..., retained], self._azimuth_strength)
            if inverse:
                result, ladj = scalar.inverse_and_ladj(value[..., changed])
            else:
                result, ladj = scalar.call_and_ladj(value[..., changed])
            value = (
                torch.stack((value[..., 0], result), -1)
                if changed == 1
                else torch.stack((result, value[..., 1]), -1)
            )
            log_det = log_det + ladj
        return value, log_det

    def _local_coordinates(self, outgoing: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        directions = torch.as_tensor(outgoing, device=self.device, dtype=torch.float64)
        if directions.ndim < 1 or directions.shape[-1] != 3:
            raise ValueError("directions must have shape (...,3)")
        length = torch.linalg.vector_norm(directions, dim=-1)
        valid = (
            torch.isfinite(directions).all(dim=-1)
            & (length > 0)
            & ((length - 1).abs() <= 2e-5)
        )
        if self.validate_args and not bool(valid.all()):
            raise ValueError("directions must be finite unit vectors (within 2e-5)")
        safe_length = torch.where(valid, length, torch.ones_like(length))
        direction = directions / safe_length[..., None]
        mu = direction[..., 2].clamp(-1, 1)
        phi = torch.atan2(direction[..., 1], direction[..., 0])
        polar = (direction[..., :2] == 0).all(dim=-1)
        phi = torch.where(polar, torch.zeros_like(phi), phi)
        v = ((phi + math.pi) / (2 * math.pi)).remainder(1)
        return mu, v, valid

    def log_prob(self, outgoing: Tensor) -> Tensor:
        """Evaluate log q relative to solid angle, for arbitrary local directions."""
        mu, v, valid = self._local_coordinates(outgoing)
        t, log_hg = hg_cdf_and_log_prob(mu, self.hg_g)
        _, log_r = self._run_couplings(torch.stack((t, v), -1), inverse=False)
        result = log_hg + log_r.to(torch.float64)
        return torch.where(valid, result, torch.full_like(result, torch.nan))

    def pdf(self, outgoing: Tensor) -> Tensor:
        return self.log_prob(outgoing).exp()

    def sample_from_uniform(self, uniforms: Tensor) -> tuple[Tensor, Tensor]:
        """Use two uniforms (u0 in (0,1), u1 in [0,1)) without branch splitting.

        Both output directions and output log PDFs have dtype float64.
        The batch shapes are (...,3) and (...) for uniforms of shape (...,2).
        """
        provided = torch.as_tensor(uniforms, device=self.device, dtype=torch.float64)
        if provided.ndim < 1 or provided.shape[-1] != 2:
            raise ValueError("uniforms must have shape (...,2)")
        valid_input = (
            torch.isfinite(provided).all(dim=-1)
            & (provided[..., 0] > 0)
            & (provided[..., 0] < 1)
            & (provided[..., 1] >= 0)
            & (provided[..., 1] < 1)
        )
        if self.validate_args and not bool(valid_input.all()):
            raise ValueError("uniforms require u0 in (0,1) and u1 in [0,1)")
        u = provided.to(dtype=self.dtype)
        valid_u = valid_input & (u[..., 0] > 0) & (u[..., 0] < 1) & (u[..., 1] < 1)
        if self.validate_args and not bool(valid_u.all()):
            raise FloatingPointError(
                "valid uniforms rounded to an excluded endpoint in the model dtype; "
                "use float64 or generate uniforms on the model's representable midpoint grid"
            )
        cylinder, log_inverse_det = self._run_couplings(u, inverse=True)
        cylinder = cylinder.to(torch.float64)
        mu = hg_icdf(cylinder[..., 0], self.hg_g)
        phi = 2 * math.pi * (cylinder[..., 1] - 0.5)
        radius = ((1 - mu) * (1 + mu)).sqrt()
        directions = torch.stack((radius * phi.cos(), radius * phi.sin(), mu), -1)
        log_prob = hg_log_prob(mu, self.hg_g) - log_inverse_det.to(torch.float64)
        valid = (
            valid_u
            & torch.isfinite(directions).all(dim=-1)
            & torch.isfinite(log_prob)
            & (cylinder[..., 0] > 0)
            & (cylinder[..., 0] < 1)
            & (mu.abs() < 1)
        )
        if self.validate_args and not bool(valid.all()):
            raise FloatingPointError(
                "sampling produced a nonfinite result or rounded spherical pole; "
                "check precision and spline/HG concentration. No clipping or resampling is applied."
            )
        return (
            torch.where(valid[..., None], directions, torch.full_like(directions, torch.nan)),
            torch.where(valid, log_prob, torch.full_like(log_prob, torch.nan)),
        )

    def sample_and_log_prob(
        self, num_samples: int = 1, *, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor]:
        """Draw reproducible uniform midpoint floats, then transform them."""
        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1:
            raise ValueError("num_samples must be a positive integer")
        bits = 23 if self.dtype == torch.float32 else 52
        integers = torch.randint(
            1 << bits, (num_samples, 2), device=self.device, generator=generator
        )
        uniforms = (integers.to(self.dtype) + 0.5) * (2.0**-bits)
        return self.sample_from_uniform(uniforms)

    def forward(self, outgoing: Tensor) -> Tensor:
        """Module-call shorthand for solid-angle log probability."""
        return self.log_prob(outgoing)
