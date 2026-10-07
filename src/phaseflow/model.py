"""A conditional HG base and a reflection-symmetric spherical residual flow.

The learned normalizing map N maps the folded HG probability square to a
uniform square. Sampling uses N^{-1}, then the analytic HG quantile warp.
Equivalently, this is an HG-base flow with correction W_g o N^{-1} o W_g^{-1}.
All returned direction densities are per steradian, not per angle pair.

The construction is a folded chart with two reflection branches, not a global
smooth diffeomorphism of S^2. Values at the poles use the phi=0 convention.
"""

import math
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, fields

import torch
from torch import Tensor, nn
from torch.distributions import Distribution, constraints
from zuko.lazy import LazyDistribution
from zuko.nn import MLP

from .coupling import AxialSymmetricCouplingTransform
from .encoding import ConditionEncoder
from .hg import hg_cdf_and_log_prob, hg_icdf, hg_log_prob
from .splines import identity_derivative_logit


@dataclass(frozen=True)
class ModelConfig:
    wavelength_min_nm: float = 380.0
    wavelength_max_nm: float = 720.0
    g_limit: float = 0.999
    g_hidden_features: tuple[int, ...] = (32, 32)
    hidden_features: tuple[int, ...] = (64, 64)
    num_coupling_layers: int = 4
    num_bins: int = 16
    one_blob_bins: int = 16
    min_bin_width: float = 1e-5
    min_bin_height: float = 1e-5
    min_derivative: float = 1e-4
    activation: str = "relu"
    include_g_context: bool = True

    def __post_init__(self):
        object.__setattr__(self, "g_hidden_features", tuple(self.g_hidden_features))
        object.__setattr__(self, "hidden_features", tuple(self.hidden_features))
        if not 0 < self.wavelength_min_nm < self.wavelength_max_nm:
            raise ValueError("wavelength bounds must be positive and increasing")
        if not math.isfinite(self.wavelength_max_nm):
            raise ValueError("wavelength bounds must be finite")
        if not 0 < self.g_limit < 1:
            raise ValueError("g_limit must be strictly between zero and one")
        for name in ("num_coupling_layers", "num_bins", "one_blob_bins"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if not isinstance(self.include_g_context, bool):
            raise ValueError("include_g_context must be a boolean")
        for widths in (self.g_hidden_features, self.hidden_features):
            if not widths or any(
                isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in widths
            ):
                raise ValueError("hidden_features must be a nonempty sequence of positive integers")
        if self.num_coupling_layers < 2 or self.num_coupling_layers % 2:
            raise ValueError("num_coupling_layers must be even and at least two")
        if self.num_bins < 2 or self.one_blob_bins < 0:
            raise ValueError("num_bins must be >=2 and one_blob_bins must be >=0")
        if not 0 < self.min_bin_width < 1 / self.num_bins:
            raise ValueError("min_bin_width must be in (0, 1/num_bins)")
        if not 0 < self.min_bin_height < 1 / self.num_bins:
            raise ValueError("min_bin_height must be in (0, 1/num_bins)")
        if not 0 < self.min_derivative < 1:
            raise ValueError("min_derivative must be in (0,1) to permit identity initialization")
        if self.activation != "relu":
            raise ValueError("schema v1 supports only relu, including in native inference")

    def to_dict(self) -> dict:
        result = asdict(self)
        for key in ("hidden_features", "g_hidden_features"):
            result[key] = list(result[key])
        return result

    @classmethod
    def from_dict(cls, value: Mapping) -> "ModelConfig":
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown model configuration fields: {sorted(unknown)}")
        return cls(**dict(value))


class HGHead(nn.Module):
    def __init__(self, in_features: int, config: ModelConfig):
        super().__init__()
        self.g_limit = config.g_limit
        self.mlp = MLP(in_features, 1, hidden_features=config.g_hidden_features, activation=nn.ReLU)
        # The initial model is HG(g=0). Moment pretraining supplies the actual g(c).
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, encoded_conditions: Tensor) -> Tensor:
        return self.g_limit * torch.tanh(self.mlp(encoded_conditions).squeeze(-1))


class PhaseFlow(LazyDistribution):
    """Conditional surface density for axisymmetric, mirror-symmetric particles.

    Conditions have last dimension 2: [wavelength_nm, incident_cosine].
    Directions have last dimension 3 in the scattering frame with incident
    propagation along +z and the particle axis in the xz plane, x>=0.

    ``model(c)`` returns a torch Distribution of 3-component unit vectors with
    a density relative to solid angle. The model has two continuous random
    degrees of freedom; the embedding dimension is not the density dimension.

    Network and RQS arithmetic follow the model dtype (float32 or float64).
    HG and spherical coordinate arithmetic use float64, and sampled directions
    are returned in float64 so sharp HG lobes are not collapsed at the poles.
    """

    def __init__(self, config: ModelConfig | None = None, *, validate_args: bool = True):
        super().__init__()
        self.config = config or ModelConfig()
        self.validate_args = validate_args
        cfg = self.config
        self.encoder = ConditionEncoder(
            cfg.wavelength_min_nm, cfg.wavelength_max_nm, cfg.one_blob_bins
        )
        self.g_head = HGHead(self.encoder.out_features, cfg)
        context_features = self.encoder.out_features + int(cfg.include_g_context)
        self.couplings = nn.ModuleList()
        for layer_index in range(cfg.num_coupling_layers):
            # True means retained. First transform changes y1 and retains y2.
            retained = 1 if layer_index % 2 == 0 else 0
            layer = AxialSymmetricCouplingTransform(
                features=2,
                context=context_features,
                mask=[retained == 0, retained == 1],
                num_bins=cfg.num_bins,
                min_bin_width=cfg.min_bin_width,
                min_bin_height=cfg.min_bin_height,
                min_derivative=cfg.min_derivative,
                hidden_features=cfg.hidden_features,
                activation=nn.ReLU,
            )
            nn.init.zeros_(layer.hyper[-1].weight)
            nn.init.zeros_(layer.hyper[-1].bias)
            with torch.no_grad():
                layer.hyper[-1].bias[2 * cfg.num_bins :] = identity_derivative_logit(
                    cfg.min_derivative
                )
            self.couplings.append(layer)

    @property
    def dtype(self) -> torch.dtype:
        return self.g_head.mlp[0].weight.dtype

    @property
    def device(self) -> torch.device:
        return self.g_head.mlp[0].weight.device

    def _conditions(self, conditions: Tensor) -> Tensor:
        c = torch.as_tensor(conditions, device=self.device, dtype=self.dtype)
        if c.ndim < 1 or c.shape[-1] != 2:
            raise ValueError("conditions must have shape (..., 2): wavelength_nm, incident_cosine")
        if self.dtype not in (torch.float32, torch.float64):
            raise TypeError("phaseflow v1 supports float32 and float64 model parameters")
        if self.validate_args:
            if not torch.isfinite(c).all():
                raise ValueError("conditions must be finite")
            lo, hi = self.config.wavelength_min_nm, self.config.wavelength_max_nm
            if ((c[..., 0] < lo) | (c[..., 0] > hi)).any():
                raise ValueError(
                    "wavelength is outside the configured range; extrapolation is disabled"
                )
            if (c[..., 1].abs() > 1).any():
                raise ValueError("incident_cosine must be in [-1,1], not an angle in radians")
        return c

    def _context(self, conditions: Tensor) -> tuple[Tensor, Tensor]:
        c = self._conditions(conditions)
        encoded = self.encoder(c)
        g = self.g_head(encoded)
        context = (
            torch.cat((encoded, g[..., None]), -1) if self.config.include_g_context else encoded
        )
        return context, g

    def hg_g(self, conditions: Tensor) -> Tensor:
        return self._context(conditions)[1]

    def hg_parameters(self) -> Iterator[nn.Parameter]:
        return self.g_head.parameters()

    def residual_parameters(self) -> Iterator[nn.Parameter]:
        return self.couplings.parameters()

    def freeze_hg(self, freeze: bool = True) -> None:
        for parameter in self.hg_parameters():
            parameter.requires_grad_(not freeze)

    def _run_couplings(
        self, value: Tensor, context: Tensor, *, inverse: bool
    ) -> tuple[Tensor, Tensor]:
        """Evaluate each conditioner exactly once per layer in either direction.

        Holding the scalar transform returned by ``meta`` avoids the additional
        conditioner evaluation in a generic inverse+Jacobian implementation.
        """
        batch_shape = torch.broadcast_shapes(value.shape[:-1], context.shape[:-1])
        value = value.to(dtype=self.dtype).expand(*batch_shape, 2)
        log_det = torch.zeros(value.shape[:-1], dtype=value.dtype, device=value.device)
        indices = range(len(self.couplings) - 1, -1, -1) if inverse else range(len(self.couplings))
        for layer_index in indices:
            layer = self.couplings[layer_index]
            retained = 1 if layer_index % 2 == 0 else 0
            changed = 1 - retained
            keep = value[..., retained : retained + 1]
            move = value[..., changed : changed + 1]
            scalar = layer.meta(context, keep)
            if inverse:
                result, ladj = scalar.inv.call_and_ladj(move)
            else:
                result, ladj = scalar.call_and_ladj(move)
            value = (
                torch.cat((result, keep), -1) if retained == 1 else torch.cat((keep, result), -1)
            )
            log_det = log_det + ladj
        return value, log_det

    def _local_coordinates(self, directions: Tensor) -> tuple[Tensor, Tensor]:
        d = torch.as_tensor(directions, device=self.device, dtype=torch.float64)
        if d.ndim < 1 or d.shape[-1] != 3:
            raise ValueError("directions must have shape (...,3)")
        length = torch.linalg.vector_norm(d, dim=-1)
        if self.validate_args:
            if not torch.isfinite(d).all() or (length == 0).any():
                raise ValueError("directions must be finite and nonzero")
            if ((length - 1).abs() > 2e-5).any():
                raise ValueError("directions must be unit vectors (within 2e-5)")
        # Normalize rounding in a float32 unit vector using float64 arithmetic.
        d = d / length[..., None]
        mu = d[..., 2].clamp(-1, 1)
        phi = torch.atan2(d[..., 1], d[..., 0]).abs()
        phi = torch.where((d[..., :2] == 0).all(dim=-1), torch.zeros_like(phi), phi)
        return mu, phi / math.pi

    def log_prob(self, outgoing: Tensor, conditions: Tensor) -> Tensor:
        """Evaluate log density relative to dOmega at arbitrary local directions."""
        context, g = self._context(conditions)
        mu, folded_phi = self._local_coordinates(outgoing)
        y1, log_hg = hg_cdf_and_log_prob(mu, g.to(torch.float64))
        y1, folded_phi = torch.broadcast_tensors(y1, folded_phi)
        square = torch.stack((y1, folded_phi), -1).to(self.dtype)
        _, log_r = self._run_couplings(square, context, inverse=False)
        return log_hg + log_r.to(torch.float64)

    def pdf(self, outgoing: Tensor, conditions: Tensor) -> Tensor:
        return self.log_prob(outgoing, conditions).exp()

    def sample_from_uniform(self, uniforms: Tensor, conditions: Tensor) -> tuple[Tensor, Tensor]:
        """Map two uniforms to a direction and its PDF, as a log value.

        u1 must be in (0,1), u2 in [0,1). u2<1/2 picks the negative reflection
        branch; the remaining fraction supplies the second square coordinate.
        The pole endpoints are excluded to give a single unambiguous chart.
        """
        context, g = self._context(conditions)
        u = torch.as_tensor(uniforms, device=self.device, dtype=self.dtype)
        if u.ndim < 1 or u.shape[-1] != 2:
            raise ValueError("uniforms must have shape (...,2)")
        if self.validate_args:
            valid = (
                torch.isfinite(u).all()
                and ((u[..., 0] > 0) & (u[..., 0] < 1)).all()
                and ((u[..., 1] >= 0) & (u[..., 1] < 1)).all()
            )
            if not valid:
                raise ValueError("uniforms require u1 in (0,1), u2 in [0,1)")
        batch_shape = torch.broadcast_shapes(u.shape[:-1], context.shape[:-1])
        u = u.expand(*batch_shape, 2)
        sign = torch.where(u[..., 1] < 0.5, -1.0, 1.0)
        folded_uniform = torch.where(u[..., 1] < 0.5, 2 * u[..., 1], 2 * u[..., 1] - 1)
        base = torch.stack((u[..., 0], folded_uniform), dim=-1)
        y, log_inverse_det = self._run_couplings(base, context, inverse=True)
        y = y.to(torch.float64)
        g64 = g.to(torch.float64)
        mu = hg_icdf(y[..., 0], g64)
        # Multiply pi in float64; a scalar-valued torch.where sign otherwise
        # inherits the default dtype and can round pi before promotion.
        phi = (math.pi * y[..., 1]) * sign
        transverse = torch.sqrt(((1 - mu) * (1 + mu)).clamp_min(0))
        direction = torch.stack((transverse * phi.cos(), transverse * phi.sin(), mu), -1)
        log_prob = hg_log_prob(mu, g64) - log_inverse_det.to(torch.float64)
        # An interior probability under a strictly increasing flow cannot
        # produce an exact pole in real arithmetic. Detect numerical collapse
        # instead of presenting a finite-probability atom as a surface PDF.
        valid_sample = (
            torch.isfinite(direction).all(dim=-1)
            & torch.isfinite(log_prob)
            & (y[..., 0] > 0)
            & (y[..., 0] < 1)
            & (mu.abs() < 1)
        )
        if self.validate_args and not bool(valid_sample.all()):
            raise FloatingPointError(
                "sampling produced a nonfinite result or rounded spherical pole; "
                "check numerical precision and excessive spline/HG concentration. "
                "No endpoint clamp or automatic resampling is applied."
            )
        direction = torch.where(
            valid_sample[..., None], direction, torch.full_like(direction, torch.nan)
        )
        log_prob = torch.where(valid_sample, log_prob, torch.full_like(log_prob, torch.nan))
        return direction, log_prob

    def sample_and_log_prob(
        self, conditions: Tensor, num_samples: int = 1, *, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor]:
        c = self._conditions(conditions)
        if not isinstance(num_samples, int) or num_samples < 1:
            raise ValueError("num_samples must be a positive integer")
        shape = (num_samples, *c.shape[:-1], 2)
        # Uniformly chosen midpoint floats, with no clamped endpoint atom.
        bits = 23 if self.dtype == torch.float32 else 52
        integers = torch.randint(1 << bits, shape, device=self.device, generator=generator)
        u = (integers.to(self.dtype) + 0.5) * (2.0**-bits)
        return self.sample_from_uniform(u, c)

    def forward(self, c: Tensor) -> "SphericalPhaseDistribution":
        return SphericalPhaseDistribution(self, self._conditions(c))


class SphericalPhaseDistribution(Distribution):
    """Manifold-valued Distribution; log_prob is per steradian on S^2."""

    arg_constraints: dict = {}
    support = constraints.dependent
    has_rsample = True

    def __init__(self, model: PhaseFlow, conditions: Tensor):
        self.model = model
        self.conditions = conditions
        super().__init__(conditions.shape[:-1], (3,), validate_args=False)

    def log_prob(self, value: Tensor) -> Tensor:
        return self.model.log_prob(value, self.conditions)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        return self.rsample_and_log_prob(sample_shape)[0]

    def rsample_and_log_prob(
        self, sample_shape: torch.Size = torch.Size()
    ) -> tuple[Tensor, Tensor]:
        sample_shape = torch.Size(sample_shape)
        directions, log_prob = self.model.sample_and_log_prob(
            self.conditions, math.prod(sample_shape)
        )
        return (
            directions.reshape(*sample_shape, *self.batch_shape, 3),
            log_prob.reshape(sample_shape + self.batch_shape),
        )

    def expand(self, batch_shape: torch.Size, _instance=None) -> "SphericalPhaseDistribution":
        if _instance is not None:
            raise ValueError("_instance is not supported")
        return SphericalPhaseDistribution(self.model, self.conditions.expand(*batch_shape, 2))
