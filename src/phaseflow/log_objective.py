"""Versioned, solid-angle log-density regression for a fixed Rainbow record.

The training query distribution is the target (NLL control only), an equal
target/uniform mixture (objective schema 1), or spherical uniform alone (schema
2). Appropriate importance weights recover the target NLL; the squared log-PDF
error is always averaged under uniform solid angle. The target, HG coefficient,
flow architecture and normalization are unchanged.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
import torch

from .rainbow import RainbowReference

LOG_UNIFORM = -math.log(4 * math.pi)
_UNIFORM_STREAMS = {"train": 5, "validation": 6, "test": 7}
_ZERO_SCAN_ELEMENTS = 1_000_000


@dataclass(frozen=True)
class LogObjectiveConfig:
    schema_version: int = 1
    nll_weight: float = 1.0
    beta: float = 0.1
    sampling: str = "target_uniform"
    target_fraction: float = 0.5
    selection_metric: str = "log_rmse"
    validation_uniform_samples: int = 32768
    test_uniform_samples: int = 65536
    zero_policy: str = "error"

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version not in (1, 2):
            raise ValueError("log objective schema_version must be 1 or 2")
        for name in ("nll_weight", "beta", "target_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or (
                not math.isfinite(value) or value < 0
            ):
                raise ValueError(f"log objective {name} must be finite and nonnegative")
            object.__setattr__(self, name, float(value))
        if self.nll_weight == 0 and self.beta == 0:
            raise ValueError("at least one log objective coefficient must be positive")
        if self.schema_version == 1:
            if self.target_fraction != 0.5:
                raise ValueError("log objective target_fraction must be exactly 0.5")
            if self.sampling not in ("target", "target_uniform"):
                raise ValueError("log objective sampling must be target or target_uniform")
            if self.sampling == "target" and self.beta != 0:
                raise ValueError("target sampling is an NLL control and requires beta=0")
        else:
            if self.sampling != "uniform":
                raise ValueError("log objective schema 2 requires sampling='uniform'")
            if self.target_fraction != 0.0:
                raise ValueError("uniform sampling requires target_fraction=0.0")
        if self.selection_metric not in ("nll", "log_rmse"):
            raise ValueError("log objective selection_metric must be nll or log_rmse")
        for name in ("validation_uniform_samples", "test_uniform_samples"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 2:
                raise ValueError(f"log objective {name} must be an integer of at least two")
        if self.zero_policy != "error":
            raise ValueError("log objective zero_policy must be error; no density floor is used")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> LogObjectiveConfig:
        if not isinstance(values, dict):
            raise ValueError("log objective must be a JSON object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown log objective fields: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate_training_count(self, count: int) -> None:
        _count_and_seed(count)
        if self.sampling == "target_uniform" and count % 2:
            raise ValueError("target_uniform requires an even train_samples for the half mixture")


@dataclass(frozen=True)
class TrainingPool:
    """Actual geometry-cast query points and teacher values evaluated there."""

    directions: np.ndarray
    log_p: np.ndarray
    components: np.ndarray
    provenance: dict[str, Any]


def _count_and_seed(count: int, seed: int | None = None) -> None:
    if type(count) is not int or count < 1:
        raise ValueError("point count must be a positive integer")
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
        raise ValueError("point seed must be an integer in [0,2^63)")


def _numpy_dtype(geometry_dtype: Any) -> np.dtype:
    if geometry_dtype == torch.float32:
        return np.dtype(np.float32)
    if geometry_dtype == torch.float64:
        return np.dtype(np.float64)
    try:
        dtype = np.dtype(geometry_dtype)
    except TypeError as error:
        raise ValueError("geometry_dtype must be float32 or float64") from error
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("geometry_dtype must be float32 or float64")
    return dtype


def verify_positive_teacher(reference: RainbowReference) -> dict[str, Any]:
    """Scan every stored cell before the whole-sphere log workflow starts.

    Checking only queried points cannot prove positivity. A zero marginal makes
    all cells in that azimuth row zero even if its unused conditional is positive.
    Temporaries contain at most ``_ZERO_SCAN_ELEMENTS`` cell differences.
    """
    reference._require_open()
    marginal = np.diff(reference.phi_cdf)
    delta_u = np.diff(reference.u_edges.astype(np.longdouble))
    zero_area = np.longdouble(0)
    zero_cells = 0
    columns = min(reference.n_theta, _ZERO_SCAN_ELEMENTS)
    rows = max(1, _ZERO_SCAN_ELEMENTS // columns)
    for j in range(0, reference.n_phi, rows):
        end_j = min(j + rows, reference.n_phi)
        zero_marginal = marginal[j:end_j] == 0
        for i in range(0, reference.n_theta, columns):
            end_i = min(i + columns, reference.n_theta)
            difference = np.diff(reference.theta_cdf[j:end_j, i : end_i + 1], axis=1)
            zeros = (difference == 0) | zero_marginal[:, None]
            counts = zeros.sum(axis=0, dtype=np.int64)
            zero_cells += int(counts.sum())
            zero_area += np.sum(counts * delta_u[i:end_i], dtype=np.longdouble)
    fraction = float(zero_area / reference.n_phi)
    report = {
        "density_measure": "solid_angle_sr",
        "zero_policy": "error",
        "reference_zero_solid_angle_fraction": fraction,
        "zero_cell_count": zero_cells,
        "scanned_cell_count": reference.n_phi * reference.n_theta,
        "scope": "all_stored_cdf_cells",
    }
    if zero_cells:
        raise ValueError(
            "whole-sphere log-PDF error requires a positive teacher in every cell; "
            f"zero_solid_angle_fraction={fraction:.17g}, zero_cell_count={zero_cells}. "
            "True zeros are not floored, discarded or resampled. Use the legacy NLL "
            "workflow for a teacher containing zero-density cells."
        )
    return report


def _uniform_directions(reference: RainbowReference, count: int, seed: int, stream: str | int):
    _count_and_seed(count, seed)
    if isinstance(stream, str):
        if stream not in _UNIFORM_STREAMS:
            raise ValueError("uniform stream must be train, validation or test")
        stream_id = _UNIFORM_STREAMS[stream]
    elif type(stream) is int and stream in _UNIFORM_STREAMS.values():
        stream_id = stream
    else:
        raise ValueError("uniform stream ID must be 5, 6 or 7")
    reference._require_open()
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, stream_id])))
    # Same specified open midpoint grid as CDF queries. These coordinates are
    # geometric spherical area and azimuth, NOT the HG probability coordinate.
    uniforms = (generator.integers(0, 2**52, size=(count, 2), dtype=np.int64) + 0.5) / 2**52
    u, phi = uniforms[:, 0], 2 * np.pi * uniforms[:, 1] - np.pi
    radius = 2 * np.sqrt(u) * np.sqrt(1 - u)
    source = np.stack((radius * np.cos(phi), radius * np.sin(phi), 1 - 2 * u), axis=-1)
    return source @ reference.source_to_nf.T


def _teacher_at_cast(reference: RainbowReference, raw_directions: np.ndarray, geometry_dtype):
    directions = np.asarray(raw_directions, dtype=_numpy_dtype(geometry_dtype))
    log_p = np.asarray(reference.log_prob(directions), dtype=np.float64)
    if not np.isfinite(directions).all() or not np.isfinite(log_p).all():
        raise FloatingPointError(
            "log workflow query points must have finite directions and positive finite "
            "teacher PDFs at the actual geometry-cast directions; no points are replaced"
        )
    return directions, log_p


def make_uniform_points(
    reference: RainbowReference,
    count: int,
    seed: int,
    stream: str | int,
    geometry_dtype: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent uniform-solid-angle queries, labelled after geometry casting.

    Call ``verify_positive_teacher`` once before preparing the new workflow.
    The stream IDs are 5 (training), 6 (validation), and 7 (final test).
    """
    raw = _uniform_directions(reference, count, seed, stream)
    return _teacher_at_cast(reference, raw, geometry_dtype)


def make_training_pool(
    reference: RainbowReference,
    count: int,
    seed: int,
    objective: LogObjectiveConfig,
    geometry_dtype: Any,
) -> TrainingPool:
    """Build the complete fixed pool; all beta candidates share its identity.

    Call ``verify_positive_teacher`` once before preparing the new workflow.
    Mixed CDF/uniform points are interleaved, starting with CDF, so every even
    prefix is balanced. Uniform-only pools use the same uniform stream and no
    CDF sampling. Labels are always evaluated at the returned points.
    """
    # Delayed import avoids a trainer/objective import cycle and deliberately
    # reuses the existing CDF RNG stream without changing its historical bytes.
    from .single_condition import _array_hash, _points

    if not isinstance(objective, LogObjectiveConfig):
        raise ValueError("objective must be a LogObjectiveConfig")
    objective.validate_training_count(count)
    _count_and_seed(count, seed)
    mixture = objective.sampling == "target_uniform"
    uniform_only = objective.sampling == "uniform"
    if uniform_only:
        target_count, uniform_count = 0, count
        raw = _uniform_directions(reference, count, seed, "train")
        raw_labels = reference.log_prob(raw)
        components = np.ones(count, dtype=np.uint8)
    else:
        target_count = count // 2 if mixture else count
        uniform_count = count - target_count
        raw_p, label_p = _points(reference, target_count, seed, "train")
        components = np.zeros(count, dtype=np.uint8)
        if mixture:
            raw_u = _uniform_directions(reference, uniform_count, seed, "train")
            label_u = reference.log_prob(raw_u)
            raw = np.empty((count, 3), dtype=np.float64)
            raw_labels = np.empty(count, dtype=np.float64)
            raw[0::2], raw[1::2] = raw_p, raw_u
            raw_labels[0::2], raw_labels[1::2] = label_p, label_u
            components[1::2] = 1
        else:
            raw, raw_labels = raw_p, label_p
    if not np.isfinite(raw_labels).all():
        raise FloatingPointError("log workflow requires positive teacher PDF at every raw query")
    directions, log_p = _teacher_at_cast(reference, raw, geometry_dtype)
    provenance = {
        "schema": "phaseflow.training_pool.v1",
        "sampling": objective.sampling,
        "target_fraction": 0.0 if uniform_only else (0.5 if mixture else 1.0),
        "density_measure": "solid_angle_sr",
        "sample_count": count,
        "component_counts": {"cdf": target_count, "uniform": uniform_count},
        "component_codes": {"cdf": 0, "uniform": 1},
        "order": "uniform" if uniform_only else ("interleaved_cdf_uniform" if mixture else "cdf"),
        "data_seed": seed,
        "source_streams": {"cdf": None if uniform_only else 0, "uniform": 5 if uniform_count else None},
        "rng": "numpy.PCG64(SeedSequence([data_seed,stream_id]))",
        "uniforms": "52-bit open midpoint grid",
        "geometry_dtype": directions.dtype.name,
        "raw_points_sha256": _array_hash(raw, raw_labels, components),
        "runtime_points_sha256": _array_hash(directions, log_p, components),
        "raw_hash_covers": "FP64 directions_nf, FP64 teacher_log_pdf, uint8 components",
        "runtime_hash_covers": "geometry-cast directions_nf, FP64 teacher_log_pdf, uint8 components",
        "teacher_value_query": "actual_geometry_cast_directions_nf",
        "dataset_fingerprint": reference.fingerprint(),
    }
    return TrainingPool(directions, log_p, components, provenance)


def loss_terms(
    log_q: torch.Tensor,
    log_p: torch.Tensor,
    objective: LogObjectiveConfig,
    *,
    report_components: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute the query-weighted objective in the model's log-PDF dtype.

    Teacher labels/importance weights are detached. There is no minibatch weight
    renormalization or device-to-host synchronization. The caller validates the
    fixed teacher pool once and checks finite optimization blocks before saving.
    Disabled terms return zero and are not evaluated (in particular, no 0*inf).
    ``report_components=True`` evaluates both raw terms for positive mixed or
    uniform teacher pools, including terms whose coefficient is zero. Such
    diagnostics never enter the loss through an inactive multiplication by zero.
    For uniform-only queries, log MSE is an ordinary unweighted sample mean;
    optional NLL uses p/u, without clipping its potentially unbounded weights.
    """
    if not isinstance(objective, LogObjectiveConfig):
        raise ValueError("objective must be a LogObjectiveConfig")
    if type(report_components) is not bool:
        raise ValueError("report_components must be boolean")
    if log_q.ndim != 1 or log_q.numel() == 0 or log_p.shape != log_q.shape:
        raise ValueError("log_q and log_p must be matching nonempty vectors")
    if log_q.dtype not in (torch.float32, torch.float64):
        raise ValueError("training log_q must be float32 or float64")
    teacher = log_p.detach().to(dtype=log_q.dtype, device=log_q.device)
    zero = log_q.new_zeros(())
    if objective.sampling == "target":
        nll = -log_q.mean() if objective.nll_weight else zero
        return {"loss": objective.nll_weight * nll, "nll": nll, "log_mse": zero}
    log_u = teacher.new_full((), LOG_UNIFORM)
    if objective.sampling == "uniform":
        nll = (
            -(torch.exp(teacher - log_u) * log_q).mean()
            if objective.nll_weight or report_components else zero
        )
        log_mse = (
            (log_q - teacher).square().mean()
            if objective.beta or report_components else zero
        )
        loss = objective.nll_weight * nll if objective.nll_weight else zero
        if objective.beta:
            loss = loss + objective.beta * log_mse
        return {"loss": loss, "nll": nll, "log_mse": log_mse}
    # Shift each logaddexp to avoid cancellation in log p - log r when the
    # density spans an extreme range. These are exactly p/r and u/r for the
    # same r=(p+u)/2; neither weights nor minibatches are renormalized.
    weight_p = torch.exp(math.log(2.0) - torch.logaddexp(zero, log_u - teacher))
    weight_u = torch.exp(math.log(2.0) - torch.logaddexp(zero, teacher - log_u))
    nll = (
        -(weight_p * log_q).mean() if objective.nll_weight or report_components else zero
    )
    log_mse = (
        (weight_u * (log_q - teacher).square()).mean()
        if objective.beta or report_components else zero
    )
    loss = objective.nll_weight * nll if objective.nll_weight else zero
    if objective.beta:
        loss = loss + objective.beta * log_mse
    return {
        "loss": loss,
        "nll": nll,
        "log_mse": log_mse,
    }


@torch.no_grad()
def evaluate_log_shape(
    model,
    reference: RainbowReference,
    points: tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor],
    batch_size: int,
) -> dict[str, Any]:
    """Log-shape and RMS relative error on independent spherical-uniform points.

    NF calls use the explicit model geometry dtype. Teacher values and all
    statistical reductions use FP64 on the selected device. No evaluation point
    is clipped or omitted. Relative RMS is accumulated in log space, and only
    overflow of the final FP64 value is represented by null plus a status.
    """
    reference._require_open()
    if model.hg_g != reference.g or model.incident_cosine != float(reference.condition[1]):
        raise ValueError("model HG g / incidence does not match the supplied Rainbow record")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("evaluation batch_size must be a positive integer")
    directions = torch.as_tensor(points[0], device=model.device, dtype=model.geometry_dtype)
    teacher = torch.as_tensor(points[1], device=model.device, dtype=torch.float64).detach()
    if directions.ndim != 2 or directions.shape[1] != 3 or teacher.shape != (len(directions),):
        raise ValueError("uniform points require directions [N,3] and log_p [N]")
    count = len(directions)
    if count < 2:
        raise ValueError("uniform log-shape evaluation requires at least two points")
    log_q = torch.empty_like(teacher)
    for start in range(0, count, batch_size):
        outgoing = directions[start : start + batch_size]
        values = model.log_prob(outgoing)
        if values.shape != (len(outgoing),):
            raise FloatingPointError("model log_prob has the wrong shape")
        log_q[start : start + len(outgoing)] = values
    error = log_q - teacher
    squared = error.square()
    # log|exp(error)-1|, stable for either sign and even error > log(DBL_MAX).
    # An exact zero error has log absolute relative error = -inf, as required.
    log_relative = torch.where(
        error > 0,
        error + torch.log(-torch.expm1(-error)),
        torch.log(-torch.expm1(error)),
    )
    log_relative_rmse = 0.5 * (torch.logsumexp(2 * log_relative, dim=0) - math.log(count))
    statistics = torch.stack((
        squared.mean(), squared.std(correction=1) / math.sqrt(count), log_relative_rmse,
    )).cpu().numpy()
    log_mse, log_mse_se, log_relative_rms = statistics.tolist()
    if not math.isfinite(log_mse) or not math.isfinite(log_mse_se):
        raise FloatingPointError("nonfinite uniform log-PDF statistic; no points are omitted")
    if math.isnan(log_relative_rms):
        raise FloatingPointError("invalid uniform relative-error statistic")
    try:
        relative_rmse = math.exp(log_relative_rms)
    except OverflowError:
        relative_rmse = math.inf
    relative_finite = math.isfinite(relative_rmse)
    return {
        "log_mse": log_mse,
        "log_rmse": math.sqrt(log_mse),
        "log_mse_standard_error": log_mse_se,
        "relative_rmse": relative_rmse if relative_finite else None,
        "relative_rmse_status": "finite" if relative_finite else "overflow_float64",
        "uniform_sample_count": count,
        "log_shape_measure": "uniform_solid_angle",
        "log_base": "natural",
    }
