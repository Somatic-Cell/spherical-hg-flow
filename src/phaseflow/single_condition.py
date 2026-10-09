"""One-record Rainbow training with a fixed, externally supplied HG moment.

This is a distinct versioned workflow from the legacy conditional folded-chart
model. Training points, validation points, final test points and proposal samples
have separate reproducible random streams. All likelihoods are per steradian.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
import zuko

from . import __version__
from .log_objective import (
    LogObjectiveConfig,
    evaluate_log_shape,
    loss_terms,
    make_training_pool,
    make_uniform_points,
    verify_positive_teacher,
)
from .monitoring import TrainingMonitor, check_tensorboard, plot_training_history
from .rainbow import RainbowReference
from .sphere_model import SingleConditionSphereFlow, SphereFlowConfig
from .training import (
    _atomic_torch_save,
    _dtype,
    _restore_rng,
    _rng_state,
    _runtime_metadata,
    _setup_runtime,
    atomic_json,
)

if TYPE_CHECKING:
    from .plotting import RainbowPlotConfig

FAMILY = "rainbow_single_condition"
CHECKPOINT_VERSION = 4
LOG_CHECKPOINT_VERSION = 5
READABLE_CHECKPOINT_VERSIONS = (2, 3, 4, 5)
_STREAMS = {"train": 0, "validation": 1, "minibatches": 2, "test": 3, "proposal": 4}


@dataclass
class SingleTrainingConfig:
    seed: int = 415
    data_seed: int | None = None
    device: str = "cuda"
    dtype: str = "float32"
    train_samples: int = 65536
    validation_samples: int = 32768
    test_samples: int = 65536
    proposal_samples: int = 16384
    batch_size: int = 1024
    steps: int = 2000
    learning_rate: float = 1e-3
    eval_every: int = 100
    checkpoint_every: int = 100
    eval_batch_size: int = 4096
    grad_clip_norm: float | None = 10.0
    deterministic: bool = True
    cpu_threads: int = 1
    log_every: int = 20
    tensorboard: bool = False
    train_monitor_samples: int = 8192

    def __post_init__(self) -> None:
        positive = (
            "train_samples",
            "batch_size",
            "eval_every",
            "checkpoint_every",
            "eval_batch_size",
            "cpu_threads",
            "log_every",
            "train_monitor_samples",
        )
        for name in (
            *positive,
            "seed",
            "steps",
            "validation_samples",
            "test_samples",
            "proposal_samples",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise ValueError(f"{name} must be an integer")
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.seed < 2**63 or self.steps < 0:
            raise ValueError("seed must be in [0, 2**63); steps must be nonnegative")
        if self.data_seed is not None and (
            type(self.data_seed) is not int or not 0 <= self.data_seed < 2**63
        ):
            raise ValueError("data_seed must be null or an integer in [0, 2**63)")
        if min(self.validation_samples, self.test_samples) < 2:
            raise ValueError("validation_samples and test_samples must be at least two")
        if self.proposal_samples != 0 and self.proposal_samples < 2:
            raise ValueError("proposal_samples must be zero or at least two")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a nonempty string")
        if (
            isinstance(self.learning_rate, bool)
            or not math.isfinite(self.learning_rate)
            or (self.learning_rate <= 0)
        ):
            raise ValueError("learning_rate must be finite and positive")
        if self.grad_clip_norm is not None and (
            isinstance(self.grad_clip_norm, bool)
            or not math.isfinite(self.grad_clip_norm)
            or self.grad_clip_norm <= 0
        ):
            raise ValueError("grad_clip_norm must be positive or null")
        if type(self.deterministic) is not bool:
            raise ValueError("deterministic must be boolean")
        if type(self.tensorboard) is not bool:
            raise ValueError("tensorboard must be boolean")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> SingleTrainingConfig:
        if not isinstance(values, dict):
            raise ValueError("training config must be an object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown single-condition training fields: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _generator(seed: int, stream: str) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, _STREAMS[stream]])))


def _minibatch_generator(seed: int, device: torch.device) -> torch.Generator:
    # A dedicated stream, independent of model initialization and all CDF pools.
    batch_seed = int(_generator(seed, "minibatches").integers(0, 2**63, dtype=np.int64))
    return torch.Generator(device=device).manual_seed(batch_seed)


def _resolve_device(device: str | torch.device) -> torch.device:
    selected = torch.device(device)
    if selected.type not in ("cpu", "cuda"):
        raise ValueError("the Rainbow workflow supports cpu and cuda devices")
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required by default but is not available. Install a CUDA-enabled "
                "PyTorch build for your GPU/driver; use --device cpu only for an explicit "
                "small correctness check. CPU fallback is not automatic."
            )
        index = torch.cuda.current_device() if selected.index is None else selected.index
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is unavailable")
        return torch.device("cuda", index)
    return torch.device("cpu")


def _device_points(
    points: tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    geometry_dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Transfer once, then take views/index selections on the selected device.
    # Teacher values and statistical reductions retain reference precision;
    # repeated NF evaluation uses the model's explicit geometry precision.
    return (
        torch.as_tensor(points[0], dtype=geometry_dtype, device=device),
        torch.as_tensor(points[1], dtype=torch.float64, device=device),
    )


def _uniforms(generator: np.random.Generator, count: int, bits: int = 52) -> np.ndarray:
    # A specified open 52-bit midpoint grid, not endpoint clipping or retries.
    return (generator.integers(0, 2**bits, size=(count, 2), dtype=np.int64) + 0.5) / 2**bits


def _points(reference: RainbowReference, count: int, seed: int, stream: str):
    directions, log_p = reference.sample(_uniforms(_generator(seed, stream), count))
    if not np.isfinite(directions).all() or not np.isfinite(log_p).all():
        raise FloatingPointError("CDF-generated target samples must have finite directions/PDFs")
    return directions, log_p


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _code_hash() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for name in (
        "single_condition.py",
        "rainbow.py",
        "sphere_model.py",
        "sphere_splines.py",
        "splines.py",
        "hg.py",
        "geometry.py",
        "training.py",
        "monitoring.py",
        "log_objective.py",
    ):
        digest.update(name.encode("ascii"))
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _runtime(device: torch.device, dtype: str) -> dict[str, Any]:
    result = _runtime_metadata(device, dtype)
    result.update(package_version=__version__, zuko_version=str(zuko.__version__))
    result["cpu_threads"] = torch.get_num_threads()
    if device.type == "cuda":
        result["cuda_version"] = torch.version.cuda
        result["gpu_name"] = torch.cuda.get_device_name(device)
        result["gpu_capability"] = list(torch.cuda.get_device_capability(device))
        result["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    result["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    result["matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    return result


def _estimate(values: np.ndarray) -> tuple[float, float | None]:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError("nonfinite likelihood statistic")
    return float(values.mean()), (
        float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else None
    )


def _check_model_reference(model: SingleConditionSphereFlow, reference: RainbowReference) -> None:
    if model.hg_g != reference.g or model.incident_cosine != float(reference.condition[1]):
        raise ValueError("model HG g / incidence does not match the supplied Rainbow record")


@torch.no_grad()
def _likelihood_metrics(
    model: SingleConditionSphereFlow,
    reference: RainbowReference,
    points: tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor],
    batch_size: int,
) -> dict[str, Any]:
    _check_model_reference(model, reference)
    directions, log_p = _device_points(points, model.device, model.geometry_dtype)
    log_q, log_hg = torch.empty_like(log_p), torch.empty_like(log_p)
    for start in range(0, len(directions), batch_size):
        x = directions[start : start + batch_size]
        q = model.log_prob(x)
        if q.shape != (len(x),):
            raise FloatingPointError("model log_prob has the wrong shape")
        log_q[start : start + len(x)] = q
        log_hg[start : start + len(x)] = model.hg_base_direction_log_prob(x)
    statistics = torch.stack(
        (-log_q, log_p - log_q, -log_hg, log_p - log_hg, log_q - log_hg, -log_p)
    )
    # Reduce on the GPU and transfer only twelve statistics once per evaluation.
    summary = torch.stack(
        (statistics.mean(-1), statistics.std(-1, correction=1) / math.sqrt(len(directions))),
        dim=-1,
    ).cpu().numpy()
    if not np.isfinite(summary).all():
        raise FloatingPointError("nonfinite likelihood statistic")
    (nll, nll_se), (kl, kl_se), (hg_nll, _), (hg_kl, hg_kl_se), (
        improvement,
        improvement_se,
    ), (entropy, _) = summary.tolist()
    return {
        "sample_count": len(directions),
        "density_measure": "solid_angle_sr",
        "nll": nll,
        "nll_standard_error": nll_se,
        "forward_kl_estimate": kl,
        "forward_kl_standard_error": kl_se,
        "hg_nll": hg_nll,
        "hg_forward_kl_estimate": hg_kl,
        "hg_forward_kl_standard_error": hg_kl_se,
        "nll_improvement_over_hg": improvement,
        "nll_improvement_standard_error": improvement_se,
        "target_entropy_estimate": entropy,
        "base_g": model.hg_g,
        "target_g": reference.g,
    }


@torch.no_grad()
def _proposal_metrics(
    model: SingleConditionSphereFlow,
    reference: RainbowReference,
    count: int,
    seed: int,
    batch_size: int,
) -> dict[str, Any] | None:
    if count == 0:
        return None
    generator = _generator(seed, "proposal")
    bits = 23 if torch.float32 in (model.spline_dtype, model.geometry_dtype) else 52
    uniforms = torch.as_tensor(
        _uniforms(generator, count, bits=bits), dtype=model.geometry_dtype, device=model.device
    )
    generated = torch.empty((count, 5), dtype=torch.float64, device=model.device)
    for start in range(0, count, batch_size):
        x, log_q = model.sample_from_uniform(uniforms[start : start + batch_size])
        evaluated = model.log_prob(x)
        generated[start : start + len(x)] = torch.cat(
            (x, log_q[:, None], evaluated[:, None]), dim=-1
        )
    values = generated.cpu().numpy()
    if not np.isfinite(values).all():
        raise FloatingPointError("nonfinite NF sample or PDF; samples are never discarded")
    x_np, log_q_np, evaluated_np = values[:, :3], values[:, 3], values[:, 4]
    p = reference.log_prob(x_np)
    if np.isnan(p).any() or np.isposinf(p).any():
        raise FloatingPointError("invalid reference PDF at an NF sample")
    lw = p - log_q_np
    mu = x_np[:, 2] / np.linalg.norm(x_np, axis=-1)
    max_discrepancy = float(np.max(np.abs(log_q_np - evaluated_np)))
    moment, moment_se = _estimate(mu)
    positive = np.isfinite(lw)
    relative_ess = log_mean_weight = mean_weight = None
    if positive.any():
        maximum = float(lw[positive].max())
        w = np.exp(lw - maximum)
        relative_ess = float(w.sum() ** 2 / (count * np.dot(w, w)))
        log_mean_weight = maximum + math.log(float(w.mean()))
        mean_weight = math.exp(log_mean_weight) if log_mean_weight < 709 else None
    return {
        "sample_count": count,
        "relative_ess": relative_ess,
        "uniform_midpoint_bits": bits,
        "importance_weight_mean": mean_weight,
        "log_importance_weight_mean": log_mean_weight,
        "reference_support_hits": int(positive.sum()),
        "status": "ok" if positive.any() else "no_nonzero_importance_weights",
        "learned_moment_estimate": moment,
        "learned_moment_standard_error": moment_se,
        "learned_moment_error": moment - reference.g,
        "sample_eval_log_pdf_max_abs_error": max_discrepancy,
        "ess_definition": "sum(w)^2 / (N * sum(w^2)); w=p_omega/q_omega",
    }


@torch.no_grad()
def evaluate_single_condition(
    model: SingleConditionSphereFlow,
    reference: RainbowReference,
    *,
    samples: int = 65536,
    seed: int = 2026,
    batch_size: int = 4096,
    proposal_samples: int | None = None,
    uniform_samples: int = 0,
) -> dict[str, Any]:
    """Independent same-condition likelihood/KL and proposal-weight diagnostics.

    Forward KL is a paired Monte Carlo estimate and can be slightly negative.
    ESS describes importance sampling of the normalized phase alone, not the
    variance of a complete renderer with visibility, illumination, or MIS.
    """
    for name, value in (("samples", samples), ("batch_size", batch_size)):
        if type(value) is not int or value < (2 if name == "samples" else 1):
            raise ValueError(f"invalid {name}")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be in [0, 2**63)")
    if proposal_samples is None:
        proposal_samples = samples
    if type(proposal_samples) is not int or (proposal_samples != 0 and proposal_samples < 2):
        raise ValueError("proposal_samples must be zero or at least two")
    if type(uniform_samples) is not int or (uniform_samples != 0 and uniform_samples < 2):
        raise ValueError("uniform_samples must be zero or at least two")
    if uniform_samples:
        verify_positive_teacher(reference)
    _check_model_reference(model, reference)
    previous_mode = model.training
    previous_validation = model.validate_args
    model.eval()
    # The CDF pool and generated lanes are checked in aggregate; avoid scalar
    # device reads inside every coupling/sample call during batched evaluation.
    model.validate_args = False
    try:
        result = _likelihood_metrics(
            model, reference, _points(reference, samples, seed, "test"), batch_size
        )
        result.update(
            scope="independent_points_same_condition",
            seed=seed,
            condition={
                "wavelength_nm": float(reference.condition[0]),
                "incident_cosine": float(reference.condition[1]),
            },
            dataset_fingerprint=reference.fingerprint(),
            code_fingerprint=_code_hash(),
            runtime=_runtime(model.device, str(model.dtype).removeprefix("torch.")),
            precision={
                "conditioner": str(model.dtype).removeprefix("torch."),
                "spline": str(model.spline_dtype).removeprefix("torch."),
                "hg_geometry_log_pdf": str(model.geometry_dtype).removeprefix("torch."),
                "teacher_and_statistical_reduction": "float64",
            },
            proposal=_proposal_metrics(model, reference, proposal_samples, seed, batch_size),
        )
        if uniform_samples:
            uniform_points = make_uniform_points(
                reference, uniform_samples, seed, "test", model.geometry_dtype
            )
            result.update(evaluate_log_shape(model, reference, uniform_points, batch_size))
            result["uniform_test_points_sha256"] = _array_hash(*uniform_points)
        return result
    finally:
        model.train(previous_mode)
        model.validate_args = previous_validation


def read_single_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("checkpoint_version") not in READABLE_CHECKPOINT_VERSIONS
        or (payload.get("family") != FAMILY)
    ):
        raise ValueError("not a supported version-2/3/4/5 Rainbow single-condition checkpoint")
    required = {
        "kind",
        "model_config",
        "model_state",
        "physics",
        "dtype",
        "global_step",
        "dataset_fingerprint",
        "code_fingerprint",
    }
    if not required <= payload.keys() or payload["kind"] not in ("training", "inference"):
        raise ValueError("malformed Rainbow single-condition checkpoint")
    if payload["checkpoint_version"] == LOG_CHECKPOINT_VERSION:
        LogObjectiveConfig.from_dict(payload.get("objective_config"))
        if payload.get("selection_metric") not in ("nll", "log_rmse"):
            raise ValueError("version-5 checkpoint requires an explicit selection metric")
    return payload


def load_single_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> tuple[SingleConditionSphereFlow, dict[str, Any]]:
    payload = read_single_checkpoint(path)
    device = _resolve_device(device)
    physics = payload["physics"]
    model = SingleConditionSphereFlow(
        physics["hg_g"],
        physics["incident_cosine"],
        SphereFlowConfig.from_dict(payload["model_config"]),
        dtype=_dtype(payload["dtype"]),
        device=device,
    )
    model.load_state_dict(payload["model_state"], strict=True)
    if model.hg_g != physics["hg_g"] or model.incident_cosine != physics["incident_cosine"]:
        raise ValueError("checkpoint model state and physics metadata disagree")
    model.eval()
    return model, payload


@dataclass
class SingleTrainResult:
    model: SingleConditionSphereFlow
    checkpoint_path: Path
    best_path: Path
    metrics: dict[str, Any]
    global_step: int
    complete: bool


def _validate_selections(
    selections: dict[str, Any], history: list[dict[str, Any]], step: int,
    objective: LogObjectiveConfig, best_step: int, best_validation: dict[str, Any],
    best_state: dict[str, Any],
) -> None:
    """Reject inconsistent dual-selection metadata before a resumed run writes."""
    from .training_scatter import _state_equal

    events = [entry for entry in history if "validation" in entry]
    if not events or set(selections) != {"nll", "log_rmse"}:
        raise ValueError("invalid version-5 selection history")
    for metric in ("nll", "log_rmse"):
        if any(
            type(entry.get("global_step")) is not int
            or not 0 <= entry["global_step"] <= step
            or not isinstance(entry["validation"].get(metric), (float, int))
            or not math.isfinite(entry["validation"][metric])
            for entry in events
        ):
            raise ValueError("invalid validation event in version-5 selection history")
        chosen = min(events, key=lambda entry: entry["validation"][metric])
        saved = selections[metric]
        if not isinstance(saved, dict) or set(saved) != {"step", "validation", "model_state"}:
            raise ValueError("malformed version-5 selection state")
        if (saved["step"] != chosen["global_step"]
                or saved["validation"] != chosen["validation"]):
            raise ValueError("saved selection does not match first minimum validation history")
    primary = selections[objective.selection_metric]
    if (best_step != primary["step"] or best_validation != primary["validation"]
            or not _state_equal(best_state, primary["model_state"])):
        raise ValueError("primary best checkpoint differs from its explicit selection state")


def train_single_condition(
    reference: RainbowReference,
    model_config: SphereFlowConfig,
    training_config: SingleTrainingConfig,
    output_directory: str | Path,
    *,
    resume: str | Path | None = None,
    max_steps_this_run: int | None = None,
    callback: Callable[[dict[str, Any]], None] | None = None,
    make_plots: bool = True,
    plot_config: RainbowPlotConfig | None = None,
    objective_config: LogObjectiveConfig | None = None,
) -> SingleTrainResult:
    """Train a fixed pool with an explicitly versioned density objective.

    ``checkpoint.pt`` resumes optimization; ``best.pt`` is inference-only.
    Without ``objective_config``, the legacy NLL/version-4 contract is unchanged.
    The optional version-5 workflow supervises log density under solid angle,
    retains independent target NLL/KL validation, and saves both selections.
    The final independent test set is evaluated only when the planned steps are
    complete. A controlled interruption does not add checkpoint-selection events.
    """
    if max_steps_this_run is not None and (
        type(max_steps_this_run) is not int or max_steps_this_run < 0
    ):
        raise ValueError("max_steps_this_run must be a nonnegative integer")
    if type(make_plots) is not bool:
        raise ValueError("make_plots must be boolean")
    if objective_config is not None and not isinstance(objective_config, LogObjectiveConfig):
        raise ValueError("objective_config must be a LogObjectiveConfig or None")
    objective = objective_config
    checkpoint_version = CHECKPOINT_VERSION if objective is None else LOG_CHECKPOINT_VERSION
    from .plotting import RainbowPlotConfig

    plotting = plot_config if plot_config is not None else RainbowPlotConfig()
    if not isinstance(plotting, RainbowPlotConfig):
        raise ValueError("plot_config must be a RainbowPlotConfig")
    cfg = training_config
    if objective is not None:
        objective.validate_training_count(cfg.train_samples)
    if cfg.tensorboard:
        check_tensorboard()  # Fail before expensive data preparation or optimization.
    output = Path(output_directory)
    previous = read_single_checkpoint(resume) if resume is not None else None
    if output.exists() and any(output.iterdir()) and previous is None:
        raise FileExistsError("output directory is not empty; resume or choose a new directory")
    if (
        previous is not None
        and output.exists()
        and any(output.iterdir())
        and (Path(resume).resolve().parent != output.resolve())
    ):
        raise FileExistsError("a different resume destination must be empty")
    fingerprint, code_fingerprint = reference.fingerprint(), _code_hash()
    physics = {
        "hg_g": reference.g,
        "incident_cosine": float(reference.condition[1]),
        "wavelength_nm": float(reference.condition[0]),
    }
    if previous is not None:
        if previous["kind"] != "training":
            raise ValueError("best.pt is inference-only; resume from checkpoint.pt")
        if previous["checkpoint_version"] != checkpoint_version:
            raise ValueError(
                f"exact resume requires a version-{checkpoint_version} checkpoint from this "
                "objective and implementation; older checkpoints remain readable for evaluation"
            )
        required = {
            "training_config",
            "optimizer_state",
            "rng_state",
            "minibatch_rng_state",
            "runtime",
            "sample_split",
            "history",
            "best_state",
            "best_step",
            "best_validation",
            "latest_validation",
        }
        if not required <= previous.keys():
            raise ValueError("training checkpoint is missing resume state")
        if objective is not None and (
            previous.get("objective_config") != objective.to_dict()
            or previous.get("selection_metric") != objective.selection_metric
            or not isinstance(previous.get("selections"), dict)
        ):
            raise ValueError("exact resume requires the original objective and selection state")
        if previous["model_config"] != model_config.to_dict() or (
            previous["training_config"] != cfg.to_dict()
        ):
            raise ValueError("exact resume requires the original model and training configuration")
        if previous["dataset_fingerprint"] != fingerprint or previous["physics"] != physics:
            raise ValueError("resume Rainbow record differs from checkpoint")
        if previous["code_fingerprint"] != code_fingerprint:
            raise ValueError("resume implementation differs from checkpoint")
        if (
            type(previous["global_step"]) is not int
            or not 0 <= previous["global_step"] <= cfg.steps
        ):
            raise ValueError("invalid checkpoint step")
    # Reuse the legacy workflow's deterministic runtime primitives, not its model
    # or HG warmup. Both configuration types provide these runtime-only fields.
    if torch.device(cfg.device).type == "cuda" and cfg.deterministic:
        workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if workspace not in (":4096:8", ":16:8"):
            raise ValueError(
                "deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8 "
                "before starting Python"
            )
    device = _resolve_device(cfg.device)
    _setup_runtime(cfg)
    runtime = _runtime(device, cfg.dtype)
    if previous is not None and previous["runtime"] != runtime:
        raise ValueError("exact resume requires the recorded runtime/device/dtype")
    model = SingleConditionSphereFlow(
        reference.g,
        physics["incident_cosine"],
        model_config,
        dtype=_dtype(cfg.dtype),
        device=device,
        validate_args=False,
    )
    data_seed = cfg.seed if cfg.data_seed is None else cfg.data_seed
    pool = None
    uniform_validation = None
    if objective is None:
        training_points = _points(reference, cfg.train_samples, data_seed, "train")
    else:
        verify_positive_teacher(reference)
        pool = make_training_pool(
            reference, cfg.train_samples, data_seed, objective, model.geometry_dtype
        )
        training_points = (pool.directions, pool.log_p)
        uniform_validation = make_uniform_points(
            reference, objective.validation_uniform_samples, data_seed,
            "validation", model.geometry_dtype,
        )
    validation_points = _points(reference, cfg.validation_samples, data_seed, "validation")
    split = {
        "scope": "one_fixed_condition; independent point streams; no condition holdout",
        "seed": cfg.seed,
        "data_seed": data_seed,
        "rng": "numpy.PCG64 with SeedSequence([seed, stream_id])",
        "minibatch_rng": {
            "engine": "torch.Generator",
            "device": str(device),
            "seed_derivation": "first PCG64 integer in [0,2^63) from stream minibatches",
        },
        "uniforms": "52-bit open midpoint grid",
        "streams": _STREAMS,
        "train_samples": cfg.train_samples,
        "validation_samples": cfg.validation_samples,
        "test_samples": cfg.test_samples,
        "training_points_sha256": _array_hash(*training_points),
        "validation_points_sha256": _array_hash(*validation_points),
    }
    if pool is not None:
        split["training_pool"] = pool.provenance
        split["training_points_sha256"] = _array_hash(
            pool.directions, pool.log_p, pool.components
        )
        split["validation_uniform_samples"] = objective.validation_uniform_samples
        split["uniform_validation_points_sha256"] = _array_hash(*uniform_validation)
    if previous is not None and previous["sample_split"] != split:
        raise ValueError("regenerated training/validation points differ from checkpoint")
    # Retain one fixed pool on the device; the optimization loop has no NumPy
    # indexing, CDF inversion, or host-to-device point transfers.
    training_directions = torch.as_tensor(
        training_points[0], dtype=model.geometry_dtype, device=device
    )
    monitor_count = min(cfg.train_samples, cfg.train_monitor_samples)
    if objective is not None and objective.sampling == "target_uniform":
        # The pool interleaves its two source strata. Keep the prefix balanced.
        monitor_count = max(2, monitor_count - monitor_count % 2)
    training_monitor = (
        training_directions[:monitor_count],
        torch.as_tensor(training_points[1][:monitor_count], dtype=torch.float64, device=device),
    )
    validation_device = _device_points(validation_points, device, model.geometry_dtype)
    training_labels = None
    uniform_validation_device = None
    if objective is not None:
        # No FP64 loss arithmetic or per-minibatch CPU copies: cast fixed teacher
        # labels once, while the validation reductions retain reference precision.
        training_labels = torch.as_tensor(training_points[1], dtype=model.dtype, device=device)
        uniform_validation_device = _device_points(
            uniform_validation, device, model.geometry_dtype
        )
    del training_points, validation_points
    del pool, uniform_validation

    def validation_metrics() -> dict[str, Any]:
        values = _likelihood_metrics(model, reference, validation_device, cfg.eval_batch_size)
        if objective is not None:
            values.update(evaluate_log_shape(
                model, reference, uniform_validation_device, cfg.eval_batch_size
            ))
        return values

    @torch.no_grad()
    def monitor_metrics() -> dict[str, Any]:
        if objective is None:
            return _likelihood_metrics(model, reference, training_monitor, cfg.eval_batch_size)
        # This monitor uses the saved query distribution. For uniform-only
        # queries, log MSE has unit weights and the NLL diagnostic uses p/u;
        # neither query pool is interpreted as target-distributed samples.
        accumulated = torch.zeros(3, dtype=torch.float64, device=device)
        directions, labels = training_monitor
        for start in range(0, monitor_count, cfg.eval_batch_size):
            end = min(start + cfg.eval_batch_size, monitor_count)
            values = loss_terms(
                model.log_prob(directions[start:end]), labels[start:end], objective,
                report_components=objective.sampling in ("target_uniform", "uniform"),
            )
            accumulated += torch.stack([
                values["loss"], values["nll"], values["log_mse"]
            ]).to(torch.float64) * (end - start)
        values = (accumulated / monitor_count).cpu().tolist()
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError("nonfinite fixed-pool objective monitor")
        report = dict(zip(("loss", "nll", "log_mse"), values, strict=True)) | {
            "sample_count": monitor_count,
            "objective": "log_density",
            "sampling": objective.sampling,
            "components_measured": objective.sampling in ("target_uniform", "uniform"),
            "beta": objective.beta,
            "nll_weight": objective.nll_weight,
            "scope": "weighted_fixed_training_pool_subset_not_independent_validation",
        }
        if objective.sampling == "target":
            report.pop("log_mse")
        return report
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(
        parameters, lr=cfg.learning_rate, foreach=device.type == "cuda"
    )
    minibatches = _minibatch_generator(cfg.seed, device)
    history: list[dict[str, Any]] = []
    step = best_step = 0
    best_state: dict[str, Any]
    selections: dict[str, dict[str, Any]] = {}
    if previous is None:
        latest = validation_metrics()
        best_validation = copy.deepcopy(latest)
        best_state = copy.deepcopy(model.state_dict())
        history.append({
            "event": "initial", "global_step": 0, "examples_seen": 0,
            "effective_passes": 0.0, "validation": latest,
            "train_monitor": monitor_metrics(),
        })
        if objective is not None:
            selections = {
                metric: {"step": 0, "validation": copy.deepcopy(latest),
                         "model_state": copy.deepcopy(best_state)}
                for metric in ("nll", "log_rmse")
            }
    else:
        model.load_state_dict(previous["model_state"], strict=True)
        _check_model_reference(model, reference)
        optimizer.load_state_dict(previous["optimizer_state"])
        minibatches.set_state(previous["minibatch_rng_state"].cpu())
        _restore_rng(previous["rng_state"])
        step, best_step = previous["global_step"], previous["best_step"]
        history = copy.deepcopy(previous["history"])
        best_state = copy.deepcopy(previous["best_state"])
        best_validation = copy.deepcopy(previous["best_validation"])
        latest = copy.deepcopy(previous["latest_validation"])
        if objective is not None:
            selections = copy.deepcopy(previous["selections"])
            _validate_selections(selections, history, step, objective, best_step,
                                 best_validation, best_state)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path, best_path = output / "checkpoint.pt", output / "best.pt"
    atomic_json(
        output / "config.json",
        {
            "schema_version": 2 if objective is None else 3,
            "family": FAMILY,
            "model": model_config.to_dict(),
            "training": cfg.to_dict(),
            "visualization": {"enabled": make_plots, **plotting.to_dict()},
            **({"objective": objective.to_dict()} if objective is not None else {}),
        },
    )
    atomic_json(output / "data_summary.json", reference.summary())
    atomic_json(output / "sample_split.json", split)

    def common() -> dict[str, Any]:
        return {
            "checkpoint_version": checkpoint_version,
            "family": FAMILY,
            "model_config": model_config.to_dict(),
            "dtype": cfg.dtype,
            "physics": physics,
            "data_provenance": reference.provenance,
            "dataset_fingerprint": fingerprint,
            "code_fingerprint": code_fingerprint,
            "runtime": runtime,
            **({"objective_config": objective.to_dict(),
                "selection_metric": objective.selection_metric} if objective is not None else {}),
        }

    def save() -> None:
        payload = {
            **common(),
            "kind": "training",
            "model_state": model.state_dict(),
            "training_config": cfg.to_dict(),
            "optimizer_state": optimizer.state_dict(),
            "global_step": step,
            "minibatch_rng_state": minibatches.get_state(),
            "rng_state": _rng_state(),
            "sample_split": split,
            "history": history,
            "best_state": best_state,
            "best_step": best_step,
            "best_validation": best_validation,
            "latest_validation": latest,
            **({"selections": selections} if objective is not None else {}),
        }
        _atomic_torch_save(checkpoint_path, payload)
        _atomic_torch_save(
            best_path,
            {
                **common(),
                "kind": "inference",
                "global_step": best_step,
                "model_state": best_state,
                "validation": best_validation,
            },
        )
        if objective is not None:
            for metric, selection in selections.items():
                suffix = "log" if metric == "log_rmse" else "nll"
                _atomic_torch_save(output / f"best_by_{suffix}.pt", {
                    **common(), "kind": "inference", "selection_metric": metric,
                    "global_step": selection["step"],
                    "model_state": selection["model_state"],
                    "validation": selection["validation"],
                })
        atomic_json(output / "history.json", history)

    pending: list[tuple[int, torch.Tensor]] = []

    def flush_updates() -> None:
        """Check before publication; synchronize a block, not every parameter.

        Each update retains only its detached loss and gradient norm. A failed
        block is never written over the last valid atomic checkpoint.
        """
        if not pending:
            return
        summaries = torch.stack([values for _, values in pending]).cpu().numpy()
        invalid = ~np.isfinite(summaries).all(axis=-1)
        if invalid.any():
            failed_step = pending[int(np.flatnonzero(invalid)[0])][0]
            raise FloatingPointError(
                f"nonfinite training loss/gradient at update {failed_step}; "
                "the last valid checkpoint was retained"
            )
        finite_parameters = torch.stack([torch.isfinite(p).all() for p in parameters]).all()
        if not bool(finite_parameters):
            raise FloatingPointError(
                f"optimizer produced nonfinite parameters by update {step}; "
                "the last valid checkpoint was retained"
            )
        for (update, _), values in zip(pending, summaries, strict=True):
            entry = {
                "global_step": update,
                "loss": float(values[0]),
                "gradient_norm_before_clip": float(values[1]),
                "learning_rate": cfg.learning_rate,
                "examples_seen": update * cfg.batch_size,
                "effective_passes": update * cfg.batch_size / cfg.train_samples,
            }
            if objective is not None:
                entry.update(nll=float(values[2]),
                             beta=objective.beta, nll_weight=objective.nll_weight,
                             objective="log_density", sampling=objective.sampling,
                             components_measured=objective.sampling in ("target_uniform", "uniform"))
                if objective.sampling in ("target_uniform", "uniform"):
                    entry["log_mse"] = float(values[3])
            history.append(entry)
        pending.clear()

    if previous is None or checkpoint_path.resolve() != Path(resume).resolve():
        save()
    if previous is None and callback is not None:
        callback(history[0])
    stop = cfg.steps if max_steps_this_run is None else min(cfg.steps, step + max_steps_this_run)
    model.train()
    with TrainingMonitor(
        output, history, tensorboard=cfg.tensorboard, start_step=step,
        batch_size=cfg.batch_size,
        metadata={
            "model": model_config.to_dict(), "training": cfg.to_dict(),
            "physics": physics, "dataset_fingerprint": fingerprint,
            "train_monitor_subset": "first min(train_monitor_samples, train_samples) fixed points",
            **({"objective": objective.to_dict()} if objective is not None else {}),
        },
    ) as monitor:
        published = len(history)
        while step < stop:
            indices = torch.randint(
                cfg.train_samples, (cfg.batch_size,), generator=minibatches, device=device
            )
            outgoing = training_directions[indices]
            optimizer.zero_grad(set_to_none=True)
            log_q = model.log_prob(outgoing)
            if log_q.shape != (cfg.batch_size,):
                raise FloatingPointError("malformed training log_prob")
            if objective is None:
                loss = -log_q.mean()  # Target-distributed samples have unit loss weight.
            else:
                terms = loss_terms(
                    log_q, training_labels[indices], objective,
                    report_components=objective.sampling in ("target_uniform", "uniform"),
                )
                loss = terms["loss"]
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                cfg.grad_clip_norm if cfg.grad_clip_norm is not None else math.inf,
                error_if_nonfinite=False,
                foreach=device.type == "cuda",
            )
            optimizer.step()
            step += 1
            scalars = [loss.detach(), gradient_norm.detach().to(loss.dtype)]
            if objective is not None:
                scalars.extend([terms["nll"].detach(), terms["log_mse"].detach()])
            pending.append((step, torch.stack(scalars)))
            evaluation_due = step % cfg.eval_every == 0 or step == cfg.steps
            checkpoint_due = step % cfg.checkpoint_every == 0 or step == cfg.steps
            logging_due = step % cfg.log_every == 0
            boundary = evaluation_due or checkpoint_due or logging_due or step == stop
            if boundary:
                flush_updates()
            if evaluation_due:
                entry = history[-1]
                latest = validation_metrics()
                entry["validation"] = latest
                entry["train_monitor"] = monitor_metrics()
                if objective is not None:
                    for metric in selections:
                        if latest[metric] < selections[metric]["validation"][metric]:
                            selections[metric] = {
                                "step": step, "validation": copy.deepcopy(latest),
                                "model_state": copy.deepcopy(model.state_dict()),
                            }
                    primary = selections[objective.selection_metric]
                    best_step, best_validation = primary["step"], primary["validation"]
                    best_state = primary["model_state"]
                elif latest["nll"] < best_validation["nll"]:
                    best_step, best_validation = step, copy.deepcopy(latest)
                    best_state = copy.deepcopy(model.state_dict())
            if boundary:
                monitor.publish(history[published:])
                published = len(history)
                monitor.telemetry(step)
                if callback is not None:
                    callback(history[-1])
            if checkpoint_due:
                save()
        # An interrupt inside an update leaves the preceding atomic checkpoint.
        save()
    if make_plots or cfg.tensorboard:
        plot_training_history(output / "history.json", output / "learning_curves.png")
    selected = SingleConditionSphereFlow(
        reference.g,
        physics["incident_cosine"],
        model_config,
        dtype=_dtype(cfg.dtype),
        device=device,
    )
    selected.load_state_dict(best_state, strict=True)
    selected.eval()
    final_test = None
    if step == cfg.steps:
        final_test = evaluate_single_condition(
            selected,
            reference,
            samples=cfg.test_samples,
            seed=data_seed,
            batch_size=cfg.eval_batch_size,
            proposal_samples=cfg.proposal_samples,
            uniform_samples=0 if objective is None else objective.test_uniform_samples,
        )
        final_test["scope"] = "final_independent_test_points_same_condition"
    metrics = {
        "family": FAMILY,
        "complete": step == cfg.steps,
        "global_step": step,
        "selected_step": best_step,
        "selection": "minimum validation NLL at scheduled evaluations",
        "initial_validation": history[0]["validation"],
        "best_validation": best_validation,
        "test": final_test,
        "condition": physics,
        "dataset_fingerprint": fingerprint,
        "generalization_scope": "same condition only; no claim about unseen wavelength/incidence",
    }
    if objective is not None:
        metrics.update(
            objective=objective.to_dict(), selection_metric=objective.selection_metric,
            selection=f"first minimum validation {objective.selection_metric} at scheduled evaluations",
            selections={metric: {"step": value["step"], "validation": value["validation"]}
                        for metric, value in selections.items()},
        )
    atomic_json(output / "metrics.json", metrics)
    if make_plots and step == cfg.steps:
        from .plotting import plot_rainbow_comparison

        # The selected model, not the last optimization iterate, supplies the
        # inference map. Plotting has its own RNG and never selects checkpoints.
        plot_rainbow_comparison(
            selected,
            reference,
            output / "plots",
            config=plotting,
            checkpoint_path=best_path,
            title=f"Validation-selected model (step {best_step})",
            selected_step=best_step,
        )
    return SingleTrainResult(selected, checkpoint_path, best_path, metrics, step, step == cfg.steps)
