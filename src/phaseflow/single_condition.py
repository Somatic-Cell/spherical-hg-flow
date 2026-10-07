"""One-record Rainbow training with a fixed, externally supplied HG moment.

This is a distinct versioned workflow from the legacy conditional folded-chart
model. Training points, validation points, final test points and proposal samples
have separate reproducible random streams. All likelihoods are per steradian.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import zuko

from . import __version__
from .hg import hg_log_prob
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

FAMILY = "rainbow_single_condition"
CHECKPOINT_VERSION = 2
_STREAMS = {"train": 0, "validation": 1, "minibatches": 2, "test": 3, "proposal": 4}


@dataclass
class SingleTrainingConfig:
    seed: int = 415
    device: str = "cpu"
    dtype: str = "float64"
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

    def __post_init__(self) -> None:
        positive = (
            "train_samples",
            "batch_size",
            "eval_every",
            "checkpoint_every",
            "eval_batch_size",
            "cpu_threads",
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
    points: tuple[np.ndarray, np.ndarray],
    batch_size: int,
) -> dict[str, Any]:
    _check_model_reference(model, reference)
    directions, log_p = points
    device = next(model.parameters()).device
    log_q, log_hg = [], []
    for start in range(0, len(directions), batch_size):
        x = torch.as_tensor(
            directions[start : start + batch_size], dtype=torch.float64, device=device
        )
        q = model.log_prob(x)
        if q.shape != (len(x),) or not torch.isfinite(q).all():
            raise FloatingPointError("model log_prob is nonfinite or has the wrong shape")
        log_q.append(q.cpu().double().numpy())
        mu = x[:, 2] / torch.linalg.vector_norm(x, dim=-1)
        log_hg.append(hg_log_prob(mu, model.hg_g, validate_args=True).cpu().numpy())
    q, h = np.concatenate(log_q), np.concatenate(log_hg)
    nll, nll_se = _estimate(-q)
    kl, kl_se = _estimate(log_p - q)
    hg_nll, _ = _estimate(-h)
    hg_kl, hg_kl_se = _estimate(log_p - h)
    improvement, improvement_se = _estimate(q - h)
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
        "target_entropy_estimate": float(-log_p.mean()),
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
    device = next(model.parameters()).device
    generator = _generator(seed, "proposal")
    log_weights, cosines = [], []
    max_discrepancy = 0.0
    for start in range(0, count, batch_size):
        uniforms = torch.as_tensor(
            _uniforms(
                generator,
                min(batch_size, count - start),
                bits=23 if model.dtype == torch.float32 else 52,
            ),
            dtype=torch.float64,
            device=device,
        )
        x, log_q = model.sample_from_uniform(uniforms)
        evaluated = model.log_prob(x)
        if (
            not torch.isfinite(x).all()
            or not torch.isfinite(log_q).all()
            or (not torch.isfinite(evaluated).all())
        ):
            raise FloatingPointError("nonfinite NF sample or PDF; samples are never discarded")
        max_discrepancy = max(max_discrepancy, float((log_q - evaluated).abs().max().item()))
        x_np = x.cpu().numpy()
        p = reference.log_prob(x_np)
        if np.isnan(p).any() or np.isposinf(p).any():
            raise FloatingPointError("invalid reference PDF at an NF sample")
        log_weights.append(p - log_q.cpu().double().numpy())
        cosines.append(x_np[:, 2] / np.linalg.norm(x_np, axis=-1))
    lw, mu = np.concatenate(log_weights), np.concatenate(cosines)
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
        "uniform_midpoint_bits": 23 if model.dtype == torch.float32 else 52,
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
    _check_model_reference(model, reference)
    previous_mode = model.training
    model.eval()
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
            proposal=_proposal_metrics(model, reference, proposal_samples, seed, batch_size),
        )
        return result
    finally:
        model.train(previous_mode)


def read_single_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("checkpoint_version") != CHECKPOINT_VERSION
        or (payload.get("family") != FAMILY)
    ):
        raise ValueError("not a version-2 Rainbow single-condition checkpoint")
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
    return payload


def load_single_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[SingleConditionSphereFlow, dict[str, Any]]:
    payload = read_single_checkpoint(path)
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


def train_single_condition(
    reference: RainbowReference,
    model_config: SphereFlowConfig,
    training_config: SingleTrainingConfig,
    output_directory: str | Path,
    *,
    resume: str | Path | None = None,
    max_steps_this_run: int | None = None,
    callback: Callable[[dict[str, Any]], None] | None = None,
) -> SingleTrainResult:
    """Train a fixed point pool; choose the best checkpoint only by validation NLL.

    ``checkpoint.pt`` resumes optimization; ``best.pt`` is inference-only.
    The final independent test set is evaluated only when the planned steps are
    complete. A controlled interruption does not add checkpoint-selection events.
    """
    if max_steps_this_run is not None and (
        type(max_steps_this_run) is not int or max_steps_this_run < 0
    ):
        raise ValueError("max_steps_this_run must be a nonnegative integer")
    cfg = training_config
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
    device = _setup_runtime(cfg)
    runtime = _runtime(device, cfg.dtype)
    if previous is not None and previous["runtime"] != runtime:
        raise ValueError("exact resume requires the recorded runtime/device/dtype")
    model = SingleConditionSphereFlow(
        reference.g,
        physics["incident_cosine"],
        model_config,
        dtype=_dtype(cfg.dtype),
        device=device,
    )
    training_points = _points(reference, cfg.train_samples, cfg.seed, "train")
    validation_points = _points(reference, cfg.validation_samples, cfg.seed, "validation")
    split = {
        "scope": "one_fixed_condition; independent point streams; no condition holdout",
        "seed": cfg.seed,
        "rng": "numpy.PCG64 with SeedSequence([seed, stream_id])",
        "uniforms": "52-bit open midpoint grid",
        "streams": _STREAMS,
        "train_samples": cfg.train_samples,
        "validation_samples": cfg.validation_samples,
        "test_samples": cfg.test_samples,
        "training_points_sha256": _array_hash(*training_points),
        "validation_points_sha256": _array_hash(*validation_points),
    }
    if previous is not None and previous["sample_split"] != split:
        raise ValueError("regenerated training/validation points differ from checkpoint")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    minibatches = _generator(cfg.seed, "minibatches")
    history: list[dict[str, Any]] = []
    step = best_step = 0
    best_state: dict[str, Any]
    if previous is None:
        latest = _likelihood_metrics(model, reference, validation_points, cfg.eval_batch_size)
        best_validation = copy.deepcopy(latest)
        best_state = copy.deepcopy(model.state_dict())
        history.append({"event": "initial", "global_step": 0, "validation": latest})
    else:
        model.load_state_dict(previous["model_state"], strict=True)
        _check_model_reference(model, reference)
        optimizer.load_state_dict(previous["optimizer_state"])
        minibatches.bit_generator.state = previous["minibatch_rng_state"]
        _restore_rng(previous["rng_state"])
        step, best_step = previous["global_step"], previous["best_step"]
        history = copy.deepcopy(previous["history"])
        best_state = copy.deepcopy(previous["best_state"])
        best_validation = copy.deepcopy(previous["best_validation"])
        latest = copy.deepcopy(previous["latest_validation"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path, best_path = output / "checkpoint.pt", output / "best.pt"
    atomic_json(
        output / "config.json",
        {
            "schema_version": 2,
            "family": FAMILY,
            "model": model_config.to_dict(),
            "training": cfg.to_dict(),
        },
    )
    atomic_json(output / "data_summary.json", reference.summary())
    atomic_json(output / "sample_split.json", split)

    def common() -> dict[str, Any]:
        return {
            "checkpoint_version": CHECKPOINT_VERSION,
            "family": FAMILY,
            "model_config": model_config.to_dict(),
            "dtype": cfg.dtype,
            "physics": physics,
            "data_provenance": reference.provenance,
            "dataset_fingerprint": fingerprint,
            "code_fingerprint": code_fingerprint,
            "runtime": runtime,
        }

    def save() -> None:
        payload = {
            **common(),
            "kind": "training",
            "model_state": model.state_dict(),
            "training_config": cfg.to_dict(),
            "optimizer_state": optimizer.state_dict(),
            "global_step": step,
            "minibatch_rng_state": minibatches.bit_generator.state,
            "rng_state": _rng_state(),
            "sample_split": split,
            "history": history,
            "best_state": best_state,
            "best_step": best_step,
            "best_validation": best_validation,
            "latest_validation": latest,
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
        atomic_json(output / "history.json", history)

    save()
    if previous is None and callback is not None:
        callback(history[0])
    stop = cfg.steps if max_steps_this_run is None else min(cfg.steps, step + max_steps_this_run)
    model.train()
    while step < stop:
        indices = minibatches.integers(0, cfg.train_samples, size=cfg.batch_size)
        outgoing = torch.as_tensor(training_points[0][indices], dtype=torch.float64, device=device)
        optimizer.zero_grad(set_to_none=True)
        log_q = model.log_prob(outgoing)
        if log_q.shape != (cfg.batch_size,) or not torch.isfinite(log_q).all():
            raise FloatingPointError("nonfinite/malformed training log_prob")
        loss = -log_q.mean()  # Target-distributed samples have unit loss weight.
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("nonfinite training gradient")
        if cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip_norm, error_if_nonfinite=True
            )
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError("optimizer produced nonfinite parameters")
        step += 1
        entry = {"global_step": step, "loss": float(loss.detach().item())}
        if step % cfg.eval_every == 0 or step == cfg.steps:
            latest = _likelihood_metrics(model, reference, validation_points, cfg.eval_batch_size)
            entry["validation"] = latest
            if latest["nll"] < best_validation["nll"]:
                best_step, best_validation = step, copy.deepcopy(latest)
                best_state = copy.deepcopy(model.state_dict())
            if callback is not None:
                callback(entry)
        history.append(entry)
        if step % cfg.checkpoint_every == 0 or step == cfg.steps:
            save()
    # Save only completed update boundaries. KeyboardInterrupt within an update
    # propagates, leaving the preceding atomic checkpoint intact.
    save()
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
            seed=cfg.seed,
            batch_size=cfg.eval_batch_size,
            proposal_samples=cfg.proposal_samples,
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
    atomic_json(output / "metrics.json", metrics)
    return SingleTrainResult(selected, checkpoint_path, best_path, metrics, step, step == cfg.steps)
