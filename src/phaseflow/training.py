"""Reproducible conditional training with an explicit HG warmup/freeze schedule.

The objective balances condition groups equally, rather than weighting groups
by the number of points or the unnormalized total scattering strength. Every
reported likelihood is a density with respect to solid angle, in sr^-1.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .data import PhasePointCloud, PointSampler, iter_group_chunks, split_conditions
from .hg import hg_log_prob
from .model import ModelConfig, PhaseFlow

CHECKPOINT_VERSION = 1


@dataclass
class TrainingConfig:
    seed: int = 415
    device: str = "cpu"
    dtype: str = "float32"
    batch_size: int = 1024
    warmup_steps: int = 300
    residual_steps: int = 2000
    joint_steps: int = 0
    lr_g: float = 1e-3
    lr_flow: float = 1e-3
    lr_joint: float = 1e-4
    moment_regularization: float = 1.0
    validation_fraction: float = 0.2
    point_sampling: str = "mass"
    eval_every: int = 100
    checkpoint_every: int = 100
    eval_batch_size: int = 8192
    grad_clip_norm: float | None = 10.0
    deterministic: bool = True
    cpu_threads: int = 1

    def __post_init__(self) -> None:
        integer_fields = (
            "seed",
            "batch_size",
            "warmup_steps",
            "residual_steps",
            "joint_steps",
            "eval_every",
            "checkpoint_every",
            "eval_batch_size",
            "cpu_threads",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        for name in ("batch_size", "eval_batch_size", "cpu_threads"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in (
            "warmup_steps",
            "residual_steps",
            "joint_steps",
            "eval_every",
            "checkpoint_every",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        for name in ("lr_g", "lr_flow", "lr_joint"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.moment_regularization) or self.moment_regularization < 0:
            raise ValueError("moment_regularization must be finite and nonnegative")
        if self.joint_steps and self.moment_regularization <= 0:
            raise ValueError(
                "joint training requires moment_regularization > 0 to anchor the otherwise non-identifiable HG base"
            )
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must lie in [0, 1)")
        if self.point_sampling not in ("mass", "uniform"):
            raise ValueError("point_sampling must be mass or uniform")
        if self.grad_clip_norm is not None and (
            not math.isfinite(self.grad_clip_norm) or self.grad_clip_norm <= 0
        ):
            raise ValueError("grad_clip_norm must be positive or null")
        if not isinstance(self.deterministic, bool):
            raise ValueError("deterministic must be a boolean")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainingConfig":
        if not isinstance(values, dict):
            raise ValueError("training config must be a JSON object")
        unknown = set(values).difference(field.name for field in fields(cls))
        if unknown:
            raise ValueError(f"unknown training config fields: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainResult:
    model: PhaseFlow
    history: list[dict[str, Any]]
    split: dict[str, list[int]]
    checkpoint_path: Path
    metrics: dict[str, Any]
    completed: dict[str, int]


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = handle.name
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _setup_runtime(config: TrainingConfig) -> torch.device:
    device = torch.device(config.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("training supports cpu and cuda devices")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; CPU fallback is not automatic")
    torch.set_num_threads(config.cpu_threads)
    if config.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(config.deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = config.deterministic
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(config.seed)
    np.random.seed(config.seed % (2**32))
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    return device


def _rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [
            numpy_state[0],
            numpy_state[1].tolist(),
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        ],
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def _runtime_metadata(device: torch.device, dtype: str) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "device": str(device),
        "dtype": dtype,
    }


def read_checkpoint(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Read only tensor/primitive payloads; no arbitrary pickled model objects."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if isinstance(checkpoint, dict) and checkpoint.get("family") == "rainbow_single_condition":
        raise ValueError(
            "This is a Rainbow single-condition checkpoint. Use evaluate-rainbow for evaluation. "
            "Native/OptiX export of the new circular model is not implemented; export v1 only "
            "supports the legacy folded model."
        )
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION
    ):
        raise ValueError("unsupported or malformed checkpoint")
    required = {
        "model_config",
        "training_config",
        "model_state",
        "completed",
        "split",
        "dataset_fingerprint",
        "sampler_state",
        "rng_state",
    }
    if not required.issubset(checkpoint):
        raise ValueError(
            f"checkpoint is missing required fields: {sorted(required.difference(checkpoint))}"
        )
    return checkpoint


def load_model_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[PhaseFlow, dict[str, Any]]:
    checkpoint = read_checkpoint(path, map_location=device)
    model = PhaseFlow(ModelConfig.from_dict(checkpoint["model_config"]))
    model = model.to(device=device, dtype=_dtype(checkpoint["training_config"]["dtype"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint


def _validate_model_domain(cloud: PhasePointCloud, config: ModelConfig) -> None:
    wavelength = cloud.conditions[:, 0]
    if np.any(wavelength < config.wavelength_min_nm) or np.any(
        wavelength > config.wavelength_max_nm
    ):
        raise ValueError(
            "dataset contains wavelengths outside the configured model domain; extrapolation is not implicit"
        )


@torch.no_grad()
def evaluate_model(
    model: PhaseFlow,
    cloud: PhasePointCloud,
    *,
    groups: list[int] | None = None,
    batch_size: int = 8192,
    sample_count: int = 0,
    seed: int = 2025,
    scope: str = "specified_conditions",
) -> dict[str, Any]:
    """Condition-balanced cross entropy and base-moment diagnostics.

    The point cloud alone does not specify the reference entropy, so this routine
    deliberately reports NLL, not a fabricated KL divergence. A quadrature
    cloud is evaluated by the supplied masses; target samples give an empirical
    estimate. ``sample_count`` additionally checks the learned final moment and
    sample/log_prob agreement, without changing training RNG state.
    """
    _validate_model_domain(cloud, model.config)
    if groups is None:
        groups = list(range(cloud.num_conditions))
    if (
        not groups
        or len(set(groups)) != len(groups)
        or min(groups) < 0
        or max(groups) >= cloud.num_conditions
    ):
        raise ValueError("evaluation groups must be nonempty, distinct, and valid")
    if sample_count < 0:
        raise ValueError("sample_count must be nonnegative")
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    was_training = model.training
    model.eval()
    targets = cloud.moment_targets()
    reports = []
    generator = torch.Generator(device=device).manual_seed(seed)
    try:
        for ci in groups:
            condition = torch.tensor(cloud.conditions[ci].copy(), dtype=dtype, device=device)
            g_tensor = model.hg_g(condition).to(torch.float64)
            predicted_g = float(g_tensor.item())
            nll, base_nll = 0.0, 0.0
            for directions, mass in iter_group_chunks(cloud, ci, batch_size):
                outgoing = torch.tensor(directions, dtype=torch.float64, device=device)
                context = condition.expand(len(outgoing), -1)
                log_q = model.log_prob(outgoing, context)
                if log_q.shape != (len(outgoing),) or not torch.isfinite(log_q).all():
                    raise FloatingPointError(f"non-finite or malformed log_prob in condition {ci}")
                nll -= float(np.dot(mass, log_q.detach().cpu().double().numpy()))
                # Match the model's float64 normalization of rounded unit
                # vectors, and share its stable HG kernel near |g|=1.
                length = torch.linalg.vector_norm(outgoing, dim=-1)
                mu = (outgoing[:, 2] / length).clamp(-1.0, 1.0)
                log_hg = hg_log_prob(mu, g_tensor)
                if not torch.isfinite(log_hg).all():
                    raise FloatingPointError(f"non-finite HG baseline in condition {ci}")
                base_nll -= float(np.dot(mass, log_hg.detach().cpu().numpy()))
            report: dict[str, Any] = {
                "condition_index": ci,
                "wavelength_nm": float(cloud.conditions[ci, 0]),
                "incident_cosine": float(cloud.conditions[ci, 1]),
                "nll": nll,
                "fitted_hg_nll": base_nll,
                "nll_improvement_over_fitted_hg": base_nll - nll,
                "target_moment_g": float(targets[ci]),
                "base_g": predicted_g,
                "base_g_error": predicted_g - float(targets[ci]),
            }
            if sample_count:
                sampled, sampled_log_q = model.sample_and_log_prob(
                    condition.unsqueeze(0), num_samples=sample_count, generator=generator
                )
                sampled = sampled.reshape(sample_count, 3)
                sampled_log_q = sampled_log_q.reshape(sample_count)
                evaluated = model.log_prob(sampled, condition.expand(sample_count, -1))
                if (
                    not torch.isfinite(sampled).all()
                    or not torch.isfinite(sampled_log_q).all()
                    or not torch.isfinite(evaluated).all()
                ):
                    raise FloatingPointError(f"non-finite sample or PDF in condition {ci}")
                cosine = sampled[:, 2].double()
                report.update(
                    {
                        "sample_count": sample_count,
                        "learned_moment_estimate": float(cosine.mean().item()),
                        "learned_moment_standard_error": float(
                            cosine.std(unbiased=True).item() / math.sqrt(sample_count)
                        )
                        if sample_count > 1
                        else None,
                        "sample_eval_log_pdf_max_abs_error": float(
                            (sampled_log_q - evaluated).abs().max().item()
                        ),
                    }
                )
            reports.append(report)
    finally:
        model.train(was_training)
    return {
        "scope": scope,
        "reference_kind": "empirical_target_samples"
        if cloud.mode == "target_samples"
        else "weighted_quadrature",
        "condition_weighting": "equal_per_condition",
        "density_measure": "solid_angle_sr",
        "num_conditions": len(groups),
        "mean_nll": float(np.mean([report["nll"] for report in reports])),
        "mean_fitted_hg_nll": float(np.mean([report["fitted_hg_nll"] for report in reports])),
        "base_g_mse": float(np.mean([report["base_g_error"] ** 2 for report in reports])),
        "conditions": reports,
        "data_source": cloud.metadata,
    }


def train_model(
    cloud: PhasePointCloud,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    output_dir: str | Path,
    *,
    resume: str | Path | None = None,
    max_steps_this_run: int | None = None,
    callback: Callable[[dict[str, Any]], None] | None = None,
) -> TrainResult:
    """Train, checkpoint, and optionally resume exactly in the same runtime.

    Stage 1 fits HG g to per-condition mean-cosine targets by squared error.
    Stage 2 freezes all HG parameters and minimizes the full spherical NLL.
    Optional stage 3 jointly minimizes that same full NLL plus a g-moment MSE;
    it never drops the HG log-density or its dependence through the flow.

    ``max_steps_this_run`` is an explicit interruption budget, useful for a
    controlled checkpoint/resume workflow. It does not alter the stored plan.
    """
    if max_steps_this_run is not None and (
        isinstance(max_steps_this_run, bool)
        or not isinstance(max_steps_this_run, int)
        or max_steps_this_run < 0
    ):
        raise ValueError("max_steps_this_run must be a nonnegative integer")
    _validate_model_domain(cloud, model_config)
    dtype = _dtype(training_config.dtype)
    represented_g_limit = float(torch.tensor(model_config.g_limit, dtype=dtype).item())
    if not 0.0 < represented_g_limit < 1.0:
        raise ValueError(
            f"g_limit={model_config.g_limit!r} is not strictly between 0 and 1 in "
            f"the requested {training_config.dtype}; it rounds to {represented_g_limit!r}. "
            "Choose a representable limit or use float64; no limit clamp is applied."
        )
    split = split_conditions(
        cloud.num_conditions, training_config.validation_fraction, training_config.seed + 1
    )
    moment_targets = cloud.moment_targets()
    if training_config.warmup_steps or training_config.joint_steps:
        training_ids = np.asarray(split["train"], dtype=np.int64)
        invalid_ids = training_ids[np.abs(moment_targets[training_ids]) >= model_config.g_limit]
        if len(invalid_ids):
            raise ValueError(
                "HG moment supervision requires abs(target_g) < "
                f"g_limit={model_config.g_limit!r} for every TRAIN condition; "
                f"violating condition indices={invalid_ids.tolist()}. "
                "Moment targets are not clamped."
            )
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "checkpoint.pt"
    if checkpoint_path.exists() and resume is None:
        raise FileExistsError(
            f"{checkpoint_path} already exists; provide resume or choose a new output directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _setup_runtime(training_config)
    model = PhaseFlow(model_config).to(device=device, dtype=dtype)
    fingerprint = cloud.fingerprint()
    sampler = PointSampler(
        cloud,
        split["train"],
        seed=training_config.seed + 2,
        point_sampling=training_config.point_sampling,
    )  # type: ignore[arg-type]
    completed = {"warmup": 0, "residual": 0, "joint": 0}
    history: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    if resume is not None:
        previous = read_checkpoint(resume, map_location=device)
        if (
            previous["model_config"] != model_config.to_dict()
            or previous["training_config"] != training_config.to_dict()
        ):
            raise ValueError(
                "resume configuration differs from checkpoint; an exact resume requires the original plan"
            )
        if previous["dataset_fingerprint"] != fingerprint:
            raise ValueError("resume dataset content/order differs from checkpoint")
        if previous["split"] != split:
            raise ValueError("resume condition split differs from checkpoint")
        if previous.get("runtime") != _runtime_metadata(device, training_config.dtype):
            raise ValueError(
                "resume runtime differs from checkpoint; exact continuation requires the recorded Python, PyTorch, NumPy, device, and dtype"
            )
        model.load_state_dict(previous["model_state"], strict=True)
        completed = previous["completed"].copy()
        history = previous.get("history", []).copy()
        sampler.load_state_dict(previous["sampler_state"])
        _restore_rng(previous["rng_state"])
    atomic_json(
        output_dir / "config.json",
        {
            "schema_version": 1,
            "model": model_config.to_dict(),
            "training": training_config.to_dict(),
        },
    )
    atomic_json(output_dir / "data_summary.json", cloud.summary())
    atomic_json(output_dir / "split.json", split)
    condition_tensor = torch.tensor(
        cloud.conditions[split["train"]].copy(), device=device, dtype=dtype
    )
    moment_tensor = torch.tensor(moment_targets[split["train"]], device=device, dtype=dtype)
    g_parameters = list(model.hg_parameters())
    flow_parameters = list(model.residual_parameters())
    if not g_parameters or not flow_parameters:
        raise ValueError("model must expose nonempty HG and residual parameter groups")
    if {id(p) for p in g_parameters}.intersection(id(p) for p in flow_parameters):
        raise ValueError("HG and residual optimizer parameter groups must be disjoint")
    stage_lengths = {
        "warmup": training_config.warmup_steps,
        "residual": training_config.residual_steps,
        "joint": training_config.joint_steps,
    }
    if set(completed) != set(stage_lengths):
        raise ValueError("checkpoint has invalid completed-stage keys")
    earlier_incomplete = False
    for name, count in completed.items():
        if (
            name not in stage_lengths
            or not isinstance(count, int)
            or not 0 <= count <= stage_lengths[name]
        ):
            raise ValueError("checkpoint has invalid completed-stage counters")
    for name, total in stage_lengths.items():
        if earlier_incomplete and completed[name] > 0:
            raise ValueError("checkpoint stage progress violates the warmup/residual/joint order")
        earlier_incomplete = earlier_incomplete or completed[name] < total
    if previous is not None and previous.get("global_step") != sum(completed.values()):
        raise ValueError("checkpoint global_step disagrees with its stage counters")
    optimizer: torch.optim.Optimizer | None = None
    optimizer_stage: str | None = None
    if previous is not None:
        optimizer_stage = previous.get("optimizer_stage")
    metrics: dict[str, Any] = {}
    steps_this_run = 0

    def evaluations() -> dict[str, Any]:
        result = {
            "train": evaluate_model(
                model,
                cloud,
                groups=split["train"],
                batch_size=training_config.eval_batch_size,
                scope="in_sample_conditions",
            ),
        }
        if split["validation"]:
            result["validation"] = evaluate_model(
                model,
                cloud,
                groups=split["validation"],
                batch_size=training_config.eval_batch_size,
                scope="held_out_conditions",
            )
        else:
            result["validation"] = None
            result["validation_note"] = (
                "No condition holdout was requested; training-condition metrics do not demonstrate generalization."
            )
        return result

    def save() -> None:
        if optimizer is None:
            stored_optimizer = previous.get("optimizer_state") if previous is not None else None
        else:
            stored_optimizer = optimizer.state_dict()
        payload = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "model_config": model_config.to_dict(),
            "training_config": training_config.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": stored_optimizer,
            "optimizer_stage": optimizer_stage,
            "completed": completed.copy(),
            "global_step": sum(completed.values()),
            "split": split,
            "dataset_fingerprint": fingerprint,
            "sampler_state": sampler.state_dict(),
            "rng_state": _rng_state(),
            "history": history,
            "metrics": metrics,
            "runtime": _runtime_metadata(device, training_config.dtype),
        }
        _atomic_torch_save(checkpoint_path, payload)
        atomic_json(output_dir / "history.json", history)
        atomic_json(output_dir / "metrics.json", metrics)

    if previous is None:
        metrics = evaluations()
        history.append({"event": "initial", "global_step": 0, "metrics": metrics})
        save()
    else:
        metrics = previous.get("metrics", {})

    try:
        for stage, total_steps in stage_lengths.items():
            if completed[stage] >= total_steps:
                continue
            if max_steps_this_run is not None and steps_this_run >= max_steps_this_run:
                break
            model.freeze_hg(stage == "residual")
            for parameter in flow_parameters:
                parameter.requires_grad_(stage != "warmup")
            if stage == "warmup":
                parameters, learning_rate = g_parameters, training_config.lr_g
            elif stage == "residual":
                parameters, learning_rate = flow_parameters, training_config.lr_flow
            else:
                parameters, learning_rate = g_parameters + flow_parameters, training_config.lr_joint
            optimizer = torch.optim.Adam(parameters, lr=learning_rate)
            optimizer_stage = stage
            if (
                previous is not None
                and previous.get("optimizer_stage") == stage
                and previous.get("optimizer_state") is not None
            ):
                optimizer.load_state_dict(previous["optimizer_state"])
            model.train()
            while completed[stage] < total_steps:
                if max_steps_this_run is not None and steps_this_run >= max_steps_this_run:
                    break
                optimizer.zero_grad(set_to_none=True)
                moment_loss: torch.Tensor | None = None
                nll: torch.Tensor | None = None
                if stage in ("warmup", "joint"):
                    moment_loss = (model.hg_g(condition_tensor) - moment_tensor).square().mean()
                if stage in ("residual", "joint"):
                    batch = sampler.sample(training_config.batch_size, device=device, dtype=dtype)
                    log_probability = model.log_prob(batch.outgoing, batch.conditions)
                    if log_probability.shape != (training_config.batch_size,):
                        raise ValueError("model.log_prob must return one scalar per training point")
                    if not torch.isfinite(log_probability).all():
                        raise FloatingPointError(f"non-finite model.log_prob during {stage}")
                    nll = -(batch.loss_weight * log_probability).mean()
                if stage == "warmup":
                    assert moment_loss is not None
                    loss = moment_loss
                elif stage == "residual":
                    assert nll is not None
                    loss = nll
                else:
                    assert nll is not None and moment_loss is not None
                    loss = nll + training_config.moment_regularization * moment_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss during {stage}")
                loss.backward()
                for parameter in parameters:
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f"non-finite gradient during {stage}")
                grad_norm = None
                if training_config.grad_clip_norm is not None:
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            parameters, training_config.grad_clip_norm, error_if_nonfinite=True
                        ).item()
                    )
                optimizer.step()
                if any(not torch.isfinite(parameter).all() for parameter in parameters):
                    raise FloatingPointError(
                        f"optimizer produced non-finite parameters during {stage}"
                    )
                completed[stage] += 1
                steps_this_run += 1
                global_step = sum(completed.values())
                entry: dict[str, Any] = {
                    "stage": stage,
                    "stage_step": completed[stage],
                    "global_step": global_step,
                    "loss": float(loss.detach().item()),
                }
                if nll is not None:
                    entry["nll"] = float(nll.detach().item())
                if moment_loss is not None:
                    entry["g_moment_mse"] = float(moment_loss.detach().item())
                if grad_norm is not None:
                    entry["gradient_norm_before_clip"] = grad_norm
                boundary = completed[stage] == total_steps
                if boundary or (
                    training_config.eval_every and global_step % training_config.eval_every == 0
                ):
                    metrics = evaluations()
                    entry["metrics"] = metrics
                history.append(entry)
                if callback is not None and (boundary or "metrics" in entry):
                    callback(entry)
                if boundary or (
                    training_config.checkpoint_every
                    and global_step % training_config.checkpoint_every == 0
                ):
                    save()
            if max_steps_this_run is not None and steps_this_run >= max_steps_this_run:
                break
        metrics = evaluations()
        save()
    except KeyboardInterrupt:
        # Preserve the last atomically completed checkpoint. An interruption in
        # backward/optimizer.step is not necessarily an exact update boundary;
        # saving it would replace a valid state with a potentially partial one.
        # max_steps_this_run is the supported controlled interruption mechanism.
        raise
    model.eval()
    return TrainResult(
        model=model,
        history=history,
        split=split,
        checkpoint_path=checkpoint_path,
        metrics=metrics,
        completed=completed.copy(),
    )
