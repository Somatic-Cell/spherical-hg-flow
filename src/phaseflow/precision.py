"""Paired numerical diagnostics for one saved Rainbow flow.

This evaluates fixed weights; it does not retrain, export an OptiX model, or
emulate CoopVec arithmetic.  In particular, the FP16 diagnostic rounds neural
parameters to IEEE float16 and promotes them to float32 for all MLP arithmetic.
The original checkpoint, physical constants and stored teacher are unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .rainbow import RainbowReference
from .single_condition import (
    _array_hash,
    _check_model_reference,
    _code_hash,
    _resolve_device,
    _runtime,
    _uniforms,
    load_single_checkpoint,
)
from .sphere_model import SingleConditionSphereFlow
from .training import atomic_json

__all__ = ["evaluate_precision"]

# Separate from training's streams 0--4 and plotting's stream 100.
_TARGET_STREAM = 101
_PROPOSAL_STREAM = 102
_COMMON_UNIFORM_BITS = 23


def _file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _generator(seed: int, stream: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, stream])))


def _finite(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise FloatingPointError(f"nonfinite {name}; no values are clipped or discarded")
    return result


def _estimate(values: np.ndarray) -> dict[str, float]:
    values = _finite(values, "paired statistic").reshape(-1)
    return {
        "mean": float(values.mean()),
        "standard_error": float(values.std(ddof=1) / math.sqrt(values.size)),
    }


def _errors(values: np.ndarray) -> dict[str, float]:
    values = _finite(values, "numerical difference").reshape(-1)
    absolute = np.abs(values)
    return {
        **_estimate(values),
        "mean_abs": float(absolute.mean()),
        "median_abs": float(np.quantile(absolute, 0.5)),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "p99_abs": float(np.quantile(absolute, 0.99)),
        "max_abs": float(absolute.max()),
        "rms": float(np.sqrt(np.mean(values * values))),
    }


def _clone_for_math(
    model: SingleConditionSphereFlow, *, dtype: torch.dtype
) -> SingleConditionSphereFlow:
    # Both diagnostic precisions use the same stable HG endpoint formulas.
    # The native variant alone retains legacy cosine-based geometry, if saved.
    config = replace(
        model.config,
        geometry_dtype="model",
        spline_dtype="float64" if dtype == torch.float64 else "model",
    )
    state = copy.deepcopy(model.state_dict())
    state["_extra_state"]["config"] = config.to_dict()
    clone = SingleConditionSphereFlow(
        model.hg_g,
        model.incident_cosine,
        config,
        dtype=dtype,
        device=model.device,
        validate_args=False,
    )
    clone.load_state_dict(state, strict=True)
    clone.eval()
    return clone


def _round_neural_parameters(model: SingleConditionSphereFlow) -> dict[str, Any]:
    """Round only a disposable FP32 clone; never change HG or other metadata."""
    limit = float(torch.finfo(torch.float16).max)
    before, after = [], []
    for name, parameter in model.named_parameters():
        values = _finite(parameter.detach().cpu().numpy(), f"parameter {name}")
        if np.any(np.abs(values) > limit):
            raise FloatingPointError(
                f"parameter {name} exceeds the finite FP16 range [-{limit}, {limit}]; "
                "FP16 diagnostics refuse clipping or overflow"
            )
        # CPU conversion makes the documented IEEE storage-rounding experiment
        # independent of GPU half arithmetic/flush-to-zero behavior.
        rounded = parameter.detach().cpu().to(torch.float16).to(torch.float32)
        quantized = _finite(rounded.numpy(), f"FP16-rounded parameter {name}")
        before.append(values.reshape(-1))
        after.append(quantized.reshape(-1))
        parameter.copy_(rounded.to(device=model.device))
    original, quantized = np.concatenate(before), np.concatenate(after)
    return {
        "scope": "all_neural_parameters_including_biases",
        "parameter_count": int(original.size),
        "fp32_parameter_bytes": int(4 * original.size),
        "fp16_parameter_bytes": int(2 * original.size),
        "byte_count_excludes": "HG/condition metadata, alignment and native matrix layouts",
        "source_max_abs": float(np.abs(original).max()),
        "fp16_finite_max": limit,
        "rounded_to_zero_count": int(np.count_nonzero((original != 0) & (quantized == 0))),
        "fp16_subnormal_count": int(
            np.count_nonzero(
                (quantized != 0) & (np.abs(quantized) < float(torch.finfo(torch.float16).tiny))
            )
        ),
        "rounding_error": _errors(quantized - original),
        "emulates_coopvec": False,
    }


def _evaluate_variant(
    model: SingleConditionSphereFlow,
    directions: np.ndarray,
    log_p: np.ndarray,
    uniforms: np.ndarray,
    batch_size: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    model.validate_args = False
    count = len(directions)
    x = torch.as_tensor(directions, dtype=model.geometry_dtype, device=model.device)
    u = torch.as_tensor(uniforms, dtype=model.spline_dtype, device=model.device)
    likelihood = torch.empty((count, 2), dtype=model.geometry_dtype, device=model.device)
    generated = torch.empty((count, 5), dtype=model.geometry_dtype, device=model.device)
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        likelihood[start:end, 0] = model.log_prob(x[start:end])
        likelihood[start:end, 1] = model.hg_base_direction_log_prob(x[start:end])
        output, sampled = model.sample_from_uniform(u[start:end])
        evaluated = model.log_prob(output)
        generated[start:end] = torch.cat((output, sampled[:, None], evaluated[:, None]), dim=-1)
    # Reductions here are CPU FP64 reference statistics; FP32 variants contain
    # no FP64 GPU arithmetic. There are no per-parameter/per-lane device reads.
    density = _finite(likelihood.cpu().numpy(), "evaluated log PDF")
    values = _finite(generated.cpu().numpy(), "generated directions or log PDF")
    log_q, log_hg = density[:, 0], density[:, 1]
    nll, kl = _estimate(-log_q), _estimate(log_p - log_q)
    metadata = {
        "precision": {
            "conditioner": str(model.dtype).removeprefix("torch."),
            "spline": str(model.spline_dtype).removeprefix("torch."),
            "hg_geometry_log_pdf": str(model.geometry_dtype).removeprefix("torch."),
            "hg_formulation": (
                "endpoint_distances" if model.config.geometry_dtype == "model" else "legacy_cosine"
            ),
        },
        "model_config": model.config.to_dict(),
        "base_g": model.hg_g,
        "nll": nll["mean"],
        "nll_standard_error": nll["standard_error"],
        "forward_kl_estimate": kl["mean"],
        "forward_kl_standard_error": kl["standard_error"],
        "hg_nll": _estimate(-log_hg)["mean"],
        "nll_improvement_over_hg": _estimate(log_q - log_hg),
        "sample_eval_log_pdf_error": _errors(values[:, 4] - values[:, 3]),
        "sample_unit_length_error": _errors(np.linalg.norm(values[:, :3], axis=-1) - 1),
    }
    return metadata, log_q, values[:, :3], values[:, 3]


def _compare(
    baseline: tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray],
    candidate: tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    _, base_log_q, base_xyz, base_sample_log_q = baseline
    _, log_q, xyz, sample_log_q = candidate
    angle = np.arctan2(
        np.linalg.norm(np.cross(base_xyz, xyz), axis=-1), np.sum(base_xyz * xyz, axis=-1)
    )
    return {
        "nll_increase": _estimate(base_log_q - log_q),
        "forward_kl_increase": _estimate(base_log_q - log_q),
        "heldout_log_pdf_difference": _errors(log_q - base_log_q),
        "same_uniform_direction_angle_rad": _errors(angle),
        "same_uniform_sample_log_pdf_difference": _errors(sample_log_q - base_sample_log_q),
        "sign_convention": "candidate minus baseline; positive nll_increase is worse",
        "sample_log_pdf_comparison": (
            "at corresponding uniforms and their respective output directions, "
            "not two densities evaluated at the same direction"
        ),
    }


@torch.no_grad()
def evaluate_precision(
    checkpoint: str | Path,
    reference: RainbowReference,
    output: str | Path | None = None,
    *,
    device: str | torch.device = "cuda",
    samples: int = 4096,
    seed: int = 2028,
    batch_size: int = 4096,
) -> dict[str, Any]:
    """Compare checkpoint, FP64, FP32, and FP16-rounded weights on paired inputs.

    ``reference`` must be the exact source record recorded in the checkpoint.
    ``samples`` controls both independent teacher points and shared model
    uniforms. All variants see the same original teacher directions; conversion
    to their geometry dtype is part of the numerical diagnostic. The 23-bit
    midpoint model uniforms are exactly representable in both FP32 and FP64.

    The FP64/FP32 candidates use identical stable HG formulas. Native inference
    preserves the checkpoint's geometry/RQS choices, including old mixed
    precision. The fourth variant isolates parameter rounding from arithmetic
    changes by comparing it to an unrounded FP32 candidate. No universal error
    tolerance or automatic precision selection is imposed.

    CUDA is required unless CPU is explicitly selected. FP64 work is restricted
    to this diagnostic; it does not change the training precision. Optional
    ``output`` is an atomic JSON report, never a modified checkpoint.
    """
    for name, value, minimum in (("samples", samples, 2), ("batch_size", batch_size, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be in [0, 2**63)")
    checkpoint = Path(checkpoint).resolve()
    destination = Path(output).resolve() if output is not None else None
    protected = {checkpoint} | {
        (reference.directory / name).resolve()
        for name in ("metadata.json", "phi_cdf.npy", "theta_given_phi_cdf.npy", "u_edges.npy")
    }
    if destination in protected:
        raise ValueError("precision output must not overwrite the checkpoint or reference files")
    selected = _resolve_device(device)
    cuda_devices = [selected.index] if selected.type == "cuda" else []
    previous_matmul_precision = torch.get_float32_matmul_precision()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        # Constructors initialize disposable MLPs before loading fixed weights.
        # Restore that incidental RNG consumption, including on error.
        with (
            torch.random.fork_rng(devices=cuda_devices),
            torch.autocast(device_type=selected.type, enabled=False),
        ):
            native, payload = load_single_checkpoint(checkpoint, device=selected)
            if payload["dataset_fingerprint"] != reference.fingerprint():
                raise ValueError("checkpoint and Rainbow reference fingerprint differ")
            _check_model_reference(native, reference)
            for name, parameter in native.named_parameters():
                _finite(parameter.detach().cpu().numpy(), f"checkpoint parameter {name}")
            double = _clone_for_math(native, dtype=torch.float64)
            single = _clone_for_math(native, dtype=torch.float32)
            rounded = _clone_for_math(single, dtype=torch.float32)
            quantization = _round_neural_parameters(rounded)
            directions, log_p = reference.sample(
                _uniforms(_generator(seed, _TARGET_STREAM), samples, bits=52)
            )
            directions = _finite(directions, "CDF-generated direction")
            log_p = _finite(log_p, "CDF-generated teacher log PDF")
            uniforms = _uniforms(
                _generator(seed, _PROPOSAL_STREAM), samples, bits=_COMMON_UNIFORM_BITS
            )
            models = {
                "native_checkpoint": native,
                "float64_reference": double,
                "float32_math": single,
                "fp16_weights_float32_math": rounded,
            }
            measured = {
                name: _evaluate_variant(model, directions, log_p, uniforms, batch_size)
                for name, model in models.items()
            }
            report = {
                "schema": "phaseflow.precision_diagnostic.v1",
                "scope": "same_checkpoint_independent_points_same_condition",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _file_hash(checkpoint),
                "checkpoint_kind": payload["kind"],
                "checkpoint_step": payload["global_step"],
                "checkpoint_code_fingerprint": payload["code_fingerprint"],
                "model_code_fingerprint": _code_hash(),
                "precision_code_sha256": _file_hash(Path(__file__)),
                "dataset_fingerprint": reference.fingerprint(),
                "source_provenance": reference.provenance,
                "physics": copy.deepcopy(payload["physics"]),
                "density_measure": "solid_angle_sr",
                "logarithm": "natural",
                "sample_count": samples,
                "seed": seed,
                "batch_size": batch_size,
                "streams": {"teacher": _TARGET_STREAM, "model_uniforms": _PROPOSAL_STREAM},
                "teacher_uniform_midpoint_bits": 52,
                "common_model_uniform_midpoint_bits": _COMMON_UNIFORM_BITS,
                "teacher_points_sha256": _array_hash(directions, log_p),
                "model_uniforms_sha256": _array_hash(uniforms),
                "target_entropy_estimate": _estimate(-log_p),
                "runtime": _runtime(selected, str(native.dtype).removeprefix("torch.")),
                "autocast_enabled": False,
                "variants": {name: result[0] for name, result in measured.items()},
                "comparisons": {
                    "native_vs_float64": _compare(
                        measured["float64_reference"], measured["native_checkpoint"]
                    ),
                    "float32_vs_float64": _compare(
                        measured["float64_reference"], measured["float32_math"]
                    ),
                    "fp16_weights_vs_float32": _compare(
                        measured["float32_math"], measured["fp16_weights_float32_math"]
                    ),
                },
                "weight_quantization": quantization,
                "limitations": [
                    "No retraining or checkpoint selection is performed.",
                    "FP16 parameters are promoted to FP32 before MLP evaluation; "
                    "this does not emulate CoopVec accumulation, activations or rounding.",
                    "HG and physical metadata remain external fixed constants; "
                    "only neural parameters, including biases, are quantized.",
                    "The stable HG endpoint formulation is used for FP64/FP32 candidates; "
                    "the native variant alone preserves legacy cosine geometry if saved.",
                    "Random-point diagnostics do not certify worst-case seam/pole behavior, "
                    "integral normalization, CUDA speed, or renderer image accuracy.",
                    "Standard errors measure paired Monte Carlo uncertainty, "
                    "not variation from independent training runs.",
                ],
            }
    finally:
        torch.set_float32_matmul_precision(previous_matmul_precision)
    if destination is not None:
        atomic_json(destination, report)
    return report
