"""Recover a recorded fixed training pool for faithful, read-only plotting.

Version-4 runs record the hash of the FP64 CDF directions and teacher log PDF,
not the device tensor itself. Regeneration is accepted only when that complete
pool hash agrees. Directions are then cast exactly as in the saved training
configuration. This is verified regeneration, not a claim to read archived GPU
memory. It neither changes training state nor requires the old runtime to run.

Version-5 log-density runs additionally record component labels and the teacher
log PDF evaluated at the actual dtype-cast points. Their raw and runtime pool
identities are both checked before any point is displayed. The recorded
objective selects the historical CDF/mixed pool or an all-uniform query pool;
replotting never substitutes one sampling distribution for another.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from .rainbow import RainbowReference
from .single_condition import (
    FAMILY,
    SingleTrainingConfig,
    _array_hash,
    _points,
    read_single_checkpoint,
)
from .sphere_model import SingleConditionSphereFlow, SphereFlowConfig

TRAINING_SCATTER_SCHEMA = "phaseflow.training_scatter.v1"


@dataclass(frozen=True)
class TrainingScatter:
    """Every pool entry once, in its training geometry dtype and original order."""

    directions_nf: NDArray[np.float32] | NDArray[np.float64]
    source_phi_degrees: NDArray[np.float64]
    theta_degrees: NDArray[np.float64]
    provenance: dict[str, Any]
    components: NDArray[np.uint8] | None = None
    teacher_log_pdf: NDArray[np.float64] | None = None


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Training scatter requires the saved run file: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Saved run file must contain a JSON object: {path}")
    return value


def _state_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and left.keys() == right.keys()
            and all(_state_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_state_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def _identity(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    identity = {
        "path": str(path),
        "sha256": _file_hash(path),
        "kind": payload["kind"],
        "checkpoint_version": payload["checkpoint_version"],
        "global_step": payload["global_step"],
        "code_fingerprint": payload["code_fingerprint"],
    }
    if payload["checkpoint_version"] == 5 and payload["kind"] == "inference":
        identity["selection_metric"] = payload["selection_metric"]
    return identity


def resolve_training_scatter(
    model: SingleConditionSphereFlow,
    reference: RainbowReference,
    checkpoint_path: str | Path | None,
    *,
    run_directory: str | Path | None = None,
) -> TrainingScatter:
    """Verify saved run/checkpoint identity and reconstruct the entire pool.

    The plotted checkpoint, resumable ``checkpoint.pt``, ``config.json`` and
    ``sample_split.json`` must describe one run. A passed inference model must
    match its checkpoint, which in turn must match the recorded best state and
    validation selection. Training checkpoints are supported but explicitly
    identified as such. Read-only plotting may use a different device/runtime;
    the saved model configuration and geometry precision must still agree.
    """
    reference._require_open()
    if checkpoint_path is None:
        raise ValueError(
            "Training scatter requires a checkpoint and its saved run directory; "
            "use scatter_mode='independent' explicitly for a diagnostic CDF sample"
        )
    if not isinstance(model, SingleConditionSphereFlow):
        raise ValueError("Training scatter requires the saved single-condition sphere model")
    plotted_path = Path(checkpoint_path).resolve()
    run = Path(run_directory).resolve() if run_directory is not None else plotted_path.parent
    training_path = run / "checkpoint.pt"
    for path in (plotted_path, training_path):
        if not path.is_file():
            raise ValueError(f"Training scatter requires the saved checkpoint: {path}")
    plotted = read_single_checkpoint(plotted_path)
    training = plotted if plotted_path == training_path else read_single_checkpoint(training_path)
    version = training["checkpoint_version"]
    if training["kind"] != "training" or version not in (4, 5):
        raise ValueError("Training scatter requires a version-4 or version-5 training checkpoint.pt")
    required = {"training_config", "sample_split", "best_state", "best_step", "best_validation"}
    if not required <= training.keys():
        raise ValueError("Training checkpoint is missing fixed-pool or selection provenance")
    config_path, split_path = run / "config.json", run / "sample_split.json"
    saved_config, saved_split = _read_object(config_path), _read_object(split_path)
    expected_schema = 3 if version == 5 else 2
    if saved_config.get("schema_version") != expected_schema or saved_config.get("family") != FAMILY:
        raise ValueError(
            f"Training scatter requires a saved schema-{expected_schema} single-condition config"
        )
    objective = None
    if version == 5:
        from .log_objective import LogObjectiveConfig

        if not {"objective_config", "selections"} <= training.keys():
            raise ValueError("Log-density checkpoint is missing objective or selection provenance")
        if saved_config.get("objective") != training["objective_config"] or (
            plotted.get("objective_config") != training["objective_config"]
        ):
            raise ValueError("Saved log-density objective differs from the plotted checkpoint")
        objective = LogObjectiveConfig.from_dict(training["objective_config"])
    if saved_config.get("training") != training["training_config"]:
        raise ValueError("Saved training configuration differs from checkpoint.pt")
    if saved_split != training["sample_split"]:
        raise ValueError("Saved sample_split.json differs from checkpoint.pt")
    cfg = SingleTrainingConfig.from_dict(training["training_config"])
    model_config = SphereFlowConfig.from_dict(training["model_config"])
    if not isinstance(saved_config.get("model"), dict) or (
        SphereFlowConfig.from_dict(saved_config["model"]) != model_config
    ):
        raise ValueError("Saved model configuration differs from checkpoint.pt")
    if model.config != model_config:
        raise ValueError("Passed model configuration differs from the saved training model")
    for key in ("checkpoint_version", "family", "dtype", "physics", "dataset_fingerprint",
                "code_fingerprint"):
        if plotted.get(key) != training.get(key):
            raise ValueError(f"Plotted checkpoint and training checkpoint differ in {key}")
    if SphereFlowConfig.from_dict(plotted["model_config"]) != model_config:
        raise ValueError("Plotted checkpoint model configuration differs from checkpoint.pt")
    fingerprint = reference.fingerprint()
    physics = {
        "hg_g": reference.g,
        "incident_cosine": float(reference.condition[1]),
        "wavelength_nm": float(reference.condition[0]),
    }
    if training["dataset_fingerprint"] != fingerprint or training["physics"] != physics:
        raise ValueError("Training scatter reference record differs from the saved checkpoint")
    if model.hg_g != reference.g or model.incident_cosine != physics["incident_cosine"]:
        raise ValueError("Passed model physics differs from the saved reference")
    dtype = {"float32": torch.float32, "float64": torch.float64}[cfg.dtype]
    geometry_dtype = torch.float64 if model_config.geometry_dtype == "float64" else dtype
    if training["dtype"] != cfg.dtype or model.dtype != dtype or model.geometry_dtype != geometry_dtype:
        raise ValueError("Passed model dtype differs from the saved training precision")
    if type(training["global_step"]) is not int or not 0 <= training["global_step"] <= cfg.steps:
        raise ValueError("Invalid saved training checkpoint step")
    if type(training["best_step"]) is not int or not (
        0 <= training["best_step"] <= training["global_step"]
    ):
        raise ValueError("Invalid saved best checkpoint step")
    if plotted["kind"] == "inference":
        if version == 5:
            selection_metric = plotted.get("selection_metric")
            selection = training["selections"].get(selection_metric)
            if selection_metric not in ("nll", "log_rmse") or not isinstance(selection, dict):
                raise ValueError("Inference checkpoint has no matching recorded selection metric")
            if not {"model_state", "step", "validation"} <= selection.keys():
                raise ValueError("Recorded selection is missing state, step or validation")
            best_state, best_step = selection["model_state"], selection["step"]
            best_validation = selection["validation"]
        else:
            best_state, best_step = training["best_state"], training["best_step"]
            best_validation = training["best_validation"]
        if type(best_step) is not int or not 0 <= best_step <= training["global_step"]:
            raise ValueError("Invalid saved selection checkpoint step")
        if plotted["global_step"] != best_step or not _state_equal(
            plotted["model_state"], best_state
        ):
            raise ValueError("Inference checkpoint differs from the recorded best state or step")
        if plotted.get("validation") != best_validation:
            raise ValueError("Inference checkpoint differs from the recorded best validation")
        scope = "validation_selected_checkpoint"
    else:
        if plotted["global_step"] != training["global_step"] or not _state_equal(
            plotted["model_state"], training["model_state"]
        ):
            raise ValueError("Plotted training checkpoint differs from the saved run state")
        scope = "training_checkpoint_iterate_not_necessarily_validation_selected"
    if not _state_equal(model.state_dict(), plotted["model_state"]):
        raise ValueError("Passed model state differs from the plotted checkpoint")

    data_seed = cfg.seed if cfg.data_seed is None else cfg.data_seed
    if (
        saved_split.get("seed") != cfg.seed
        or saved_split.get("data_seed") != data_seed
        or saved_split.get("train_samples") != cfg.train_samples
        or saved_split.get("streams", {}).get("train") != 0
        or saved_split.get("uniforms") != "52-bit open midpoint grid"
    ):
        raise ValueError("Saved fixed-pool seed, count or sampling stream is inconsistent")
    recorded_hash = saved_split.get("training_points_sha256")
    if not isinstance(recorded_hash, str) or len(recorded_hash) != 64:
        raise ValueError("Saved training pool is missing its SHA-256 identity")
    components, teacher_log_pdf, pool_provenance = None, None, None
    if version == 5:
        from .log_objective import make_training_pool

        pool = make_training_pool(
            reference, cfg.train_samples, data_seed, objective,
            geometry_dtype=model.geometry_dtype,
        )
        if pool.provenance != saved_split.get("training_pool"):
            raise ValueError("Regenerated training pool differs from saved component provenance")
        directions, log_pdf = pool.directions, pool.log_p
        components, teacher_log_pdf = pool.components, pool.log_p
        pool_provenance = pool.provenance
        regenerated_hash = _array_hash(directions, log_pdf, components)
    else:
        directions, log_pdf = _points(reference, cfg.train_samples, data_seed, "train")
        regenerated_hash = _array_hash(directions, log_pdf)
    if regenerated_hash != recorded_hash:
        raise ValueError(
            "Regenerated fixed training pool does not match training_points_sha256; "
            "refusing to plot a different point cloud"
        )
    # Match the trainer's one-time NumPy -> model-device geometry cast exactly.
    cast_directions = torch.as_tensor(
        directions, dtype=model.geometry_dtype, device=model.device
    ).detach().cpu().numpy().copy()
    source = cast_directions.astype(np.float64) @ reference.source_to_nf
    source_phi = np.arctan2(source[:, 1], source[:, 0])
    source_phi = (source_phi + np.pi) % (2 * np.pi) - np.pi
    theta = np.arctan2(np.hypot(source[:, 0], source[:, 1]), source[:, 2])
    source_phi_degrees, theta_degrees = np.rad2deg(source_phi), np.rad2deg(theta)
    in_band = (theta_degrees >= 120.0) & (theta_degrees <= 150.0)
    provenance = {
        "schema": TRAINING_SCATTER_SCHEMA,
        "source": "verified_regeneration_of_recorded_fixed_training_pool",
        "scope": "each_fixed_pool_entry_once_not_minibatch_repetitions",
        "sample_count": cfg.train_samples,
        "displayed_sample_count": int(cast_directions.shape[0]),
        "downsampling": False,
        "seed": cfg.seed,
        "data_seed": data_seed,
        "stream_id": 0,
        "rng": "numpy.PCG64(SeedSequence([data_seed,stream_id]))",
        "uniform_midpoint_bits": 52,
        "training_points_sha256": recorded_hash,
        "regenerated_training_points_sha256": regenerated_hash,
        "hash_verified": True,
        "pool_hash_covers": "FP64 directions_nf and FP64 teacher_log_pdf, before device cast",
        "geometry_dtype": str(geometry_dtype).removeprefix("torch."),
        "directions_nf_sha256": _array_hash(cast_directions),
        "source_phi_degrees_sha256": _array_hash(source_phi_degrees),
        "theta_degrees_sha256": _array_hash(theta_degrees),
        "precision_provenance": (
            "Regenerated FP64 pool matches the saved hash; the plotted NF directions use "
            "the saved training geometry cast. Original device tensor bytes were not archived."
        ),
        "dataset_fingerprint": fingerprint,
        "training_run_directory": str(run),
        "plotted_checkpoint": _identity(plotted_path, plotted),
        "training_checkpoint": _identity(training_path, training),
        "plotted_model_scope": scope,
        "model_state_verified": True,
        "implementation_sha256": _file_hash(Path(__file__)),
        "configuration": {"path": str(config_path), "sha256": _file_hash(config_path)},
        "sample_split": {"path": str(split_path), "sha256": _file_hash(split_path)},
        "band": {
            "theta_degrees": [120.0, 150.0],
            "boundary": "inclusive endpoints",
            "azimuth": "full circle",
            "observed_train_samples": int(np.count_nonzero(in_band)),
            "observed_train_fraction": float(np.mean(in_band)),
            "count_scope": "actual dtype-cast fixed pool; not an expected count",
        },
    }
    if version == 5:
        provenance.update({
            "schema": "phaseflow.training_scatter.v2",
            "sampling_distribution": pool_provenance["sampling"],
            "training_pool": pool_provenance,
            "component_counts": pool_provenance["component_counts"],
            "component_codes": pool_provenance["component_codes"],
            "source_streams": pool_provenance["source_streams"],
            "pool_hash_covers": (
                "actual training geometry directions, FP64 teacher_log_pdf evaluated at "
                "those directions, and uint8 component labels; raw FP64 pool hash is also verified"
            ),
            "components_sha256": _array_hash(components),
            "teacher_log_pdf_sha256": _array_hash(teacher_log_pdf),
            "precision_provenance": (
                "Regenerated raw and runtime pool hashes agree with the saved run. Each "
                "plotted direction and teacher label matches the actual training geometry dtype."
            ),
            "objective": objective.to_dict(),
        })
        # The component provenance identifies the streams actually used. Mixed
        # pools use two; all-uniform pools use only the uniform stream and do
        # not generate any CDF-distributed training points.
        provenance.pop("stream_id")
        provenance["band"]["observed_component_samples"] = {
            name: int(np.count_nonzero(in_band & (components == code)))
            for name, code in pool_provenance["component_codes"].items()
        }
    return TrainingScatter(
        cast_directions, source_phi_degrees, theta_degrees, provenance,
        components=components, teacher_log_pdf=teacher_log_pdf,
    )
