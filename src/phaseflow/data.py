"""Explicit point-cloud adapters for a normalized conditional phase distribution.

There are two distinct data modes:

``target_samples``
    Each point is an independent sample from the desired conditional density.
    Point weights are forbidden; applying the density again would square it.
``quadrature``
    ``weights`` are nonnegative integration masses, for example p(omega) dOmega.
    They are NOT bare density values unless the supplied quadrature measure has
    already been included. Their scale may differ between condition groups.

All directions use an incident-aligned right-handed frame: +z is the incident
light's propagation direction, +x is the projected particle symmetry axis,
and +y = +z cross +x. An arbitrary consistent x axis is used at axial incidence.
No jitter, smoothing, symmetry averaging, or direction renormalization is done.
An NPZ file is only one adapter; a CDF generator can instantiate PhasePointCloud
directly from its in-memory samples without adopting a particular CDF format.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np
import torch
from numpy.typing import NDArray

SCHEMA_VERSION = 1
COORDINATE_FRAME = "incident_z_particle_axis_x"
DATA_MODES = ("target_samples", "quadrature")


@dataclass
class PointBatch:
    """A training batch; ``loss_weight`` belongs in the NLL exactly once.

    Directions retain float64 geometry even when network conditions and weights
    use float32. This avoids collapsing narrowly separated polar directions.
    """

    outgoing: torch.Tensor
    conditions: torch.Tensor
    condition_index: torch.Tensor
    loss_weight: torch.Tensor


@dataclass
class PhasePointCloud:
    """Validated full-sphere conditional samples or weighted quadrature points.

    ``conditions[c]`` is ``[wavelength_nm, incident_cosine]``; the latter is the
    cosine relative to the particle symmetry axis. ``outgoing[n, 2]`` is instead
    the scattering-angle cosine relative to the incident propagation direction.
    Duplicate condition rows are rejected to prevent train/validation leakage.
    Positive total mass is required for every condition group.
    """

    conditions: NDArray[Any]
    outgoing: NDArray[Any]
    condition_index: NDArray[Any]
    mode: Literal["target_samples", "quadrature"] = "target_samples"
    weights: NDArray[Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    coordinate_frame: str = COORDINATE_FRAME
    coverage: str = "full_sphere"
    _indices: tuple[NDArray[np.int64], ...] = field(init=False, repr=False)
    _masses: tuple[NDArray[np.float64], ...] = field(init=False, repr=False)
    _cdfs: tuple[NDArray[np.float64], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in DATA_MODES:
            raise ValueError(f"data_mode must be one of {DATA_MODES}, got {self.mode!r}")
        if self.coordinate_frame != COORDINATE_FRAME:
            raise ValueError(f"coordinate_frame must be {COORDINATE_FRAME!r}")
        if self.coverage != "full_sphere":
            raise ValueError(
                "coverage must be 'full_sphere'; a cropped rainbow CDF is not a full phase distribution"
            )
        self.conditions = np.array(self.conditions, dtype=np.float64, order="C", copy=True)
        raw_outgoing = np.asarray(self.outgoing)
        if raw_outgoing.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raw_outgoing = raw_outgoing.astype(np.float64)
        self.outgoing = np.array(raw_outgoing, order="C", copy=True)
        raw_index = np.asarray(self.condition_index)
        if raw_index.dtype.kind not in "iu":
            raise ValueError("condition_index must have an integer dtype")
        self.condition_index = np.array(raw_index, dtype=np.int64, order="C", copy=True)
        if self.conditions.ndim != 2 or self.conditions.shape[1] != 2 or len(self.conditions) == 0:
            raise ValueError("conditions must have shape [C, 2] with C > 0")
        if self.outgoing.ndim != 2 or self.outgoing.shape[1] != 3 or len(self.outgoing) == 0:
            raise ValueError("outgoing must have shape [N, 3] with N > 0")
        if self.condition_index.shape != (len(self.outgoing),):
            raise ValueError("condition_index must have shape [N]")
        if not np.isfinite(self.conditions).all() or not np.isfinite(self.outgoing).all():
            raise ValueError("conditions and outgoing must contain only finite values")
        if np.any(self.conditions[:, 0] <= 0):
            raise ValueError("wavelength_nm must be positive")
        if np.any(np.abs(self.conditions[:, 1]) > 1):
            raise ValueError("incident_cosine must lie in [-1, 1]")
        if len(np.unique(self.conditions, axis=0)) != len(self.conditions):
            raise ValueError(
                "conditions must be unique; combine repeated condition groups before splitting"
            )
        lengths = np.linalg.norm(self.outgoing.astype(np.float64), axis=-1)
        if np.any(np.abs(lengths - 1.0) > 2e-5):
            raise ValueError(
                "outgoing directions must be unit vectors within 2e-5; this adapter does not renormalize them"
            )
        if np.any(np.abs(self.outgoing[:, 2]) > 1.0 + 1e-7):
            raise ValueError(
                "outgoing z coordinates must lie in [-1, 1] up to floating-point roundoff"
            )
        if np.any(self.condition_index < 0) or np.any(self.condition_index >= len(self.conditions)):
            raise ValueError("condition_index contains an index outside [0, C)")
        counts = np.bincount(self.condition_index, minlength=len(self.conditions))
        if np.any(counts == 0):
            raise ValueError("every condition must have at least one point")
        if self.mode == "target_samples":
            if self.weights is not None:
                raise ValueError(
                    "weights are forbidden in target_samples mode; samples already follow the target density"
                )
        else:
            if self.weights is None:
                raise ValueError("quadrature mode requires explicit integration masses in weights")
            self.weights = np.array(self.weights, dtype=np.float64, order="C", copy=True)
            if self.weights.shape != (len(self.outgoing),):
                raise ValueError("weights must have shape [N]")
            if not np.isfinite(self.weights).all() or np.any(self.weights < 0):
                raise ValueError("weights must be finite and nonnegative")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a JSON object")
        self.metadata = json.loads(json.dumps(self.metadata, allow_nan=False))
        order = np.argsort(self.condition_index, kind="stable")
        boundaries = np.cumsum(counts)[:-1]
        indices = tuple(np.split(order, boundaries))
        masses = []
        cdfs = []
        for ci, group in enumerate(indices):
            mass = (
                np.ones(len(group), dtype=np.float64)
                if self.weights is None
                else self.weights[group].copy()
            )
            # Scaling first avoids overflow for a valid collection of large masses.
            scale = mass.max()
            if not scale > 0:
                raise ValueError(f"condition {ci} has zero total integration mass")
            mass /= scale
            mass /= mass.sum(dtype=np.float64)
            cdf = np.cumsum(mass, dtype=np.float64)
            cdf[-1] = 1.0
            masses.append(mass)
            cdfs.append(cdf)
        self._indices = indices
        self._masses = tuple(masses)
        self._cdfs = tuple(cdfs)
        for array in self._indices + self._masses + self._cdfs:
            array.setflags(write=False)
        for array in (self.conditions, self.outgoing, self.condition_index, self.weights):
            if array is not None:
                array.setflags(write=False)

    @property
    def num_conditions(self) -> int:
        return len(self.conditions)

    @property
    def num_points(self) -> int:
        return len(self.outgoing)

    def group_indices(self, condition: int) -> NDArray[np.int64]:
        return self._indices[condition]

    def group_masses(self, condition: int) -> NDArray[np.float64]:
        """Normalized discrete/quadrature mass, never a density per steradian."""
        return self._masses[condition]

    def moment_targets(self) -> NDArray[np.float64]:
        """Estimated E[cos(scattering angle)] per condition; no fitting of HG KL."""
        # Use the same angular cosine as inference, accounting for tolerated
        # unit-vector roundoff without changing any stored direction values.
        directions = self.outgoing.astype(np.float64)
        cosine = np.clip(directions[:, 2] / np.linalg.norm(directions, axis=-1), -1.0, 1.0)
        return np.asarray(
            [np.dot(mass, cosine[ids]) for ids, mass in zip(self._indices, self._masses)]
        )

    def fingerprint(self) -> str:
        """Content hash including point order, which influences deterministic sampling."""
        digest = hashlib.sha256()
        for value in (
            SCHEMA_VERSION,
            self.mode,
            self.coordinate_frame,
            self.coverage,
            self.metadata,
        ):
            digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
        for array in (self.conditions, self.outgoing, self.condition_index, self.weights):
            if array is None:
                digest.update(b"null")
                continue
            digest.update(str(array.dtype).encode())
            digest.update(json.dumps(array.shape).encode())
            digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
        return digest.hexdigest()

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "data_mode": self.mode,
            "coordinate_frame": self.coordinate_frame,
            "coverage": self.coverage,
            "num_conditions": self.num_conditions,
            "num_points": self.num_points,
            "points_per_condition": [len(x) for x in self._indices],
            "moment_g_targets": self.moment_targets().tolist(),
            "metadata": self.metadata,
            "fingerprint": self.fingerprint(),
        }

    def save_npz(self, path: str | Path) -> None:
        """Atomically write a portable, pickle-free adapter file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields: dict[str, Any] = {
            "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
            "data_mode": np.asarray(self.mode),
            "coordinate_frame": np.asarray(self.coordinate_frame),
            "coverage": np.asarray(self.coverage),
            "conditions": self.conditions,
            "outgoing": self.outgoing,
            "condition_index": self.condition_index,
            "metadata_json": np.asarray(json.dumps(self.metadata, sort_keys=True, allow_nan=False)),
        }
        if self.weights is not None:
            fields["weights"] = self.weights
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
            ) as handle:
                temporary = handle.name
                np.savez_compressed(handle, **fields)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load_npz(cls, path: str | Path) -> "PhasePointCloud":
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "schema_version",
                "data_mode",
                "coordinate_frame",
                "coverage",
                "conditions",
                "outgoing",
                "condition_index",
            }
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"NPZ is missing required fields: {sorted(missing)}")
            version = archive["schema_version"]
            if (
                version.shape != ()
                or version.dtype.kind not in "iu"
                or int(version) != SCHEMA_VERSION
            ):
                raise ValueError(
                    f"unsupported NPZ schema_version; expected scalar {SCHEMA_VERSION}"
                )
            strings = {}
            for name in ("data_mode", "coordinate_frame", "coverage", "metadata_json"):
                if name not in archive:
                    continue
                value = archive[name]
                if value.shape != () or value.dtype.kind not in "US":
                    raise ValueError(f"{name} must be a scalar string")
                item = value.item()
                strings[name] = item.decode("utf-8") if isinstance(item, bytes) else str(item)
            return cls(
                conditions=archive["conditions"],
                outgoing=archive["outgoing"],
                condition_index=archive["condition_index"],
                mode=strings["data_mode"],  # type: ignore[arg-type]
                weights=archive["weights"] if "weights" in archive else None,
                metadata=json.loads(strings.get("metadata_json", "{}")),
                coordinate_frame=strings["coordinate_frame"],
                coverage=strings["coverage"],
            )


def split_conditions(
    num_conditions: int, validation_fraction: float, seed: int
) -> dict[str, list[int]]:
    """Split entire condition groups; a zero fraction explicitly disables holdout."""
    if num_conditions < 1:
        raise ValueError("at least one condition is required")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must lie in [0, 1)")
    if validation_fraction == 0:
        return {"train": list(range(num_conditions)), "validation": []}
    if num_conditions < 2:
        raise ValueError(
            "a single condition cannot be held out; set validation_fraction=0 for an explicit in-sample experiment"
        )
    count = min(num_conditions - 1, max(1, int(round(validation_fraction * num_conditions))))
    permutation = np.random.default_rng(seed).permutation(num_conditions)
    return {
        "train": sorted(permutation[count:].tolist()),
        "validation": sorted(permutation[:count].tolist()),
    }


class PointSampler:
    """Condition-balanced stochastic batches with checkpointable sampling state.

    In default ``mass`` mode, quadrature points are drawn from their normalized
    masses, and every loss weight is one. In ``uniform`` mode, points are drawn
    uniformly within each condition and weighted by N_c m_i. Both estimate the
    same objective with equal weight for each selected condition.
    """

    def __init__(
        self,
        cloud: PhasePointCloud,
        groups: list[int] | NDArray[Any],
        *,
        seed: int,
        point_sampling: Literal["mass", "uniform"] = "mass",
    ) -> None:
        self.cloud = cloud
        self.groups = np.asarray(groups, dtype=np.int64)
        if self.groups.ndim != 1 or len(self.groups) == 0:
            raise ValueError("sampler needs a nonempty one-dimensional group list")
        if (
            len(np.unique(self.groups)) != len(self.groups)
            or np.any(self.groups < 0)
            or np.any(self.groups >= cloud.num_conditions)
        ):
            raise ValueError("sampler groups must be distinct valid condition indices")
        if point_sampling not in ("mass", "uniform"):
            raise ValueError("point_sampling must be 'mass' or 'uniform'")
        self.point_sampling = point_sampling
        self.rng = np.random.default_rng(seed)

    def sample(
        self,
        batch_size: int,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> PointBatch:
        """``dtype`` controls conditions/weights; outgoing geometry is float64."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        group_ids = self.groups[self.rng.integers(len(self.groups), size=batch_size)]
        point_ids = np.empty(batch_size, dtype=np.int64)
        loss_weight = np.ones(batch_size, dtype=np.float64)
        for ci in np.unique(group_ids):
            positions = np.flatnonzero(group_ids == ci)
            indices = self.cloud._indices[int(ci)]
            masses = self.cloud._masses[int(ci)]
            if self.cloud.mode == "target_samples" or self.point_sampling == "uniform":
                local = self.rng.integers(len(indices), size=len(positions))
                if self.cloud.mode == "quadrature":
                    loss_weight[positions] = len(indices) * masses[local]
            else:
                local = np.searchsorted(
                    self.cloud._cdfs[int(ci)], self.rng.random(len(positions)), side="right"
                )
                local = np.minimum(local, len(indices) - 1)
            point_ids[positions] = indices[local]
        return PointBatch(
            outgoing=torch.as_tensor(
                self.cloud.outgoing[point_ids].copy(), dtype=torch.float64, device=device
            ),
            conditions=torch.as_tensor(
                self.cloud.conditions[group_ids].copy(), dtype=dtype, device=device
            ),
            condition_index=torch.as_tensor(group_ids, dtype=torch.long, device=device),
            loss_weight=torch.as_tensor(loss_weight, dtype=dtype, device=device),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "groups": self.groups.tolist(),
            "point_sampling": self.point_sampling,
            "rng": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (
            state["groups"] != self.groups.tolist()
            or state["point_sampling"] != self.point_sampling
        ):
            raise ValueError("sampler state does not match condition split or sampling mode")
        self.rng.bit_generator.state = state["rng"]


def iter_group_chunks(
    cloud: PhasePointCloud, condition: int, batch_size: int
) -> Iterator[tuple[NDArray[Any], NDArray[np.float64]]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    indices, masses = cloud._indices[condition], cloud._masses[condition]
    for start in range(0, len(indices), batch_size):
        stop = start + batch_size
        yield cloud.outgoing[indices[start:stop]], masses[start:stop]
