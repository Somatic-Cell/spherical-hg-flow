"""Validated access to one ``rainbow`` solver CDF record.

The teacher is the saved, piecewise constant *solid-angle* density.  This module
does not filter, renormalize, floor, fold, or otherwise change that density.
It implements the public ``rainbow.phase_cdf.numpy.v2`` coordinate contract at
https://github.com/Somatic-Cell/rainbow/tree/a9538941dcb9df2de6a4130fa66f625f0133c949
without importing or executing code from the data producer.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

REFERENCE_SCHEMA = "phaseflow.rainbow_reference.v1"
RAINBOW_SCHEMA = "rainbow.phase_cdf.numpy.v2"
_FILES = ("metadata.json", "phi_cdf.npy", "theta_given_phi_cdf.npy", "u_edges.npy")
_BLOCK_ELEMENTS = 1 << 20
_FRAME_TOLERANCE = 2e-12


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate metadata key: {key}")
        result[key] = value
    return result


def _invalid_json_constant(value: str) -> Any:
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _finite_scalar(value: Any, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite real scalar") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _real_array(value: ArrayLike, width: int, name: str) -> NDArray[np.float64]:
    array = np.asarray(value)
    if array.ndim < 1 or array.shape[-1] != width or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real array of shape [...,{width}]")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


class RainbowReference:
    """One complete, validated solver record and its reference distribution.

    ``condition`` is ``[wavelength_nm, dot(shape_axis, incident)]``; ``g`` is
    ``metadata.hg.g``, checked against the exact first moment of the saved cells.
    ``frame`` retains the recorded local-to-particle matrix byte-for-value.
    ``nf_frame`` has incident +z and the projected oriented particle axis +x.
    Only an exactly axial incident direction uses the recorded e1 as fallback x.

    ``sample`` takes two independent uniform variates in [0,1), ordered as
    [phi marginal, u conditional on phi].  CDF plateaus are skipped by a right
    generalized inverse.  Cell boundaries use their right-hand cell, except the
    final u boundary, which uses the final cell.  The phi seam belongs to the
    first phi cell.  At an explicitly requested pole, source phi=0 is the PDF
    representative; the returned PDF uses the same rule as ``log_prob``.
    At an explicit zero variate, a measure-zero boundary representative may have
    zero PDF (including a neighboring cell selected by angular roundoff). An
    interior quantile that numerically collapses to a pole or returns a
    zero-density direction is rejected instead of moving the direction or
    inventing a positive PDF.

    All directions and log PDFs are float64; PDFs are per steradian.  CDF
    validation and inversion use bounded temporaries, never an N-sample by
    N-theta gather.  Mapped arrays must not be used after ``close`` and source
    files must not be modified while the reference is open.
    """

    def __init__(
        self, directory: str | Path, *, validation_block_elements: int = _BLOCK_ELEMENTS
    ) -> None:
        self.directory = Path(directory).resolve()
        if self.directory.name.endswith(".part"):
            raise ValueError("An incomplete .part directory is not a dataset record")
        if type(validation_block_elements) is not int or validation_block_elements < 1:
            raise ValueError("validation_block_elements must be a positive integer")
        self._closed = False
        self._mapped_arrays: list[np.memmap] = []
        try:
            metadata_path = self.directory / "metadata.json"
            if metadata_path.stat().st_size > 1024 * 1024:
                raise ValueError("Unexpectedly large metadata.json")
            metadata_bytes = metadata_path.read_bytes()
            self._metadata = json.loads(
                metadata_bytes,
                object_pairs_hook=_json_object,
                parse_constant=_invalid_json_constant,
            )
            self._validate_metadata()
            self.phi_cdf = self._load_array("phi_cdf.npy", (self.n_phi + 1,))
            self.theta_cdf = self._load_array(
                "theta_given_phi_cdf.npy", (self.n_phi, self.n_theta + 1)
            )
            self.u_edges = self._load_array("u_edges.npy", (self.n_theta + 1,))
            self._theta_flat = self.theta_cdf.reshape(-1)
            self._validate_cdfs(validation_block_elements)
            self._make_frames()
            self._make_provenance(metadata_bytes)
        except BaseException:
            # In particular, release rejected NPYs on Windows before propagating
            # the error.  A partially initialized object owns its mapped files.
            self.close()
            raise

    def __enter__(self) -> RainbowReference:
        self._require_open()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            for array in self._mapped_arrays:
                array._mmap.close()
            self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("RainbowReference has been closed")

    @property
    def metadata(self) -> dict[str, Any]:
        """Unmodified generator metadata, including its physical-quality fields."""
        return copy.deepcopy(self._metadata)

    @property
    def provenance(self) -> dict[str, Any]:
        return copy.deepcopy(self._provenance)

    def fingerprint(self) -> str:
        """Path-independent SHA-256 over file identities and adapter semantics."""
        return self._fingerprint

    def summary(self) -> dict[str, Any]:
        return {
            "schema": REFERENCE_SCHEMA,
            "condition": self.condition.tolist(),
            "g": self.g,
            "theta_count": self.n_theta,
            "phi_count": self.n_phi,
            "fingerprint": self.fingerprint(),
            "provenance": self.provenance,
        }

    @staticmethod
    def _dimension(value: Any, name: str) -> int:
        if type(value) is not int or not 1 <= value <= 0x7FFFFFFF:
            raise ValueError(f"{name} must be an angular dimension in [1, 2**31-1]")
        return value

    def _validate_metadata(self) -> None:
        m = self._metadata
        if not isinstance(m, dict):
            raise ValueError("metadata.json must contain a JSON object")
        if m.get("schema") != RAINBOW_SCHEMA or m.get("complete") is not True:
            raise ValueError("Unsupported Rainbow schema or incomplete record")
        for key, expected in {
            "dtype": "<f8",
            "order": "C",
            "density_measure": "solid_angle_sr",
            "cell_model": "constant_density_per_spherical_cell",
            "coordinates": "u=(1-cos(theta))/2; v=(phi+pi)/(2*pi)",
            "normalization": "integral_p_domega_equals_one",
            "input_polarization": "unpolarized",
            "coordinate_contract_id": "rainbow.phase_cdf.coordinates.v1",
            "direction_convention": "physical_propagation",
            "particle_frame_handedness": "right",
            "sampling_frame_layout": "rows_xyz_columns_e0_e1_k",
            "sampling_frame_map": "local_column_to_particle_column",
            "theta_zero": "forward",
            "theta_pi": "backward",
        }.items():
            if m.get(key) != expected:
                raise ValueError(f"Unsupported {key}: {m.get(key)!r}")
        if m.get("azimuth_periodic") is not True:
            raise ValueError("azimuth_periodic must be true")
        if m.get("cdf_axis_order") != ["phi_cell", "theta_edge"]:
            raise ValueError("Unsupported cdf_axis_order")
        if m.get("shape_polar_axis") != [0, -1, 0] or m.get("particle_up_axis") != [0, 1, 0]:
            raise ValueError("Unsupported oriented particle axes")
        phi_range = np.asarray(m.get("phi_range_rad"), dtype=np.float64)
        if phi_range.shape != (2,) or not np.allclose(
            phi_range, [-np.pi, np.pi], rtol=0, atol=2e-15
        ):
            raise ValueError("phi_range_rad must cover [-pi, pi]")
        self.n_theta = self._dimension(m.get("theta_count"), "theta_count")
        self.n_phi = self._dimension(m.get("phi_count"), "phi_count")
        self.wavelength_nm = _finite_scalar(m.get("wavelength_nm"), "wavelength_nm")
        if self.wavelength_nm <= 0:
            raise ValueError("wavelength_nm must be positive")
        material = m.get("material")
        if not isinstance(material, dict) or material.get("wavelength_convention") != "vacuum_nm":
            raise ValueError("The wavelength convention must be vacuum_nm")
        hg = m.get("hg")
        if not isinstance(hg, dict):
            raise ValueError("Missing HG metadata")
        for key, expected in {
            "method": "first_moment_of_saved_cell_pdf",
            "target": "saved_cdf",
            "cosine_convention": "dot(incident_propagation,outgoing_propagation)",
        }.items():
            if hg.get(key) != expected:
                raise ValueError(f"Unsupported HG {key}")
        self.g = _finite_scalar(hg.get("g"), "hg.g")
        if not -1 < self.g < 1:
            raise ValueError("hg.g must lie strictly between -1 and 1; clipping is not allowed")

    def _load_array(self, name: str, shape: tuple[int, ...]) -> np.memmap:
        path = self.directory / name
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if isinstance(array, np.memmap):
            self._mapped_arrays.append(array)
        elif isinstance(array, np.lib.npyio.NpzFile):
            array.close()
        if (
            not isinstance(array, np.memmap)
            or array.dtype.str != "<f8"
            or array.shape != shape
            or not array.flags.c_contiguous
        ):
            raise ValueError(f"Unexpected shape/dtype/order in {name}")
        if path.stat().st_size != array.offset + array.nbytes:
            raise ValueError(f"Truncated or trailing data in {name}")
        return array

    def _validate_cdfs(self, block_elements: int) -> None:
        edges = self.u_edges
        if (
            not np.isfinite(edges).all()
            or edges[0] != 0
            or edges[-1] != 1
            or not np.all(np.diff(edges) > 0)
        ):
            raise ValueError("u_edges must increase strictly from 0 to 1")
        marginal = self.phi_cdf
        if (
            not np.isfinite(marginal).all()
            or marginal[0] != 0
            or marginal[-1] != 1
            or np.any(np.diff(marginal) < 0)
        ):
            raise ValueError("Invalid marginal phi CDF")
        stride = self.n_theta + 1
        for start in range(0, self._theta_flat.size, block_elements):
            stop = min(start + block_elements, self._theta_flat.size)
            values = self._theta_flat[start:stop]
            if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
                raise ValueError("Conditional CDF contains nonfinite or out-of-range data")
            positions = np.arange(start, stop, dtype=np.int64)
            in_row = positions % stride
            if np.any(values[in_row == 0] != 0) or np.any(values[in_row == self.n_theta] != 1):
                raise ValueError("Conditional CDF endpoints must be 0 and 1")
            check = in_row != 0
            if np.any(values[check] < self._theta_flat[positions[check] - 1]):
                raise ValueError("Conditional CDF decreases; refusing to repair it")
        mass, moment = self.mass_and_g(block_elements=block_elements)
        if abs(mass - 1.0) > 1e-10 or abs(moment - self.g) > 5e-12:
            raise ValueError("HG first moment is inconsistent with the saved CDF")
        self._mass_check = mass
        self._moment_check = moment

    def mass_and_g(self, *, block_elements: int = _BLOCK_ELEMENTS) -> tuple[float, float]:
        """Exact saved-cell mass and first moment, without point sampling."""
        self._require_open()
        if type(block_elements) is not int or block_elements <= 0:
            raise ValueError("block_elements must be a positive integer")
        mean_mu = 1.0 - (self.u_edges[:-1] + self.u_edges[1:])
        mass = np.longdouble(0)
        axial = np.longdouble(0)
        for j in range(self.n_phi):
            marginal_mass = np.longdouble(self.phi_cdf[j + 1] - self.phi_cdf[j])
            for start in range(0, self.n_theta, block_elements):
                stop = min(start + block_elements, self.n_theta)
                weights = (
                    np.diff(self.theta_cdf[j, start : stop + 1]).astype(np.longdouble)
                    * marginal_mass
                )
                mass += weights.sum(dtype=np.longdouble)
                axial += (weights * mean_mu[start:stop]).sum(dtype=np.longdouble)
        if not np.isfinite(mass) or mass <= 0 or not np.isfinite(axial):
            raise ValueError("Invalid reconstructed probability mass")
        return float(mass), float(axial / mass)

    def _make_frames(self) -> None:
        m = self._metadata
        self.frame = np.asarray(m.get("sampling_frame_columns"), dtype=np.float64)
        if (
            self.frame.shape != (3, 3)
            or not np.isfinite(self.frame).all()
            or not np.allclose(self.frame.T @ self.frame, np.eye(3), rtol=0, atol=_FRAME_TOLERANCE)
            or not np.isclose(np.linalg.det(self.frame), 1.0, rtol=0, atol=_FRAME_TOLERANCE)
        ):
            raise ValueError("Sampling frame must be a right-handed orthonormal matrix")
        incident = self.frame[:, 2]
        hg_axis = np.asarray(m["hg"].get("axis_particle_frame"), dtype=np.float64)
        if hg_axis.shape != (3,) or not np.allclose(
            hg_axis, incident, rtol=0, atol=_FRAME_TOLERANCE
        ):
            raise ValueError("HG axis is inconsistent with the sampling frame incident direction")
        inclination = _finite_scalar(
            m.get("incident_inclination_degrees"), "incident_inclination_degrees"
        )
        if not -90 <= inclination <= 90:
            raise ValueError("incident_inclination_degrees must lie in [-90,90]")
        alpha = np.deg2rad(inclination)
        expected_incident = np.array([np.cos(alpha), -np.sin(alpha), 0.0])
        # The solver first stores an FP32 axis; M records its FP64 normalization.
        # Use M for geometry and conditioning, never overwrite it by the ideal axis.
        if not np.allclose(incident, expected_incident, rtol=0, atol=2e-7):
            raise ValueError("Incident inclination is inconsistent with the sampling frame")
        eta = -incident[1]
        if not -1 <= eta <= 1:
            raise ValueError("Recorded incident cosine is outside [-1,1]")
        self.condition = np.array([self.wavelength_nm, eta], dtype=np.float64)
        # Projection in the *recorded* transverse basis is stable near axial
        # incidence and introduces only a 2D rotation, without z-axis mixing.
        transverse = -self.frame[1, :2]
        transverse_length = np.hypot(transverse[0], transverse[1])
        if transverse_length == 0:
            self._rotation_cos, self._rotation_sin = 0.0, 1.0
        else:
            self._rotation_cos, self._rotation_sin = transverse / transverse_length
        c, s = self._rotation_cos, self._rotation_sin
        self.source_to_nf = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
        self.nf_frame = self.frame @ self.source_to_nf.T
        for array in (self.frame, self.condition, self.source_to_nf, self.nf_frame):
            array.setflags(write=False)

    def _make_provenance(self, metadata_bytes: bytes) -> None:
        files = {}
        for name in _FILES:
            path = self.directory / name
            if name == "metadata.json":
                digest = hashlib.sha256(metadata_bytes)
            else:
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    while block := stream.read(1024 * 1024):
                        digest.update(block)
            files[name] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
        semantics = {
            "schema": REFERENCE_SCHEMA,
            "files": files,
            "source_to_nf": self.source_to_nf.tolist(),
            "uniform_order": ["phi_marginal", "u_given_phi"],
            "uniform_domain": "[0,1)",
            "pole_pdf_representative": "source_phi_zero",
            "density_measure": "solid_angle_sr",
        }
        payload = json.dumps(semantics, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self._fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        self._provenance = {
            **semantics,
            "record_directory": str(self.directory),
            "source_repository": "https://github.com/Somatic-Cell/rainbow",
            "metadata": copy.deepcopy(self._metadata),
            "saved_mass_check": self._mass_check,
            "saved_g_check": self._moment_check,
            "nf_frame_columns": self.nf_frame.tolist(),
        }

    def _conditional_cells(
        self, phi_cells: NDArray[np.int64], quantiles: NDArray[np.float64]
    ) -> NDArray[np.int64]:
        # Largest i for which T[j,i] <= r, skipping every zero-mass plateau.
        # Bounds and each gather are O(number of samples), even for long rows.
        lower = np.zeros(quantiles.shape, dtype=np.int64)
        upper = np.full(quantiles.shape, self.n_theta, dtype=np.int64)
        while np.any(lower < upper):
            middle = (lower + upper + 1) // 2
            move_right = self.theta_cdf[phi_cells, middle] <= quantiles
            lower = np.where(move_right, middle, lower)
            upper = np.where(move_right, upper, middle - 1)
        return lower

    def sample(self, uniforms: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Inverse the saved CDF using [phi marginal, conditional u] in [0,1)."""
        self._require_open()
        uniforms = _real_array(uniforms, 2, "uniforms")
        if np.any(uniforms < 0) or np.any(uniforms >= 1):
            raise ValueError("uniforms must lie in [0,1); endpoints are not clamped")
        shape = uniforms.shape[:-1]
        values = uniforms.reshape(-1, 2)
        phi_cells = np.searchsorted(self.phi_cdf, values[:, 0], side="right") - 1
        phi_fraction = (values[:, 0] - self.phi_cdf[phi_cells]) / (
            self.phi_cdf[phi_cells + 1] - self.phi_cdf[phi_cells]
        )
        theta_cells = self._conditional_cells(phi_cells, values[:, 1])
        lower_cdf = self.theta_cdf[phi_cells, theta_cells]
        upper_cdf = self.theta_cdf[phi_cells, theta_cells + 1]
        theta_fraction = (values[:, 1] - lower_cdf) / (upper_cdf - lower_cdf)
        u = self.u_edges[theta_cells] + theta_fraction * (
            self.u_edges[theta_cells + 1] - self.u_edges[theta_cells]
        )
        if np.any((values[:, 1] > 0) & ((u <= 0) | (u >= 1))):
            raise FloatingPointError("An interior CDF quantile rounded to a pole")
        phi = -np.pi + 2 * np.pi * ((phi_cells + phi_fraction) / self.n_phi)
        radius = 2 * np.sqrt(u) * np.sqrt(1 - u)
        source_x, source_y = radius * np.cos(phi), radius * np.sin(phi)
        c, s = self._rotation_cos, self._rotation_sin
        directions = np.stack(
            (c * source_x + s * source_y, -s * source_x + c * source_y, 1 - 2 * u), axis=-1
        ).reshape(*shape, 3)
        # Evaluate the actual returned direction to share the seam/pole/boundary
        # representative with arbitrary queries, including explicit zero uniforms.
        log_pdf = self.log_prob(directions)
        interior = np.all(values > 0, axis=-1).reshape(shape)
        if np.any(interior & ~np.isfinite(log_pdf)):
            raise FloatingPointError("An interior CDF quantile rounded into a zero-density cell")
        return directions, log_pdf

    def log_prob(self, outgoing_nf: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the saved solid-angle density at arbitrary NF-frame directions.

        Unit-vector representation error up to 2e-5 is accepted, as in model
        inference. Angles use the direction's scale-invariant geometry in double
        precision; the caller's directions and the saved distribution are unchanged.
        """
        self._require_open()
        directions = _real_array(outgoing_nf, 3, "outgoing_nf")
        x, y, z = np.moveaxis(directions, -1, 0)
        radial = np.hypot(x, y)
        norm = np.hypot(radial, z)
        if np.any(np.abs(norm - 1) > 2e-5):
            raise ValueError("outgoing_nf must contain unit vectors within 2e-5")
        radial, mu = radial / norm, z / norm
        # Stable half-angle formula preserves a small polar angle even when mu
        # has rounded to +1 but nonzero transverse components retain that angle.
        half_angle = 0.5 * radial * (radial / (1 + np.abs(mu)))
        u = np.where(mu >= 0, half_angle, 1 - half_angle)
        c, s = self._rotation_cos, self._rotation_sin
        source_x, source_y = c * x - s * y, s * x + c * y
        phi = np.arctan2(source_y, source_x)
        phi = np.where(radial == 0, 0.0, phi)
        v = np.remainder((phi + np.pi) / (2 * np.pi), 1.0)
        phi_cells = np.minimum((v * self.n_phi).astype(np.int64), self.n_phi - 1)
        theta_cells = np.searchsorted(self.u_edges, u, side="right") - 1
        theta_cells = np.minimum(theta_cells, self.n_theta - 1)
        phi_mass = self.phi_cdf[phi_cells + 1] - self.phi_cdf[phi_cells]
        conditional_mass = (
            self.theta_cdf[phi_cells, theta_cells + 1] - self.theta_cdf[phi_cells, theta_cells]
        )
        delta_u = self.u_edges[theta_cells + 1] - self.u_edges[theta_cells]
        # Separate logarithms retain very small positive masses; true zero cells
        # return -inf and are never replaced by an arbitrary density floor.
        with np.errstate(divide="ignore"):
            return np.asarray(
                np.log(phi_mass)
                + np.log(conditional_mass)
                - np.log(4 * np.pi)
                + np.log(self.n_phi)
                - np.log(delta_u)
            )
