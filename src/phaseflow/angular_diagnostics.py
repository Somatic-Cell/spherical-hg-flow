"""Read-only angular diagnostics for a saved Rainbow CDF and a fitted flow.

The configurable theta band is a reporting region, never a cropped training
target or a model-selection objective. Target masses integrate the actual
piecewise-constant solid-angle cells. Profiles retain native cells and the
recorded source frame; their NF/HG values are point evaluations, not cell
averages. No log-density floor, target smoothing or density renormalization is
applied. The fitted model is evaluated on its existing device in bounded batches.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .plotting import _positive_pdf
from .training import atomic_json

if TYPE_CHECKING:
    from .rainbow import RainbowReference
    from .sphere_model import SingleConditionSphereFlow


ANGULAR_DIAGNOSTICS_SCHEMA = "phaseflow.angular_diagnostics.v1"
_ZERO_SCAN_ELEMENTS = 1 << 20
_ZERO_COLOR = "#d2d6dc"


@dataclass(frozen=True)
class AngularDiagnosticConfig:
    """Reporting coordinates and figure settings; none alter training."""

    theta_band_degrees: tuple[float, float] = (120.0, 150.0)
    source_phi_degrees: tuple[float, ...] = (-90.0, 0.0, 90.0)
    eval_batch_size: int = 4096
    dpi: int = 140

    def __post_init__(self) -> None:
        for name in ("theta_band_degrees", "source_phi_degrees"):
            value = getattr(self, name)
            if not isinstance(value, (list, tuple)) or any(
                isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                for x in value
            ):
                raise ValueError(f"{name} must be a finite numeric sequence")
            object.__setattr__(self, name, tuple(float(x) for x in value))
        band = self.theta_band_degrees
        if len(band) != 2 or not 0 <= band[0] < band[1] <= 180:
            raise ValueError("theta_band_degrees must satisfy 0 <= lower < upper <= 180")
        if not self.source_phi_degrees or any(
            not -180 <= phi <= 180 for phi in self.source_phi_degrees
        ):
            raise ValueError("source_phi_degrees must be nonempty and lie in [-180, 180]")
        for name in ("eval_batch_size", "dpi"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        low, high = _band_u(band)
        if not low < high:
            raise ValueError("theta band boundaries are indistinguishable in float64 u")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["theta_band_degrees"] = list(self.theta_band_degrees)
        result["source_phi_degrees"] = list(self.source_phi_degrees)
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AngularDiagnosticConfig:
        if not isinstance(value, dict):
            raise ValueError("angular diagnostic configuration must be an object")
        try:
            return cls(**value)
        except TypeError as error:
            raise ValueError(f"Invalid angular diagnostic configuration: {error}") from error


def _band_u(band: tuple[float, float]) -> tuple[float, float]:
    return tuple(math.sin(math.radians(theta) / 2) ** 2 for theta in band)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _conditional_band_mass(reference: RainbowReference, low: float, high: float) -> np.ndarray:
    """Stable difference of piecewise-linear saved CDF endpoint values.

    Evaluate partial boundary-cell masses separately: adding a tiny partial
    mass to a CDF already near one would lose it before the two CDF values
    could be subtracted. Complete interior cells telescope to a saved-edge
    difference. This remains O(N_phi), including bands crossing many cells.
    """
    first = int(np.searchsorted(reference.u_edges, low, side="right")) - 1
    last = int(np.searchsorted(reference.u_edges, high, side="left")) - 1
    edges = reference.u_edges
    first_cdf = reference.theta_cdf[:, first : first + 2].astype(np.longdouble)
    first_mass = first_cdf[:, 1] - first_cdf[:, 0]
    first_width = np.longdouble(edges[first + 1]) - edges[first]
    if first == last:
        return first_mass * ((np.longdouble(high) - low) / first_width)
    last_cdf = reference.theta_cdf[:, last : last + 2].astype(np.longdouble)
    last_mass = last_cdf[:, 1] - last_cdf[:, 0]
    last_width = np.longdouble(edges[last + 1]) - edges[last]
    first_fraction = (np.longdouble(edges[first + 1]) - low) / first_width
    last_fraction = (np.longdouble(high) - edges[last]) / last_width
    return (
        first_mass * first_fraction + (last_cdf[:, 0] - first_cdf[:, 1]) + last_mass * last_fraction
    )


def _reference_band_and_zeros(
    reference: RainbowReference, low: float, high: float
) -> dict[str, float]:
    """Saved-cell integrals with bounded temporary CDF-difference arrays."""
    marginal = np.diff(reference.phi_cdf.astype(np.longdouble))
    conditional_band = _conditional_band_mass(reference, low, high)
    mass = np.sum(marginal * conditional_band, dtype=np.longdouble)
    u_edges = reference.u_edges.astype(np.longdouble)
    delta_u = np.diff(u_edges)
    overlap_u = np.maximum(0, np.minimum(u_edges[1:], high) - np.maximum(u_edges[:-1], low))
    zero_area_fraction = np.longdouble(0)
    zero_band_area_fraction = np.longdouble(0)
    # A zero phi marginal makes its entire column zero, regardless of the
    # arbitrary normalized conditional CDF saved in that unused column.
    columns = min(reference.n_theta, _ZERO_SCAN_ELEMENTS)
    rows = max(1, _ZERO_SCAN_ELEMENTS // columns)
    for j in range(0, reference.n_phi, rows):
        end_j = min(reference.n_phi, j + rows)
        zero_marginal = marginal[j:end_j] == 0
        for i in range(0, reference.n_theta, columns):
            end_i = min(reference.n_theta, i + columns)
            differences = np.diff(reference.theta_cdf[j:end_j, i : end_i + 1], axis=1)
            zeros = (differences == 0) | zero_marginal[:, None]
            zero_phi_counts = zeros.sum(axis=0, dtype=np.int64)
            zero_area_fraction += np.sum(zero_phi_counts * delta_u[i:end_i], dtype=np.longdouble)
            zero_band_area_fraction += np.sum(
                zero_phi_counts * overlap_u[i:end_i], dtype=np.longdouble
            )
    return {
        "reference_band_mass_exact": float(mass),
        "reference_zero_solid_angle_fraction": float(zero_area_fraction / reference.n_phi),
        "reference_band_zero_solid_angle_fraction": float(
            zero_band_area_fraction / (reference.n_phi * (np.longdouble(high) - low))
        ),
    }


def _hg_band_probability(g: float, low: float, high: float) -> float:
    """H_g(cos(theta_low))-H_g(cos(theta_high)), with no tail subtraction.

    Rationalizing the difference of reciprocal square roots gives the formula
    below, including g=0 exactly. This is analytic HG, not a small-g expansion.
    """
    g64 = np.longdouble(g)
    a, b = 1 - g64, 1 + g64
    u = np.asarray([low, high], dtype=np.longdouble)
    denominator = np.sqrt(a * a + 4 * g64 * u) if g >= 0 else np.sqrt(b * b - 4 * g64 * (1 - u))
    probability = (
        a
        * b
        * 2
        * (np.longdouble(high) - low)
        / (denominator[0] * denominator[1] * denominator.sum())
    )
    return float(probability)


def _profiles(
    model: SingleConditionSphereFlow,
    reference: RainbowReference,
    config: AngularDiagnosticConfig,
) -> dict[str, np.ndarray]:
    phi_centers = -180 + (np.arange(reference.n_phi, dtype=np.float64) + 0.5) * (
        360 / reference.n_phi
    )
    requested = np.asarray(config.source_phi_degrees, dtype=np.float64)
    cells = np.asarray(
        [np.argmin(np.abs((phi_centers - phi + 180) % 360 - 180)) for phi in requested],
        dtype=np.int64,
    )
    actual = phi_centers[cells]
    edges = np.asarray(reference.u_edges)
    u = edges[:-1] / 2 + edges[1:] / 2
    one_minus_u = (1 - edges[:-1]) / 2 + (1 - edges[1:]) / 2
    radius = 2 * np.sqrt(u * one_minus_u)
    mu = one_minus_u - u
    phi = np.deg2rad(actual)
    theta_mid = np.rad2deg(2 * np.arctan2(np.sqrt(u), np.sqrt(one_minus_u)))
    theta_edges = np.rad2deg(2 * np.arctan2(np.sqrt(edges), np.sqrt(1 - edges)))
    with np.errstate(divide="ignore"):
        reference_log_pdf = (
            np.log(reference.phi_cdf[cells + 1] - reference.phi_cdf[cells])[:, None]
            + np.log(np.diff(reference.theta_cdf[cells], axis=1))
            - np.log((4 * math.pi / reference.n_phi) * np.diff(edges))[None, :]
        )
    a, b = 1 - reference.g, 1 + reference.g
    denominator_squared = (
        a * a + 4 * reference.g * u if reference.g >= 0 else b * b - 4 * reference.g * one_minus_u
    )
    hg_log_pdf = np.broadcast_to(
        math.log(a) + math.log(b) - math.log(4 * math.pi) - 1.5 * np.log(denominator_squared),
        reference_log_pdf.shape,
    ).copy()
    nf_log_pdf = np.empty(reference_log_pdf.size, dtype=np.float64)
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, nf_log_pdf.size, config.eval_batch_size):
                stop = min(nf_log_pdf.size, start + config.eval_batch_size)
                index = np.arange(start, stop, dtype=np.int64)
                j, i = index // reference.n_theta, index % reference.n_theta
                source = np.column_stack(
                    (radius[i] * np.cos(phi[j]), radius[i] * np.sin(phi[j]), mu[i])
                )
                directions = torch.as_tensor(
                    source @ reference.source_to_nf.T,
                    device=model.device,
                    dtype=getattr(model, "geometry_dtype", torch.float64),
                )
                value = model.log_prob(directions)
                if value.shape != (stop - start,):
                    raise ValueError("model log_prob must return one value per profile direction")
                nf_log_pdf[start:stop] = value.detach().to(dtype=torch.float64).cpu().numpy()
    finally:
        for module, mode in modes:
            module.training = mode
    return {
        "u_edges": edges.copy(),
        "theta_edges_degrees": theta_edges,
        "theta_midpoints_degrees": theta_mid,
        "requested_source_phi_degrees": requested,
        "actual_source_phi_degrees": actual,
        "phi_cell_indices": cells,
        "reference_log_pdf": reference_log_pdf,
        "nf_log_pdf": nf_log_pdf.reshape(reference_log_pdf.shape),
        "hg_log_pdf": hg_log_pdf,
    }


def _draw_profiles(
    profiles: dict[str, np.ndarray], config: AngularDiagnosticConfig, heading: str, path: Path
) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    teacher = _positive_pdf(profiles["reference_log_pdf"], name="Reference", allow_zero=True)
    nf = _positive_pdf(profiles["nf_log_pdf"], name="NF", allow_zero=False)
    hg = _positive_pdf(profiles["hg_log_pdf"], name="HG", allow_zero=False)
    columns = len(config.source_phi_degrees)
    figure = Figure(figsize=(4.7 * columns, 7.3))
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, columns, squeeze=False, sharey="row")
    edges, mid = profiles["theta_edges_degrees"], profiles["theta_midpoints_degrees"]
    for j in range(columns):
        for row, limits in enumerate(((0, 180), config.theta_band_degrees)):
            ax = axes[row, j]
            ax.stairs(
                np.where(teacher[j] > 0, teacher[j], np.nan),
                edges,
                baseline=None,
                color="#242424",
                linewidth=1.2,
            )
            ax.plot(mid, nf[j], color="#d55e00", linewidth=1.3)
            ax.plot(mid, hg[j], color="#0072b2", linestyle="--", linewidth=1.1)
            for i in np.flatnonzero(teacher[j] == 0):
                ax.axvspan(edges[i], edges[i + 1], color=_ZERO_COLOR, alpha=0.65, zorder=-1)
            ax.set_yscale("log")
            ax.set_xlim(*limits)
            ax.grid(visible=True, which="major", alpha=0.22)
            ax.set_xlabel("Scattering angle theta [degrees]")
            if j == 0:
                ax.set_ylabel(r"PDF [sr$^{-1}$], logarithmic scale")
            actual = profiles["actual_source_phi_degrees"][j]
            requested = profiles["requested_source_phi_degrees"][j]
            scope = "Full angular range" if row == 0 else "Reporting-band zoom"
            ax.set_title(
                f"{scope} | source phi = {actual:g} degrees\n(requested {requested:g})", fontsize=10
            )
    # Match limits within each row, using only cells intersecting its displayed
    # range. In particular the zoom is not flattened by an off-screen front lobe.
    for row, limits in enumerate(((0, 180), config.theta_band_degrees)):
        selected = (edges[:-1] < limits[1]) & (edges[1:] > limits[0])
        visible = np.concatenate(
            (teacher[:, selected].ravel(), nf[:, selected].ravel(), hg[:, selected].ravel())
        )
        positive = visible[visible > 0]
        low, high = float(positive.min()), float(positive.max())
        lower, upper = low / 1.4, high * 1.4
        axes[row, 0].set_ylim(lower if lower > 0 else low, upper if math.isfinite(upper) else high)
    legend = [
        Line2D([], [], color="#242424", label="Stored-CDF PDF (exact cell values)"),
        Line2D([], [], color="#d55e00", label="NF PDF (cell-midpoint evaluations)"),
        Line2D([], [], color="#0072b2", linestyle="--", label="HG base (original g)"),
    ]
    if np.any(teacher == 0):
        legend.append(Patch(facecolor=_ZERO_COLOR, label="Zero teacher PDF (no log floor)"))
    figure.legend(
        handles=legend,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.045),
        fontsize=9,
    )
    figure.suptitle(heading, fontsize=13, y=0.985)
    figure.text(
        0.5,
        0.016,
        "Native source-cell profiles; NF/HG curves can miss structure inside a cell. "
        "The reporting band does not crop the training target.",
        ha="center",
        fontsize=8,
    )
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.19, top=0.865, hspace=0.4, wspace=0.12)
    temporary = path.with_suffix(".png.tmp")
    figure.savefig(temporary, format="png", dpi=config.dpi, facecolor="white")
    temporary.replace(path)
    figure.clear()


def evaluate_angular_diagnostics(
    model: SingleConditionSphereFlow,
    reference: RainbowReference,
    output_directory: str | Path,
    *,
    train_samples: int,
    batch_size: int,
    config: AngularDiagnosticConfig | None = None,
    checkpoint_path: str | Path | None = None,
    objective_config=None,
) -> dict[str, Any]:
    """Save exact target band mass, expected counts, and aligned PDF cuts.

    This function consumes no train/validation/test samples and changes neither
    weights nor selection. Expected counts use the recorded training query
    distribution, not observed counts in a particular fixed pool. The optional
    checkpoint identity records the caller-supplied file; the caller must pass
    the file used to load ``model``.
    """
    config = AngularDiagnosticConfig() if config is None else config
    if objective_config is not None:
        from .log_objective import LogObjectiveConfig

        if not isinstance(objective_config, LogObjectiveConfig):
            raise ValueError("objective_config must be a LogObjectiveConfig or None")
        objective_config.validate_training_count(train_samples)
    if not isinstance(config, AngularDiagnosticConfig):
        raise ValueError("config must be an AngularDiagnosticConfig")
    for name, value in (("train_samples", train_samples), ("batch_size", batch_size)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    reference._require_open()
    if model.hg_g != reference.g or model.incident_cosine != float(reference.condition[1]):
        raise ValueError("Model HG g or incident condition does not match the reference record")
    checkpoint = None
    if checkpoint_path is not None:
        path = Path(checkpoint_path).resolve()
        checkpoint = {"path": str(path), "sha256": _file_hash(path)}
    low, high = _band_u(config.theta_band_degrees)
    metrics = _reference_band_and_zeros(reference, low, high)
    mass = metrics["reference_band_mass_exact"]
    sampling = "target" if objective_config is None else objective_config.sampling
    target_fraction = {"target": 1.0, "target_uniform": 0.5, "uniform": 0.0}[sampling]
    uniform_mass = high - low
    query_mass = target_fraction * mass + (1 - target_fraction) * uniform_mass
    metrics.update(
        expected_train_band_samples=train_samples * query_mass,
        expected_minibatch_band_samples=batch_size * query_mass,
        hg_band_probability_width=_hg_band_probability(reference.g, low, high),
    )
    if objective_config is not None:
        metrics.update(
            uniform_band_mass_exact=uniform_mass,
            training_query_band_mass=query_mass,
            expected_cdf_train_band_samples=train_samples * target_fraction * mass,
            expected_uniform_train_band_samples=train_samples * (1 - target_fraction) * uniform_mass,
        )
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("nonfinite angular diagnostic statistic")
    profiles = _profiles(model, reference, config)
    # Validate every curve before creating any artifact, retaining true zero
    # teacher cells while rejecting nonfinite model values and PDF underflow.
    for name in ("reference", "nf", "hg"):
        _positive_pdf(profiles[f"{name}_log_pdf"], name=name, allow_zero=name == "reference")
    metadata = reference.metadata
    synthetic = metadata.get("synthetic") is True or any(
        "synthetic" in str(metadata.get(key, "")).lower() for key in ("purpose", "source_commit")
    )
    kind = "Synthetic fixture" if synthetic else "Rainbow CDF record"
    heading = (
        f"{kind} | {reference.wavelength_nm:g} nm; incident inclination "
        f"{metadata['incident_inclination_degrees']:g} degrees\nAngular diagnostics"
    )
    if checkpoint is not None:
        heading += f" | checkpoint {Path(checkpoint['path']).name}"
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    png, arrays = output / "angular_profiles.png", output / "angular_profiles.npz"
    _draw_profiles(profiles, config, heading, png)
    temporary = arrays.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **profiles)
    temporary.replace(arrays)
    configuration = config.to_dict()
    serialized = json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False)
    model_config = getattr(model, "config", None)
    report: dict[str, Any] = {
        "schema": (ANGULAR_DIAGNOSTICS_SCHEMA if objective_config is None
                   else "phaseflow.angular_diagnostics.v2"),
        **metrics,
        "scope": "report_only_same_condition_no_model_selection",
        "source_kind": "synthetic_fixture" if synthetic else "rainbow_cdf_record",
        "configuration": configuration,
        "configuration_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "implementation_sha256": _file_hash(Path(__file__)),
        "dataset_fingerprint": reference.fingerprint(),
        "data_provenance": reference.provenance,
        "checkpoint": checkpoint,
        "condition": {
            "wavelength_nm": reference.wavelength_nm,
            "incident_cosine": float(reference.condition[1]),
            "incident_inclination_degrees": metadata["incident_inclination_degrees"],
            "hg_g": reference.g,
        },
        "model": {
            "family": getattr(model, "model_family", type(model).__name__),
            "configuration": model_config.to_dict() if model_config is not None else None,
            "device": str(model.device),
            "parameter_dtype": str(model.dtype),
            "density_evaluation_dtype": str(getattr(model, "geometry_dtype", torch.float64)),
        },
        "coordinates": {
            "frame": "recorded_solver_source",
            "density_measure": "solid_angle_sr",
            "theta_zero": "forward",
            "theta_pi": "backward",
            "source_to_nf": reference.source_to_nf.tolist(),
            "u": "(1-cos(theta))/2",
            "solid_angle": "4*pi*du*dv_source",
        },
        "band": {
            "theta_degrees": list(config.theta_band_degrees),
            "u_boundaries": [low, high],
            "source_phi_degrees": [-180.0, 180.0],
            "solid_angle_sr": 4 * math.pi * (high - low),
            "target_mass_method": "saved_phi_marginal_times_exact_piecewise_linear_u_CDF_difference",
            "exactness": "exact_saved_cell_integral_up_to_floating_point_roundoff",
            "hg_method": "analytic_HG_CDF_difference_rationalized_without_tail_subtraction",
            "zero_fraction_denominator": "solid_angle_of_the_reporting_band",
            "band_choice": "configurable_reporting_region_not_a_certified_rainbow_boundary",
            "nf_band_mass": None,
            "conditional_kl": None,
        },
        "expected_counts": {
            "train_samples": train_samples,
            "batch_size": batch_size,
            "method": {
                "target": "N*p(R) and B*p(R), not observed counts in a fixed pool",
                "target_uniform": (
                    "N*(0.5*p(R)+0.5*u(R)) and B*(0.5*p(R)+0.5*u(R)); "
                    "unconditional expectations, not observed counts"
                ),
                "uniform": (
                    "N*u(R) and B*u(R); uniform solid-angle queries, "
                    "not observed counts in a fixed pool"
                ),
            }[sampling],
            "observed_train_band_samples": None,
            **({"sampling": objective_config.sampling,
                "target_fraction": target_fraction,
                "cdf_samples": int(train_samples * target_fraction),
                "uniform_samples": int(train_samples * (1 - target_fraction))}
               if objective_config is not None else {}),
        },
        "profiles": {
            "requested_source_phi_degrees": profiles["requested_source_phi_degrees"].tolist(),
            "actual_source_phi_degrees": profiles["actual_source_phi_degrees"].tolist(),
            "phi_cell_indices": profiles["phi_cell_indices"].tolist(),
            "phi_choice": "nearest_periodic_native_cell_center_ties_choose_lowest_cell_index",
            "array_order": ["requested_phi_profile", "native_theta_cell"],
            "theta_count": reference.n_theta,
            "reference_pdf": "exact_saved_cell_mass_divided_by_cell_solid_angle",
            "nf_and_hg_pdf": "point_evaluation_at_native_u_midpoints_not_cell_averages",
            "no_within_cell_resolution_claim": True,
            "density_floor": None,
            "zero_log_pdf": "negative_infinity_in_NPZ_omitted_curve_with_gray_cell_span_in_figure",
            "zero_reference_cells_per_profile": np.isneginf(profiles["reference_log_pdf"])
            .sum(axis=1)
            .tolist(),
        },
        "artifacts": {
            "figure": {"path": png.name, "sha256": _file_hash(png)},
            "arrays": {"path": arrays.name, "sha256": _file_hash(arrays)},
            "report": {"path": "angular_diagnostics.json"},
        },
    }
    atomic_json(output / "angular_diagnostics.json", report)
    return report
