"""Read-only, finite-sample checks against the stored spherical-cell teacher.

Histograms aggregate exact cell masses; they do not smooth or change the teacher
used for training. Confidence intervals are per-region descriptive intervals,
not simultaneous coverage, a global goodness-of-fit test, or model selection.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .rainbow import RainbowReference
from .single_condition import _array_hash, _points, load_single_checkpoint
from .training_scatter import resolve_training_scatter

SCHEMA = "phaseflow.training_sampling_audit.v1"
_REGIONS = (
    ("rainbow_band", (120.0, 150.0), (-180.0, 180.0)),
    ("theta_150_neighborhood", (149.0, 151.0), (-180.0, 180.0)),
    ("theta_90_phi_0_neighborhood", (85.0, 95.0), (-5.0, 5.0)),
    ("theta_90_phi_0_one_degree", (89.5, 90.5), (-0.5, 0.5)),
)
_HYPOTHETICAL_COUNTS = (65536, 262144, 1048576)


def _overlap_fractions(old_edges: np.ndarray, new_edges: np.ndarray) -> np.ndarray:
    lower = np.maximum(new_edges[:-1, None], old_edges[None, :-1])
    upper = np.minimum(new_edges[1:, None], old_edges[None, 1:])
    return np.maximum(upper - lower, 0.0) / np.diff(old_edges)[None, :]


def integrated_histogram(
    reference: RainbowReference, u_edges: np.ndarray, phi_edges_degrees: np.ndarray,
) -> np.ndarray:
    """Integrate rectangular bins by saved-cell overlap, not PDF quadrature.

Returns probability masses indexed [u_bin, phi_bin]. Angular edges are in the
recorded source frame. Phi intervals must be within [-180,180]; split any
seam-crossing interval explicitly. Source u edges may be nonuniform.
    """
    reference._require_open()
    u_edges, phi_edges = np.asarray(u_edges), np.asarray(phi_edges_degrees)
    for edges, low, high in ((u_edges, 0, 1), (phi_edges, -180, 180)):
        if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
                or np.any(np.diff(edges) <= 0) or edges[0] < low or edges[-1] > high):
            raise ValueError("Histogram edges must be finite, increasing, and inside the domain")
    u_fraction = _overlap_fractions(reference.u_edges, u_edges)
    phi_fraction = _overlap_fractions(
        np.linspace(-180.0, 180.0, reference.n_phi + 1), phi_edges,
    )
    result = np.zeros((len(phi_edges) - 1, len(u_edges) - 1), dtype=np.float64)
    marginal = np.diff(reference.phi_cdf)
    # Bounded source-table temporaries, including for large producer grids.
    for start in range(0, reference.n_phi, 128):
        stop = min(start + 128, reference.n_phi)
        mass = marginal[start:stop, None] * np.diff(reference.theta_cdf[start:stop], axis=1)
        result += phi_fraction[:, start:stop] @ (mass @ u_fraction.T)
    return result.T


def _coordinates(directions: np.ndarray, reference: RainbowReference):
    source = directions.astype(np.float64) @ reference.source_to_nf
    radial = np.hypot(source[:, 0], source[:, 1])
    theta = np.arctan2(radial, source[:, 2])
    phi = (np.rad2deg(np.arctan2(source[:, 1], source[:, 0])) + 180) % 360 - 180
    return np.rad2deg(theta), phi, np.sin(theta / 2) ** 2


def _wilson_interval(count: int, total: int) -> list[float]:
    z = 1.959963984540054
    estimate, correction = count / total, z * z / total
    center = (estimate + correction / 2) / (1 + correction)
    half = z * math.sqrt(estimate * (1 - estimate) / total + z * z / (4 * total**2))
    half /= 1 + correction
    return [0.0 if count == 0 else center - half, 1.0 if count == total else center + half]


def _region_statistics(probability: float, count: int, total: int) -> dict[str, Any]:
    expected = total * probability
    sigma = math.sqrt(total * probability * (1 - probability))
    return {
        "observed_count": count,
        "observed_fraction": count / total,
        "expected_count": expected,
        "binomial_count_standard_deviation": sigma,
        "standardized_count_residual": (count - expected) / sigma if sigma > 0 else None,
        "wilson_95_percent_interval": _wilson_interval(count, total),
    }


def _atomic_write(path: Path, writer) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".audit-", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("wb") as stream:
            writer(stream)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _plot_histogram(path: Path, u_edges, phi_edges, mass, observed, total, source_kind) -> None:
    from matplotlib import colormaps
    from matplotlib.colors import LogNorm
    from matplotlib.figure import Figure

    from .plotting import _save_figure

    solid_angles = 2 * np.diff(u_edges)[:, None] * np.deg2rad(np.diff(phi_edges))[None, :]
    densities = (mass / solid_angles, observed / (total * solid_angles))
    positive = np.concatenate([value[value > 0] for value in densities])
    low, high = float(positive.min()), float(positive.max())
    if low == high:
        low, high = low / 2, high * 2
    figure = Figure(figsize=(16, 5), constrained_layout=True)
    axes = figure.subplots(1, 3)
    cmap = colormaps["viridis"].with_extremes(bad="#d2d6dc")
    for axis, values, title in zip(axes[:2], densities, (
        "Exact teacher mass / solid angle", f"Fixed-pool histogram (N = {total:,})",
    ), strict=True):
        image = axis.pcolormesh(phi_edges, u_edges, np.ma.masked_equal(values, 0),
                               cmap=cmap, norm=LogNorm(low, high), shading="flat")
        axis.set_title(title)
    figure.colorbar(image, ax=list(axes[:2]), location="bottom", label="PDF [sr^-1], log scale")
    expected, variance = total * mass, total * mass * (1 - mass)
    supported = (expected >= 5) & (total - expected >= 5)
    residual = np.divide(observed - expected, np.sqrt(variance),
                         out=np.zeros_like(mass), where=variance > 0)
    limit = max(1.0, float(np.max(np.abs(residual[supported]), initial=0)))
    image = axes[2].pcolormesh(
        phi_edges, u_edges, np.ma.masked_where(~supported, residual), shading="flat",
        cmap=colormaps["coolwarm"].with_extremes(bad="#d2d6dc"), vmin=-limit, vmax=limit,
    )
    axes[2].set_title("Per-bin standardized count residual")
    figure.colorbar(image, ax=axes[2], location="bottom", label="(K - N p) / sqrt(N p (1-p))")
    for axis in axes:
        axis.set(xlabel="Source azimuth phi [degrees]", ylabel="u = (1 - cos(theta)) / 2",
                 xlim=(-180, 180), ylim=(1, 0))
    figure.suptitle(
        f"{source_kind} | Equal-solid-angle bins; FP64 counts before the training cast\n"
        "Gray PDF bins: zero density/count. Gray residuals: expected count or complement < 5.\n"
        "Coarse counts are a sampler diagnostic; no smoothing or global pass/fail test."
    )
    _save_figure(figure, path, dpi=140, write_pdf=False)


def audit_training_sampling(
    reference: RainbowReference, run_directory: str | Path, output_directory: str | Path,
    *, device: str = "cuda", u_bins: int = 32, phi_bins: int = 64, plot: bool = True,
) -> dict[str, Any]:
    """Audit every entry of the verified pool without altering the saved run.

Default ROIs are descriptive neighborhoods, not extracted physical feature
boundaries. Binomial statistics refer to the pre-cast sampler; post-cast counts
are reported separately because rounding can move points across boundaries.
Existing output may be replaced only when its recorded audit identity matches.
A running identity record is saved before artifacts, allowing an interrupted
audit to retry without treating incomplete output as a completed measurement.
    """
    for name, value in (("u_bins", u_bins), ("phi_bins", phi_bins)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(plot) is not bool:
        raise ValueError("plot must be boolean")
    run, output = Path(run_directory).resolve(), Path(output_directory).resolve()
    if output == run or run.is_relative_to(output) or output.is_relative_to(run):
        raise ValueError("Audit output must be a separate directory with a disjoint tree from the run")
    model, _ = load_single_checkpoint(run / "best.pt", device=device)
    pool = resolve_training_scatter(model, reference, run / "best.pt", run_directory=run)
    count, data_seed = pool.provenance["sample_count"], pool.provenance["data_seed"]
    points = _points(reference, count, data_seed, "train")
    if _array_hash(*points) != pool.provenance["training_points_sha256"]:
        raise ValueError("Pre-cast training pool does not match its saved hash")
    before = _coordinates(points[0], reference)
    after = _coordinates(pool.directions_nf, reference)
    identity = {
        "schema": SCHEMA, "dataset_fingerprint": reference.fingerprint(),
        "training_points_sha256": pool.provenance["training_points_sha256"],
        "checkpoint_sha256": pool.provenance["plotted_checkpoint"]["sha256"],
        "geometry_dtype": pool.provenance["geometry_dtype"],
        "u_bins": u_bins, "phi_bins": phi_bins, "plot": plot,
    }
    report_path = output / "sampling_audit.json"
    artifact_names = {"sampling_audit.json", "sampling_histogram.npz", "sampling_histogram.png"}
    if output.exists() and any(output.iterdir()):
        if not report_path.is_file():
            raise ValueError("Nonempty audit output must contain a matching sampling_audit.json")
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("identity") != identity:
            raise ValueError("Existing audit output belongs to a different run or configuration")
        entries = list(output.iterdir())
        if any(path.is_symlink() for path in entries):
            raise ValueError("Audit artifacts must not be symbolic links")
        orphans = [path for path in entries if path.name not in artifact_names]
        if any(not (previous.get("status") == "running" and path.is_file() and (
            path.name.startswith(".audit-")
            or (path.name.startswith(".sampling_histogram.png.") and path.suffix == ".tmp")
        )) for path in orphans):
            raise ValueError("Audit output contains unrelated files; refusing to overwrite it")
        # Only the matching unfinished audit owns these temporary files. This
        # also recovers a process interruption that bypassed finally cleanup.
        for path in orphans:
            path.unlink()

    regions = []
    for name, theta_bounds, phi_bounds in _REGIONS:
        u_bounds = np.sin(np.deg2rad(theta_bounds) / 2) ** 2
        probability = float(integrated_histogram(reference, u_bounds, np.array(phi_bounds))[0, 0])
        if not 0 <= probability <= 1:
            raise FloatingPointError("Integrated ROI probability is outside [0,1]")
        counts = [int(np.count_nonzero(
            (theta >= theta_bounds[0]) & (theta < theta_bounds[1])
            & (phi >= phi_bounds[0]) & (phi < phi_bounds[1])
        )) for theta, phi, _ in (before, after)]
        regions.append({
            "name": name, "theta_degrees": list(theta_bounds), "phi_degrees": list(phi_bounds),
            "reference_probability": probability,
            "precast_fp64": _region_statistics(probability, counts[0], count),
            "training_cast": {"observed_count": counts[1], "observed_fraction": counts[1] / count,
                              "count_change_from_precast": counts[1] - counts[0]},
            "hypothetical_pool_sizes": [{
                "sample_count": n, "expected_count": n * probability,
                "probability_of_no_points": (
                    0.0 if probability == 1 else math.exp(n * math.log1p(-probability))
                ),
            } for n in _HYPOTHETICAL_COUNTS],
        })
    u_edges, phi_edges = np.linspace(0, 1, u_bins + 1), np.linspace(-180, 180, phi_bins + 1)
    mass = integrated_histogram(reference, u_edges, phi_edges)
    histograms = [np.histogram2d(u, phi, bins=(u_edges, phi_edges))[0].astype(np.int64)
                  for _, phi, u in (before, after)]
    if any(int(hist.sum()) != count for hist in histograms):
        raise FloatingPointError("Histogram failed to account for every training-pool entry")
    expected = count * mass
    metadata = reference.metadata
    synthetic = metadata.get("synthetic") is True or any(
        "synthetic" in str(metadata.get(key, "")).lower() for key in ("purpose", "source_commit")
    )
    source_kind = "Synthetic CDF arithmetic fixture" if synthetic else "Stored Rainbow CDF"
    report = {
        "schema": SCHEMA, "identity": identity, "status": "complete",
        "source_kind": "synthetic_fixture" if synthetic else "rainbow_solver_record",
        "scope": "read_only_actual_training_pool_sampler_diagnostic; no_model_selection",
        "reference_integration": "stored_cell_overlap_fractions_in_u_and_source_phi; no_PDF_quadrature",
        "statistical_scope": (
            "Per-ROI Wilson 95% intervals and binomial residuals for FP64 pre-cast counts. "
            "ROIs overlap; intervals are not simultaneous. No global p-value or pass/fail. "
            "Coarse agreement does not establish fine-fringe resolution or solver convergence."
        ),
        "sample_count": count, "downsampling": False, "pool": pool.provenance,
        "region_boundary_convention": "lower inclusive, upper exclusive in theta/source phi",
        "regions": regions,
        "histogram": {
            "shape": [u_bins, phi_bins], "axis_order": ["u_bin", "source_phi_bin"],
            "cell_measure": "equal_solid_angle", "reference_mass_sum": float(mass.sum()),
            "precast_count_sum": int(histograms[0].sum()),
            "training_cast_count_sum": int(histograms[1].sum()),
            "bins_with_expected_count_below_5": int(np.count_nonzero(expected < 5)),
            "observed_precast_points_in_zero_mass_bins": int(histograms[0][mass == 0].sum()),
            "observed_training_cast_points_in_zero_mass_bins": int(histograms[1][mass == 0].sum()),
            "bins_changed_by_training_cast": int(np.count_nonzero(histograms[0] != histograms[1])),
            "global_p_value": None,
        },
        "artifacts": {"arrays": "sampling_histogram.npz", "report": "sampling_audit.json",
                      "figure": "sampling_histogram.png" if plot else None},
    }
    output.mkdir(parents=True, exist_ok=True)
    running = (json.dumps({"schema": SCHEMA, "identity": identity, "status": "running"},
                          indent=2, allow_nan=False) + "\n").encode("utf-8")
    _atomic_write(report_path, lambda stream: stream.write(running))
    _atomic_write(output / "sampling_histogram.npz", lambda stream: np.savez_compressed(
        stream, u_edges=u_edges, phi_edges_degrees=phi_edges, reference_probability=mass,
        expected_count=expected, precast_fp64_counts=histograms[0],
        training_cast_counts=histograms[1],
        cell_solid_angles=2 * np.diff(u_edges)[:, None] * np.deg2rad(np.diff(phi_edges))[None, :],
    ))
    if plot:
        _plot_histogram(output / "sampling_histogram.png", u_edges, phi_edges, mass,
                        histograms[0], count, source_kind)
    encoded = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _atomic_write(report_path, lambda stream: stream.write(encoded))
    return report
