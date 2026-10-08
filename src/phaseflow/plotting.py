"""Aligned scientific plots for one saved Rainbow CDF and a fitted sphere flow.

Colors always represent density per steradian. The displayed (phi, theta)
rectangle is not equal area: raw point density on it also includes sin(theta).
The source CDF grid is retained exactly, including its nonuniform u boundaries.
Matplotlib is imported only when figures are requested; no pyplot/global backend
or global random generator is changed.
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
from numpy.typing import NDArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .rainbow import RainbowReference
    from .sphere_model import SingleConditionSphereFlow


PLOT_SCHEMA = "phaseflow.rainbow_plots.v1"
_CDF_PLOT_STREAM = 100
_ZERO_COLOR = "#d2d6dc"
_DENSITY_LABEL = r"PDF [sr$^{-1}$], logarithmic scale"
_CHART_NOTE = (
    "Equal angular scaling; this chart is not equal area. "
    "CDF point density on the chart includes sin(theta)."
)


@dataclass(frozen=True)
class RainbowPlotConfig:
    """Display and batching settings; none change the learned distribution."""

    cdf_samples: int = 32768
    seed: int = 2027
    eval_batch_size: int = 16384
    dpi: int = 160
    write_pdf: bool = False

    def __post_init__(self) -> None:
        for name in ("cdf_samples", "eval_batch_size", "dpi"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("plot seed must be a nonnegative integer")
        if type(self.write_pdf) is not bool:
            raise ValueError("write_pdf must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RainbowPlotConfig:
        if not isinstance(value, dict):
            raise ValueError("plotting configuration must be an object")
        try:
            return cls(**value)
        except TypeError as error:
            raise ValueError(f"Invalid plotting configuration: {error}") from error


@dataclass(frozen=True)
class RainbowGrid:
    """Native cell values, with image arrays indexed [theta_cell, phi_cell].

    ``reference_pdf`` is the exact saved piecewise-constant density. ``nf_pdf``
    is a point evaluation at each cell's midpoint in (u, phi), not a cell average.
    Their solid-angle weighted sums therefore have different interpretations.
    """

    phi_edges_degrees: NDArray[np.float64]
    theta_edges_degrees: NDArray[np.float64]
    reference_pdf: NDArray[np.float64]
    nf_pdf: NDArray[np.float64]
    cell_solid_angles: NDArray[np.float64]
    reference_mass_exact: float
    nf_mass_midpoint_estimate: float


def _positive_pdf(log_pdf: NDArray[np.float64], *, name: str, allow_zero: bool) -> np.ndarray:
    valid = np.isfinite(log_pdf)
    if allow_zero:
        valid |= np.isneginf(log_pdf)
    if not valid.all():
        raise FloatingPointError(f"{name} has invalid log PDF values")
    with np.errstate(under="ignore", over="ignore"):
        pdf = np.exp(log_pdf)
    if not np.isfinite(pdf).all() or np.any((pdf == 0) & np.isfinite(log_pdf)):
        raise FloatingPointError(
            f"{name} PDF is outside the positive finite float64 plotting range; "
            "refusing to turn underflow into zero cells or apply a density floor"
        )
    return pdf


def evaluate_rainbow_grid(
    model: SingleConditionSphereFlow,
    reference: RainbowReference,
    *,
    batch_size: int = 16384,
) -> RainbowGrid:
    """Evaluate every native CDF cell using bounded model-device batches.

    CPU arrays hold the display grid; inference executes on ``model.device``.
    Directions and geometry stay float64 even with a float32 conditioner. The
    recorded source-to-NF rotation is applied before PDF evaluation. No global
    RNG is used, and the caller's training/evaluation mode is restored on errors.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    # Check open ownership before accessing memmaps, including on a closed record.
    reference._require_open()
    n_phi, n_theta = reference.n_phi, reference.n_theta
    u_edges = np.asarray(reference.u_edges)
    delta_u = np.diff(u_edges)
    solid_angle = (4 * np.pi / n_phi) * delta_u
    phi_edges = np.linspace(-np.pi, np.pi, n_phi + 1, dtype=np.float64)
    theta_edges = 2 * np.arctan2(np.sqrt(u_edges), np.sqrt(1 - u_edges))

    # Take the logs separately to retain tiny nonzero marginal/conditional masses
    # until representability of the plotted density has been checked explicitly.
    with np.errstate(divide="ignore"):
        reference_log_pdf = (
            np.log(np.diff(reference.phi_cdf))[:, None]
            + np.log(np.diff(reference.theta_cdf, axis=1))
            - np.log(solid_angle)[None, :]
        ).T
    reference_pdf = _positive_pdf(reference_log_pdf, name="Reference", allow_zero=True)
    del reference_log_pdf

    # Form u and 1-u midpoints separately. If a very narrow terminal cell has a
    # rounded midpoint u=1, the nonzero transverse direction is still retained.
    u_mid = u_edges[:-1] / 2 + u_edges[1:] / 2
    one_minus_u_mid = (1 - u_edges[:-1]) / 2 + (1 - u_edges[1:]) / 2
    radius = 2 * np.sqrt(u_mid * one_minus_u_mid)
    mu = one_minus_u_mid - u_mid
    phi_mid = (phi_edges[:-1] + phi_edges[1:]) / 2
    nf_log_pdf = np.empty(n_theta * n_phi, dtype=np.float64)
    training_modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, nf_log_pdf.size, batch_size):
                stop = min(start + batch_size, nf_log_pdf.size)
                index = np.arange(start, stop, dtype=np.int64)
                i, j = index // n_phi, index % n_phi
                source = np.column_stack(
                    (
                        radius[i] * np.cos(phi_mid[j]),
                        radius[i] * np.sin(phi_mid[j]),
                        mu[i],
                    )
                )
                outgoing = torch.as_tensor(
                    source @ reference.source_to_nf.T,
                    dtype=getattr(model, "geometry_dtype", torch.float64),
                    device=model.device,
                )
                values = model.log_prob(outgoing)
                if values.shape != (stop - start,):
                    raise ValueError("Model log_prob must return one value per grid direction")
                nf_log_pdf[start:stop] = values.detach().to(dtype=torch.float64).cpu().numpy()
    finally:
        # Preserve deliberately mixed submodule modes as well as the root flag.
        for module, training in training_modes:
            module.training = training
    nf_pdf = _positive_pdf(nf_log_pdf.reshape(n_theta, n_phi), name="NF", allow_zero=False)

    reference_mass = float(
        np.sum(reference_pdf.astype(np.longdouble) * solid_angle[:, None], dtype=np.longdouble)
    )
    nf_mass = float(
        np.sum(nf_pdf.astype(np.longdouble) * solid_angle[:, None], dtype=np.longdouble)
    )
    if not np.isfinite(reference_mass) or not np.isfinite(nf_mass):
        raise FloatingPointError("A solid-angle weighted plotting-grid mass is nonfinite")
    return RainbowGrid(
        phi_edges_degrees=np.rad2deg(phi_edges),
        theta_edges_degrees=np.rad2deg(theta_edges),
        reference_pdf=reference_pdf,
        nf_pdf=nf_pdf,
        cell_solid_angles=solid_angle,
        reference_mass_exact=reference_mass,
        nf_mass_midpoint_estimate=nf_mass,
    )


def cdf_scatter_coordinates(
    reference: RainbowReference, *, samples: int = 32768, seed: int = 2027
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate independent CDF points and return source (phi, theta) degrees.

    This is a diagnostic stream, not the fixed training pool or a model-selection
    stream. Both uniform variates are 52-bit midpoint values strictly inside (0,1).
    """
    if type(samples) is not int or samples < 1:
        raise ValueError("samples must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, _CDF_PLOT_STREAM])))
    uniforms = (rng.integers(0, 1 << 52, size=(samples, 2)) + 0.5) / (1 << 52)
    outgoing_nf, _ = reference.sample(uniforms)
    source = outgoing_nf @ reference.source_to_nf
    phi = np.arctan2(source[:, 1], source[:, 0])
    phi = (phi + np.pi) % (2 * np.pi) - np.pi
    theta = np.arctan2(np.hypot(source[:, 0], source[:, 1]), source[:, 2])
    return np.rad2deg(phi), np.rad2deg(theta)


def _shared_color_limits(grid: RainbowGrid) -> tuple[float, float]:
    positive_reference = grid.reference_pdf[grid.reference_pdf > 0]
    low = min(float(positive_reference.min()), float(grid.nf_pdf.min()))
    high = max(float(positive_reference.max()), float(grid.nf_pdf.max()))
    # Constant (or rounding-level constant) fields need a nonzero display range.
    # This changes the visual scale only, never the density arrays themselves.
    if math.log(high) - math.log(low) < 1e-12:
        padded_low, padded_high = low / math.sqrt(10), high * math.sqrt(10)
        low = padded_low if padded_low > 0 else low
        high = padded_high if math.isfinite(padded_high) else high
    if not 0 < low < high < math.inf:
        raise FloatingPointError("Cannot construct a finite positive shared log color scale")
    return low, high


def _configure_axes(ax: Axes) -> None:
    ax.set_xlim(-180, 180)
    ax.set_ylim(180, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.set_xlabel("Source azimuth phi [degrees]")
    ax.set_ylabel("Scattering angle theta [degrees]")
    ax.tick_params(labelsize=10)


def _array_hash(value: NDArray[np.float64]) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").view(np.uint8)).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _save_figure(fig: Figure, path: Path, *, dpi: int, write_pdf: bool) -> list[str]:
    paths = [path.name]
    try:
        # Deliberately avoid bbox_inches='tight': all individual output canvases
        # and data rectangles must have exactly the same size and placement.
        fig.savefig(path, dpi=dpi, facecolor="white")
        if write_pdf:
            pdf_path = path.with_suffix(".pdf")
            fig.savefig(pdf_path, facecolor="white")
            paths.append(pdf_path.name)
    finally:
        fig.clear()
    return paths


def plot_rainbow_comparison(
    model: SingleConditionSphereFlow,
    reference: RainbowReference,
    output_directory: str | Path,
    *,
    config: RainbowPlotConfig | None = None,
    checkpoint_path: str | Path | None = None,
    selected_step: int | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Save three aligned maps, a combined figure and a provenance manifest.

    Both PDF maps share one LogNorm over all positive values, without percentile
    clipping, smoothing, extra normalization, or density floors. True zero CDF
    cells have a separate gray color. Inference covers every native source cell;
    raster display pixels need not resolve subpixel cells, so native numerical
    grid statistics and resolution are recorded in ``plots.json``.

    ``checkpoint_path`` records the exact file hash when supplied. The function
    plots the passed model's current weights; callers loading a checkpoint must
    validate the record identity before calling, as the CLI does.
    """
    # Headless object-oriented rendering does not set a global backend or import
    # pyplot, and keeps CLI help / non-plotting commands free of matplotlib work.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LogNorm
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    config = config or RainbowPlotConfig()
    if not isinstance(config, RainbowPlotConfig):
        raise ValueError("config must be a RainbowPlotConfig")
    if selected_step is not None and (type(selected_step) is not int or selected_step < 0):
        raise ValueError("selected_step must be a nonnegative integer or None")
    if model.hg_g != reference.g or model.incident_cosine != float(reference.condition[1]):
        raise ValueError("Model HG g or incident condition does not match the reference record")
    checkpoint = None
    if checkpoint_path is not None:
        checkpoint_file = Path(checkpoint_path).resolve()
        checkpoint = {"path": str(checkpoint_file), "sha256": _file_hash(checkpoint_file)}

    grid = evaluate_rainbow_grid(model, reference, batch_size=config.eval_batch_size)
    scatter_phi, scatter_theta = cdf_scatter_coordinates(
        reference, samples=config.cdf_samples, seed=config.seed
    )
    vmin, vmax = _shared_color_limits(grid)
    norm = LogNorm(vmin=vmin, vmax=vmax, clip=False)
    # Use a perceptually ordered map and reserve gray for actual zero mass.
    from matplotlib import colormaps

    cmap = colormaps["viridis"].with_extremes(bad=_ZERO_COLOR)
    metadata = reference.metadata
    synthetic = metadata.get("synthetic") is True or any(
        "synthetic" in str(metadata.get(key, "")).lower() for key in ("purpose", "source_commit")
    )
    source_kind = "Synthetic fixture" if synthetic else "Rainbow CDF record"
    condition_label = (
        f"{reference.wavelength_nm:g} nm; "
        f"incident inclination {metadata['incident_inclination_degrees']:g} degrees"
    )
    heading = f"{source_kind} | {condition_label}"
    if title is not None:
        heading += f"\n{title}"
    elif selected_step is not None:
        heading += f"\nValidation-selected model (step {selected_step})"
    panel_titles = (
        "Reference PDF from the stored CDF",
        f"Samples from the stored CDF (N = {config.cdf_samples:,})",
        "NF PDF evaluated at source cell centers",
    )

    def draw_panel(ax: Axes, panel: int):
        if panel == 1:
            artist = ax.scatter(
                scatter_phi,
                scatter_theta,
                s=1.3,
                c="#172a3a",
                alpha=0.3,
                linewidths=0,
                rasterized=True,
            )
            ax.set_facecolor("#fafbfd")
        else:
            values = grid.reference_pdf if panel == 0 else grid.nf_pdf
            artist = ax.pcolormesh(
                grid.phi_edges_degrees,
                grid.theta_edges_degrees,
                np.ma.masked_equal(values, 0),
                norm=norm,
                cmap=cmap,
                shading="flat",
                antialiased=False,
                rasterized=True,
            )
        _configure_axes(ax)
        ax.set_title(panel_titles[panel], fontsize=13, pad=12)
        return artist

    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    files: dict[str, list[str]] = {}
    # Fixed rectangles ensure the scatter canvas has exactly the same dimensions
    # as either PDF canvas, including the space reserved for their colorbars.
    single_size = (12.0, 7.4)
    rect = [0.09, 0.19, 0.77, 0.77 * single_size[0] / 2 / single_size[1]]
    for panel, stem in enumerate(("reference_pdf", "cdf_samples", "nf_pdf")):
        fig = Figure(figsize=single_size)
        FigureCanvasAgg(fig)
        ax = fig.add_axes(rect)
        artist = draw_panel(ax, panel)
        fig.suptitle(heading, x=0.5, y=0.965, fontsize=13)
        if panel != 1:
            colorbar_axis = fig.add_axes([0.880, rect[1], 0.018, rect[3]])
            fig.colorbar(artist, cax=colorbar_axis, label=_DENSITY_LABEL)
        if panel == 0 and np.any(grid.reference_pdf == 0):
            fig.legend(
                handles=[Patch(facecolor=_ZERO_COLOR, label="Zero PDF (not a log-scale floor)")],
                loc="lower center",
                bbox_to_anchor=(0.5, 0.065),
                frameon=False,
                fontsize=10,
            )
        fig.text(0.5, 0.038, _CHART_NOTE, ha="center", fontsize=9)
        files[stem] = _save_figure(
            fig, output / f"{stem}.png", dpi=config.dpi, write_pdf=config.write_pdf
        )

    fig = Figure(figsize=(21.0, 6.2))
    FigureCanvasAgg(fig)
    panel_width = 0.276
    panel_height = panel_width * 21 / 2 / 6.2
    comparison_rectangles = []
    image_artist = None
    for panel, left in enumerate((0.038, 0.350, 0.662)):
        comparison_rectangles.append([left, 0.29, panel_width, panel_height])
        ax = fig.add_axes(comparison_rectangles[-1])
        artist = draw_panel(ax, panel)
        ax.title.set_fontsize(12)
        if panel != 1:
            image_artist = artist
    colorbar_axis = fig.add_axes([0.26, 0.12, 0.48, 0.025])
    fig.colorbar(image_artist, cax=colorbar_axis, orientation="horizontal", label=_DENSITY_LABEL)
    fig.suptitle(heading, x=0.5, y=0.925, fontsize=16)
    fig.text(0.5, 0.021, _CHART_NOTE, ha="center", fontsize=11)
    if np.any(grid.reference_pdf == 0):
        fig.legend(
            handles=[Patch(facecolor=_ZERO_COLOR, label="Zero reference PDF")],
            loc="lower right",
            bbox_to_anchor=(0.976, 0.105),
            frameon=False,
            fontsize=10,
        )
    files["comparison"] = _save_figure(
        fig, output / "comparison.png", dpi=config.dpi, write_pdf=config.write_pdf
    )

    config_value = getattr(model, "config", None)
    manifest: dict[str, Any] = {
        "schema": PLOT_SCHEMA,
        "source_kind": "synthetic_fixture" if synthetic else "rainbow_cdf_record",
        "title": heading,
        "configuration": config.to_dict(),
        "condition": reference.condition.tolist(),
        "base_g": reference.g,
        "selected_step": selected_step,
        "dataset_fingerprint": reference.fingerprint(),
        "data_provenance": reference.provenance,
        "checkpoint": checkpoint,
        "model": {
            "family": getattr(model, "model_family", type(model).__name__),
            "configuration": config_value.to_dict() if config_value is not None else None,
            "device": str(model.device),
            "parameter_dtype": str(model.dtype),
            "density_evaluation_dtype": str(getattr(model, "geometry_dtype", torch.float64)),
        },
        "coordinates": {
            "frame": "recorded_solver_source",
            "x": "source_azimuth_phi_degrees",
            "x_limits": [-180.0, 180.0],
            "y": "scattering_theta_degrees",
            "y_limits": [180.0, 0.0],
            "theta_zero": "forward_at_top",
            "data_aspect": "one_degree_x_equals_one_degree_y",
            "axes_width_over_height": 2.0,
            "equal_area": False,
            "density_measure": "solid_angle_sr",
            "source_to_nf": reference.source_to_nf.tolist(),
            "scatter_chart_density": "p_per_sr * sin(theta); not p_per_sr alone",
        },
        "grid": {
            "resolution": "full_native_cdf_cells_no_angular_downsampling",
            "array_order": ["theta_cell", "phi_cell"],
            "theta_count": reference.n_theta,
            "phi_count": reference.n_phi,
            "point_evaluation": "midpoint_in_u_and_source_phi",
            "theta_edges": "2*atan2(sqrt(u_edges),sqrt(1-u_edges))",
            "reference_pdf": "exact_saved_cell_mass_divided_by_cell_solid_angle",
            "nf_pdf": "point_values_not_cell_averages",
            "reference_mass_exact": grid.reference_mass_exact,
            "nf_mass_midpoint_estimate": grid.nf_mass_midpoint_estimate,
            "nf_mass_warning": (
                "Native-cell midpoint quadrature only; this is not an exact normalization test "
                "and can miss NF structure inside a cell. No density renormalization is applied."
            ),
            "zero_reference_cells": int(np.count_nonzero(grid.reference_pdf == 0)),
            "reference_pdf_sha256": _array_hash(grid.reference_pdf),
            "nf_pdf_sha256": _array_hash(grid.nf_pdf),
        },
        "scatter": {
            "source": "stored_cdf_independent_diagnostic_stream",
            "sample_count": config.cdf_samples,
            "seed": config.seed,
            "stream_id": _CDF_PLOT_STREAM,
            "rng": "numpy.PCG64(SeedSequence([seed,stream_id]))",
            "uniform_midpoint_bits": 52,
            "source_phi_degrees_sha256": _array_hash(scatter_phi),
            "theta_degrees_sha256": _array_hash(scatter_theta),
        },
        "color_scale": {
            "normalization": "matplotlib.colors.LogNorm",
            "shared_between": ["reference_pdf", "nf_pdf"],
            "vmin": vmin,
            "vmax": vmax,
            "limits": "joint_positive_extrema; rounding-level_constant_fields_padded_only_for_display",
            "colormap": "viridis",
            "zero_color": _ZERO_COLOR,
            "percentile_clipping": False,
            "density_floor": None,
        },
        "figures": {
            "individual_size_inches": list(single_size),
            "individual_axes_rectangle": rect,
            "comparison_size_inches": [21.0, 6.2],
            "comparison_axes_rectangles": comparison_rectangles,
            "display_note": "PNG raster pixels may combine native cells narrower than a pixel.",
        },
        "files": files,
    }
    temporary = output / "plots.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "plots.json")
    return manifest
