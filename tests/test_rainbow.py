"""Synthetic CDF arithmetic fixtures, not optical solver reference solutions."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from phaseflow.rainbow import RainbowReference


def write_rainbow_fixture(
    directory: Path,
    *,
    masses: np.ndarray | None = None,
    u_edges: np.ndarray | None = None,
    inclination: float = 20.0,
    frame_rotation: float = 0.0,
    exact_axis: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Write a small, deliberately nonuniform synthetic saved-cell distribution."""
    if masses is None:
        masses = np.array([[0.05, 0.0, 0.15], [0, 0, 0], [0.1, 0.2, 0], [0, 0.05, 0.45]])
    if u_edges is None:
        u_edges = np.array([0.0, 0.1, 0.7, 1.0])
    masses = np.asarray(masses, dtype=np.float64)
    u_edges = np.asarray(u_edges, dtype=np.float64)
    n_phi, n_theta = masses.shape
    assert u_edges.shape == (n_theta + 1,)
    np.testing.assert_allclose(masses.sum(), 1.0, rtol=0, atol=1e-15)
    marginal = masses.sum(axis=1)
    phi_cdf = np.concatenate(([0.0], np.cumsum(marginal)))
    phi_cdf[-1] = 1
    theta_cdf = np.empty((n_phi, n_theta + 1))
    for j in range(n_phi):
        if marginal[j] == 0:
            theta_cdf[j] = u_edges
        else:
            theta_cdf[j] = np.concatenate(([0.0], np.cumsum(masses[j] / marginal[j])))
            theta_cdf[j, -1] = 1
    alpha = np.deg2rad(inclination)
    incident = np.array([np.cos(alpha), -np.sin(alpha), 0.0])
    if exact_axis:
        assert abs(inclination) == 90
        incident = np.array([0.0, -np.sign(inclination), 0.0])
    e0 = np.array([0.0, 0.0, 1.0])
    canonical_frame = np.column_stack((e0, np.cross(incident, e0), incident))
    c, s = np.cos(frame_rotation), np.sin(frame_rotation)
    frame = canonical_frame @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    g = float(np.sum(masses * (1 - u_edges[:-1] - u_edges[1:])[None, :]))
    metadata = {
        "purpose": "synthetic arithmetic fixture; not a rainbow solver result",
        "schema": "rainbow.phase_cdf.numpy.v2",
        "complete": True,
        "dtype": "<f8",
        "order": "C",
        "theta_count": n_theta,
        "phi_count": n_phi,
        "density_measure": "solid_angle_sr",
        "cell_model": "constant_density_per_spherical_cell",
        "coordinates": "u=(1-cos(theta))/2; v=(phi+pi)/(2*pi)",
        "normalization": "integral_p_domega_equals_one",
        "input_polarization": "unpolarized",
        "coordinate_contract_id": "rainbow.phase_cdf.coordinates.v1",
        "direction_convention": "physical_propagation",
        "particle_frame_handedness": "right",
        "particle_up_axis": [0, 1, 0],
        "shape_polar_axis": [0, -1, 0],
        "sampling_frame_layout": "rows_xyz_columns_e0_e1_k",
        "sampling_frame_map": "local_column_to_particle_column",
        "theta_zero": "forward",
        "theta_pi": "backward",
        "azimuth_periodic": True,
        "cdf_axis_order": ["phi_cell", "theta_edge"],
        "phi_range_rad": [-np.pi, np.pi],
        "sampling_frame_columns": frame.tolist(),
        "wavelength_nm": 550.0,
        "material": {"wavelength_convention": "vacuum_nm"},
        "incident_inclination_degrees": inclination,
        "hg": {
            "g": g,
            "method": "first_moment_of_saved_cell_pdf",
            "target": "saved_cdf",
            "cosine_convention": "dot(incident_propagation,outgoing_propagation)",
            "axis_particle_frame": incident.tolist(),
            "g_source": 0.73,
        },
        "source_commit": "synthetic_fixture_no_solver_commit",
        "source_dirty_at_configure": False,
        "quality": {"angular_convergence_certified": False, "allow_underresolved": True},
        "storage_processing": {"additional_filter": "none", "coarsening_tv": 0.04},
        "diffraction_policy": {"computed": False},
    }
    directory.mkdir(parents=True)
    _write_metadata(directory, metadata)
    np.save(directory / "phi_cdf.npy", phi_cdf.astype("<f8"))
    np.save(directory / "theta_given_phi_cdf.npy", theta_cdf.astype("<f8"))
    np.save(directory / "u_edges.npy", u_edges.astype("<f8"))
    return masses, u_edges, metadata


def _write_metadata(directory: Path, metadata: dict) -> None:
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _nf_direction(u: np.ndarray, source_phi: np.ndarray) -> np.ndarray:
    radius = 2 * np.sqrt(u * (1 - u))
    # Independently derived source -> canonical NF map: (x,y,z) -> (y,-x,z).
    return np.stack((radius * np.sin(source_phi), -radius * np.cos(source_phi), 1 - 2 * u), -1)


def test_exact_cell_densities_moment_and_frame(tmp_path) -> None:
    masses, edges, metadata = write_rainbow_fixture(tmp_path / "record")
    with RainbowReference(tmp_path / "record", validation_block_elements=2) as record:
        np.testing.assert_array_equal(record.frame, metadata["sampling_frame_columns"])
        np.testing.assert_allclose(record.condition, [550, np.sin(np.deg2rad(20))], atol=1e-15)
        assert abs(record.g + 0.235) < 1e-15
        assert record.g != metadata["hg"]["g_source"]
        np.testing.assert_allclose(record.mass_and_g(block_elements=1), [1, -0.235], atol=1e-15)
        source_phi, u = np.meshgrid(
            -np.pi + (np.arange(4) + 0.5) * (np.pi / 2),
            (edges[:-1] + edges[1:]) / 2,
            indexing="ij",
        )
        directions = _nf_direction(u, source_phi)
        solid_angle = np.pi * np.diff(edges)
        expected = masses / solid_angle
        np.testing.assert_allclose(np.exp(record.log_prob(directions)), expected, rtol=1e-14)
        assert np.isneginf(record.log_prob(directions)[masses == 0]).all()
        np.testing.assert_allclose(np.sum(expected * solid_angle), 1, atol=1e-15)
        axis = np.array([0.0, -1.0, 0.0])
        k = record.frame[:, 2]
        projected = axis - np.dot(axis, k) * k
        projected /= np.linalg.norm(projected)
        np.testing.assert_allclose(record.nf_frame[:, 0], projected, atol=1e-15)
        np.testing.assert_allclose(record.nf_frame[:, 1], np.cross(k, projected), atol=1e-15)
        assert not record.condition.flags.writeable
        assert not record.frame.flags.writeable


def test_known_inverse_and_plateaus_without_jitter(tmp_path) -> None:
    write_rainbow_fixture(tmp_path / "record")
    with RainbowReference(tmp_path / "record") as record:
        # phi marginal .1 -> halfway through first phi cell.  Conditional .1
        # -> .1/.25 of the first u cell, so u=.04 (not a linear theta step).
        directions, log_pdf = record.sample(np.array([0.1, 0.1]))
        expected = _nf_direction(np.array(0.04), np.array(-3 * np.pi / 4))
        np.testing.assert_allclose(directions, expected, rtol=0, atol=3e-16)
        np.testing.assert_allclose(np.exp(log_pdf), 0.05 / (0.1 * np.pi), rtol=1e-14)
        # Exactly .2 skips the zero-marginal phi cell; exactly .25 skips the
        # zero conditional interval in the first row.  No random perturbation.
        points, probabilities = record.sample([[0.2, 0.0], [0.1, 0.25]])
        assert points[0, 2] == 1
        np.testing.assert_allclose(
            points[1], _nf_direction(np.array(0.7), np.array(-3 * np.pi / 4))
        )
        np.testing.assert_array_equal(probabilities, record.log_prob(points))


def test_zero_quantiles_skip_leading_zero_mass_cells(tmp_path) -> None:
    masses = np.array([[0, 0, 0], [0.05, 0, 0.15], [0.1, 0.2, 0], [0, 0.05, 0.45]])
    write_rainbow_fixture(tmp_path / "record", masses=masses)
    with RainbowReference(tmp_path / "record") as record:
        points, logp = record.sample([[0.0, 0.1], [0.8, 0.0]])
        expected = _nf_direction(np.array([0.04, 0.1]), np.array([-np.pi / 2, 0.8 * np.pi]))
        np.testing.assert_allclose(points, expected, rtol=0, atol=3e-16)
        assert np.isfinite(logp[0])
        # The second zero variate is exactly on a jump in this discontinuous
        # teacher.  Keep the PDF of the returned direction, even if rounding
        # picks the neighboring zero cell; do not jitter or invent a floor.
        np.testing.assert_array_equal(logp, record.log_prob(points))


def test_one_cell_record_is_uniform_on_solid_angle(tmp_path) -> None:
    write_rainbow_fixture(tmp_path / "record", masses=np.ones((1, 1)), u_edges=np.array([0, 1]))
    with RainbowReference(tmp_path / "record", validation_block_elements=1) as record:
        assert record.g == 0
        uniforms = np.random.default_rng(15).random((17, 2))
        points, logp = record.sample(uniforms)
        np.testing.assert_allclose(points[:, 2], 1 - 2 * uniforms[:, 1], rtol=0, atol=0)
        np.testing.assert_allclose(logp, -np.log(4 * np.pi), rtol=0, atol=0)


def test_sampling_matches_independent_cell_masses_and_axial_moment(tmp_path) -> None:
    masses, edges, _ = write_rainbow_fixture(tmp_path / "record")
    rng = np.random.default_rng(7419)
    with RainbowReference(tmp_path / "record") as record:
        values = rng.random((180_000, 2))
        directions, sample_logp = record.sample(values)
        np.testing.assert_allclose(np.linalg.norm(directions, axis=-1), 1, atol=4e-16, rtol=0)
        np.testing.assert_array_equal(sample_logp, record.log_prob(directions))
        # Independent histogram in solver coordinates, for this modest grid.
        source_phi = np.arctan2(directions[:, 0], -directions[:, 1])
        phi_indices = np.floor((source_phi + np.pi) / (np.pi / 2)).astype(int) % 4
        u = (1 - directions[:, 2]) / 2
        theta_indices = np.searchsorted(edges, u, side="right") - 1
        counts = np.bincount(3 * phi_indices + theta_indices, minlength=12).reshape(4, 3)
        observed = counts / len(values)
        sigma = np.sqrt(masses * (1 - masses) / len(values))
        assert np.all(np.abs(observed - masses) <= 5 * sigma + 1 / len(values))
        assert np.all(counts[masses == 0] == 0)
        assert abs(directions[:, 2].mean() - record.g) < 0.004
        # Both azimuth signs are preserved; the adapter does not impose symmetry.
        assert np.any(directions[:, 1] < 0) and np.any(directions[:, 1] > 0)


def test_recorded_transverse_rotation_is_respected(tmp_path) -> None:
    angle = 0.37
    masses, edges, metadata = write_rainbow_fixture(tmp_path / "record", frame_rotation=angle)
    with RainbowReference(tmp_path / "record") as record:
        np.testing.assert_array_equal(record.frame, metadata["sampling_frame_columns"])
        phi, u = np.meshgrid(
            -np.pi + (np.arange(4) + 0.5) * np.pi / 2,
            (edges[:-1] + edges[1:]) / 2,
            indexing="ij",
        )
        directions = _nf_direction(u, phi + angle)
        np.testing.assert_allclose(
            np.exp(record.log_prob(directions)), masses / (np.pi * np.diff(edges)), rtol=1e-14
        )
        particle = directions @ record.nf_frame.T
        source = particle @ record.frame
        np.testing.assert_allclose(np.arctan2(source[..., 1], source[..., 0]), phi, atol=5e-16)


@pytest.mark.parametrize("inclination", [-90.0, -20.0, 20.0, 90.0])
def test_signed_incident_condition_and_axial_frame(tmp_path, inclination) -> None:
    write_rainbow_fixture(
        tmp_path / "record", inclination=inclination, exact_axis=abs(inclination) == 90
    )
    with RainbowReference(tmp_path / "record") as record:
        assert abs(record.condition[1] - np.sin(np.deg2rad(inclination))) < 1e-15
        np.testing.assert_allclose(record.nf_frame.T @ record.nf_frame, np.eye(3), atol=4e-16)
        assert np.linalg.det(record.nf_frame) > 0


def test_seam_pole_and_query_shape_conventions(tmp_path) -> None:
    write_rainbow_fixture(tmp_path / "record")
    with RainbowReference(tmp_path / "record") as record:
        direction = _nf_direction(np.array(0.05), np.array(-np.pi))
        wrapped = _nf_direction(np.array(0.05), np.array(np.pi))
        assert record.log_prob(direction) == record.log_prob(wrapped)
        poles = np.array([[0.0, 0.0, 1.0], [-0.0, -0.0, 1.0], [0, 0, -1]])
        logp = record.log_prob(poles)
        assert logp[0] == logp[1]
        np.testing.assert_allclose(np.exp(logp[0]), 0.1 / (0.1 * np.pi), rtol=1e-14)
        assert np.isneginf(logp[-1])
        directions, densities = record.sample(np.full((2, 3, 2), 0.413))
        assert directions.shape == (2, 3, 3) and densities.shape == (2, 3)
        assert directions.dtype == densities.dtype == np.dtype(np.float64)
        np.testing.assert_allclose(record.log_prob(directions * (1 + 1e-6)), densities, rtol=1e-14)
        empty_directions, empty_pdf = record.sample(np.empty((0, 2)))
        assert empty_directions.shape == (0, 3) and empty_pdf.shape == (0,)


def test_tiny_forward_angle_does_not_disappear_when_cosine_rounds_to_one(tmp_path) -> None:
    masses = np.tile(np.array([0.125, 0.125]), (4, 1))
    write_rainbow_fixture(tmp_path / "record", masses=masses, u_edges=np.array([0, 1e-18, 1]))
    with RainbowReference(tmp_path / "record") as record:
        points, logp = record.sample([[0.13, 0.25], [0.87, 0.25]])
        assert np.all(points[:, 2] == 1)
        assert np.all(np.hypot(points[:, 0], points[:, 1]) > 0)
        np.testing.assert_allclose(np.exp(logp), 0.125 / (np.pi * 1e-18), rtol=5e-15)


def test_unrepresentable_interior_sample_is_reported_instead_of_clamped(tmp_path) -> None:
    write_rainbow_fixture(tmp_path / "record")
    with RainbowReference(tmp_path / "record") as record:
        # The last quantile lies so close to the last edge of this back cell
        # that its interpolated double u rounds to 1.  Do not synthesize a PDF
        # for that numerical atom or silently move it to a representable point.
        with pytest.raises(FloatingPointError, match="rounded to a pole"):
            record.sample([0.1, np.nextafter(1.0, 0.0)])
        # Here the mathematical sample is just above a leading zero-density
        # interval. The increment cannot be represented in u, and angular
        # evaluation lands on the zero side of the cell boundary.
        with pytest.raises(FloatingPointError, match="zero-density cell"):
            record.sample([0.5, np.nextafter(0.0, 1.0)])


@pytest.mark.parametrize(
    "uniforms",
    [
        [-0.01, 0.5],
        [1, 0.5],
        [0.5, 1],
        [np.nan, 0.5],
        [0.5, np.inf],
        [0.1],
        [0.1, 0.2, 0.3],
        [1j, 0.5],
    ],
)
def test_invalid_uniforms_are_rejected(tmp_path, uniforms) -> None:
    write_rainbow_fixture(tmp_path / "record")
    with RainbowReference(tmp_path / "record") as record, pytest.raises(ValueError):
        record.sample(uniforms)


@pytest.mark.parametrize("direction", [[0, 0, 0], [0, 0, 2], [0, 0, np.nan], [1, 0], [1j, 0, 1]])
def test_invalid_directions_are_rejected(tmp_path, direction) -> None:
    write_rainbow_fixture(tmp_path / "record")
    with RainbowReference(tmp_path / "record") as record, pytest.raises(ValueError):
        record.log_prob(direction)


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema", "rainbow.phase_cdf.numpy.v1"),
        ("complete", False),
        ("direction_convention", "away_from_event"),
        ("density_measure", "theta_phi"),
        ("cdf_axis_order", ["theta_edge", "phi_cell"]),
        ("shape_polar_axis", [0, 1, 0]),
        ("theta_count", True),
        ("wavelength_nm", -1),
        ("azimuth_periodic", False),
        ("wavelength_nm", 10**500),
    ],
)
def test_unsupported_or_incomplete_metadata_is_rejected(tmp_path, key, value) -> None:
    _, _, metadata = write_rainbow_fixture(tmp_path / "record")
    metadata[key] = value
    _write_metadata(tmp_path / "record", metadata)
    with pytest.raises(ValueError):
        RainbowReference(tmp_path / "record")


@pytest.mark.parametrize(
    "key,value",
    [
        ("g", 0.4),
        ("g", 1),
        ("g", -1),
        ("target", "source_before_coarsening"),
        ("axis_particle_frame", [1, 0, 0]),
    ],
)
def test_inconsistent_hg_metadata_is_rejected(tmp_path, key, value) -> None:
    _, _, metadata = write_rainbow_fixture(tmp_path / "record")
    metadata["hg"][key] = value
    _write_metadata(tmp_path / "record", metadata)
    with pytest.raises(ValueError, match="HG|hg.g"):
        RainbowReference(tmp_path / "record")


def test_invalid_frame_and_duplicate_json_are_rejected(tmp_path) -> None:
    _, _, metadata = write_rainbow_fixture(tmp_path / "record")
    metadata["sampling_frame_columns"][0][0] = 0.2
    _write_metadata(tmp_path / "record", metadata)
    with pytest.raises(ValueError, match="frame"):
        RainbowReference(tmp_path / "record")
    metadata_path = tmp_path / "record" / "metadata.json"
    metadata_path.write_text('{"complete":false,"complete":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        RainbowReference(tmp_path / "record")
    metadata_path.write_text('{"g":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="Nonfinite"):
        RainbowReference(tmp_path / "record")


@pytest.mark.parametrize(
    "kind",
    ["nonmonotone", "nan", "endpoint", "out_of_range", "u_duplicate", "u_endpoint", "phi_decrease"],
)
def test_invalid_cdfs_are_rejected_across_small_validation_blocks(tmp_path, kind) -> None:
    write_rainbow_fixture(tmp_path / "record")
    name = "theta_given_phi_cdf.npy"
    if kind.startswith("u_"):
        name = "u_edges.npy"
    elif kind.startswith("phi_"):
        name = "phi_cdf.npy"
    path = tmp_path / "record" / name
    array = np.load(path)
    if kind == "nonmonotone":
        array[0, 2] = 0.1
    elif kind == "nan":
        array[1, 2] = np.nan
    elif kind == "endpoint":
        array[2, 0] = 0.1
    elif kind == "out_of_range":
        array[3, 2] = 1.01
    elif kind == "u_duplicate":
        array[2] = array[1]
    elif kind == "u_endpoint":
        array[-1] = 0.9
    else:
        array[2] = 0.1
    np.save(path, array)
    with pytest.raises(ValueError):
        RainbowReference(tmp_path / "record", validation_block_elements=3)


@pytest.mark.parametrize(
    "kind", ["float32", "big_endian", "fortran", "transpose", "trailing", "truncated"]
)
def test_wrong_npy_layout_or_payload_is_rejected(tmp_path, kind) -> None:
    write_rainbow_fixture(tmp_path / "record")
    path = tmp_path / "record" / "theta_given_phi_cdf.npy"
    array = np.load(path)
    if kind == "float32":
        np.save(path, array.astype("<f4"))
    elif kind == "big_endian":
        np.save(path, array.astype(">f8"))
    elif kind == "fortran":
        np.save(path, np.asfortranarray(array))
    elif kind == "transpose":
        np.save(path, array.T)
    elif kind == "trailing":
        with path.open("ab") as stream:
            stream.write(b"trailing")
    else:
        path.write_bytes(path.read_bytes()[:-8])
    with pytest.raises(ValueError):
        RainbowReference(tmp_path / "record")


def test_partial_record_close_and_full_file_provenance(tmp_path) -> None:
    _, _, metadata = write_rainbow_fixture(tmp_path / "record")
    shutil.copytree(tmp_path / "record", tmp_path / "same_record")
    with (
        RainbowReference(tmp_path / "record") as first,
        RainbowReference(tmp_path / "same_record") as second,
    ):
        assert first.fingerprint() == second.fingerprint()
        summary = first.summary()
        assert summary["provenance"]["metadata"] == metadata
        json.dumps(summary, allow_nan=False)
        for name, info in first.provenance["files"].items():
            path = tmp_path / "record" / name
            assert info["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
            assert info["bytes"] == path.stat().st_size
        before = first.fingerprint()
        # The caller cannot mutate the internally retained metadata/provenance.
        first.metadata["hg"]["g"] = 0.9
        assert first.metadata["hg"]["g"] == metadata["hg"]["g"]
    first.close()
    with pytest.raises(RuntimeError, match="closed"):
        first.sample([0.2, 0.3])
    with pytest.raises(RuntimeError, match="closed"):
        first.log_prob([0, 0, 1])
    metadata["quality"]["allow_underresolved"] = False
    _write_metadata(tmp_path / "record", metadata)
    with RainbowReference(tmp_path / "record") as changed:
        assert changed.fingerprint() != before
    shutil.copytree(tmp_path / "record", tmp_path / "record.part")
    with pytest.raises(ValueError, match="part"):
        RainbowReference(tmp_path / "record.part")
