"""Sampler audit checks with synthetic CDFs; no Rainbow optical validation."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from test_rainbow import write_rainbow_fixture

from phaseflow.rainbow import RainbowReference
from phaseflow.sampling_audit import (
    _wilson_interval,
    audit_training_sampling,
    integrated_histogram,
)
from phaseflow.single_condition import SingleTrainingConfig, _points, train_single_condition
from phaseflow.sphere_model import SphereFlowConfig


def _train(record, run, *, count=4096, geometry_dtype="model"):
    return train_single_condition(
        record,
        SphereFlowConfig(num_coupling_layers=2, num_bins=4, hidden_features=(4,),
                         geometry_dtype=geometry_dtype, spline_dtype="model"),
        SingleTrainingConfig(device="cpu", dtype="float32", seed=17, data_seed=37,
                             train_samples=count, validation_samples=32, test_samples=32,
                             train_monitor_samples=32, proposal_samples=0, steps=0,
                             batch_size=16, eval_batch_size=128, tensorboard=False),
        run, make_plots=False,
    )


def _hashes(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()}


def test_exact_nonuniform_source_cells_and_partial_rectangles(tmp_path):
    masses, edges, _ = write_rainbow_fixture(tmp_path / "record")
    with RainbowReference(tmp_path / "record") as record:
        actual = integrated_histogram(record, edges, np.linspace(-180, 180, 5))
        np.testing.assert_allclose(actual, masses.T, rtol=1e-14, atol=1e-16)
        # Half of each occupied phi cell, with half of its first two u cells:
        # .05/4 from phi cell 0, and (.1+.2)/4 from phi cell 2.
        partial = integrated_histogram(record, np.array([0.05, 0.4]), np.array([-135, 45]))
        np.testing.assert_allclose(partial, [[0.0875]], rtol=1e-14)
        np.testing.assert_allclose(integrated_histogram(
            record, np.array([0, 1]), np.array([-180, 180]),
        ), [[1]], rtol=1e-14)


def test_uniform_solid_angle_integrates_on_nonmatching_edges(tmp_path):
    edges = np.array([0, 0.0001, 0.03, 0.2, 0.91, 1.0])
    write_rainbow_fixture(tmp_path / "record", masses=np.tile(np.diff(edges) / 7, (7, 1)),
                          u_edges=edges)
    u, phi = np.array([0, 0.1, 0.8, 1]), np.array([-180, -21.3, 34.7, 180])
    with RainbowReference(tmp_path / "record") as record:
        expected = np.diff(u)[:, None] * np.diff(phi)[None, :] / 360
        np.testing.assert_allclose(integrated_histogram(record, u, phi), expected, rtol=1e-14)


@pytest.mark.parametrize("geometry_dtype", ["model", "float64"])
def test_audit_counts_exact_recorded_pool_and_keeps_run_unchanged(tmp_path, geometry_dtype):
    write_rainbow_fixture(tmp_path / "record", frame_rotation=0.37)
    run, output = tmp_path / "run", tmp_path / "audit"
    with RainbowReference(tmp_path / "record") as record:
        _train(record, run, geometry_dtype=geometry_dtype)
        before_hashes = _hashes(run)
        report = audit_training_sampling(record, run, output, device="cpu", u_bins=7,
                                         phi_bins=9, plot=False)
        assert _hashes(run) == before_hashes
        assert report["sample_count"] == 4096 and report["downsampling"] is False
        assert report["status"] == "complete"
        assert report["pool"]["hash_verified"] is True
        assert report["histogram"]["global_p_value"] is None
        assert report["histogram"]["precast_count_sum"] == 4096
        points = _points(record, 4096, 37, "train")[0]
        # Independent 3D frame round trip, instead of the audit's 2D rotation.
        source = (points @ record.nf_frame.T) @ record.frame
        theta = np.rad2deg(np.arctan2(np.hypot(source[:, 0], source[:, 1]), source[:, 2]))
        phi = np.rad2deg(np.arctan2(source[:, 1], source[:, 0]))
        with np.load(output / "sampling_histogram.npz", allow_pickle=False) as arrays:
            observed = np.histogram2d(np.sin(np.deg2rad(theta) / 2) ** 2, phi,
                                     bins=(arrays["u_edges"], arrays["phi_edges_degrees"]))[0]
            np.testing.assert_array_equal(arrays["precast_fp64_counts"], observed)
            np.testing.assert_allclose(arrays["reference_probability"].sum(), 1, atol=1e-14)
            np.testing.assert_allclose(arrays["cell_solid_angles"].sum(), 4 * np.pi, atol=1e-14)
            assert arrays["training_cast_counts"].sum() == 4096
        for region in report["regions"]:
            lo, hi = region["theta_degrees"]
            left, right = region["phi_degrees"]
            count = np.count_nonzero((theta >= lo) & (theta < hi) & (phi >= left) & (phi < right))
            assert region["precast_fp64"]["observed_count"] == count
            for hypothetical in region["hypothetical_pool_sizes"]:
                probability = region["reference_probability"]
                assert hypothetical["expected_count"] == probability * hypothetical["sample_count"]
                assert 0 <= hypothetical["probability_of_no_points"] <= 1
        assert json.loads((output / "sampling_audit.json").read_text()) == report
        # A repeated diagnostic may refresh its own matching artifacts only.
        again = audit_training_sampling(record, run, output, device="cpu", u_bins=7,
                                        phi_bins=9, plot=False)
        assert report == again and _hashes(run) == before_hashes


def test_no_point_cap_and_plot_with_sparse_zero_count_bins(tmp_path):
    write_rainbow_fixture(tmp_path / "record", frame_rotation=-0.29)
    with RainbowReference(tmp_path / "record") as record:
        _train(record, tmp_path / "run", count=262144)
        report = audit_training_sampling(record, tmp_path / "run", tmp_path / "audit",
                                         device="cpu", u_bins=8, phi_bins=16)
    assert report["sample_count"] == 262144
    assert report["histogram"]["precast_count_sum"] == 262144
    assert report["histogram"]["training_cast_count_sum"] == 262144
    assert report["histogram"]["bins_with_expected_count_below_5"] > 0
    assert report["histogram"]["observed_precast_points_in_zero_mass_bins"] == 0
    assert (tmp_path / "audit/sampling_histogram.png").stat().st_size > 1000


def test_mismatched_run_or_configuration_refused_before_artifact_changes(tmp_path):
    write_rainbow_fixture(tmp_path / "record")
    run, output = tmp_path / "run", tmp_path / "audit"
    with RainbowReference(tmp_path / "record") as record:
        _train(record, run)
        audit_training_sampling(record, run, output, device="cpu", plot=False)
        hashes = _hashes(output)
        with pytest.raises(ValueError, match="different run or configuration"):
            audit_training_sampling(record, run, output, device="cpu", u_bins=16, plot=False)
        assert _hashes(output) == hashes
        for overlapping_output in (run, run / "audit", run.parent):
            with pytest.raises(ValueError, match="separate directory"):
                audit_training_sampling(record, run, overlapping_output, device="cpu", plot=False)
        assert not (run / "audit").exists()
        split = json.loads((run / "sample_split.json").read_text())
        split["train_samples"] += 1
        (run / "sample_split.json").write_text(json.dumps(split))
        with pytest.raises(ValueError, match="differs from checkpoint"):
            audit_training_sampling(record, run, tmp_path / "invalid", device="cpu", plot=False)
        assert not (tmp_path / "invalid").exists()


def test_failed_plot_retains_running_identity_and_identical_retry_recovers(tmp_path, monkeypatch):
    import phaseflow.sampling_audit as audit_module

    write_rainbow_fixture(tmp_path / "record")
    run, output = tmp_path / "run", tmp_path / "audit"
    with RainbowReference(tmp_path / "record") as record:
        _train(record, run)
        original_run = _hashes(run)
        original_plot = audit_module._plot_histogram

        def fail_plot(*args, **kwargs):
            raise RuntimeError("intentional plotting failure")

        monkeypatch.setattr(audit_module, "_plot_histogram", fail_plot)
        with pytest.raises(RuntimeError, match="intentional plotting failure"):
            audit_training_sampling(record, run, output, device="cpu", u_bins=4, phi_bins=8)
        incomplete = json.loads((output / "sampling_audit.json").read_text())
        assert incomplete["status"] == "running"
        assert (output / "sampling_histogram.npz").is_file()
        assert not (output / "sampling_histogram.png").exists()
        assert _hashes(run) == original_run
        # Simulate a killed writer's temporary files, which do not belong to a
        # different experiment merely because their artifact write was incomplete.
        orphans = [output / ".audit-stale", output / ".sampling_histogram.png.stale.tmp"]
        for path in orphans:
            path.write_bytes(b"interrupted output")
        incomplete_hashes = _hashes(output)
        with pytest.raises(ValueError, match="different run or configuration"):
            audit_training_sampling(record, run, output, device="cpu", u_bins=8, phi_bins=8)
        assert _hashes(output) == incomplete_hashes
        monkeypatch.setattr(audit_module, "_plot_histogram", original_plot)
        report = audit_training_sampling(record, run, output, device="cpu", u_bins=4, phi_bins=8)
        assert report["status"] == "complete"
        assert report["identity"] == incomplete["identity"]
        assert json.loads((output / "sampling_audit.json").read_text()) == report
        assert (output / "sampling_histogram.png").stat().st_size > 1000
        assert not any(path.exists() for path in orphans)
        assert _hashes(run) == original_run


def test_wilson_intervals_cover_boundary_estimates_and_known_values():
    np.testing.assert_allclose(_wilson_interval(0, 10), [0, 0.2775327998628892], rtol=1e-14)
    np.testing.assert_allclose(_wilson_interval(10, 10), [0.7224672001371107, 1], rtol=1e-14)
    np.testing.assert_allclose(_wilson_interval(5, 10), [0.236593090512564, 0.763406909487436],
                               rtol=1e-14)


@pytest.mark.parametrize("option", [{"u_bins": 0}, {"phi_bins": True}, {"plot": 1}])
def test_invalid_options_fail_before_loading_run(tmp_path, option):
    with pytest.raises(ValueError):
        audit_training_sampling(None, tmp_path / "missing", tmp_path / "output", **option)
    assert not (tmp_path / "output").exists()
