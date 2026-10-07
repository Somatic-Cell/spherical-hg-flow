"""Real native inference comparisons, including nonidentity conditional flows."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from phaseflow.export import COUPLING, DENSE, HEADER, NETWORK, export_model
from phaseflow.model import ModelConfig, PhaseFlow

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def native_cli(tmp_path_factory):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("A C++20 compiler is required for native inference tests")
    executable = tmp_path_factory.mktemp("native") / "phaseflow_cli"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            f"-I{ROOT / 'cpp/include'}",
            str(ROOT / "cpp/src/model.cpp"),
            str(ROOT / "cpp/src/phaseflow_cli.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return executable


def _model(*, blob_bins=4, include_g=True, sharp_g=None):
    torch.manual_seed(9121)
    model = PhaseFlow(
        ModelConfig(
            g_hidden_features=(9, 7),
            hidden_features=(11, 9),
            num_coupling_layers=4,
            num_bins=7,
            one_blob_bins=blob_bins,
            include_g_context=include_g,
        )
    )
    with torch.no_grad():
        model.g_head.mlp[-1].weight.normal_(0, 0.8)
        model.g_head.mlp[-1].bias.fill_(0.4)
        for coupling in model.couplings:
            # Exercise coupled, condition-dependent nonuniform bins and slopes.
            coupling.hyper[-1].weight.normal_(0, 0.18)
            coupling.hyper[-1].bias.add_(0.15 * torch.randn_like(coupling.hyper[-1].bias))
        if sharp_g is not None:
            model.g_head.mlp[-1].weight.zero_()
            model.g_head.mlp[-1].bias.fill_(float(np.arctanh(sharp_g / model.config.g_limit)))
    # Starting from FP32 ensures no hidden extra precision in the exported weights.
    return model.eval()


def _run(native_cli, binary, operation, rows):
    lines = [
        operation + " " + " ".join(format(float(value), ".17g") for value in row) for row in rows
    ]
    result = subprocess.run(
        [str(native_cli), str(binary)],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    return np.array([[float(value) for value in row.split()] for row in result.stdout.splitlines()])


@pytest.mark.integration
@pytest.mark.parametrize("blob_bins,include_g", [(4, True), (0, False), (5, False)])
def test_native_matches_python_nonidentity(tmp_path, native_cli, blob_bins, include_g):
    model = _model(blob_bins=blob_bins, include_g=include_g)
    directory = export_model(model, tmp_path / "export")
    binary = directory / "model.pflow"
    model.double()
    conditions = torch.tensor(
        [
            [380.0, -1.0],
            [720.0, 1.0],
            [517.125, -0.37],
            [619.5, 0.213],
            [550.0, -0.125],
            [400.0, 0.875],
            [700.0, -0.8],
            [430.0, 0.0],
        ],
        dtype=torch.float64,
    )
    uniforms = torch.tensor(
        [
            [0.001, 0.0],
            [0.999, 0.5],
            [0.237, 0.134],
            [0.781, 0.634],
            [0.463, 0.499],
            [0.031, 0.999],
            [0.918, 0.284],
            [0.543, 0.712],
        ],
        dtype=torch.float64,
    )
    with torch.no_grad():
        encoded = model.encoder(conditions).numpy()
        g = model.hg_g(conditions).numpy()
        directions, log_pdf = model.sample_from_uniform(uniforms, conditions)
        evaluated = model.log_prob(directions, conditions).numpy()
    native_encoded = _run(native_cli, binary, "encode", conditions.numpy())
    native_g = _run(native_cli, binary, "g", conditions.numpy())[:, 0]
    native_samples = _run(native_cli, binary, "sample", np.c_[conditions.numpy(), uniforms.numpy()])
    np.testing.assert_allclose(native_encoded, encoded, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(native_g, g, rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(native_samples[:, :3], directions.numpy(), rtol=5e-10, atol=5e-11)
    np.testing.assert_allclose(native_samples[:, 3], log_pdf.numpy(), rtol=5e-10, atol=5e-10)
    np.testing.assert_allclose(native_samples[:, 5], g, rtol=2e-11, atol=2e-12)
    native_evaluated = _run(
        native_cli, binary, "eval", np.c_[conditions.numpy(), directions.numpy()]
    )
    np.testing.assert_allclose(native_evaluated[:, 0], evaluated, rtol=5e-10, atol=5e-10)
    np.testing.assert_allclose(native_evaluated[:, 1], np.exp(evaluated), rtol=5e-10, atol=5e-10)
    native_roundtrip = _run(
        native_cli, binary, "eval", np.c_[conditions.numpy(), native_samples[:, :3]]
    )
    np.testing.assert_allclose(native_roundtrip[:, 0], native_samples[:, 3], rtol=5e-9, atol=5e-9)
    reflected = native_samples[:, :3].copy()
    reflected[:, 1] *= -1
    native_reflected = _run(native_cli, binary, "eval", np.c_[conditions.numpy(), reflected])
    np.testing.assert_allclose(native_reflected, native_roundtrip, rtol=5e-13, atol=5e-13)


@pytest.mark.integration
@pytest.mark.parametrize("sharp_g", [-0.9989, 0.9989])
def test_native_high_anisotropy_retains_double_directions(tmp_path, native_cli, sharp_g):
    model = _model(sharp_g=sharp_g)
    binary = export_model(model, tmp_path / "export") / "model.pflow"
    model.double()
    conditions = torch.tensor([[550.0, 0.3]] * 4, dtype=torch.float64)
    uniforms = torch.tensor(
        [[0.17, 0.21], [0.63, 0.84], [0.90, 0.36], [0.01, 0.50]], dtype=torch.float64
    )
    with torch.no_grad():
        directions, log_pdf = model.sample_from_uniform(uniforms, conditions)
    samples = _run(native_cli, binary, "sample", np.c_[conditions.numpy(), uniforms.numpy()])
    np.testing.assert_allclose(samples[:, :3], directions.numpy(), rtol=1e-8, atol=2e-10)
    np.testing.assert_allclose(samples[:, 3], log_pdf.numpy(), rtol=1e-8, atol=1e-7)
    evaluation = _run(native_cli, binary, "eval", np.c_[conditions.numpy(), samples[:, :3]])
    np.testing.assert_allclose(evaluation[:, 0], samples[:, 3], rtol=1e-8, atol=1e-7)
    assert np.all(np.abs(samples[:, 2]) < 1)


@pytest.mark.integration
def test_native_matches_default_fp32_network_arithmetic(tmp_path, native_cli):
    model = _model()
    binary = export_model(model, tmp_path / "export") / "model.pflow"
    generator = torch.Generator().manual_seed(701)
    conditions = torch.rand((32, 2), generator=generator)
    conditions[:, 0] = 380 + 340 * conditions[:, 0]
    conditions[:, 1] = 2 * conditions[:, 1] - 1
    uniforms = 0.01 + 0.98 * torch.rand((32, 2), generator=generator)
    with torch.no_grad():
        directions, log_pdf = model.sample_from_uniform(uniforms, conditions)
        g = model.hg_g(conditions)
        evaluated = model.log_prob(directions, conditions)
    native_samples = _run(native_cli, binary, "sample", np.c_[conditions.numpy(), uniforms.numpy()])
    native_evaluated = _run(
        native_cli, binary, "eval", np.c_[conditions.numpy(), directions.numpy()]
    )
    # These tolerances cover FP32 network and RQS rounding, not a wrong transform.
    np.testing.assert_allclose(native_samples[:, :3], directions.numpy(), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(native_samples[:, 3], log_pdf.numpy(), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(native_samples[:, 5], g.numpy(), rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(native_evaluated[:, 0], evaluated.numpy(), rtol=2e-5, atol=2e-5)


@pytest.mark.integration
def test_native_axial_incidence_is_azimuth_invariant(tmp_path, native_cli):
    binary = export_model(_model(), tmp_path / "export") / "model.pflow"
    azimuths = np.linspace(-np.pi, np.pi, 31)
    cosine = 0.271
    radius = np.sqrt((1 - cosine) * (1 + cosine))
    directions = np.c_[
        radius * np.cos(azimuths), radius * np.sin(azimuths), np.full_like(azimuths, cosine)
    ]
    for incident_cosine in (-1.0, 1.0):
        conditions = np.tile([550.0, incident_cosine], (len(azimuths), 1))
        evaluated = _run(native_cli, binary, "eval", np.c_[conditions, directions])
        np.testing.assert_allclose(evaluated[:, 0], evaluated[0, 0], rtol=2e-12, atol=2e-12)
        uniforms = np.c_[np.full_like(azimuths, 0.413), np.linspace(0.01, 0.99, len(azimuths))]
        sampled = _run(native_cli, binary, "sample", np.c_[conditions, uniforms])
        np.testing.assert_allclose(sampled[:, 2], sampled[0, 2], rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(sampled[:, 3], sampled[0, 3], rtol=2e-12, atol=2e-12)


def test_manifest_and_safetensors_match_native_payload(tmp_path):
    model = _model()
    directory = export_model(model, tmp_path / "export")
    binary = (directory / "model.pflow").read_bytes()
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["native_bytes"] == len(binary)
    assert manifest["native_sha256"] == hashlib.sha256(binary).hexdigest()
    assert (
        manifest["weights_sha256"]
        == hashlib.sha256((directory / "weights.safetensors").read_bytes()).hexdigest()
    )
    tensors = load_file(str(directory / "weights.safetensors"))
    header = HEADER.unpack_from(binary)
    parameter_count = header[4]
    dense_count, network_count, coupling_count = header[5:8]
    parameter_start = (
        HEADER.size
        + NETWORK.size * network_count
        + DENSE.size * dense_count
        + COUPLING.size * coupling_count
    )
    parameters = np.frombuffer(binary, dtype="<f4", offset=parameter_start)
    assert len(parameters) == parameter_count
    for descriptor in manifest["tensors"]:
        name = descriptor["name"]
        start = descriptor["element_offset"]
        count = descriptor["element_count"]
        np.testing.assert_array_equal(
            parameters[start : start + count], tensors[name].numpy().ravel()
        )
        assert list(tensors[name].shape) == descriptor["shape"]
        assert tensors[name].dtype == torch.float32
    assert manifest["condition_order"] == ["wavelength_nm", "incident_cosine"]
    assert manifest["encoding"]["renormalize_boundary_mass"] is False


@pytest.mark.integration
def test_native_rejects_corrupted_binary(tmp_path, native_cli):
    binary = (export_model(_model(), tmp_path / "export") / "model.pflow").read_bytes()
    header = HEADER.unpack_from(binary)
    network_count = header[6]
    dense_start = HEADER.size + NETWORK.size * network_count
    corruptions = {}
    for name, offset, packing, value in [
        ("magic", 0, "<8s", b"BADMAGIC"),
        ("version", 8, "<I", 2),
        ("sentinel", 12, "<I", 0x04030201),
        ("count", 36, "<I", 0xFFFFFFFF),
        ("spline", 48, "<I", 0),
        ("workspace", 60, "<I", 1),
        ("network_dims", HEADER.size + 8, "<I", 0),
        ("dense_dims", dense_start, "<I", 0),
        ("offset", dense_start + 8, "<Q", 0xFFFFFFFFFFFFFFFF),
        ("nonfinite_weight", len(binary) - 4, "<f", float("nan")),
    ]:
        corrupted = bytearray(binary)
        struct.pack_into(packing, corrupted, offset, value)
        corruptions[name] = corrupted
    corruptions["truncated"] = binary[:-1]
    corruptions["trailing"] = binary + b"\0"
    for name, corrupted in corruptions.items():
        path = tmp_path / f"{name}.pflow"
        path.write_bytes(corrupted)
        result = subprocess.run([str(native_cli), str(path)], capture_output=True, text=True)
        assert result.returncode == 2, name
        assert "Invalid phaseflow model:" in result.stderr, (name, result.stderr)


def test_export_rejects_nonfinite_before_writing(tmp_path):
    model = _model()
    with torch.no_grad():
        model.couplings[0].hyper[-1].weight[0, 0] = float("nan")
    destination = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="nonfinite"):
        export_model(model, destination)
    assert not destination.exists()


def test_export_rejects_unsupported_network(tmp_path):
    model = copy.deepcopy(_model())
    model.g_head.mlp[1] = torch.nn.SiLU()
    with pytest.raises(ValueError, match="must be ReLU"):
        export_model(model, tmp_path / "invalid")


@pytest.mark.integration
def test_native_rejects_invalid_queries(tmp_path, native_cli):
    binary = export_model(_model(), tmp_path / "export") / "model.pflow"
    for query in (
        "g 370 0",
        "g 550 1.001",
        "sample 550 0 0 0.2",
        "sample 550 0 0.5 1",
        "eval 550 0 0 0 0",
        "eval 550 0 0 0 2",
        "sample 550 0 0.2 0.3 unexpected",
    ):
        result = subprocess.run(
            [str(native_cli), str(binary)], input=query + "\n", capture_output=True, text=True
        )
        assert result.returncode == 2, query


@pytest.mark.integration
@pytest.mark.parametrize("sharp_g,u0", [(0.9989, np.nextafter(1.0, 0.0)), (-0.9989, 1e-16)])
def test_native_rejects_rounded_pole_samples(tmp_path, native_cli, sharp_g, u0):
    binary = export_model(_model(sharp_g=sharp_g), tmp_path / "export") / "model.pflow"
    query = f"sample 550 0.3 {u0:.17g} 0.42\n"
    result = subprocess.run(
        [str(native_cli), str(binary)], input=query, capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "Invalid sample query" in result.stderr
