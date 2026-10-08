"""Paired precision diagnostics on synthetic CDFs, not solver/CoopVec tests."""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest
import torch
from test_rainbow import write_rainbow_fixture

from phaseflow.precision import evaluate_precision
from phaseflow.rainbow import RainbowReference
from phaseflow.single_condition import (
    CHECKPOINT_VERSION,
    FAMILY,
    load_single_checkpoint,
)
from phaseflow.sphere_model import SingleConditionSphereFlow, SphereFlowConfig


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def saved_fixture(path, *, dtype=torch.float32, legacy_geometry=False):
    record_path = path / "record"
    write_rainbow_fixture(record_path)
    config = SphereFlowConfig(
        hidden_features=(8, 8),
        num_coupling_layers=2,
        num_bins=8,
        spline_dtype="float64" if legacy_geometry else "model",
        geometry_dtype="float64" if legacy_geometry else "model",
    )
    with RainbowReference(record_path) as reference, torch.random.fork_rng(devices=[]):
        torch.manual_seed(734)
        model = SingleConditionSphereFlow(
            reference.g, float(reference.condition[1]), config, dtype=dtype
        )
        # A nonidentity model is needed to exercise spline and weight-rounding
        # errors. Bounded perturbations keep this arithmetic fixture moderate.
        with torch.no_grad():
            for layer in model.couplings:
                layer.hyper[-1].weight.uniform_(-0.15, 0.15)
                layer.hyper[-1].bias.add_(torch.linspace(-0.1, 0.1, layer.hyper[-1].bias.numel()))
        checkpoint = path / "best.pt"
        torch.save(
            {
                "checkpoint_version": CHECKPOINT_VERSION,
                "family": FAMILY,
                "kind": "inference",
                "model_config": config.to_dict(),
                "model_state": model.state_dict(),
                "physics": {
                    "hg_g": reference.g,
                    "incident_cosine": float(reference.condition[1]),
                    "wavelength_nm": float(reference.condition[0]),
                },
                "dtype": str(dtype).removeprefix("torch."),
                "global_step": 7,
                "dataset_fingerprint": reference.fingerprint(),
                "code_fingerprint": "synthetic_fixed_weights_not_a_training_run",
            },
            checkpoint,
        )
    return checkpoint, record_path


def test_report_preserves_checkpoint_rng_and_precision_runtime(tmp_path):
    checkpoint, record_path = saved_fixture(tmp_path)
    before = checkpoint.read_bytes()
    rng_before = torch.get_rng_state().clone()
    previous_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("medium")
        with RainbowReference(record_path) as reference:
            report = evaluate_precision(
                checkpoint,
                reference,
                tmp_path / "report.json",
                device="cpu",
                samples=129,
                batch_size=31,
            )
        assert torch.get_float32_matmul_precision() == "medium"
    finally:
        torch.set_float32_matmul_precision(previous_precision)
    torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
    assert checkpoint.read_bytes() == before
    assert json.loads((tmp_path / "report.json").read_text()) == report
    assert report["runtime"]["matmul_allow_tf32"] is False
    assert report["checkpoint_step"] == 7
    assert report["weight_quantization"]["emulates_coopvec"] is False
    assert report["common_model_uniform_midpoint_bits"] == 23
    assert report["weight_quantization"]["rounding_error"]["max_abs"] > 0
    for name, variant in report["variants"].items():
        expected = "float64" if name == "float64_reference" else "float32"
        assert set(
            variant["precision"][key] for key in ("conditioner", "spline", "hg_geometry_log_pdf")
        ) == {expected}
        assert variant["base_g"] == report["physics"]["hg_g"]
        # Representative moderate fixture only: not a universal runtime gate.
        assert variant["sample_eval_log_pdf_error"]["max_abs"] < 2e-5
        assert variant["sample_unit_length_error"]["max_abs"] < 5e-7


def test_quantization_delta_and_standard_error_are_paired(tmp_path):
    checkpoint, record_path = saved_fixture(tmp_path)
    count, seed = 257, 590
    with RainbowReference(record_path) as reference:
        report = evaluate_precision(
            checkpoint, reference, device="cpu", samples=count, seed=seed, batch_size=64
        )
        # Independently re-evaluate both models at the documented, common test
        # inputs. Comparing separately generated test sets would give a different SE.
        stream = report["streams"]["teacher"]
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, stream])))
        uniforms = (rng.integers(0, 2**52, size=(count, 2), dtype=np.int64) + 0.5) / 2**52
        points, _ = reference.sample(uniforms)
        original, _ = load_single_checkpoint(checkpoint, device="cpu")
        rounded = copy.deepcopy(original)
        with torch.no_grad():
            for parameter in rounded.parameters():
                parameter.copy_(parameter.half().float())
            x = torch.as_tensor(points, dtype=torch.float32)
            # Match the documented GPU evaluation batches, which can have
            # slightly different FP32 reduction order from one large MLP call.
            deltas = [
                (original.log_prob(x[i : i + 64]) - rounded.log_prob(x[i : i + 64])).double()
                for i in range(0, count, 64)
            ]
        delta = torch.cat(deltas).numpy()
    paired = report["comparisons"]["fp16_weights_vs_float32"]["nll_increase"]
    assert paired["mean"] == pytest.approx(delta.mean(), abs=2e-8)
    assert paired["standard_error"] == pytest.approx(delta.std(ddof=1) / math.sqrt(count), abs=2e-8)
    models = report["variants"]
    assert paired["mean"] == pytest.approx(
        models["fp16_weights_float32_math"]["nll"] - models["float32_math"]["nll"], abs=1e-14
    )


def test_double_reference_retains_fixed_weights_and_common_samples(tmp_path):
    checkpoint, record_path = saved_fixture(tmp_path, dtype=torch.float64)
    with RainbowReference(record_path) as reference:
        first = evaluate_precision(checkpoint, reference, device="cpu", samples=97, batch_size=29)
        repeated = evaluate_precision(
            checkpoint, reference, device="cpu", samples=97, batch_size=29
        )
        another_seed = evaluate_precision(
            checkpoint, reference, device="cpu", samples=97, batch_size=29, seed=2030
        )
    assert first == repeated
    assert first["teacher_points_sha256"] != another_seed["teacher_points_sha256"]
    assert first["model_uniforms_sha256"] != another_seed["model_uniforms_sha256"]
    paired = first["comparisons"]["native_vs_float64"]
    assert paired["heldout_log_pdf_difference"]["max_abs"] == 0
    assert paired["same_uniform_direction_angle_rad"]["max_abs"] == 0


def test_old_mixed_precision_is_labelled_and_not_silently_reloaded_as_fp32(tmp_path):
    checkpoint, record_path = saved_fixture(tmp_path, legacy_geometry=True)
    with RainbowReference(record_path) as reference:
        report = evaluate_precision(checkpoint, reference, device="cpu", samples=67, batch_size=32)
    native = report["variants"]["native_checkpoint"]["precision"]
    assert native == {
        "conditioner": "float32",
        "spline": "float64",
        "hg_geometry_log_pdf": "float64",
        "hg_formulation": "legacy_cosine",
    }
    assert report["variants"]["float32_math"]["precision"]["hg_geometry_log_pdf"] == "float32"
    assert (
        report["variants"]["float64_reference"]["precision"]["hg_formulation"]
        == "endpoint_distances"
    )


def test_rejects_wrong_source_and_protects_checkpoint_and_teacher(tmp_path):
    checkpoint, record_path = saved_fixture(tmp_path)
    write_rainbow_fixture(tmp_path / "other", inclination=25)
    output = tmp_path / "report.json"
    with RainbowReference(tmp_path / "other") as other:
        with pytest.raises(ValueError, match="fingerprint differ"):
            evaluate_precision(checkpoint, other, output, device="cpu", samples=4)
    assert not output.exists()
    with RainbowReference(record_path) as reference:
        for protected in (checkpoint, record_path / "metadata.json", record_path / "phi_cdf.npy"):
            original = protected.read_bytes()
            with pytest.raises(ValueError, match="must not overwrite"):
                evaluate_precision(checkpoint, reference, protected, device="cpu", samples=4)
            assert protected.read_bytes() == original


@pytest.mark.parametrize("invalid, message", [(float("nan"), "nonfinite"), (70_000, "FP16 range")])
def test_unsafe_half_weights_fail_without_clipping_or_output(tmp_path, invalid, message):
    checkpoint, record_path = saved_fixture(tmp_path)
    payload = torch.load(checkpoint, weights_only=True)
    payload["model_state"]["couplings.0.hyper.0.weight"].fill_(invalid)
    torch.save(payload, checkpoint)
    before = checkpoint.read_bytes()
    output = tmp_path / "report.json"
    with (
        RainbowReference(record_path) as reference,
        pytest.raises(FloatingPointError, match=message),
    ):
        evaluate_precision(checkpoint, reference, output, device="cpu", samples=4)
    assert checkpoint.read_bytes() == before
    assert not output.exists()


def test_requested_cuda_is_never_replaced_by_cpu(tmp_path, monkeypatch):
    checkpoint, record_path = saved_fixture(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with RainbowReference(record_path) as reference, pytest.raises(RuntimeError, match="CUDA"):
        evaluate_precision(checkpoint, reference, samples=4)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device is unavailable")
def test_precision_comparison_on_cuda(tmp_path):
    checkpoint, record_path = saved_fixture(tmp_path)
    with RainbowReference(record_path) as reference:
        report = evaluate_precision(
            checkpoint, reference, device="cuda", samples=257, batch_size=64
        )
    assert report["runtime"]["device"].startswith("cuda")
    for variant in report["variants"].values():
        assert math.isfinite(variant["nll"])
        assert variant["sample_eval_log_pdf_error"]["max_abs"] < 5e-5
