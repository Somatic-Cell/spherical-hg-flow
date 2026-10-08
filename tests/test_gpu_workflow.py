"""GPU execution contracts and CUDA integration gates for synthetic Rainbow data.

CUDA-marked tests require real hardware. CPU tests validate fail-fast defaults
and legacy checkpoint interpretation; they do not establish CUDA performance.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from test_single_condition import smooth_teacher

import phaseflow.single_condition as workflow
from phaseflow.rainbow import RainbowReference
from phaseflow.single_condition import (
    FAMILY,
    SingleTrainingConfig,
    evaluate_single_condition,
    load_single_checkpoint,
    read_single_checkpoint,
    train_single_condition,
)
from phaseflow.sphere_model import SingleConditionSphereFlow, SphereFlowConfig

CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device; CPU is not a substitute"
)


def gpu_model_config(fp32_geometry: bool = False) -> SphereFlowConfig:
    return SphereFlowConfig(
        hidden_features=(16, 16), num_coupling_layers=2, num_bins=8,
        spline_dtype="model" if fp32_geometry else "float64",
        geometry_dtype="model" if fp32_geometry else "float64",
    )


def gpu_training_config(*, steps: int, eval_every: int = 4) -> SingleTrainingConfig:
    return SingleTrainingConfig(
        seed=415,
        device="cuda:0",
        dtype="float32",
        train_samples=1024,
        validation_samples=1024,
        test_samples=1024,
        proposal_samples=128,
        batch_size=256,
        steps=steps,
        learning_rate=0.005,
        eval_every=eval_every,
        checkpoint_every=eval_every,
        eval_batch_size=256,
    )


def assert_tensor_state_equal(left, right) -> None:
    """Require exact state equality, including optimizer and generator tensors."""
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tensor_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            assert_tensor_state_equal(first, second)
    else:
        assert left == right


def test_shipped_training_defaults_require_cuda_and_float32_conditioners():
    defaults = SingleTrainingConfig()
    assert defaults.device == "cuda" and defaults.dtype == "float32"
    path = Path(__file__).resolve().parents[1] / "configs" / "rainbow_single.json"
    shipped = json.loads(path.read_text(encoding="utf-8"))
    assert shipped["training"]["device"] == "cuda"
    assert shipped["training"]["dtype"] == "float32"
    assert shipped["model"]["spline_dtype"] == "model"
    assert shipped["model"]["geometry_dtype"] == "model"


def test_default_training_fails_without_cuda_before_sampling_or_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    def unexpected_sampling(*_args, **_kwargs):
        pytest.fail("CUDA validation must precede generation of the large CDF pools")

    monkeypatch.setattr(workflow, "_points", unexpected_sampling)
    output = tmp_path / "run"
    with RainbowReference(smooth_teacher(tmp_path / "record")) as record:
        with pytest.raises(RuntimeError, match="CPU fallback is not automatic"):
            train_single_condition(
                record, gpu_model_config(), SingleTrainingConfig(), output, make_plots=False
            )
    assert not output.exists()


def test_invalid_deterministic_cublas_configuration_fails_before_device_setup(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":invalid")

    def unexpected_device_setup(*_args, **_kwargs):
        pytest.fail("invalid deterministic cuBLAS configuration must fail before CUDA setup")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_device_setup)
    with RainbowReference(smooth_teacher(tmp_path / "record")) as record:
        with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
            train_single_condition(
                record,
                gpu_model_config(),
                SingleTrainingConfig(),
                tmp_path / "run",
                make_plots=False,
            )
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_version2_checkpoint_loads_on_explicit_cpu_without_reinterpreting_precision(
    tmp_path, monkeypatch, dtype
):
    # This primitive/tensor fixture uses a real model state in the old format;
    # it deliberately omits the newly introduced spline_dtype configuration.
    torch.manual_seed(214)
    config = SphereFlowConfig(hidden_features=(8,), num_coupling_layers=2, num_bins=8)
    g = 0.198_721_096_450_612_6
    eta = -0.138_276_463_281_792
    original = SingleConditionSphereFlow(g, eta, config, dtype=dtype)
    with torch.no_grad():
        for coupling in original.couplings:
            coupling.hyper[-1].weight.normal_(0, 0.04)
    old_config = config.to_dict()
    old_config.pop("spline_dtype")
    state = copy.deepcopy(original.state_dict())
    state["_extra_state"]["config"].pop("spline_dtype")
    path = tmp_path / "version2.pt"
    torch.save(
        {
            "checkpoint_version": 2,
            "family": FAMILY,
            "kind": "inference",
            "model_config": old_config,
            "model_state": state,
            "physics": {"hg_g": g, "incident_cosine": eta, "wavelength_nm": 550.0},
            "dtype": str(dtype).removeprefix("torch."),
            "global_step": 0,
            "dataset_fingerprint": "synthetic-version2-compatibility-fixture",
            "code_fingerprint": "synthetic-version2-compatibility-fixture",
        },
        path,
    )
    restored, payload = load_single_checkpoint(path, device="cpu")
    assert payload["checkpoint_version"] == 2
    assert restored.device == torch.device("cpu")
    assert restored.dtype == restored.spline_dtype == dtype
    assert restored.config.spline_dtype == "model"
    assert restored.hg_g == g and restored.incident_cosine == eta
    uniforms = torch.tensor([[0.13, 0.24], [0.46, 0.68], [0.87, 0.92]], dtype=dtype)
    for expected, actual in zip(
        original.sample_from_uniform(uniforms), restored.sample_from_uniform(uniforms), strict=True
    ):
        assert torch.equal(expected, actual)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU fallback is not automatic"):
        load_single_checkpoint(path)


@pytest.mark.cuda
@CUDA_REQUIRED
@pytest.mark.parametrize("fp32_geometry", [False, True])
def test_cuda_resident_point_pools_learning_and_evaluation(tmp_path, monkeypatch, fp32_geometry):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device("cuda", 0)
    coordinate_dtype = torch.float32 if fp32_geometry else torch.float64
    streams = Counter()
    transfers = Counter()
    training_calls = []
    original_points = workflow._points
    original_device_points = workflow._device_points
    original_log_prob = SingleConditionSphereFlow.log_prob

    def track_points(reference, count, seed, stream):
        streams[stream] += 1
        return original_points(reference, count, seed, stream)

    def check_pool_transfer(points, destination, geometry_dtype=torch.float64):
        assert destination == device
        assert geometry_dtype == coordinate_dtype
        result = original_device_points(points, destination, geometry_dtype)
        for source, resident, dtype in zip(
            points, result, (coordinate_dtype, torch.float64), strict=True
        ):
            assert resident.device == device and resident.dtype == dtype
            if isinstance(source, torch.Tensor):
                # Validation visits must reuse the original device pool.
                assert source.device == device and source.data_ptr() == resident.data_ptr()
                transfers["resident"] += 1
            else:
                assert isinstance(source, np.ndarray)
                transfers["host"] += 1
        return result

    def check_model_inputs(model, outgoing):
        assert outgoing.device == device and outgoing.dtype == coordinate_dtype
        assert model.dtype == torch.float32 and model.spline_dtype == coordinate_dtype
        assert model.geometry_dtype == coordinate_dtype
        if torch.is_grad_enabled():
            training_calls.append(len(outgoing))
            assert not model.validate_args
        return original_log_prob(model, outgoing)

    monkeypatch.setattr(workflow, "_points", track_points)
    monkeypatch.setattr(workflow, "_device_points", check_pool_transfer)
    monkeypatch.setattr(SingleConditionSphereFlow, "log_prob", check_model_inputs)
    cfg = gpu_training_config(steps=32, eval_every=8)
    with RainbowReference(smooth_teacher(tmp_path / "record")) as record:
        result = train_single_condition(
            record, gpu_model_config(fp32_geometry), cfg, tmp_path / "run", make_plots=False
        )
        assert result.complete and result.model.device == device
        assert result.model.hg_g == record.g
        assert training_calls == [cfg.batch_size] * cfg.steps
        assert streams == {"train": 1, "validation": 1, "test": 1}
        # Both arrays of validation/test transfer once, regardless of updates.
        # Training transfers directions directly, without unused teacher PDFs.
        assert transfers["host"] == 4 and transfers["resident"] >= 2
        initial = result.metrics["initial_validation"]
        selected = result.metrics["best_validation"]
        assert selected["nll"] < initial["nll"] - 0.005
        assert result.metrics["test"]["nll_improvement_over_hg"] > 0.005
        report = evaluate_single_condition(
            result.model, record, samples=256, proposal_samples=128, batch_size=64, seed=913
        )
        assert report["runtime"]["device"] == "cuda:0"
        assert report["precision"] == {
            "conditioner": "float32",
            "spline": str(coordinate_dtype).removeprefix("torch."),
            "hg_geometry_log_pdf": str(coordinate_dtype).removeprefix("torch."),
            "teacher_and_statistical_reduction": "float64",
        }
        assert report["base_g"] == record.g
        assert report["proposal"]["uniform_midpoint_bits"] == (23 if fp32_geometry else 52)
        assert 0 < report["proposal"]["relative_ess"] <= 1.00000000001
        assert report["proposal"]["sample_eval_log_pdf_max_abs_error"] < (1e-4 if fp32_geometry else 1e-5)


@pytest.mark.cuda
@CUDA_REQUIRED
@pytest.mark.parametrize("fp32_geometry", [False, True])
def test_cuda_resume_restores_optimizer_and_generator_exactly_at_unscheduled_boundary(
    tmp_path, monkeypatch, fp32_geometry
):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    checked_optimizer_loads = []
    original_load = torch.optim.Adam.load_state_dict

    def check_loaded_optimizer(optimizer, state_dict):
        result = original_load(optimizer, state_dict)
        assert optimizer.state
        for parameter, state in optimizer.state.items():
            assert parameter.device == torch.device("cuda", 0)
            assert parameter.dtype == torch.float32
            for name in ("exp_avg", "exp_avg_sq"):
                assert state[name].device == parameter.device
                assert state[name].dtype == parameter.dtype
            # Adam's step counter may remain CPU when capturable=False.
            assert int(state["step"]) == 5
        checked_optimizer_loads.append(True)
        return result

    monkeypatch.setattr(torch.optim.Adam, "load_state_dict", check_loaded_optimizer)
    cfg = gpu_training_config(steps=12)
    with RainbowReference(smooth_teacher(tmp_path / "record")) as record:
        full = train_single_condition(
            record, gpu_model_config(fp32_geometry), cfg, tmp_path / "full", make_plots=False
        )
        partial = train_single_condition(
            record,
            gpu_model_config(fp32_geometry),
            cfg,
            tmp_path / "resumed",
            max_steps_this_run=5,
            make_plots=False,
        )
        assert not partial.complete and partial.metrics["test"] is None
        interrupted = read_single_checkpoint(partial.checkpoint_path)
        assert interrupted["checkpoint_version"] == workflow.CHECKPOINT_VERSION
        assert interrupted["minibatch_rng_state"].dtype == torch.uint8
        assert interrupted["sample_split"]["minibatch_rng"]["device"] == "cuda:0"
        assert [entry["global_step"] for entry in interrupted["history"] if "validation" in entry] == [
            0, 4
        ]
        continued = train_single_condition(
            record,
            gpu_model_config(fp32_geometry),
            cfg,
            tmp_path / "resumed",
            resume=partial.checkpoint_path,
            make_plots=False,
        )
        assert checked_optimizer_loads == [True]
        expected = read_single_checkpoint(full.checkpoint_path)
        actual = read_single_checkpoint(continued.checkpoint_path)
        for key in (
            "model_state", "best_state", "optimizer_state", "minibatch_rng_state", "rng_state"
        ):
            assert_tensor_state_equal(expected[key], actual[key])
        assert expected["history"] == actual["history"]
        assert full.metrics == continued.metrics
        assert_tensor_state_equal(full.model.state_dict(), continued.model.state_dict())
        restored, _ = load_single_checkpoint(continued.best_path, device="cuda:0")
        assert restored.hg_g == record.g
        assert restored.dtype == torch.float32
        assert restored.spline_dtype == (torch.float32 if fp32_geometry else torch.float64)
        assert_tensor_state_equal(restored.state_dict(), continued.model.state_dict())
