from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from phaseflow.data import PhasePointCloud, split_conditions
from phaseflow.model import ModelConfig, PhaseFlow
from phaseflow.synthetic import make_demo
from phaseflow.training import TrainingConfig, evaluate_model, load_model_checkpoint, train_model


def _small_model() -> ModelConfig:
    return ModelConfig(
        g_hidden_features=(12,),
        hidden_features=(24, 24),
        num_coupling_layers=2,
        num_bins=8,
        one_blob_bins=8,
    )


def _cloud():
    return make_demo(
        mode="quadrature",
        wavelengths_nm=(450.0, 650.0),
        incident_cosines=(-0.4, 0.5),
        n_mu=24,
        n_phi=32,
    )


def test_hg_warmup_and_residual_training_reduce_their_actual_objectives(tmp_path) -> None:
    config = TrainingConfig(
        seed=31,
        batch_size=384,
        warmup_steps=60,
        residual_steps=100,
        lr_g=0.01,
        lr_flow=0.004,
        validation_fraction=0.25,
        eval_every=0,
        checkpoint_every=0,
    )
    result = train_model(_cloud(), _small_model(), config, tmp_path / "learn")
    initial = result.history[0]["metrics"]["train"]
    warmup = next(item for item in reversed(result.history) if item.get("stage") == "warmup")[
        "metrics"
    ]["train"]
    final = result.metrics["train"]
    assert warmup["base_g_mse"] < initial["base_g_mse"] * 0.15
    assert final["mean_nll"] < warmup["mean_nll"] - 0.005
    assert final["mean_nll"] < final["mean_fitted_hg_nll"] - 0.005
    assert abs(final["base_g_mse"] - warmup["base_g_mse"]) < 1e-14
    assert result.metrics["validation"]["scope"] == "held_out_conditions"
    assert not set(result.split["train"]).intersection(result.split["validation"])
    model, checkpoint = load_model_checkpoint(result.checkpoint_path)
    assert checkpoint["completed"] == {"warmup": 60, "residual": 100, "joint": 0}
    evaluated = evaluate_model(model, _cloud(), groups=result.split["validation"], sample_count=64)
    assert np.isfinite(evaluated["mean_nll"])
    assert all(
        item["sample_eval_log_pdf_max_abs_error"] < 0.005 for item in evaluated["conditions"]
    )


@pytest.mark.parametrize(
    ("dtype", "joint_steps", "stop_at"), [("float64", 0, 7), ("float32", 4, 12)]
)
def test_resume_matches_uninterrupted_cpu_updates_bit_for_bit(
    tmp_path, dtype, joint_steps, stop_at
) -> None:
    config = TrainingConfig(
        seed=17,
        dtype=dtype,
        batch_size=64,
        warmup_steps=4,
        residual_steps=6,
        joint_steps=joint_steps,
        lr_g=0.004,
        lr_flow=0.002,
        validation_fraction=0.25,
        eval_every=3,
        checkpoint_every=2,
    )
    cloud = _cloud()
    uninterrupted = train_model(cloud, _small_model(), config, tmp_path / "uninterrupted")
    interrupted = train_model(
        cloud, _small_model(), config, tmp_path / "resumed", max_steps_this_run=stop_at
    )
    assert sum(interrupted.completed.values()) == stop_at
    assert interrupted.completed["joint"] == (2 if joint_steps else 0)
    resumed = train_model(
        cloud, _small_model(), config, tmp_path / "resumed", resume=interrupted.checkpoint_path
    )
    for name, tensor in uninterrupted.model.state_dict().items():
        assert torch.equal(tensor, resumed.model.state_dict()[name]), name
    assert uninterrupted.metrics == resumed.metrics
    assert uninterrupted.completed == resumed.completed


def test_resume_rejects_changed_plan_or_dataset(tmp_path) -> None:
    cloud = _cloud()
    config = TrainingConfig(
        warmup_steps=1, residual_steps=1, batch_size=32, eval_every=0, checkpoint_every=0
    )
    interrupted = train_model(
        cloud, _small_model(), config, tmp_path / "original", max_steps_this_run=1
    )
    with pytest.raises(ValueError, match="configuration differs"):
        train_model(
            cloud,
            _small_model(),
            replace(config, lr_flow=0.01),
            tmp_path / "wrong_config",
            resume=interrupted.checkpoint_path,
        )
    changed = make_demo(
        mode="quadrature",
        wavelengths_nm=(450.0, 650.0),
        incident_cosines=(-0.4, 0.5),
        n_mu=24,
        n_phi=32,
        seed=99,
    )
    with pytest.raises(ValueError, match="dataset content"):
        train_model(
            changed,
            _small_model(),
            config,
            tmp_path / "wrong_data",
            resume=interrupted.checkpoint_path,
        )


def test_single_condition_metrics_are_not_claimed_as_generalization(tmp_path) -> None:
    cloud = make_demo(
        mode="quadrature", wavelengths_nm=(550.0,), incident_cosines=(0.5,), n_mu=12, n_phi=16
    )
    config = TrainingConfig(
        batch_size=32,
        warmup_steps=1,
        residual_steps=1,
        validation_fraction=0.0,
        eval_every=0,
        checkpoint_every=0,
    )
    result = train_model(cloud, _small_model(), config, tmp_path / "single")
    assert result.metrics["validation"] is None
    assert result.metrics["train"]["scope"] == "in_sample_conditions"
    assert "do not demonstrate generalization" in result.metrics["validation_note"]


def test_joint_stage_uses_full_nll_and_explicit_moment_regularizer(tmp_path) -> None:
    with pytest.raises(ValueError, match="anchor"):
        TrainingConfig(joint_steps=1, moment_regularization=0.0)
    config = TrainingConfig(
        batch_size=32,
        warmup_steps=1,
        residual_steps=1,
        joint_steps=2,
        moment_regularization=0.37,
        eval_every=0,
        checkpoint_every=0,
    )
    result = train_model(_cloud(), _small_model(), config, tmp_path / "joint")
    joint = [entry for entry in result.history if entry.get("stage") == "joint"]
    assert len(joint) == 2
    for entry in joint:
        assert entry["loss"] == pytest.approx(entry["nll"] + 0.37 * entry["g_moment_mse"], abs=2e-6)
    assert any(parameter.requires_grad for parameter in result.model.hg_parameters())


def test_hg_baseline_matches_identity_flow_for_sharp_lobes_and_rounded_vectors() -> None:
    model = PhaseFlow(replace(_small_model(), g_limit=1.0 - 1e-12)).double()
    with torch.no_grad():
        model.g_head.mlp[-1].bias.fill_(math.atanh((1.0 - 1e-9) / model.config.g_limit))
    z = np.array([1.0, 1.0 - 1e-7, 0.3], dtype=np.float64)
    direction = np.stack((np.sqrt((1.0 - z) * (1.0 + z)), np.zeros_like(z), z), axis=-1)
    direction *= np.array([1.0 + 5e-8, 1.0 + 3e-8, 1.0 - 1e-5])[:, None]
    cloud = PhasePointCloud(
        conditions=np.array([[550.0, 0.3]]),
        outgoing=direction,
        condition_index=np.zeros(3, dtype=np.int64),
    )
    report = evaluate_model(model, cloud)
    assert math.isfinite(report["mean_fitted_hg_nll"])
    # The nominal identity spline parameters are initialized in float32 before
    # .double(); permit that initialization rounding, not a different HG law.
    assert report["mean_fitted_hg_nll"] == pytest.approx(report["mean_nll"], abs=1e-7)


@pytest.mark.parametrize("target_g", [-0.9995, -0.999, 0.999, 0.9995])
@pytest.mark.parametrize("stage", ["warmup", "joint"])
def test_unattainable_training_moment_is_rejected_without_clamping(
    tmp_path, target_g, stage
) -> None:
    cloud = PhasePointCloud(
        conditions=np.array([[550.0, 0.3]]),
        outgoing=np.array([[math.sqrt((1.0 - target_g) * (1.0 + target_g)), 0.0, target_g]]),
        condition_index=np.zeros(1, dtype=np.int64),
    )
    config = TrainingConfig(
        validation_fraction=0.0,
        warmup_steps=1 if stage == "warmup" else 0,
        residual_steps=0,
        joint_steps=1 if stage == "joint" else 0,
    )
    with pytest.raises(ValueError, match="TRAIN condition"):
        train_model(cloud, _small_model(), config, tmp_path / "invalid_moment")
    assert cloud.moment_targets()[0] == target_g
    assert not (tmp_path / "invalid_moment").exists()


def test_unattainable_validation_moment_does_not_leak_into_training_gate(tmp_path) -> None:
    config = TrainingConfig(
        seed=8, validation_fraction=0.5, warmup_steps=1, residual_steps=0, eval_every=0
    )
    split = split_conditions(2, config.validation_fraction, config.seed + 1)
    z = np.full(2, 0.2)
    z[split["validation"]] = 0.9995
    cloud = PhasePointCloud(
        conditions=np.array([[500.0, -0.3], [600.0, 0.3]]),
        outgoing=np.stack((np.sqrt((1.0 - z) * (1.0 + z)), np.zeros_like(z), z), axis=-1),
        condition_index=np.arange(2, dtype=np.int64),
    )
    result = train_model(cloud, _small_model(), config, tmp_path / "validation_only")
    assert result.metrics["validation"]["conditions"][0]["target_moment_g"] == 0.9995


def test_float32_g_limit_rounding_to_one_is_rejected_before_training(tmp_path) -> None:
    model_config = replace(_small_model(), g_limit=1.0 - 1e-9)
    config = TrainingConfig(dtype="float32", warmup_steps=1, residual_steps=0)
    with pytest.raises(ValueError, match="rounds to 1.0"):
        train_model(_cloud(), model_config, config, tmp_path / "invalid_limit")
    assert not (tmp_path / "invalid_limit").exists()
