"""Static Windows launch wiring and actual Python planning; not CMD execution."""

from __future__ import annotations

import json
from pathlib import Path

from phaseflow.log_sweep import main

ROOT = Path(__file__).resolve().parents[1]


def test_default_plan_is_seven_fixed_capacity_trials(tmp_path, capsys):
    output = tmp_path / "runs"
    assert main([
        "plan", "--record", str(tmp_path / "missing_cdf"),
        "--config", str(ROOT / "configs/rainbow_log_loss.json"),
        "--sweep-config", str(ROOT / "configs/rainbow_log_sweep.json"),
        "--output", str(output),
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["planned_total_updates"] == 140000
    assert len(result["trials"]) == 7
    assert [row["beta"] for row in result["trials"] if row["kind"] == "beta"] == [
        0.0, 0.01, 0.03, 0.1, 0.3, 1.0,
    ]
    assert result["model"]["num_coupling_layers"] == 16
    assert result["model"]["num_bins"] == 64
    assert result["model"]["hidden_features"] == [64, 64]
    assert result["model"]["geometry_dtype"] == result["model"]["spline_dtype"] == "model"
    assert result["training"]["dtype"] == "float32"
    assert result["training"]["device"] == "cuda"
    assert result["training"]["learning_rate"] == 0.001
    assert result["training"]["train_samples"] == 262144
    assert not output.exists()


def test_batch_uses_shared_interpreter_cuda_preflight_and_standalone_modes():
    source = (ROOT / "sweep_log_loss.bat").read_text(encoding="utf-8")
    assert "setlocal EnableExtensions DisableDelayedExpansion" in source
    assert "call environment.bat" in source
    assert 'check_environment.py" check --device "%DEVICE%"' in source
    assert 'set "DEVICE=cuda:0"' in source
    assert 'pushd "%~dp0"' in source and "popd" in source
    assert "pip install" not in source and ".venv" not in source
    for mode in ("run", "plan", "replot"):
        assert f"-m phaseflow.log_sweep {mode}" in source
    assert 'if not "%ERRORLEVEL%"=="0" goto :failed' in source
    assert 'exit /b %EXIT_CODE%' in source
