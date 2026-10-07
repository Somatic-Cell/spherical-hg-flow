"""Reproduce a complete synthetic point-cloud CLI workflow and native comparison.

This measures implementation behavior, not rainbow physics or renderer speed.
Run from an installed source checkout: python scripts/run_smoke.py --output runs/smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

from phaseflow.cli import load_config
from phaseflow.data import PhasePointCloud
from phaseflow.training import atomic_json, evaluate_model, load_model_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.json"))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists():
        parser.error("output must be a new directory so previous results are preserved")
    output.mkdir(parents=True)
    config_path = args.config.resolve()
    model_config, training_config = load_config(config_path)
    torch.set_num_threads(training_config.cpu_threads)
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(training_config.cpu_threads)
    environment["MKL_NUM_THREADS"] = str(training_config.cpu_threads)
    commands = []

    def cli(*arguments: str) -> dict:
        command = [sys.executable, "-m", "phaseflow", *map(str, arguments)]
        commands.append(command)
        completed = subprocess.run(
            command, cwd=project, env=environment, text=True, capture_output=True, check=True
        )
        return json.loads(completed.stdout)

    data = output / "points.npz"
    run = output / "training"
    exported = output / "export"
    cli(
        "demo",
        "--output",
        data,
        "--wavelengths",
        "550",
        "--incident-cosines",
        "-0.75",
        "-0.25",
        "0.25",
        "0.75",
        "--points-per-condition",
        "2048",
    )
    start = time.perf_counter()
    trained = cli("train", "--data", data, "--config", config_path, "--output", run, "--quiet")
    training_seconds = time.perf_counter() - start
    model, checkpoint = load_model_checkpoint(run / "checkpoint.pt")
    expected_steps = {
        "warmup": training_config.warmup_steps,
        "residual": training_config.residual_steps,
        "joint": training_config.joint_steps,
    }
    if checkpoint["completed"] != expected_steps or trained["completed"] != expected_steps:
        raise AssertionError("saved checkpoint did not reach the requested final step")
    cloud = PhasePointCloud.load_npz(data)
    split = checkpoint["split"]
    evaluated_split = "validation" if split["validation"] else "train"
    reloaded = evaluate_model(
        model,
        cloud,
        groups=split[evaluated_split],
        batch_size=training_config.eval_batch_size,
    )
    reload_error = abs(reloaded["mean_nll"] - trained["metrics"][evaluated_split]["mean_nll"])
    if reload_error > 1e-10:
        raise AssertionError(f"checkpoint reload changed the NLL: {reload_error}")
    evaluated = cli(
        "evaluate",
        "--data",
        data,
        "--checkpoint",
        run / "checkpoint.pt",
        "--split",
        evaluated_split,
        "--sample-count",
        "2048",
        "--output",
        output / "evaluation.json",
    )
    cli("export", "--checkpoint", run / "checkpoint.pt", "--output", exported)
    compiler = shutil.which("g++") or shutil.which("clang++")
    native = {"status": "skipped_no_cxx_compiler"}
    if compiler:
        executable = output / "phaseflow_cli"
        compile_command = [
            compiler,
            "-std=c++20",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-I" + str(project / "cpp/include"),
            str(project / "cpp/src/model.cpp"),
            str(project / "cpp/src/phaseflow_cli.cpp"),
            "-o",
            str(executable),
        ]
        subprocess.run(compile_command, text=True, capture_output=True, check=True)
        # Export quantizes every learned tensor to FP32. This run already trained
        # FP32; round explicitly so custom float64 configs use the same weights.
        reference = model.float().double()
        conditions = torch.tensor(
            [[550.0, -0.75], [550.0, -0.25], [550.0, 0.25], [550.0, 0.75]],
            dtype=torch.float64,
        )
        uniforms = torch.tensor(
            [[0.13, 0.17], [0.49, 0.51], [0.78, 0.34], [0.91, 0.89]],
            dtype=torch.float64,
        )
        with torch.no_grad():
            directions, log_pdf = reference.sample_from_uniform(uniforms, conditions)
            evaluated_log_pdf = reference.log_prob(directions, conditions)
        lines = []
        for condition, uniform in zip(conditions.tolist(), uniforms.tolist()):
            lines.append("sample " + " ".join(map(repr, condition + uniform)))
        for condition, direction in zip(conditions.tolist(), directions.tolist()):
            lines.append("eval " + " ".join(map(repr, condition + direction)))
        result = subprocess.run(
            [str(executable), str(exported / "model.pflow")],
            input="\n".join(lines) + "\n",
            text=True,
            capture_output=True,
            check=True,
        )
        parsed = [list(map(float, line.split())) for line in result.stdout.splitlines()]
        if len(parsed) != 8:
            raise AssertionError("native CLI did not return every requested result")
        samples = torch.tensor(parsed[:4], dtype=torch.float64)
        evaluations = torch.tensor(parsed[4:], dtype=torch.float64)
        direction_error = float((samples[:, :3] - directions).abs().max())
        sample_error = float((samples[:, 3] - log_pdf).abs().max())
        eval_error = float((evaluations[:, 0] - evaluated_log_pdf).abs().max())
        if max(direction_error, sample_error, eval_error) > 1e-8:
            raise AssertionError("trained-weight native comparison exceeded its tolerance")
        native = {
            "status": "passed",
            "queries": 8,
            "comparison": "FP32_weights_with_double_arithmetic",
            "max_abs_direction_error": direction_error,
            "max_abs_sample_log_pdf_error": sample_error,
            "max_abs_eval_log_pdf_error": eval_error,
            "compiler": subprocess.run(
                [compiler, "--version"], text=True, capture_output=True, check=True
            ).stdout.splitlines()[0],
        }
    source_hash = hashlib.sha256()
    for path in sorted((project / "src/phaseflow").glob("*.py")):
        source_hash.update(path.name.encode())
        source_hash.update(path.read_bytes())
    report = {
        "purpose": "synthetic_implementation_validation_not_rainbow_physics",
        "model": model_config.to_dict(),
        "training": training_config.to_dict(),
        "runtime": checkpoint["runtime"],
        "source_sha256": source_hash.hexdigest(),
        "data": cloud.summary(),
        "split": split,
        "completed": checkpoint["completed"],
        "initial_metrics": checkpoint["history"][0]["metrics"],
        "final_metrics": trained["metrics"],
        "evaluation": evaluated,
        "checkpoint_reload_nll_abs_error": reload_error,
        "training_cli_wall_seconds": training_seconds,
        "timing_scope": "CPU CLI including imports, setup, evaluations and checkpoint writes",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "export_bytes": (exported / "model.pflow").stat().st_size,
        "native": native,
        "commands": commands,
        "cuda_optix_validation": "not_run",
    }
    atomic_json(output / "workflow_report.json", report)
    print(
        json.dumps(
            {
                "report": str(output / "workflow_report.json"),
                "completed": report["completed"],
                "validation_hg_nll": evaluated["mean_fitted_hg_nll"],
                "validation_nf_nll": evaluated["mean_nll"],
                "reload_nll_abs_error": reload_error,
                "native": native,
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
