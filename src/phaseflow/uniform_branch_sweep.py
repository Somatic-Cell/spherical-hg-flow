"""Fork a completed uniform/pure-log training checkpoint without resetting Adam.

This is an additive workflow. It does not modify the trainer, spline, loss, data
streams, or their fingerprint. The *declared* transition changes only the final
step and the post-fork learning rate (in both config and optimizer param groups).
The original checkpoint is copied byte-for-byte and never overwritten.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

SCHEMA = "phaseflow.uniform_branch_sweep.v1"
STATE_FILES = (
    "checkpoint.pt", "best.pt", "best_by_log.pt", "best_by_nll.pt",
    "config.json", "sample_split.json", "history.json", "metrics.json", "branch.json",
)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8-sig") as f:
        return json.load(f)


def equal_state(a: Any, b: Any) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                and a.dtype == b.dtype and a.shape == b.shape
                and torch.equal(a.cpu(), b.cpu()))
    if isinstance(a, dict) or isinstance(b, dict):
        return (isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys()
                and all(equal_state(a[k], b[k]) for k in a))
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return (type(a) is type(b) and len(a) == len(b)
                and all(equal_state(x, y) for x, y in zip(a, b, strict=True)))
    return a == b


def load_payload(path: Path) -> dict[str, Any]:
    p = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(p, dict):
        raise ValueError(f"Not a checkpoint dictionary: {path}")
    return p


def validate_parent(p: dict[str, Any], expected_step: int) -> None:
    required = {
        "checkpoint_version", "family", "kind", "global_step", "model_config",
        "model_state", "training_config", "optimizer_state", "rng_state",
        "minibatch_rng_state", "sample_split", "history", "best_state", "best_step",
        "best_validation", "latest_validation", "runtime", "dtype", "physics",
        "dataset_fingerprint", "code_fingerprint", "objective_config", "selection_metric",
        "selections", "data_provenance",
    }
    missing = required - p.keys()
    if missing:
        raise ValueError(f"Training checkpoint lacks {sorted(missing)}; use checkpoint.pt.")
    if (p["checkpoint_version"] != 5 or p["family"] != "rainbow_single_condition"
            or p["kind"] != "training"):
        raise ValueError("A version-5 resumable Rainbow checkpoint.pt is required, not best.pt.")
    if type(expected_step) is not int or expected_step <= 0:
        raise ValueError("expected parent step must be a positive integer")
    cfg, obj = p["training_config"], p["objective_config"]
    if (type(p["global_step"]) is not int or p["global_step"] != expected_step
            or cfg["steps"] != expected_step):
        raise ValueError(f"Parent must have completed exactly {expected_step} updates.")
    if (obj.get("sampling") != "uniform" or obj.get("target_fraction") != 0.0
            or obj.get("nll_weight") != 0.0 or obj.get("beta") != 1.0
            or obj.get("selection_metric") != "log_rmse"
            or p["selection_metric"] != "log_rmse"):
        raise ValueError("This comparison requires uniform-only pure log, beta=1, log_rmse selection.")
    if p["dtype"] != "float32" or cfg["dtype"] != "float32":
        raise ValueError("This sweep preserves the existing FP32 experiment.")
    if any(p["model_config"].get(k, "model") != "model"
           for k in ("geometry_dtype", "spline_dtype")):
        raise ValueError("Expected model-precision (FP32) geometry and spline arithmetic.")
    groups = p["optimizer_state"].get("param_groups", [])
    states = p["optimizer_state"].get("state", {})
    if not groups or not states:
        raise ValueError("Saved Adam state is empty; resetting Adam is not supported.")
    for group in groups:
        if group.get("lr") != cfg["learning_rate"]:
            raise ValueError("Parent optimizer rate differs from its config; no scheduler assumed.")
        if "betas" not in group:
            raise ValueError("Expected Adam param groups.")
    for state in states.values():
        if not {"step", "exp_avg", "exp_avg_sq"} <= state.keys():
            raise ValueError("Incomplete Adam moment/counter state.")
        if float(state["step"]) != expected_step:
            raise ValueError("Adam update counters do not match the parent step.")
    history = p["history"]
    steps = [row.get("global_step") for row in history]
    if (not steps or steps[0] != 0 or steps[-1] != expected_step
            or any(type(s) is not int for s in steps)
            or any(b <= a for a, b in zip(steps, steps[1:]))):
        raise ValueError("Parent history is missing, non-monotone, or not at the parent step.")
    if "validation" not in history[-1] or history[-1]["validation"] != p["latest_validation"]:
        raise ValueError("Fork at a saved validation boundary; latest validation must match.")
    counts = p["sample_split"].get("training_pool", {}).get("component_counts")
    if counts != {"cdf": 0, "uniform": cfg["train_samples"]}:
        raise ValueError("Saved pool is not all-uniform or its count differs from config.")


def branch_payload(parent: dict[str, Any], *, total_steps: int, learning_rate: float
                   ) -> dict[str, Any]:
    """Copy all state; change only the two declared future-training settings."""
    validate_parent(parent, parent["global_step"])
    if type(total_steps) is not int or total_steps <= parent["global_step"]:
        raise ValueError("total_steps must exceed the completed parent step")
    if (isinstance(learning_rate, bool) or not isinstance(learning_rate, (float, int))
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("learning rate must be finite and positive")
    child = copy.deepcopy(parent)
    child["training_config"]["steps"] = total_steps
    child["training_config"]["learning_rate"] = float(learning_rate)
    for group in child["optimizer_state"]["param_groups"]:
        group["lr"] = float(learning_rate)
    # A reversible whitelist check proves there was no reset or hidden state edit.
    restored = copy.deepcopy(child)
    restored["training_config"] = copy.deepcopy(parent["training_config"])
    for group, old in zip(restored["optimizer_state"]["param_groups"],
                          parent["optimizer_state"]["param_groups"], strict=True):
        group["lr"] = old["lr"]
    if not equal_state(restored, parent):
        raise RuntimeError("Unexpected checkpoint edit outside the declared branch settings")
    return child


def branch_name(rate: float) -> str:
    return "lr_" + format(rate, ".12g")


def make_plan(parent: Path, output: Path, payload: dict[str, Any], *,
              total_steps: int, milestones: list[int], rates: list[float]) -> dict[str, Any]:
    parent, output = parent.resolve(), output.resolve()
    parent_run = parent.parent
    if output == parent_run or parent_run in output.parents or output in parent_run.parents:
        raise ValueError("OUTPUT and the parent run must be disjoint; keep the original run read-only.")
    start = payload["global_step"]
    if (not milestones or milestones != sorted(set(milestones))
            or milestones[0] <= start or milestones[-1] != total_steps
            or any(type(s) is not int for s in milestones)):
        raise ValueError("Milestones must increase after the parent and end at total_steps.")
    if any(s % payload["training_config"]["eval_every"] for s in milestones):
        raise ValueError("Milestones must be on the unchanged validation schedule.")
    if not rates or len(set(map(branch_name, rates))) != len(rates):
        raise ValueError("Learning rates must be nonempty and uniquely named.")
    for lr in rates:
        branch_payload(payload, total_steps=total_steps, learning_rate=lr)
    return {
        "schema": SCHEMA,
        "parent_checkpoint": str(parent), "parent_sha256": file_hash(parent),
        "parent_step": start, "total_steps": total_steps, "milestones": milestones,
        "learning_rates": rates, "output": str(output),
        "additional_updates_per_branch": total_steps - start,
        "additional_updates_total": (total_steps - start) * len(rates),
        "parent_training_config": payload["training_config"],
        "model_config": payload["model_config"], "objective": payload["objective_config"],
        "dataset_fingerprint": payload["dataset_fingerprint"],
        "trainer_code_fingerprint": payload["code_fingerprint"],
        "orchestrator_sha256": file_hash(Path(__file__)),
        "runtime": payload["runtime"],
        "split": payload["sample_split"],
        "selection_metric": "validation_log_rmse", "test_used_for_selection": False,
        "transition": "Preserve weights/Adam moments+step/RNG/history/pools; change horizon and lr only",
        "inherited_best": "All scheduled validation events through the milestone, including the parent",
    }


@contextmanager
def output_lock(root: Path):
    """OS-released lock: a killed process does not leave a stale logical lock."""
    root.mkdir(parents=True, exist_ok=True)
    f = (root / ".sweep.lock").open("a+b")
    try:
        if f.tell() == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, BlockingIOError) as exc:
        f.close()
        raise RuntimeError("Another process is using this sweep output") from exc
    try:
        yield
    finally:
        f.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()


def _api():
    # Deferred imports let plan/metadata tests run without the training package dependencies.
    from . import single_condition as train
    from .log_objective import LogObjectiveConfig
    from .rainbow import RainbowReference
    from .sphere_model import SphereFlowConfig
    return train, LogObjectiveConfig, RainbowReference, SphereFlowConfig


def preflight(parent: dict[str, Any], reference: Any, device: str, *, cpu_test: bool) -> None:
    train, Objective, _, ModelConfig = _api()
    cfg = train.SingleTrainingConfig.from_dict(parent["training_config"])
    objective = Objective.from_dict(parent["objective_config"])
    model_cfg = ModelConfig.from_dict(parent["model_config"])
    if model_cfg.to_dict() != parent["model_config"]:
        raise ValueError("Model config round-trip changed fields; refusing a different architecture.")
    if parent["code_fingerprint"] != train._code_hash():
        raise ValueError(
            "Trainer/model fingerprint differs from the parent. Use the exact checkout used for "
            "uniform training, plus this additive patch. Do not overwrite the saved fingerprint."
        )
    if torch.device(device).type != "cuda" and not cpu_test:
        raise ValueError("CUDA is required; --allow-cpu-test is only for explicit synthetic tests.")
    if torch.device(cfg.device).type == "cuda" and cfg.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    selected = train._resolve_device(device)
    recorded = train._resolve_device(cfg.device)
    if selected != recorded:
        raise ValueError("Device differs from the parent; exact CUDA RNG continuation requires it.")
    train._setup_runtime(cfg)
    if train._runtime(selected, cfg.dtype) != parent["runtime"]:
        raise ValueError("Runtime/library versions differ from the parent. No automatic conversion.")
    if cfg.tensorboard:
        train.check_tensorboard()
    physics = {"hg_g": reference.g, "incident_cosine": float(reference.condition[1]),
               "wavelength_nm": float(reference.condition[0])}
    if physics != parent["physics"] or reference.fingerprint() != parent["dataset_fingerprint"]:
        raise ValueError("CDF/physics differs from the original run")
    train._validate_selections(parent["selections"], parent["history"], parent["global_step"],
                               objective, parent["best_step"], parent["best_validation"],
                               parent["best_state"])
    train.verify_positive_teacher(reference)
    seed = cfg.seed if cfg.data_seed is None else cfg.data_seed
    pool = train.make_training_pool(reference, cfg.train_samples, seed, objective, torch.float32)
    split = parent["sample_split"]
    if (train._array_hash(pool.directions, pool.log_p, pool.components)
            != split["training_points_sha256"]
            or pool.provenance != split["training_pool"]):
        raise ValueError("Regenerated uniform training points/labels do not match parent")
    val = train._points(reference, cfg.validation_samples, seed, "validation")
    uv = train.make_uniform_points(reference, objective.validation_uniform_samples, seed,
                                    "validation", torch.float32)
    if (train._array_hash(*val) != split["validation_points_sha256"]
            or train._array_hash(*uv) != split["uniform_validation_points_sha256"]):
        raise ValueError("Regenerated validation pools differ from the parent")


def init_root(plan: dict[str, Any], root: Path, parent: Path) -> None:
    manifest = root / "manifest.json"
    if manifest.exists():
        if read_json(manifest) != plan:
            raise ValueError("Sweep plan/source/code changed; use a different OUTPUT.")
    else:
        unexpected = [p.name for p in root.iterdir() if p.name != ".sweep.lock"]
        if unexpected:
            raise ValueError(f"Output is not an empty new sweep: {unexpected}")
        atomic_json(manifest, plan)
    snapshot = root / "parent_checkpoint.pt"
    if snapshot.exists():
        if file_hash(snapshot) != plan["parent_sha256"]:
            raise ValueError("Immutable parent snapshot has changed")
    else:
        temporary = root / "parent_checkpoint.pt.tmp"
        shutil.copyfile(parent, temporary)
        if file_hash(temporary) != plan["parent_sha256"]:
            raise ValueError("Parent changed during snapshot creation")
        os.replace(temporary, snapshot)


def init_branch(root: Path, plan: dict[str, Any], parent: dict[str, Any], rate: float) -> Path:
    run = root / branch_name(rate)
    provenance = {"schema": SCHEMA, "parent_sha256": plan["parent_sha256"],
                  "parent_step": parent["global_step"],
                  "learning_rate_before": parent["training_config"]["learning_rate"],
                  "learning_rate_after": rate, "total_steps": plan["total_steps"],
                  "orchestrator_sha256": plan["orchestrator_sha256"],
                  "transition_applies_from_step": parent["global_step"] + 1,
                  "trainer_code_fingerprint": parent["code_fingerprint"],
                  "all_other_optimizer_fields_preserved": True,
                  "no_knot_constraint_added": True}
    if run.exists():
        saved = read_json(run / "branch.json")
        if any(saved.get(k) != v for k, v in provenance.items()):
            raise ValueError(f"Branch provenance mismatch: {run}")
        if file_hash(run / "initial_checkpoint.pt") != saved["initial_checkpoint_sha256"]:
            raise ValueError("Adapted initial checkpoint has changed")
        expected = branch_payload(parent, total_steps=plan["total_steps"], learning_rate=rate)
        if not equal_state(load_payload(run / "initial_checkpoint.pt"), expected):
            raise ValueError("Initial branch state no longer equals the declared parent transition")
        return run
    temporary = Path(tempfile.mkdtemp(prefix=branch_name(rate) + ".tmp-", dir=root))
    try:
        adapted = branch_payload(parent, total_steps=plan["total_steps"], learning_rate=rate)
        torch.save(adapted, temporary / "initial_checkpoint.pt")
        shutil.copyfile(temporary / "initial_checkpoint.pt", temporary / "checkpoint.pt")
        provenance["initial_checkpoint_sha256"] = file_hash(temporary / "initial_checkpoint.pt")
        atomic_json(temporary / "branch.json", provenance)
        os.replace(temporary, run)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return run


def validate_child(child: dict[str, Any], parent: dict[str, Any], plan: dict[str, Any],
                   rate: float) -> None:
    expected = branch_payload(parent, total_steps=plan["total_steps"], learning_rate=rate)
    for k in ("family", "kind", "checkpoint_version", "model_config", "training_config",
              "physics", "dtype", "runtime", "code_fingerprint", "objective_config",
              "selection_metric", "sample_split", "dataset_fingerprint"):
        if not equal_state(child.get(k), expected[k]):
            raise ValueError(f"Branch checkpoint changed {k}")
    if not parent["global_step"] <= child["global_step"] <= plan["total_steps"]:
        raise ValueError("Child step outside the declared continuation interval")
    if not equal_state(child["history"][:len(parent["history"])], parent["history"]):
        raise ValueError("Inherited parent history has been altered")
    for entry in child["history"][len(parent["history"]):]:
        if "learning_rate" in entry and entry["learning_rate"] != rate:
            raise ValueError("History contains an undeclared learning-rate change")
    for group in child["optimizer_state"]["param_groups"]:
        if group["lr"] != rate:
            raise ValueError("Optimizer learning rate changed")


def snapshot(run: Path, step: int) -> Path:
    dest = run / "milestones" / f"updates_{step}"
    if dest.exists():
        validate_snapshot(dest)
        return dest
    p = load_payload(run / "checkpoint.pt")
    if p["global_step"] != step:
        raise ValueError("A milestone must be saved at its exact iterate; cannot relabel later weights")
    metrics = read_json(run / "metrics.json")
    if metrics["global_step"] != step or metrics["selected_step"] != p["best_step"]:
        raise ValueError("Milestone metrics are stale; finalize this exact checkpoint first")
    if read_json(run / "history.json") != p["history"]:
        raise ValueError("Milestone history differs from its resumable checkpoint")
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"updates_{step}.tmp-", dir=dest.parent))
    try:
        for name in STATE_FILES:
            shutil.copyfile(run / name, temporary / name)
        atomic_json(temporary / "snapshot.json", {
            "schema": SCHEMA, "step": step,
            "state_sha256": {name: file_hash(temporary / name) for name in STATE_FILES},
        })
        os.replace(temporary, dest)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return dest


def validate_snapshot(dest: Path) -> dict[str, Any]:
    report = read_json(dest / "snapshot.json")
    if report.get("schema") != SCHEMA or set(report["state_sha256"]) != set(STATE_FILES):
        raise ValueError("Unknown milestone snapshot schema")
    for name, sha in report["state_sha256"].items():
        if file_hash(dest / name) != sha:
            raise ValueError(f"Immutable milestone file changed: {dest / name}")
    return report


def render_snapshot(dest: Path, reference: Any, device: str, dpi: int) -> None:
    from .angular_diagnostics import AngularDiagnosticConfig, evaluate_angular_diagnostics
    from .log_objective import LogObjectiveConfig
    from .plotting import RainbowPlotConfig, plot_rainbow_comparison
    from .single_condition import load_single_checkpoint

    saved = validate_snapshot(dest)
    marker = dest / "visualization_complete.json"
    identity = {"step": saved["step"], "state_sha256": saved["state_sha256"], "dpi": dpi}
    if marker.exists():
        old = read_json(marker)
        if old.get("identity") != identity:
            raise ValueError("Visualization settings changed; retain the original snapshot output")
        if all((dest / n).is_file() and file_hash(dest / n) == sha
               for n, sha in old["artifact_sha256"].items()):
            return
    cfg = read_json(dest / "config.json")
    for mode, name in (("best", "best.pt"), ("last", "checkpoint.pt")):
        path = dest / name
        model, payload = load_single_checkpoint(path, device=device)
        label = ("Validation-selected through milestone" if mode == "best"
                 else "Actual optimizer iterate; not validation-selected")
        plot_rainbow_comparison(
            model, reference, dest / f"plots_{mode}",
            config=RainbowPlotConfig(eval_batch_size=cfg["training"]["eval_batch_size"], dpi=dpi),
            checkpoint_path=path, training_run_directory=dest, scatter_mode="training",
            selected_step=payload["global_step"],
            title=f"{label} {saved['step']}; checkpoint step {payload['global_step']}",
        )
        evaluate_angular_diagnostics(
            model, reference, dest / f"diagnostics_{mode}",
            train_samples=cfg["training"]["train_samples"],
            batch_size=cfg["training"]["batch_size"],
            config=AngularDiagnosticConfig(eval_batch_size=cfg["training"]["eval_batch_size"],
                                           dpi=dpi),
            checkpoint_path=path,
            objective_config=LogObjectiveConfig.from_dict(cfg["objective"]),
        )
        # This extra zoom is display-only. Keep the normal 120--150 reporting band unchanged.
        extra_profile_zoom(dest / f"diagnostics_{mode}" / "angular_profiles.npz", dpi)
    artifacts = {}
    for sub in ("plots_best", "plots_last", "diagnostics_best", "diagnostics_last"):
        for p in (dest / sub).rglob("*"):
            if p.is_file():
                artifacts[str(p.relative_to(dest)).replace(os.sep, "/")] = file_hash(p)
    atomic_json(marker, {"identity": identity, "artifact_sha256": artifacts})


def extra_profile_zoom(path: Path, dpi: int) -> None:
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    with np.load(path, allow_pickle=False) as d:
        for i, phi in enumerate(d["actual_source_phi_degrees"]):
            fig = Figure(figsize=(8.4, 4.8))
            FigureCanvasAgg(fig)
            ax = fig.subplots()
            ax.stairs(np.exp(d["reference_log_pdf"][i]), d["theta_edges_degrees"],
                      baseline=None, label="Stored CDF (cell PDF)")
            ax.plot(d["theta_midpoints_degrees"], np.exp(d["nf_log_pdf"][i]),
                    label="NF (cell midpoint)")
            ax.plot(d["theta_midpoints_degrees"], np.exp(d["hg_log_pdf"][i]), "--", label="HG")
            ax.set(xlim=(120, 160), yscale="log", xlabel="Scattering angle [degrees]",
                   ylabel="PDF [sr^-1]", title=f"Source phi = {phi:.1f} degrees; display-only zoom")
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(path.parent / f"profile_120_160_{i + 1}.png", dpi=dpi)
            fig.clear()


def write_report(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for rate in plan["learning_rates"]:
        for step in plan["milestones"]:
            dest = root / branch_name(rate) / "milestones" / f"updates_{step}"
            if not dest.exists():
                continue
            validate_snapshot(dest)
            p = load_payload(dest / "checkpoint.pt")
            metric = read_json(dest / "metrics.json")
            best, last = p["best_validation"], p["latest_validation"]
            rows.append({
                "trial_id": branch_name(rate), "learning_rate": rate,
                "global_step": step, "additional_updates": step - plan["parent_step"],
                "selected_step": p["best_step"],
                "best_validation_log_rmse": best["log_rmse"],
                "last_validation_log_rmse": last["log_rmse"],
                "best_validation_relative_rmse": best["relative_rmse"],
                "last_validation_relative_rmse": last["relative_rmse"],
                "best_validation_kl": best["forward_kl_estimate"],
                "last_validation_kl": last["forward_kl_estimate"],
                "test_log_rmse": (metric.get("test") or {}).get("log_rmse"),
                "test_kl": (metric.get("test") or {}).get("forward_kl_estimate"),
                "directory": str(dest.relative_to(root)).replace(os.sep, "/"),
                "plots_complete": (dest / "visualization_complete.json").is_file(),
            })
    finals = [r for r in rows if r["global_step"] == plan["total_steps"]]
    complete = len(finals) == len(plan["learning_rates"])
    report = {"schema": SCHEMA, "training_complete": complete, "trials": rows,
              "selection_metric": "best_validation_log_rmse", "test_used_for_selection": False}
    if complete:
        chosen = min(finals, key=lambda r: (r["best_validation_log_rmse"], r["learning_rate"]))
        report["selection"] = chosen
        atomic_json(root / "selection.json", {"schema": SCHEMA, "selected": chosen,
                    "rule": "minimum validation log_rmse through the common final milestone",
                    "ties": "smaller post-fork learning rate", "test_used": False})
    atomic_json(root / "summary.json", report)
    if rows:
        with (root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    lines = ["# Uniform pure-log continuation", "",
             "Both branches inherit the same parent weights, Adam state, RNG, and fixed pools.",
             "Best includes the parent; last is the exact milestone iterate. Test never selects.",
             "", "| LR after fork | Updates | Best step | Best log RMSE | Last log RMSE | Results |",
             "|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        folder = r["directory"]
        links = (f"[best map]({folder}/plots_best/comparison.png), "
                 f"[last map]({folder}/plots_last/comparison.png), "
                 f"[profiles]({folder}/diagnostics_best/angular_profiles.png)")
        lines.append(f"|{r['learning_rate']}|{r['global_step']}|{r['selected_step']}|"
                     f"{r['best_validation_log_rmse']:.6g}|{r['last_validation_log_rmse']:.6g}|"
                     f"{links}|")
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_history_summary(root, plan)
    return report


def plot_history_summary(root: Path, plan: dict[str, Any]) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(9, 4.8))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    present = False
    for rate in plan["learning_rates"]:
        path = root / branch_name(rate) / "history.json"
        if not path.exists():
            continue
        history = read_json(path)
        rows = [r for r in history if "validation" in r and r["global_step"] > 0]
        ax.plot([r["global_step"] for r in rows],
                [r["validation"]["log_rmse"] for r in rows], label=f"post-fork lr={rate:g}")
        present = True
    if present:
        ax.axvline(plan["parent_step"], linestyle="--", label="common parent")
        ax.set(xlabel="Total optimizer updates", ylabel="Validation log-PDF RMSE (natural log)",
               title="Latest validation at each scheduled evaluation (not best-so-far)")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / "summary.png", dpi=160)
    fig.clear()


def run_sweep(plan: dict[str, Any], reference: Any, parent: dict[str, Any], *,
              device: str, dpi: int, plots: bool, mode: str) -> dict[str, Any]:
    train, Objective, _, ModelConfig = _api()
    root = Path(plan["output"])
    with output_lock(root):
        init_root(plan, root, Path(plan["parent_checkpoint"]))
        for rate in plan["learning_rates"]:
            if mode == "plot" and not (root / branch_name(rate)).exists():
                continue
            run = init_branch(root, plan, parent, rate)
            cfg = train.SingleTrainingConfig.from_dict(branch_payload(
                parent, total_steps=plan["total_steps"], learning_rate=rate)["training_config"])
            for step in plan["milestones"]:
                dest = run / "milestones" / f"updates_{step}"
                if dest.exists():
                    validate_snapshot(dest)
                elif mode == "run":
                    current = load_payload(run / "checkpoint.pt")
                    validate_child(current, parent, plan, rate)
                    if current["global_step"] > step:
                        raise ValueError(f"Missing past snapshot at {step}; cannot invent old weights")
                    if current["global_step"] <= step:
                        print(f"\n[{branch_name(rate)}] {current['global_step']} -> {step} "
                              f"(total target {cfg.steps})", flush=True)
                        started = time.perf_counter()

                        def callback(row):
                            if "validation" in row:
                                print(f"  step={row['global_step']} "
                                      f"val_log_rmse={row['validation']['log_rmse']:.7g}", flush=True)

                        train.train_single_condition(
                            reference, ModelConfig.from_dict(parent["model_config"]), cfg, run,
                            resume=run / "checkpoint.pt",
                            max_steps_this_run=step - current["global_step"],
                            callback=callback, make_plots=False,
                            objective_config=Objective.from_dict(parent["objective_config"]),
                        )
                        with (run / "continuation_timing.jsonl").open("a", encoding="utf-8") as f:
                            f.write(json.dumps({"from_step": current["global_step"], "to_step": step,
                                               "seconds": time.perf_counter() - started}) + "\n")
                    dest = snapshot(run, step)
                else:
                    continue
                write_report(root, plan)
                if plots:
                    print(f"  Plotting best and last: {dest}", flush=True)
                    render_snapshot(dest, reference, device, dpi)
                    write_report(root, plan)
        # A second independent read also detects accidental mutation of the original checkpoint.
        if file_hash(Path(plan["parent_checkpoint"])) != plan["parent_sha256"]:
            raise RuntimeError("Parent checkpoint changed during sweep")
        return write_report(root, plan)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "plan", "check", "plot", "report"),
                        nargs="?", default="run")
    parser.add_argument("--parent", default="runs/rainbow_log_uniform_gpu")
    parser.add_argument("--record", default=None,
                        help="Defaults to the recorded CDF directory in the parent checkpoint")
    parser.add_argument("--output", default="runs/rainbow_log_uniform_continue_gpu")
    parser.add_argument("--parent-step", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--milestones", type=int, nargs="+", default=[3000, 5000])
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.003, 0.001])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--no-plots", action="store_true", help="Explicit opt-out; default plots both")
    parser.add_argument("--allow-cpu-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        parent = Path(args.parent).resolve()
        if parent.is_dir():
            parent = parent / "checkpoint.pt"
        if not parent.is_file():
            raise FileNotFoundError(f"Missing resumable parent checkpoint: {parent}")
        p = load_payload(parent)
        validate_parent(p, args.parent_step)
        plan = make_plan(parent, Path(args.output), p, total_steps=args.steps,
                         milestones=args.milestones, rates=args.learning_rates)
        if args.dpi < 1:
            raise ValueError("DPI must be positive")
        plan["plot_dpi"] = args.dpi
        if args.mode == "plan":
            print(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False))
            return 0
        if args.mode == "report":
            root = Path(plan["output"])
            with output_lock(root):
                if read_json(root / "manifest.json") != plan:
                    raise ValueError("Report plan differs from the saved sweep")
                write_report(root, plan)
            return 0
        _, _, Reference, _ = _api()
        record = args.record or p["data_provenance"].get("record_directory")
        if not record or not Path(record).is_dir():
            raise FileNotFoundError("CDF directory is missing; set RECORD in the batch or --record.")
        with Reference(record) as reference:
            preflight(p, reference, args.device, cpu_test=args.allow_cpu_test)
            if args.mode == "check":
                print("[OK] Parent, runtime, code, CDF, all training/validation pools verified.")
                return 0
            report = run_sweep(plan, reference, p, device=args.device, dpi=args.dpi,
                               plots=not args.no_plots, mode=args.mode)
        print(f"[DONE] training_complete={report['training_complete']}; {plan['output']}")
        return 0 if report["training_complete"] else 1
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        print("The parent run is never overwritten. Correct the problem and repeat the command.",
              file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
