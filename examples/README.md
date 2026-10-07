> **Legacy v0.1 example artifacts.** The checkpoints and native exports below use
> the old HG-head / folded-chart model and remain examples for the legacy commands.
> They are not Rainbow solver output or checkpoints for the new circular-RQS model.
> Start new single-condition experiments with
> [RAINBOW_SINGLE_CONDITION.md](../docs/RAINBOW_SINGLE_CONDITION.md).

# Synthetic reference artifacts

These files come from the recorded `configs/smoke.json` run: wavelength 550 nm,
four incident cosines, 2048 samples per condition, 40 HG warmup updates and 80
residual updates. The teacher is `phaseflow.synthetic`, not a rainbow simulation.
The tiny training run is an interface example, not a converged physical model.

| File | Use |
|---|---|
| `synthetic_points.npz` | The exact point cloud used in the recorded run |
| `synthetic_checkpoint.pt` | The complete primitive/tensor training checkpoint |
| `synthetic_export/model.pflow` | Self-contained native FP32-weight artifact |
| `synthetic_export/weights.safetensors` | The same named weights |
| `synthetic_export/manifest.json` | Version, layout, units, conditions, symmetry, precision |

For example, from the repository root:

```bash
phaseflow evaluate --data examples/synthetic_points.npz \
  --checkpoint examples/synthetic_checkpoint.pt --split validation --sample-count 2048
```

The included native export uses the final `sin(inclination)` axial symmetry
constraint. The older NromFlowHG2Mie export format is not interchangeable with
this versioned format. Full results are in `reports/smoke_workflow.json` and
`reports/VALIDATION.md`.

Regenerate all artifacts from source in a new directory:

```bash
python scripts/run_smoke.py --output runs/my_smoke
```

The script checks final checkpoint progress, reloaded NLL, CLI evaluation and
export, and trained-weight native inference when a C++ compiler is available.
