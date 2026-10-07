> **Historical v0.1 development notes.** The stage schedule, folded-chart
> requirements and native parity checks below describe the legacy model. Current
> development follows [AGENTS.md](../AGENTS.md) and
> [RAINBOW_SINGLE_CONDITION.md](RAINBOW_SINGLE_CONDITION.md). General reproducibility
> and precision principles still apply; legacy tests do not validate the new
> external-g / full-circle model or its future native implementation.

# Development and verification

Install the project in editable mode and use the source tree's tests:

```bash
python -m pip install -e '.[dev]'
pytest -q
ruff check .
python -m build
```

The tests verify HG normalization/first moment and zero-g continuity, RQS inverse
and autograd Jacobians, trainable endpoint slopes, local/world geometry, exact
reflection symmetry, axial-incidence symmetry, surface normalization, arbitrary
PDF vs sample-PDF consistency, one-conditioner-call per layer, batching, dataset
semantics, meaningful synthetic learning, stage freezing, exact CPU resume, and
Python/C++ agreement after explicitly quantizing weights to FP32.

Native tests compile the C++ reference using an available g++ or clang++. A
missing compiler causes a reported skip, not a successful native validation.
Build without `-ffast-math`; the reference depends on standard floating-point
semantics. There is no nvcc gate because this project has not yet been verified
with a CUDA toolkit or OptiX host application.

## Where changes belong

- RQS numerical changes: `splines.py` plus the matching native core; verify inverse
  values, gradients and Jacobians, not just composition on a few random points.
- Symmetry/condition encoding changes: `coupling.py`, `encoding.py`, `MODEL.md`,
  native core and export manifest, with cross-language checks.
- Training changes: preserve dataset provenance, holdout rules, staged objectives,
  and optimizer/RNG state. Check an interrupted run against an uninterrupted run.
- New CDF layout: create an adapter producing `PhasePointCloud`; do not make the
  training objective depend implicitly on that layout.
- Precision or acceleration changes: compare to this reference on concentrated
  distributions before deciding acceptable tolerances for rendering.

## Versioning and provenance

Python checkpoints, data adapters, and native exports each have a version.
The native binary is self-contained, with a manifest documenting matrix/feature
order and a SHA-256 hash. A checkpoint is loaded with `weights_only=True` using
primitive/tensor state, not a pickled module object.

Runtime checks for exact resume compare Python/PyTorch/NumPy versions, device
string, dtype, and the full configs. They do not prove the same CPU/GPU model,
operating system, or driver. The actual verified scope is recorded in
`reports/VALIDATION.md`. It must be updated from measurements, not expectations.

## First physical experiments

1. Fix wavelength and sweep incident cosine. Keep whole angles held out and
   compare the fitted HG with HG+NF at the same evaluation points.
2. Check angular slices, reflection and axial limits, forward-peak mass,
   rainbow-feature locations, normalization, and sample/eval consistency.
3. Repeat with incidence fixed and wavelength swept, using explicit wavelength
   spacing/normalization matching the physical data.
4. Expand to both conditions. Compare blob counts, spline bins/layers, and
   wavelength representations as ablations, using the same train/test split.
5. In the renderer, verify solid-angle PDF conventions, MIS, export quantization,
   GPU precision and actual sample/eval timings separately. Inspect rendered
   error when substituting NF for the physical phase function itself.

Single-axis experiments still use the two-condition model; one input column is
constant. This keeps the data/export contract stable while reducing the learning
problem. Do not set equal wavelength normalization bounds for a fixed wavelength.
