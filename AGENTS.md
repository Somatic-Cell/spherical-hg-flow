# Research implementation contract

The mathematical model is part of the specification, not an implementation detail.
The current primary workflow is the **single-condition Rainbow model** documented in
`docs/RAINBOW_SINGLE_CONDITION.md`. Read it and `README.md` before changing that workflow.

## Current model and data contract

- Use the externally supplied `metadata.json` value `hg.g` as a fixed HG base
  coefficient. It is the first moment of the **stored CDF**. Do not substitute
  `g_source`, estimate it again from training points, silently clip it, or make
  an HG MLP or an HG warmup stage mandatory.
- Preserve the full-circle / interval RQS construction selected from Rezende
  et al. (2020), Sections 2.1.2, 2.2 and 2.3.1: periodic circular splines,
  positive interval endpoint slopes, periodic conditioners, and genuinely
  alternating coupling. HG probability coordinates and the physical symmetry
  constraints are explicit extensions. Do not silently revert to the v1
  absolute-azimuth chart with a randomly restored reflection branch.
- Match the documented parameter tying when mirror symmetry is enabled. The
  flow state still contains the full circle. Preserve axial-incidence azimuth
  invariance and the signed incident cosine; the particle need not be
  north/south symmetric. Finite pole density is not a claim of globally
  continuous density or a globally smooth sphere diffeomorphism.
- Preserve density with respect to solid angle. Incident condition and external
  g are fixed when computing the angular Jacobian. HG CDF coordinates and the
  HG density factor must agree in both sampling and arbitrary PDF evaluation.
- Preserve the producer's stored piecewise-constant solid-angle distribution:
  use the actual nonuniform `u_edges`, the two hierarchical CDFs, and uniform
  interpolation inside a cell in `u=(1-cos(theta))/2` and azimuth. Do not apply
  smoothing, jitter, PDF floors, extra normalization, or theta-uniform
  interpolation without an explicit, separately versioned change of target.
- Incoming and outgoing directions are light propagation directions. Read the
  recorded frame as a matrix with columns `[e0,e1,ki]`, even though the JSON
  outer arrays are matrix rows. Make the change to the NF frame explicit.
- Do not infer a full phase distribution from cropped angular data. Do not
  multiply a target-distributed training sample by its target PDF again.
  Scattering cross sections are separate from the normalized angular density.
- The first workflow fits one fixed incident direction and one wavelength.
  Keep the single-condition restriction explicit; do not label it as validated
  conditional interpolation. Multi-condition learning is a subsequent extension.

## Evaluation, reproducibility and scope

- Keep the fixed training pool, validation stream and test stream independent.
  Model selection uses validation data; final test data must not select weights.
  Record seeds, sample counts, update count, optimizer/RNG state, complete
  configuration, input file hashes and physical-source metadata.
- Use NLL / forward KL with the correct solid-angle measure. Negative continuous
  NLL is valid. A finite-sample KL estimate can fluctuate below zero; do not
  replace it by zero or rename a plain NLL as KL. Evaluate importance ESS using
  actual flow samples and their matching proposal density.
- Distinguish the HG base g, the teacher's first moment, and the final NF's first
  moment. Moment matching is not an automatic consequence of residual learning.
- Preserve the selected checkpoint separately from the resumable training
  checkpoint. Load primitive/tensor checkpoints with `weights_only=True`.
  Reject mismatched source data, configuration, model family and format versions.
- Validate CDF inversion/evaluation, frame conversions, RQS round trips and
  Jacobians, periodic seams, symmetries, normalization, sample/eval consistency,
  learning and resume after relevant changes. Prefer checks that resolve an
  actual mathematical or integration risk over implementation-mirroring tests.
- Label synthetic fixtures as such. They validate the adapter and learning
  pipeline, not the Rainbow solver's physics, convergence, or real-data accuracy.
- Report only actually executed platforms and measurements. CPU Python or host
  C++ tests do not establish CUDA training, NVCC compilation, OptiX integration,
  renderer correctness, or GPU sampling/evaluation speed.

## Legacy v1 boundary

`docs/MODEL.md`, `docs/DATA_CONTRACT.md`, `docs/EXPORT_FORMAT.md`,
`docs/DEVELOPMENT.md`, `examples/README.md` and `reports/VALIDATION.md` retain
the historical v0.1 / format-v1 implementation and its evidence. Their folded
chart and mandatory HG-head requirements apply **only to that legacy model**;
they do not override the current workflow above.

- Keep existing legacy checkpoints and native format-v1 inference compatible.
  The new model must not be exported or loaded as format v1. Future native
  support needs its own versioned layout and Python/native parity checks.
- Do not alter the historical `NromFlowHG2Mie` repository or the Rainbow data
  producer as an incidental part of this project.

Run `pytest -q` and `ruff check .` from the repository root for local checks.
Record the measured scope when reporting results; retained historical reports
do not validate new changes.
