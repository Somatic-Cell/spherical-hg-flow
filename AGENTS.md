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

- The primary training, evaluation and inference path targets CUDA by default.
  Fail explicitly when requested CUDA is unavailable; CPU is an explicit
  correctness-test/debug choice. Keep fixed training and validation tensors on
  the selected device and avoid CPU minibatch copies or per-parameter host
  synchronization in the optimization loop.
- Distinguish the MLP parameter dtype from the spline/probability-coordinate
  dtype. The v0.4 starting configuration uses FP32 weights, HG, geometry and
  RQS arithmetic (`geometry_dtype="model"`, `spline_dtype="model"`). Preserve
  externally supplied g exactly in metadata; derive runtime endpoint constants
  from it without clipping. Retain transverse direction information near poles.
  Use explicit FP64 reference diagnostics and FP64 teacher/statistical reduction
  values. Old configs without geometry_dtype retain their FP64 geometry behavior.
  Do not silently enable AMP/FP16 or TF32. FP16-rounded weights evaluated with
  FP32 arithmetic are a quantization diagnostic, not a CoopVec parity test.
- Keep the fixed training pool, validation stream and test stream independent.
  Model selection uses validation data; final test data must not select weights.
  Record seeds, sample counts, update count, optimizer/RNG state, complete
  configuration, input file hashes and physical-source metadata.
- Capacity sweeps plan the final optimizer update count before training. Archive
  each milestone's best validation state before continuing, and never reuse a
  later selected checkpoint for an earlier milestone. Sample-count comparisons
  verify and preserve their completed parent experiment. Sampler-audit output
  must be in a directory tree disjoint from the saved run or milestone archive.
- Use NLL / forward KL with the correct solid-angle measure. Negative continuous
  NLL is valid. A finite-sample KL estimate can fluctuate below zero; do not
  replace it by zero or rename a plain NLL as KL. Evaluate importance ESS using
  actual flow samples and their matching proposal density.
- Distinguish the HG base g, the teacher's first moment, and the final NF's first
  moment. Moment matching is not an automatic consequence of residual learning.
- Preserve the selected checkpoint separately from the resumable training
  checkpoint. Load primitive/tensor checkpoints with `weights_only=True`.
  Reject mismatched source data, configuration, model family and format versions.
- Keep live JSONL/TensorBoard histories consistent with the resumed checkpoint:
  remove uncheckpointed future scalar events by replaying checkpoint history.
  Wall-clock telemetry is separate from the deterministic numerical history.
- Validate CDF inversion/evaluation, frame conversions, RQS round trips and
  Jacobians, periodic seams, symmetries, normalization, sample/eval consistency,
  learning and resume after relevant changes. Prefer checks that resolve an
  actual mathematical or integration risk over implementation-mirroring tests.
- Label synthetic fixtures as such. They validate the adapter and learning
  pipeline, not the Rainbow solver's physics, convergence, or real-data accuracy.
- After completed training, visualize the selected weights together with the
  exact stored-CDF PDF and the actual configured fixed training pool. Keep all maps in the same
  recorded source frame, axes, extent and aspect. Share the two PDF maps' log
  color scale; mark true zero target density rather than flooring it. A scatter
  map in theta/azimuth coordinates is not an equal-area density estimate.
  The default scatter must show every entry of the actual fixed training pool
  once, without a display sample cap. Verify regeneration against the saved pool
  hash and apply the saved training geometry dtype before plotting. Keep old
  completed sweep artifacts intact when producing revised plots.
- Report only actually executed platforms and measurements. CPU Python or host
  C++ tests do not establish CUDA training, NVCC compilation, OptiX integration,
  renderer correctness, or GPU sampling/evaluation speed.

## Optional log-density objective (schema 3 / checkpoint 5)

- `docs/LOG_DENSITY_OBJECTIVE.md` specifies the explicit extension authorized for
  weak-scattering shape fidelity. Schema-2 training remains the original NLL
  path. Never interpret a mixed query pool as samples from the target density.
- The new objective uses target NLL plus beta times the squared log-density
  ratio averaged under uniform solid angle. An equal target/uniform mixture
  requires separate p/r and u/r weights, without minibatch self-normalization.
  HG, the stored target, symmetry constraints and flow architecture are unchanged.
- Check every stored cell for positive density before whole-sphere log metrics.
  True zeros cause an explicit error; do not add epsilon, replace query points,
  smooth the CDF or silently restrict the integration domain.
- Independently evaluate target NLL/KL and uniform-solid-angle log errors.
  Record the selected metric, preserve both NLL and log-RMSE selected weights,
  and never compare raw composite loss across different beta values for selection.
- Record query-source labels and both raw and actual-geometry pool identities.
  Evaluate teacher labels at the actual geometry-cast regression points. Mixed
  scatter plots show all entries and distinguish CDF and uniform components.
- Minibatch total objective is not NLL. Report measured raw components even
  when their coefficient is zero; do not display disabled placeholders as errors.
- Read old checkpoints for inference/plots, but require matching objective,
  version, input, code and runtime for exact continuation. The old CDF-only
  sampling audit must explicitly reject the new pool contract.

## Explicit all-uniform training queries (objective schema 2)

- `docs/UNIFORM_LOG_TRAINING.md` specifies the user-requested all-uniform
  alternative. Preserve objective schema 1's target and half-mixture behavior.
  Objective schema 2 uses `sampling="uniform"` and `target_fraction=0.0`.
- Generate every training point uniformly in solid angle using the existing
  uniform training stream 5. Use no target/CDF training samples. Query the
  unchanged stored PDF at each actual geometry-cast direction.
- The log-density term is the ordinary unweighted mean of squared log ratios.
  The default pure-log experiment sets the NLL coefficient to zero. If NLL is
  enabled or reported, use the exact p/u weight; do not treat uniform points
  as target-distributed, clip importance weights, or self-normalize them.
- Keep independent target validation/test streams for NLL/KL and independent
  uniform validation/test streams for log/relative errors. Pure log does not
  remove the model's analytic normalization or alter its HG base.
- Record actual source counts cdf=0/uniform=N, all source labels uniform, and
  source_streams cdf=null/uniform=5. Historical stream registry entries are
  not assertions that those streams contributed to the training pool.
- Plot the complete actual uniform pool, label it as spherical-uniform queries,
  and use training_samples.png for the new scatter. Retain old target/mixed
  plot compatibility and never relabel them as all-uniform.

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
On a CUDA machine, first require `python -c "import torch; assert torch.cuda.is_available()"`,
then run `python -m pytest -m cuda -q`. A CUDA-marked suite consisting only of
skips is not a GPU validation result. Do not assume a self-hosted GPU CI runner
exists or report unexecuted CUDA checks as passed.
