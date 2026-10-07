# Research implementation contract

The mathematical model is part of the specification, not an implementation detail.
Read `docs/MODEL.md`, `docs/DATA_CONTRACT.md`, and `docs/EXPORT_FORMAT.md` before changing it.

- Preserve the conditional HG base, trainable bounded RQS endpoint slopes, real
  alternating coupling, reflection folding, and smooth axial-incidence constraint.
- Preserve density with respect to solid angle. Conditions are held fixed for the
  angular Jacobian. Joint training must retain every dependence through g.
- Do not silently replace the spline, drop a density term, clip physical g targets,
  change physical axes/units, introduce jitter/smoothing, or approximate narrow HG
  lobes by a cutoff. Explicit approximations need named settings and evidence.
- Incoming and outgoing directions are light propagation directions. Preserve the
  sign of incident_cosine; the particle need not be north/south symmetric.
- Do not infer a full phase distribution from a cropped rainbow-only CDF. Do not
  multiply an already-target-distributed sample by its target PDF a second time.
- Python and native inference must share feature order, tensor layout, parameter
  constraints, symmetry gates, and density conventions. Change the format version
  when changing a released export contract.
- Distinguish HG base g, the target's first moment, and the final NF's first moment.
- Validate whole held-out conditions. Label in-sample scores and synthetic teacher
  data. Report NLL, not a made-up KL when target entropy is unavailable.
- Test round trips, Jacobians, spherical normalization, symmetry, sample/eval
  consistency, learning, resume, and cross-language inference after relevant changes.
- Keep training and export versioned and reproducible. Preserve dataset hashes,
  condition splits, optimizer/RNG state, and explicit configurations.
- This environment verified CPU Python and host C++. Do not claim CUDA/OptiX
  compilation, device inference, or rendering speed without running those checks.
- Do not alter the historical NromFlowHG2Mie repository as part of this project.

Use `pytest -q` and `ruff check .` for the local checks. See `docs/DEVELOPMENT.md`.
