# Single-condition Rainbow workflow validation

Date: 2026-10-07. Package: `spherical-hg-flow` 0.2.0.

This report covers the new single-condition workflow added to repository base
`3427d49c03019419dba5c5fea5bafb23e92a8789`. The producer contract was inspected at
`Somatic-Cell/rainbow@a9538941dcb9df2de6a4130fa66f625f0133c949`.

**The tests use synthetic records with the solver's public schema. No actual
Rainbow solver output, CUDA training, OptiX execution, or rendering benchmark
was available for this validation.** The results establish the implemented data
contract and learning path, not the accuracy or convergence of rainbow optics.

## Executed checks

| Check | Result |
|---|---|
| Complete Python/native regression suite | 219 passed; no failures or skips |
| New CDF adapter cases | 56 passed |
| New circular-spline/sphere-model cases | 44 passed |
| New training/CLI/resume cases | 16 passed |
| Retained legacy tests | 103 passed |
| Ruff | Passed |
| `git diff --check` | Passed |
| Source distribution and wheel build | Passed with `python -m build --no-isolation` |

The final full test run took 9.77 s in this environment. This duration is test
execution time, not a phase-sampling, training-throughput, or rendering benchmark.
The host C++ checks exercise only the retained **legacy v1** implementation.
They do not imply native support for the new circular model.

Evidence: [JUnit results](rainbow_pytest.xml),
[synthetic learning configuration and metrics](rainbow_synthetic_smoke.json).

Environment:

| Component | Value |
|---|---|
| Python | 3.12.14 |
| PyTorch | 2.8.0+cpu |
| NumPy | 2.5.3 |
| Zuko | 1.6.0 |
| New learning experiment | CPU, float64, one CPU thread |
| CUDA available in this environment | No |

From an activated development environment at the repository root:

```sh
ruff check .
pytest -q --junitxml=reports/rainbow_pytest.xml
python -m build --no-isolation
```

The development run used `PYTHONPATH=src`, with `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1`, to test the checkout using the existing dependency environment.

## Probability and geometry checks

The adapter tests include independent cell-mass and solid-angle calculations,
nonuniform `u_edges`, zero-mass cells and marginal rows, CDF plateaus, arbitrary
direction density evaluation, sampling frequencies, exact saved first moments,
and source-to-NF frame rotation. They reject malformed records and mismatched
HG labels without repair. They also check file hashes, lifecycle cleanup,
matrix orientation, narrow-angle evaluation, and numerical endpoint failures.

The sphere tests include circular seam derivatives, analytic inverse/log-Jacobian
agreement, gradients of the reflected full-circle parameters, and an independent
two-dimensional autograd Jacobian with both coupling directions active. Numerical
solid-angle integration checks normalization in mirrored and generic modes.
Other tests cover identity HG, both signs of g, g beyond the legacy 0.999 limit,
external-g preservation after changing neural precision, reflection, exact axial
symmetry, sample/eval agreement, and one conditioner call per layer.

The implemented family follows the circular-spline and cylinder constructions
in Rezende et al. (2020), Sections 2.1.2, 2.2 and 2.3.1. Parameter sharing for
physical reflection and the axial-incidence gate are documented project
extensions. The paper's alternative flow families and experiments are outside
this implementation. Finite density does not establish continuity at the sphere
poles; see the paper's supplement Appendix A.1 and the model specification.

## End-to-end learning check

The analytic **synthetic** teacher before cell averaging is

$$
p(\mu,\phi_s)=\frac{(1+0.6\mu)(1-0.65\cos(2\phi_s))}{4\pi}.
$$

The fixture integrates this function into 16 polar cells and 24 azimuth cells,
then writes the public hierarchical CDF format. The actual training target is
the saved cellwise-constant solid-angle distribution. Its exact stored first
moment is `g=0.1987210964506126`. The labels 550 nm and 20 degrees define the
coordinate fixture; they are not a prediction for a physical water droplet.

The test uses two alternating coupling layers, eight bins, hidden widths
`[16,16]`, 2,048 fixed training points, 2,048 independent validation points,
and 100 Adam updates at learning rate 0.003 with batch size 256. The best model
is selected only by scheduled validation NLL. Final statistics use a separate
2,048-point teacher stream and 512 NF proposal samples.

| Independent test statistic | HG base | Selected NF |
|---|---:|---:|
| Forward KL estimate, nats | 0.11011204 | 0.01308581 |
| Standard error of that estimate | 0.00965335 | 0.00339780 |
| Mean NLL, nats | 2.46762233 | 2.37059610 |

The paired NLL improvement is `0.09702623` nats, with standard error
`0.01031881`. Proposal relative ESS is `0.97667980`, the sample mean of `p/q`
is `1.00506105`, and the maximum sample/eval log-PDF difference is
`5.77315973e-15`. The NF first-moment estimate is `0.18892789`, with standard
error `0.02329390`; the fixed HG coefficient itself remains unchanged.

These standard errors describe evaluation sampling only. They do not include
variation between training seeds or optical-solver/discretization error.
Finite-sample KL estimates are not clamped to zero. ESS concerns importance
sampling of the normalized phase, without lighting, visibility, or MIS.

Reproduce the learning check with:

```sh
pytest -q tests/test_single_condition.py::test_training_improves_independent_kl_and_preserves_external_g
```

## Reproducibility and compatibility

The resume test compares an uninterrupted run with a run interrupted after
five updates, which is deliberately not a scheduled validation boundary.
Model tensors, selected tensors, histories, metrics and minibatch RNG state
agree exactly after resumption. Test points are not evaluated at the partial
stop, and stopping does not add a model-selection candidate.

The workflow rejects changed data/configuration for exact resume and rejects
using inference-only `best.pt` as an optimizer checkpoint. It uses
`torch.load(..., weights_only=True)`, preserves source hashes and provenance,
and records the code/runtime fingerprints used for training and evaluation.

CLI tests execute `inspect-rainbow`, `train-rainbow`, and `evaluate-rainbow` on
one synthetic record. They also verify that the new checkpoint/model is
explicitly rejected by the legacy native exporter before any export files
are created.

The next empirical check is to run the same documented workflow on a complete
record from the real solver and assess approximation error versus point count
and model capacity. Conditional interpolation and a new versioned native /
OptiX export are subsequent implementation stages.
