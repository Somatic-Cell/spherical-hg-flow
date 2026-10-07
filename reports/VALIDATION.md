> **Historical validation of v0.1 only.** The 103-test result and all measurements
> below were obtained for the old folded-chart model with an HG MLP. They do not
> validate the new Rainbow adapter, external-g circular-RQS model, or its learning
> accuracy. The old native comparison applies only to format v1. Current scope and
> verification requirements are in
> [RAINBOW_SINGLE_CONDITION.md](../docs/RAINBOW_SINGLE_CONDITION.md).

# Validation record — v0.1.0

Recorded on 2026-10-07. This is implementation validation using an analytic
synthetic directional distribution. It does not validate a 2012 rainbow model,
polarization, production rendering quality, or GPU performance.

## Environment

| Component | Recorded value |
|---|---|
| Python | 3.12.14 |
| PyTorch | 2.8.0+cpu |
| Zuko | 1.6.0 |
| NumPy | 2.5.3 |
| C++ | g++ 13.3.0, C++20, `-O2 -Wall -Wextra -Wpedantic` |
| Device | CPU; one thread for the recorded training run |
| Learned weights / network-RQS arithmetic | float32 |
| HG / directions in Python | float64 |
| Native reference | FP32 weights, double arithmetic |

Exact package versions are in `requirements-validation.txt`. The source and
dataset fingerprints, configs, command arguments, metrics and native errors
are preserved in `smoke_workflow.json`.

## Automated checks

```text
pytest -q --junitxml=phaseflow-pytest.xml
103 passed in 6.90s

ruff check .
All checks passed!
```

No tests were skipped in this run. The JUnit results are in `pytest.xml`.

The 103 cases include HG CDF/quantile and normalization/first-moment checks;
RQS analytic inverse and autograd Jacobians; endpoint slope learning; scattering
frames and mirror branches; axial-incidence symmetry with first-order allowed
azimuth modulation; spherical normalization; sample-vs-arbitrary-PDF consistency;
one conditioner call per layer; context/Distribution shapes; point weights and
unit-vector precision; actual synthetic learning; held-out condition separation;
frozen HG and full joint loss; exact CPU checkpoint resume; native export parity;
corrupt files and invalid inputs; and explicit numerical failure at rounded poles.

The 14 native/export test cases include Gaussian blob counts 0/4/5, raw-g context
on/off, nonidentity couplings, g near ±0.9989, and both reflection and axial
symmetry. They compile and execute the standard C++ reference. The native kernel
is not compiled with nvcc in this environment.

## Complete point-cloud workflow

Reproduce with:

```bash
python scripts/run_smoke.py --output runs/reproduced_smoke
```

The recorded run used `configs/smoke.json`: 3779 learned parameters, two coupling
layers, eight RQS bins, eight one-blob bins per condition, and a small HG head.
There are 8192 target samples: 2048 each at wavelength 550 nm and incident cosine
`[-0.75,-0.25,0.25,0.75]`. The first three conditions were used for training;
`eta=0.75` was held out from both HG warmup and residual fitting.

Training ran 40 HG moment updates and 80 residual NLL updates, with no joint
stage. Its CPU CLI wall time was about 2.74 seconds, including imports, setup,
evaluation and checkpoint writes. This single timing is not an inference
benchmark, and hardware-to-hardware performance should not be inferred from it.

| NLL with respect to solid angle | Fitted HG | HG + NF | Improvement |
|---|---:|---:|---:|
| Training conditions, equal condition weight | 2.485191 | 2.451466 | 0.033726 |
| Held-out condition, eta=0.75 | 2.415065 | 2.364542 | 0.050523 |

Lower NLL is better. The same held-out points are used for the HG/NF comparison.
These are empirical cross-entropies, not KL divergences. The run is intentionally
short and the held-out set contains only one condition; it is a functioning
learning example, not evidence of convergence or broad physical generalization.

On the held-out condition, the point cloud's mean scattering cosine is 0.261435;
the HG head predicts 0.211262. The short warmup therefore leaves a visible g error.
The final flow's sample estimate of the mean cosine is 0.219570 with estimated
standard error 0.013715 over 2048 draws. NLL improvement does not establish exact
moment matching; the final NF moment is not constrained to equal the HG base g.

The largest absolute log-PDF discrepancy between sampling and independent Python
evaluation across those 2048 draws was **1.945e-6** with FP32 networks/RQS and
FP64 angular calculations. Reloading the saved checkpoint changed the computed
validation NLL by **0.0** in the recorded check. The saved stage counters were
verified as warmup=40, residual=80, joint=0.

## Trained-weight native comparison

The workflow exported its actual trained model. The native file is **15492 bytes**
including descriptors and FP32 weights. The script then compared four sample
queries and four evaluation queries with Python using the same FP32 weights and
double reference arithmetic on both sides:

| Quantity | Maximum absolute difference |
|---|---:|
| Direction components | 2.776e-16 |
| Sample log PDF | 5.329e-15 |
| Evaluated log PDF | 2.887e-15 |

These are eight particular trained-model queries. Broader configuration and
concentrated-HG checks are covered by the native tests. These numbers are not a
claim of bitwise agreement with ordinary Python FP32 execution or any GPU kernel.

## Scope that remains unverified

- The actual CDF buffer, its interpolation/inversion, its angular coverage and
  normalization, and its 2012-paper physics.
- Accuracy on forward peaks, rainbow features and wavelength/incident-angle
  variation in that physical dataset.
- Global smoothness of the spherical map and density limits at outgoing poles.
  The implemented folded chart defines a normalized surface density, not a
  globally smooth sphere diffeomorphism.
- nvcc compilation, CUDA training/resume, device-pointer upload, OptiX program
  integration, MIS behavior in the actual renderer, and sample/eval throughput.
- CMake execution: a CMake configuration is included; the measured native build
  used g++ directly.

Open input uniforms can still round to a polar endpoint under excessive numerical
concentration. The implementation detects and rejects that event. It does not
silently retry, clamp, replace the density, or claim FP64 prevents every such case.
See `docs/MODEL.md` and `docs/EXPORT_FORMAT.md` for the inference contract.
