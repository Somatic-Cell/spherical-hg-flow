# GPU-default workflow and aligned plotting validation

Date: 2026-10-07. Package: `spherical-hg-flow` 0.3.0.

Repository base: `0956c7d52b36ce2c704fdab82e447ff80efbbfe2`.
The producer contract and the external HG / full-circle RQS model are retained
from the previous single-condition workflow.

**CUDA was unavailable in the validation environment. The executed results
below use CPU and synthetic CDF records. They establish numerical and workflow
behavior, not GPU throughput, real Rainbow approximation accuracy, or OptiX
integration. The shipped training and evaluation defaults require CUDA.**

## Executed checks

| Check | Result |
|---|---|
| Complete Python and retained native regression suite | 256 passed, 4 CUDA tests skipped |
| Ruff | Passed |
| Whitespace / patch checks | Passed |
| Source distribution and wheel | Built successfully with `python -m build --no-isolation` |
| Mixed FP32 conditioner / FP64 spline learning | Completed 100 updates on a synthetic teacher |
| Automatic post-training plots | All four PNG files and `plots.json` created |
| Visual inspection | Comparison and individual PDF maps inspected; labels, shared axes and colorbar fit |
| CLI integration | Inspect, train, independent evaluation and standalone plotting passed |
| Exact interrupted resume | CPU weights, selected weights, generator state, history and metrics matched |
| CUDA runtime / training / plotting | Not executed; four explicit hardware gates remain |

Evidence: [JUnit results](gpu_plots_pytest.xml) and
[synthetic experiment configuration, metrics and plot metadata](gpu_plots_synthetic_smoke.json).
The full suite took 13.02 seconds in this environment; that is a test duration,
not a sampling, training or renderer benchmark. The retained C++ tests concern
the legacy model only.

| Environment | Value |
|---|---|
| Python | 3.12.14 |
| PyTorch | 2.8.0+cpu |
| NumPy | 2.5.3 |
| Zuko | 1.6.0 |
| Matplotlib | 3.11.2 |
| Test device | CPU, one thread |
| Synthetic experiment precision | FP32 MLP, FP64 spline / HG / directions / log PDF |

## GPU execution changes and their checks

The default configuration uses `training.device="cuda"`,
`training.dtype="float32"`, and `model.spline_dtype="float64"`.
`evaluate-rainbow`, `plot-rainbow`, and checkpoint loading also default to CUDA.
Unavailable CUDA fails before generating the large training pools or creating
the training output directory. CPU execution requires an explicit choice.

The CDF adapter validates and samples the fixed teacher pools on the CPU once.
Training directions and validation arrays are then retained on the selected
device. Each update uses `torch.randint` with a dedicated device generator and
indexes the resident training tensor. Likelihood evaluations operate in batches,
reduce the statistics on the device, and return a small summary. NF proposal
evaluation batches the model work and then passes the generated directions to
the CPU reference CDF lookup in one transfer.

Loss and gradient-norm checks are accumulated and transferred at validation,
checkpoint or controlled-pause boundaries. Parameter finiteness is checked
before publishing that boundary. A failed block leaves the preceding valid
checkpoint intact. The trusted internal training path disables per-call scalar
input validation; invalid numerical outputs still fail the aggregate checks.
The public model retains input validation by default.

The fixed HG coefficient is cached as a nonpersistent FP64 device tensor.
Changing the weight dtype or device rebuilds it from the authoritative Python
binary64 value. Sampling, evaluation and the HG baseline reuse this cache.
Regression tests check the exact coefficient bits, checkpoint compatibility,
and absence of per-call Python-scalar tensor construction and scalar reads.

CUDA Adam uses foreach operations. TF32 and AMP / FP16 are not enabled
implicitly. Deterministic CUDA configuration validates the cuBLAS workspace
setting. The recorded runtime includes the resolved device index, GPU name and
capability when CUDA is used. Same-runtime resumption is the reproducibility
contract; bitwise equivalence across different platforms is not asserted.

The new optimizer/minibatch state uses checkpoint version 3. Version-2
checkpoints remain readable for inference, including their original spline
precision. They are not treated as exact resumptions under changed code and
random-number generation. The model's historical schema-1 extra state remains
compatible: omission of `spline_dtype` means the original `"model"` precision.

## Synthetic mixed-precision learning experiment

The analytic fixture before cell integration is

$$
p(\mu,\phi_s)=\frac{(1+0.6\mu)(1-0.65\cos(2\phi_s))}{4\pi}.
$$

It is integrated into 16 scattering-angle cells and 24 source-azimuth cells.
The learned teacher is the resulting stored piecewise-constant density per
steradian. Its stored first moment is `g=0.1987210964506126`. The labels 550 nm
and 20 degrees define a coordinate fixture, not a physical rainbow prediction.

| Experiment setting | Value |
|---|---|
| Model | 2 alternating coupling layers, 8 bins, hidden widths `[16,16]` |
| Precision | FP32 MLP, FP64 spline and HG |
| Fixed train / validation / final test | 2,048 / 2,048 / 2,048 points |
| Updates / batch size | 100 / 256 |
| Learning rate | 0.003 |
| Validation interval | 20 updates |
| Checkpoint interval | 4 updates |
| Training seed | 415 |
| Final NF proposal samples | 512 |
| Diagnostic CDF scatter | 32,768 independent samples; seed 2027, stream 100 |
| Selected step | 100, selected by validation NLL |

| Independent final-test statistic | Value |
|---|---:|
| HG forward KL estimate | 0.1101120408 |
| HG KL standard error | 0.0096533510 |
| NF forward KL estimate | 0.0095769370 |
| NF KL standard error | 0.0032730340 |
| HG NLL | 2.4676223285 |
| NF NLL | 2.3670872247 |
| Paired NLL improvement over HG | 0.1005351038 |
| Improvement standard error | 0.0094022494 |
| Relative importance ESS | 0.9756543529 |
| Mean importance weight | 1.0023216308 |
| Maximum sample/eval log-PDF discrepancy | 2.6645352591e-15 |

These values validate a small implementation example. The production starting
configuration has larger pools and a larger model; neither its parameters nor
the 100-update synthetic example have been optimized for real solver data.

The training implementation fingerprint for this run is
`372fb5c5e46a6a2a6166026ca1344f3b6d0756fd7fda753408a550b382d9e63d`.
Complete configurations and source-file hashes are recorded in the JSON report.

## Plotting conventions and validation

The requested maps are `reference_pdf.png`, `cdf_samples.png`, and `nf_pdf.png`.
`comparison.png` places them together. All individual canvases and plot
rectangles have identical dimensions; the data rectangle is 2:1. The horizontal
axis is recorded solver azimuth from -180 to 180 degrees. The vertical axis is
scattering angle from 0 to 180 degrees, with forward scattering at the top.

The teacher is reconstructed from the two CDF differences and the actual
nonuniform `u_edges`. Cell mass is divided by cell solid angle. No smoothing,
probability floor, renormalization or extra `sin(theta)` multiplier is applied
to the PDF. True zero cells have their own gray color. Invalid or unrepresentable
NF densities raise instead of becoming zero-colored pixels.

The NF is queried at every stored cell's midpoint in `(u, source_phi)`, after
the explicit source-to-NF rotation. Teacher and NF share one logarithmic color
normalization. In this experiment its limits were
`0.012247347446778996` and `0.205611032072011` per steradian. The 16-by-24
appearance of the fixture is its native CDF resolution, not display downsampling
of a high-resolution solver record.

The scatter uses an independent diagnostic CDF stream, rather than selecting
or weighting points using the NF. On a `(phi, theta)` angular chart, raw point
density includes `sin(theta)`; it is not directly proportional to the plotted
density per steradian. Figure captions and metadata make this distinction.

Tests use an independent anisotropic analytic density to detect a missing,
reversed or 90-degree-offset source rotation. They also check nonuniform cell
areas, zero cells, CDF scatter frequencies, shared color normalization, PNG/PDF
output, global RNG preservation and restoration of individual module modes.
Figures are created only after the full optimization plan completes, using
the selected `best.pt` weights.

The teacher's reconstructed mass was 1.0. The NF's native-cell midpoint
quadrature was 0.9958057756. The latter is a coarse-grid quadrature diagnostic,
not an exact NF normalization measurement or a reason to renormalize the PDF.

## Reproduce the checks

Install the source checkout and development dependencies. From its root:

```sh
python -m pytest -q --junitxml=reports/gpu_plots_pytest.xml
ruff check .
python -m build --no-isolation
```

The executed development run used `PYTHONPATH=src`, `OMP_NUM_THREADS=1`, and
`MKL_NUM_THREADS=1` with the recorded CPU environment. The synthetic plotting
experiment used the existing test harness explicitly:

```python
from pathlib import Path
from dataclasses import replace
from test_single_condition import smooth_teacher, small_model, small_config
from phaseflow.rainbow import RainbowReference
from phaseflow.single_condition import train_single_condition
from phaseflow.plotting import RainbowPlotConfig

root = Path("runs/reproduce_gpu_plots")
with RainbowReference(smooth_teacher(root / "record")) as reference:
    result = train_single_condition(
        reference,
        replace(small_model(), spline_dtype="float64"),
        small_config(device="cpu", dtype="float32", steps=100, eval_every=20),
        root / "training",
        plot_config=RainbowPlotConfig(cdf_samples=32768, dpi=160),
    )
```

For this test-harness snippet include both `src` and `tests` on `PYTHONPATH`.
It is not the real-data training command. That command is documented in the
[README](../README.md), using the CUDA-default `configs/rainbow_single.json`.

## CUDA gates still to execute

On the target GPU, first check availability, then run the marked tests:

```sh
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
python -m pytest -m cuda -q
```

The four gates cover mixed-precision sample/eval/gradient parity, resident
training and independent evaluation, exact CUDA optimizer/RNG resumption, and
CUDA/CPU plotting-grid agreement. All-skipped output is not GPU validation.
Actual timing should be measured on the intended GPU with representative
solver records. CUDA kernel compilation, native model export and OptiX renderer
integration are separate work from this Python training and plotting update.
