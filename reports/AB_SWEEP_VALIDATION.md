# A/B sweep and angular diagnostics validation

Date: 2026-10-08 UTC.

## Scope

The delivered addition compares learning rates, selects the minimum validation
NLL, and compares fixed training-pool sizes at the selected rate. It adds exact
saved-CDF reporting-band statistics and aligned PDF profiles. The v0.4 model,
NLL objective, fixed external HG coefficient, source distribution, core training
implementation and default FP32 arithmetic are retained.

The training-code fingerprint is unchanged from the delivered v0.4 baseline:

```
1280c84d98be314e9cd54aad4203d8ed01ed522f2fc8a2856e749a7e7767f474
```

The current `rainbow.py` is byte-identical to that baseline. The earlier reported
metadata error was resolved by regenerating the input with the updated solver;
no legacy CDF compatibility adapter is included here.

## Executed environment

| Component | Observed value |
|---|---|
| OS | Linux x86-64 |
| Python | 3.14.8 |
| PyTorch | 2.14.1+cpu |
| CUDA runtime / device | None / unavailable |
| Zuko | 1.6.0 |
| NumPy | 2.5.3 |

The batch launcher continues to require the user's selected CUDA environment.
All training performed for this report explicitly selected CPU and a small
synthetic fixture. No GPU training, Windows CMD execution, physical Rainbow
accuracy, OptiX or CoopVec evaluation was performed here.

## Full regression suite

Executed from the repository root:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
git diff --check
```

Results:

- **329 passed, 9 skipped in 36.86 seconds.** All nine skips require CUDA.
- Ruff: all checks passed.
- `git diff --check`: no whitespace errors. Git emitted a line-ending conversion
  notice for the pre-existing `execute.bat` working copy.

The new tests include 16 runner tests, 24 angular-diagnostic tests plus one
CUDA-only test, and two CLI integration tests. They verify:

- genuine miniature training and A-to-B selection;
- validation-only selection even when reporting-only test results prefer the
  opposite learning rate;
- common independent evaluation streams and nested training-pool hashes;
- reuse of the identical A baseline in B;
- exact checkpoint continuation after interruption;
- completed-checkpoint postprocessing recovery with zero extra optimizer steps;
- refusal to reuse changed data, configuration, code/runtime or recorded results;
- exact band masses including partial nonuniform cells and tiny positive tails;
- zero-density solid-angle fractions and preserved zero PDF values;
- source-to-NF frame conversion and periodic source-cell selection;
- stable analytic HG band probabilities, checked against independent quadrature;
- bounded model-device evaluation, model-mode and RNG preservation;
- real CLI dispatch, diagnosis of `best.pt`, output arrays and CSV integration.

## Requested candidates exercised end to end

The synthetic fixture is the existing `tests/test_single_condition.py` function
`smooth_teacher`. Its saved cells are exact integrals of a smooth positive
azimuth-dependent analytic density. It is neither a Rainbow solver output nor a
test of rainbow-peak reconstruction.

The candidate lists match the requested experiment exactly:

- A learning rates: `3e-4`, `1e-3`, `3e-3` with N = 65536.
- B training-pool sizes: 4096, 16384, 65536, 262144 at the A-selected rate.

The correctness workload used two coupling layers, eight bins, hidden sizes
16/16, FP32 throughout the model, 20 updates of batch size 128, validation 512,
test 1024 and proposal samples 512. It enabled TensorBoard, learning curves,
three-map comparisons and the real angular-diagnostic callback. The full
configuration is retained in [config.json](ab-sweep-synthetic/config.json).

The actual CLI was run with `--device cpu`. It completed **six physical trials
and seven logical rows**. A selected `0.003` for this synthetic fixture and short
budget; every B trial then used that value, and the N = 65536 row reused A.
This value is not a recommendation for the user's physical dataset.

The identical CLI command was run again. It completed through receipt reuse,
and **all 133 saved per-trial/attempt/manifest files checked remained byte
identical**, including model/optimizer checkpoints and logs. The summary outputs
were allowed to be regenerated.

An independent `diagnose-rainbow` command also evaluated the selected existing
run without further training. Its saved profile arrays matched the sweep's
profile arrays exactly.

### Retained evidence

The following files are copies of actual synthetic-run outputs:

- [Comparison table](ab-sweep-synthetic/summary.csv)
- [Structured summary](ab-sweep-synthetic/summary.json)
- [A selection](ab-sweep-synthetic/selection.json)
- [Summary plot](ab-sweep-synthetic/summary.png)
- [Angular profiles](ab-sweep-synthetic/angular_profiles.png)
- [Profile arrays](ab-sweep-synthetic/angular_profiles.npz)
- [Angular diagnostics](ab-sweep-synthetic/angular_diagnostics.json)
- [Sweep configuration](ab-sweep-synthetic/sweep_config.json)
- [Reuse and runtime verification](ab-sweep-synthetic/verification.json)

The summary and angular-profile PNGs were inspected visually. Axes, legends,
logarithmic PDF scales and angular zooms were readable without clipping. These
are demonstration outputs from the synthetic validation. Paths and checkpoint
hashes inside the records identify the original validation run; the evidence
directory is not a relocated resumable training run.

## Windows batch and packaging

`sweep.bat` was statically checked for ASCII/CRLF, quoted paths, label targets,
caret continuations, shared interpreter/device selection, parser compatibility
and nonzero exit propagation. It expects `RECORD` to be copied from the working
`execute.bat`. TensorBoard is started separately with `monitor.bat`.

The add-on is delivered as an incremental patch against the previously
delivered v0.4 working tree, plus the complete updated files. It does not require
reinstalling dependencies when the checkout is already installed in editable
mode. Use a new sweep output directory for a changed experiment, and only one
writer process per output directory.

## Scientific limits

One condition and one seed provide an exploratory comparison. Fixed update
counts do not establish convergence for every N, and error bars from Monte
Carlo integration do not measure variation between training seeds. The 120–150
degree band is a configurable reporting region; its expected counts are not
observed counts or guarantees of coverage of every fine peak. Native-midpoint
NF profiles do not resolve arbitrary within-cell behavior.

The cited Mie fitting loss is discussed in
[SWEEP_AND_RAINBOW_LOSS.md](../docs/SWEEP_AND_RAINBOW_LOSS.md). It has not been
implemented as a training objective in this addition. The user's shared image
was examined qualitatively; the underlying real CDF and training logs were not
available to measure numerical accuracy here.
