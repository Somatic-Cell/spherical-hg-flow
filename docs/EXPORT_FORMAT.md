# Native inference and export format v1

`phaseflow export --checkpoint run/checkpoint.pt --output exported` writes three
files. `export_model(model, output_directory)` provides the same Python API and
returns the output directory as a `Path`.

| File | Purpose |
| --- | --- |
| `model.pflow` | Self-contained native inference model, with explicit dimensions, little-endian descriptors, and FP32 weights. The C++ loader only needs this file. |
| `weights.safetensors` | The same named FP32 tensors for inspection or another renderer upload implementation. No pickle or executable loader is used. |
| `manifest.json` | Readable model configuration, tensor offsets and shapes, coordinate conventions, checksums, and numerical precision. |

The legacy project's useful separation between safe tensor weights and readable
metadata is retained. Its checkpoint key names, hardcoded file paths, and flow
architecture are not part of this format. The native file additionally resolves
every tensor's role, shape, and matrix layout without a JSON dependency.

## Build and run the CPU reference

```sh
g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Icpp/include cpp/src/model.cpp cpp/src/phaseflow_cli.cpp -o phaseflow_cli
./phaseflow_cli exported/model.pflow
```

When CMake 3.20 or newer is available, the equivalent build is:

```sh
cmake -S cpp -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native -j
build/native/phaseflow_cli exported/model.pflow
```

The executable reads one query per stdin line and writes one result per line:

| Query | Output columns |
| --- | --- |
| `encode 550 0.3` | Normalized raw conditions followed by both one-blob vectors |
| `g 550 0.3` | Predicted HG coefficient |
| `eval 550 0.3 0 0 1` | `log_pdf pdf g` |
| `sample 550 0.3 0.37 0.81` | `direction_x direction_y direction_z log_pdf pdf g` |

PDFs are per steradian. Wavelength is in nm. The second condition is
`incident_cosine = particle_axis dot incident_propagation_direction`, in `[-1,1]`.
It is not an angle in radians. Wavelength extrapolation is rejected.
The first sampling uniform is in `(0,1)` and the second in `[0,1)`.

The C++ host reference uses double arithmetic with FP32 exported parameters.
PyTorch normally uses FP32 for the networks and RQS, and double for the HG
and spherical geometry. Consequently, the native reference is not bitwise
identical to normal FP32 PyTorch execution. The integration tests also compare
against a double PyTorch model whose parameters start as the exported FP32
values, including nonidentity couplings, varying conditions, both reflection
branches, and strongly positive and negative HG coefficients.

## Renderer call contract

The numerical functions in `cpp/include/phaseflow/phaseflow_core.cuh` use plain
POD descriptors and caller-owned workspace. `sample_local` calls the HG network
once, each coupling conditioner once, and evaluates each scalar inverse spline
and inverse Jacobian together. Its returned density does not run a second
forward flow. `evaluate_local` traverses the forward couplings and calls each
conditioner once. In either direction the spline inverse is analytic; there is
no iterative solve.

For a host application:

```cpp
#include "phaseflow/model.hpp"
#include <stdexcept>

auto owner = phaseflow::Model::load("exported/model.pflow");
auto model = owner.view();
std::vector<double> workspace(model.workspace_scalars());
auto result = phaseflow::sample_local(
    model, 550.0, 0.3, 0.37, 0.81, workspace.data()
);
if (!result.evaluation.valid) {
    throw std::runtime_error("Invalid phase sampling query");
}
// result.direction is in the scattering frame.
// result.evaluation.pdf is the MIS proposal density per steradian.
```

The owner must outlive the view. Each concurrently active query needs its own
workspace. `Model::sample` and `Model::evaluate` allocate their workspace for
convenience; a renderer should reuse storage through the free functions.
Workspace is `context_size + 2 * max_network_width` scalars, specified by the
manifest and `ModelView::workspace_scalars()`.

The local `+z` axis is the incident **propagation** direction. The local `+x`
axis points along the projection of the particle axis onto its perpendicular
plane; local `+y = z cross x`. Rotate the output into the renderer's world frame
and rotate an evaluated direction into the same local frame. A renderer that
stores an incident direction pointing away from the scattering event must
convert it to this propagation convention. The separate Python geometry module
defines the frame and its exact axial-incidence fallback.

Evaluation accepts finite unit directions within `2e-5` length tolerance and
renormalizes that small representation error in double precision. Exact poles
use the canonical azimuth zero. The square chart with reflection branches does
not enforce a smooth, direction-independent density limit at the poles.

An interior input uniform can round to a radial endpoint in the inverse RQS,
or its HG direction can round to an exact pole when concentration is very high.
Such a sample returns `valid=false`. Returning a finite surface PDF for the
collapsed point would hide a discrete atom introduced by numerical rounding.
The reference therefore does not clamp the point, substitute its canonical
pole density, or automatically retry. Treat this as a numerical failure to
investigate through more precision or the model's concentration; double
arithmetic alone does not guarantee that every representable uniform is safe.
PyTorch raises `FloatingPointError` with validation enabled and returns NaN
lanes with validation disabled for the same failure condition. A renderer
must not silently resample failed lanes under the original proposal density.

### CUDA and OptiX integration status

The `.cuh` core has `__host__ __device__` functions and makes no allocation,
launch, CUDA runtime, or OptiX calls. The shipped loader and tests are standard
C++20. **No NVCC compilation, GPU execution, OptiX integration, or GPU timing
was performed in the development environment.** The double scalar path is the
validated numerical reference; a fast FP32 device implementation requires its
own accuracy and performance measurements.

To upload a loaded model, copy `owner.parameters()`, `owner.dense_layers()`,
`owner.networks()`, and `owner.couplings()` to device arrays, then replace those
four pointers in a copy of `owner.view()`. The descriptors contain offsets in
units of FP32 elements, never host pointers. `Config` and the remaining view
fields are copied by value. Give each active ray/thread separate double
workspace and call the same numerical functions from the relevant shader.
Do not copy the binary's descriptor bytes directly into C++ structs: use the
validated host loader, which handles serialization independently of struct
padding and host byte order. Network widths and register/local-memory costs
must be assessed on the target GPU before choosing a device layout.

The proposal density returned by this model is the sampling density for MIS.
The renderer must separately evaluate its physical scattering function if the
flow is used only as a proposal. Substituting the learned density for the
physical function is a distinct modeling decision.

## Exact numerical interpretation

Conditions are normalized as `s_lambda = (lambda-lambda_min)/(lambda_max-lambda_min)`
and `s_eta = (eta+1)/2`. Encoding order is
`[s_lambda, s_eta, lambda_blob[0:K], eta_blob[0:K]]`.
Each blob entry integrates a Gaussian of standard deviation `1/K` over
`[k/K,(k+1)/K]`; out-of-interval Gaussian mass is not renormalized.
`one_blob_bins=0` disables the blob vectors.

Network 0 predicts `g = g_limit * tanh(raw)`. NF context is the encoded vector,
with this raw coefficient appended when `include_g_context=1`.
Each coupling network takes `[retained_coordinate, context...]` and outputs
`[width_logits[K], height_logits[K], derivative_logits[K+1]]`.
All hidden layers use ReLU; final layers have no activation. Dense matrices
are stored row-major with shape `[out_features, in_features]` and applied as
`y[row] = bias[row] + sum(weight[row,col] * x[col])`.

For each spline, widths are
`min_bin_width + (1-K*min_bin_width)*softmax(width_logits)`, with the analogous
height expression. Positive bin masses are normalized before taking cumulative
sums, then the two cumulative endpoints are set exactly to zero and one.
Knot derivatives, including both endpoints, are independently learned using
`min_derivative + softplus(derivative_logits)`.
An interior knot uses its right-hand bin; one uses the final bin. There are no
identity tails. Both directions use the stable rational-quadratic formulas
shared with `phaseflow.splines`, including coefficient scaling and the analytic
quadratic inverse.

Version 1 always enforces rotational symmetry at **axial incidence**. Set
`a = sqrt((1-incident_cosine)*(1+incident_cosine))`, the sine of the incident
inclination. This allows first-order azimuth dependence as incidence moves
away from the particle axis. A coupling that changes the radial HG coordinate
receives `0.5+a*(v-0.5)` as its retained conditioner input, while carrying the
original folded azimuth `v` through unchanged. A coupling that changes `v`
uses its unchanged radial conditioner input, then blends its constrained
widths and heights as `a*mass+(1-a)/num_bins`, and derivatives as
`a*derivative+(1-a)`, before bin normalization and cumulative sums. These are
convex combinations of positive spline parameters, so monotonicity and
analytic inversion are preserved. At incident cosine `-1` or `+1`, all
azimuth transforms are identity and all radial transforms are independent of
azimuth. Intermediate incidence changes continuously without folding the
incident cosine or imposing upper/lower particle symmetry. This convention is
part of the format version and requires no extra flag in the header.

The forward flow maps data square coordinates
`[HG_CDF(mu;g), abs(phi)/pi]` to a uniform square. Sampling traverses the
couplings in reverse order with inverse splines. The second input uniform
selects negative azimuth below `0.5` and positive azimuth otherwise, and
`2*u[1] mod 1` supplies the folded coordinate. The sample is transformed back
with the analytic HG inverse CDF. Reflection probability and folded angular
Jacobian cancel, giving physical density
`HG_pdf(mu;g) * square_residual_pdf` without an additional factor of two or pi.

## Binary layout

All numeric fields are little-endian. FP32/FP64 use IEEE 754. The file contains
the following sections in this exact order:

1. A 120-byte header.
2. `network_count` network records, 16 bytes each.
3. `dense_count` dense-layer records, 24 bytes each.
4. `coupling_count` coupling records, 8 bytes each.
5. `parameter_count` FP32 values, 4 bytes each.

### Header

| Byte offset | Type | Field |
| ---: | --- | --- |
| 0 | 8 bytes | ASCII `PHFLOW01` |
| 8 | uint32 | Format version, 1 |
| 12 | uint32 | Byte-order sentinel, `0x01020304` |
| 16 | uint64 | Exact total file byte count |
| 24 | uint64 | Parameter element count |
| 32 | uint32 | Dense-layer count |
| 36 | uint32 | Network count |
| 40 | uint32 | Coupling count |
| 44 | uint32 | One-blob bin count |
| 48 | uint32 | RQS bin count |
| 52 | uint32 | Include raw g in context, 0 or 1 |
| 56 | uint32 | Activation identifier, 1 = ReLU |
| 60 | uint32 | Maximum workspace network width |
| 64, 68 | uint32 each | Reserved, both zero |
| 72 | float64 | Minimum wavelength in nm |
| 80 | float64 | Maximum wavelength in nm |
| 88 | float64 | HG coefficient limit |
| 96 | float64 | Minimum RQS bin width |
| 104 | float64 | Minimum RQS bin height |
| 112 | float64 | Minimum RQS knot derivative |

### Records and packing

A network record contains four uint32 fields:
`first_layer, num_layers, input_size, output_size`.
Network 0 is the g head. Coupling i refers to network i+1.

A dense record contains uint32 `input_size, output_size`, followed by uint64
`weight_offset, bias_offset`. Offsets are element indices in the final FP32
parameter array. Each weight matrix immediately precedes its bias vector.
Networks' layers and all parameters are packed consecutively, without padding,
gaps, sharing, or overlaps.

A coupling record contains uint32 `network_index, retained_index`.
The retained index is 0 or 1; the changed index is `1-retained_index`.
Records are in forward, data-to-base evaluation order.

The loader validates magic, version, byte order, exact size, bounds, all
dimension chains, canonical offsets, config ranges, workspace requirements,
and finite parameters before constructing an inference view. It rejects
unknown activations and trailing bytes. Format-v1 implementation limits are
512 MiB per file, 4096 networks, 16384 dense layers, 65536 features in any
layer, and 4096 bins in either encoding or spline. SHA-256 checksums in the
manifest support artifact comparison; the self-contained native loader does
not require or authenticate the manifest.
