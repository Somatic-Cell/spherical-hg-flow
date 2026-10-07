> **Legacy v1 point-cloud contract.** The content below applies to `PhasePointCloud`
> and the historical `train` / `evaluate` commands. The new `train-rainbow` workflow
> reads a validated solver record directly, uses external g, and keeps independent
> point streams for a single fixed condition. Its contract is in
> [RAINBOW_SINGLE_CONDITION.md](RAINBOW_SINGLE_CONDITION.md). In particular, the old
> instruction to treat a single condition as in-sample does not apply to it.

# Point-cloud and CDF adapter contract, version 1

The learning pipeline depends on directions and conditions, not on a particular
CDF buffer layout. The current in-memory boundary is `PhasePointCloud`.

| Field | Shape / dtype | Meaning |
|---|---|---|
| `conditions` | `[C,2]`, floating | `[wavelength_nm, incident_cosine]` |
| `outgoing` | `[N,3]`, float32/64 | Unit propagation directions in the local scattering frame |
| `condition_index` | `[N]`, integer | Row index into conditions |
| `mode` | string | `target_samples` or `quadrature` |
| `weights` | `[N]`, floating; quadrature only | Nonnegative angular integration masses |
| `coordinate_frame` | string | `incident_z_particle_axis_x` |
| `coverage` | string | `full_sphere` |
| `metadata` | JSON object | Teacher provenance, particle parameters, normalization, etc. |

Conditions are unique and all groups must have points. Directions must have
norm one to within 2e-5. The adapter checks them and does not add jitter, smooth,
or normalize them. Model inference normalizes tolerated vector roundoff in
float64. A supplied float64 direction is preserved even with float32 networks.
Moment targets use the same `z/norm(direction)` cosine as inference, without
altering the stored directions.

## Physical coordinates

For world-space incoming direction `k_i` and particle axis `a`, both are
**propagation** / oriented-axis directions. Compute `eta=dot(a,k_i)` after
normalizing them. Do not substitute the scattering angle, elevation angle in
radians, or absolute incident cosine.

```python
import torch
from phaseflow.geometry import scattering_frame

# Use float64 tensors to preserve narrow angular features.
x_axis, y_axis, z_axis = scattering_frame(incident, particle_axis, validate_args=True)
local_outgoing = torch.stack(
    (
        (outgoing_world * x_axis).sum(-1),
        (outgoing_world * y_axis).sum(-1),
        (outgoing_world * z_axis).sum(-1),
    ),
    dim=-1,
)
```

Sampled local directions are returned to world coordinates by
`local.x*x_axis + local.y*y_axis + local.z*z_axis`. A rigid rotation has unit
solid-angle Jacobian, so the PDF value is unchanged.

Only exactly parallel/antiparallel incidence uses the deterministic fallback
frame. The model's axial constraint makes its PDF independent of this arbitrary
azimuthal reference. A particle's top and bottom still remain distinct conditions.

## Target samples

The directions must already follow the desired normalized conditional phase
distribution. This can come from inverse-CDF sampling. Use

```python
cloud = PhasePointCloud(conditions, outgoing, condition_index, mode="target_samples")
```

Weights are prohibited. Multiplying sampled points by their phase density again
would change the objective. A fixed point cloud is an empirical sample of the
teacher; the sample count and seed belong in provenance. Repeated stochastic
minibatches can be drawn from it; no assumption about CDF interpolation is made.

## Weighted quadrature

If the buffer contains a directional grid and phase density values, supply
integration masses. Examples are

- A grid in scattering cosine and azimuth: `mass = p(mu,phi)*delta_mu*delta_phi`.
- A grid in polar angle and azimuth: `mass = p(theta,phi)*sin(theta)*delta_theta*delta_phi`.
- A quadrature rule: `mass = p(direction)*solid_angle_quadrature_weight`.

```python
cloud = PhasePointCloud(
    conditions,
    outgoing,
    condition_index,
    mode="quadrature",
    weights=integration_masses,
)
```

An arbitrary common scale per condition cancels on normalization. Masses must
be finite, nonnegative, and have positive sum in each condition. Point values
alone do not describe solid-angle weights on a nonuniform grid.

By default conditions are sampled uniformly, then their quadrature points are
sampled proportional to mass. The NLL loss weight is one. The explicit
`point_sampling="uniform"` alternative samples points uniformly in a selected
group and returns loss weight `N_c*m_j`. Both estimate the same condition-balanced
objective. Do not multiply the mass twice.

## CDF questions left to the physical-data producer

The producer must specify full-sphere normalization, direction conventions,
whether values are a joint/marginal/conditional CDF, interpolation and inversion
rules, and the physical model settings. In particular, a buffer restricted to a
visible-rainbow angular interval is not a complete phase function. Settle that
coverage before passing its samples as `full_sphere`.

This adapter cannot verify the physical truth of metadata or establish unseen
support from a finite point cloud. It rejects declared cropped coverage and
malformed arrays; the producer remains responsible for their semantics.

No `abs(eta)` folding is allowed merely because the particle is axisymmetric.
For the reflection model, the physical scalar distribution must be symmetric
under local `y -> -y`. If polarization or a non-axisymmetric particle breaks that
assumption, the model family must be changed explicitly.

## NPZ adapter

`save_npz` / `load_npz` store the above arrays without pickle. Required scalar
labels are `schema_version=1`, `data_mode`, `coordinate_frame`, and `coverage`.
Metadata uses `metadata_json`. The fingerprint includes actual array values,
dtypes, ordering, and metadata, because point order affects deterministic sampling.
No private CDF layout is embedded in this format.

## Training / validation

The split is made over condition rows and recorded in `split.json`; held-out
conditions are unused by both the HG warmup and residual training. The default
split tests prediction at withheld condition pairs. It does not by itself prove
generalization to new wavelength ranges, particle sizes, or physics regimes.

Use `validation_fraction=0` only for an explicitly in-sample experiment, including
a single fixed condition. To test point sampling error separately, produce an
independent cloud and use `phaseflow evaluate --split all` on it.

Quadrature masses and target samples suffice for cross-entropy/NLL. Without
the teacher's continuous density/entropy, those data do not yield an absolute
continuous KL divergence. Reports therefore label NLL directly.
