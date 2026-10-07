# Model and density contract

## Conditions and physical frame

Let `k_i` be the incident light's unit propagation direction and `a` the particle's
oriented unit symmetry axis. The two physical conditions are

\[
c=(\lambda,\eta),\qquad \eta=a\cdot k_i\in[-1,1].
\]

The outgoing propagation direction is represented in a right-handed frame whose
z axis is `k_i`, x points toward the projection of `a` perpendicular to `k_i`, and
y is z cross x. Exactly axial incidence uses an arbitrary deterministic frame;
the model below makes its density independent of that azimuthal choice.
The particle is axisymmetric, but it need not have north/south symmetry.

For a local outgoing direction `(x,y,z)`, define

\[
\mu=z,\qquad \phi=\operatorname{atan2}(y,x),\qquad d\Omega=d\mu\,d\phi.
\]

Positive g describes forward scattering under this propagation convention.

## HG head and coordinates

The condition encoder outputs normalized raw scalars followed by integrated
Gaussian one-blob features. If K blobs are enabled, their width is `sigma=1/K`:

\[
s_\lambda=\frac{\lambda-\lambda_{\min}}{\lambda_{\max}-\lambda_{\min}},\qquad
s_\eta=\frac{\eta+1}{2},\qquad
e_j(s)=\Phi\left(\frac{(j+1)/K-s}{1/K}\right)
       -\Phi\left(\frac{j/K-s}{1/K}\right).
\]

The feature order is `[s_lambda,s_eta,lambda_blobs,eta_blobs]`. Out-of-range
Gaussian mass is not renormalized. `one_blob_bins=0` keeps only the raw scalars.
There is no Fourier or periodic encoding of eta or wavelength.

The HG head is `g(c)=g_limit*tanh(MLP(encoded(c)))`. The conditional base is

\[
h_g(\mu)=\frac{1-g^2}{4\pi(1+g^2-2g\mu)^{3/2}},\qquad
H_g'(\mu)=2\pi h_g(\mu).
\]

`hg.py` implements the analytic CDF and inverse with algebraic rearrangements
that remain valid at g=0. It does not use a small-g isotropic approximation.
All finite model g satisfy `abs(g)<=g_limit<1`, including floating-point tanh saturation.

The probability coordinates are

\[
t=(t_1,t_2)=\left(H_g(\mu),\frac{|\phi|}{\pi}\right)\in[0,1]^2.
\]

The reflection sign has equal probability on the two sides. Its factor 1/2
cancels the doubled angular Jacobian of the folded chart. Therefore a square
density r produces the solid-angle density

\[
q(\omega_o\mid c)=h_{g(c)}(\mu)\,r(t\mid c).
\]

There is no additional sine, cosine, or factor-of-two term to multiply into this
expression. A density with respect to polar angle would be a different measure.

## Normalizing direction and HG as the actual base

Let N be the learned map from the data square t to the uniform base square b.
It is a composition of alternating conditional RQS couplings. Its log density is

\[
\log q(\omega_o\mid c)=\log h_{g(c)}(\mu)+\log|\det D_tN(t;c)|.
\]

Evaluation computes `t`, runs N, and accumulates the forward log determinant.
Sampling starts with b uniform, runs `N^-1` in reverse layer order, and applies
the HG quantile map. Its returned log PDF is

\[
\log q=\log h_g(\mu)-\log|\det D_bN^{-1}(b;c)|.
\]

Using an analytic uniform probability coordinate is still an HG-base flow.
If W_g is the HG quantile warp, the residual sphere transport on each branch is
`W_g composed with N^-1 composed with W_g^-1`. Applied to an HG sample, this
gives exactly the implementation above. The raw g is also appended to the
conditioner input when `include_g_context=true`; this does not change the base.

The two input uniforms suffice. `u[0]` is in (0,1). If `u[1]<1/2`, choose the
negative reflection sign and set `b[1]=2*u[1]`. Otherwise choose the positive
sign and set `b[1]=2*u[1]-1`. `u[1]` is in [0,1). The internal sampler draws
uniformly from representable midpoint values rather than clamping endpoints.

## RQS coupling and endpoint behavior

For each scalar spline, the conditioner emits K width logits, K height logits,
and K+1 slope logits. Positive constraints are

\[
w_j=\epsilon_w+(1-K\epsilon_w)\operatorname{softmax}(l_w)_j,\qquad
h_j=\epsilon_h+(1-K\epsilon_h)\operatorname{softmax}(l_h)_j,
\]

\[
d_j=\epsilon_d+\operatorname{softplus}(l_d)_j.
\]

Positions at the two ends remain 0 and 1. Their slopes are trainable. The final
conditioner layer starts at zero except for the slope biases that parameterize
slope one. Thus the residual begins at identity up to parameter rounding.
The inverse uses the analytic quadratic solution with stable algebraic branches,
not iterative numerical root-finding.

Odd-numbered couplings update t1 while retaining t2. Even-numbered couplings
update t2 while retaining t1. Each layer's conditioner is called once during
sample-with-PDF and once during arbitrary PDF evaluation. Both directions still
require sequential traversal of all layers. This is an operation-count property,
not a measured equality of sample/eval runtime.

## Physical symmetry at axial incidence

Reflection folding alone would permit arbitrary azimuth dependence at eta=±1.
To remove dependence on the undefined scattering-plane reference there, define

\[
A=\sqrt{(1-\eta)(1+\eta)}=\sin\alpha,\qquad \eta=\cos\alpha.
\]

For a radial t1 update, feed `0.5+A*(t2-0.5)` to its conditioner, while retaining
the original t2 in the state. At A=0, the radial transform cannot depend on t2.

For an azimuthal t2 update, keep the radial conditioner input unchanged, but
blend its constrained spline masses and slopes before knot normalization:

\[
w'_j=A w_j+(1-A)/K,\quad h'_j=A h_j+(1-A)/K,\quad d'_j=A d_j+(1-A).
\]

At A=0 this is exactly the identity azimuthal map. Each layer stays invertible,
and the constraint is continuous in incidence. The HG base then remains uniform
in phi while its radial density can still learn a non-HG shape. This is an
always-on v1 model contract and is identical in Python and native inference.

The first-order taper in inclination is physically motivated. The other
directional invariant is

\[
\nu=a\cdot\omega_o
 =\eta\mu+\sin\alpha\sqrt{1-\mu^2}\cos\phi.
\]

A smooth rotationally invariant, mirror-symmetric density written as
`F(mu,eta,nu)` can therefore contain a term
`alpha*sqrt(1-mu^2)*(partial F/partial nu)*cos(phi)` near alpha=0, and an
analogous first-order term near the opposite axis. Using `1-eta^2=sin(alpha)^2`
as the gate would unnecessarily suppress this allowed first-order variation.
The chosen gate permits it while still enforcing exact axial azimuth symmetry.
It does not guarantee global smoothness or the precise asymptotic order of
every higher azimuthal harmonic. The two signs of eta remain distinct conditions.

The gate's derivative with respect to eta is singular at eta=+/-1 because cosine
is the condition coordinate. Conditions are fixed in the directional Jacobian
and ordinary parameter training; g-head and flow parameter gradients remain
available at these endpoints. Gradients with respect to eta itself at an
endpoint are not guaranteed by this API.

## Training objectives

Each training condition has equal objective weight, regardless of the number of
stored points. For target samples, the objective is ordinary empirical NLL.
For quadrature, normalized masses `m_j` estimate the angular integral:

\[
L=-\frac{1}{C}\sum_c\sum_j m_{cj}\log q(\omega_{cj}\mid c),\qquad
\widehat g_c=\sum_jm_{cj}\mu_{cj}.
\]

Stage 1 minimizes mean squared error between g(c) and the target mean cosine.
This is moment matching; it is not maximum-likelihood fitting of the HG family.
Stage 2 freezes the HG head and minimizes the full spherical NLL of the flow.
These stages anchor the interpretation of g despite non-unique HG/residual
decompositions. The residual may change the final distribution's mean cosine.

An explicit optional joint stage optimizes the same full NLL plus a positive
HG moment regularizer. Do not drop `log h_g`, detach `H_g(mu)`, or treat a cached
HG coordinate as fixed when g is trainable. Conditions are fixed when forming
the angular Jacobian, but all g dependencies remain active in the training
gradient. No derivative `dg/dc` is part of the directional PDF Jacobian.

## Scope and numerical limits

This is a surface density on S² expressed through a chart and two reflection
branches. It does not implement every construction in *Normalizing Flows on
Tori and Spheres*. In particular it is not a single global smooth S² map. Polar
density limits may depend on phi; values at exactly mu=±1 use canonical phi=0.
These zero-area points do not change normalization. Physical smoothness near
them needs evaluation on the actual target distribution.

HG coordinates and sampled directions use float64. Network/RQS calculations
follow model dtype. At g=0.999, storing only a float32 cosine can collapse a
substantial number of directions onto mu=1; later upcasting cannot restore them.
Keep double direction information through training and correctness comparisons.
The finite precision grid of the RQS still limits very extreme concentration.

Open input uniforms alone cannot prevent all rounded poles: a sufficiently
compressed inverse spline may round its probability coordinate to 0 or 1, and
the HG quantile may also exhaust the angular precision. With open `u[0]`, an
exact monotone spline and `abs(g)<1` produce an interior cosine in real
arithmetic. A sampled cosine rounded to either pole is therefore a numerical
failure. With `validate_args=True`, the Python sampler raises
`FloatingPointError`; with validation disabled, affected output directions and
log PDFs are NaN while unaffected lanes remain valid. Native inference returns
an invalid sample. The sampler neither clips the result nor retries silently.

Replacing the reported PDF with the canonical pole evaluation would hide a
finite-precision atom without giving it a valid solid-angle density. Such
samples must be diagnosed before rendering use; they must not simply be
discarded. Promoting the entire model/RQS calculation to float64 can resolve a
particular failure, but it does not guarantee that arbitrarily concentrated
flows remain representable. Pole rejection is separate from evaluating a PDF
at an explicitly supplied exact-pole direction, which uses the documented
canonical coordinate convention.

No finite-angle forward cone is deleted. No negative lobe, zero-density region,
oscillation, or polarization effect is silently synthesized. A positive smooth
flow can approximate zero target density but cannot represent an exact open
zero-density region with positive base and finite derivatives.

The model represents a normalized conditional angular distribution, not a
scattering cross-section. Exact physical zeros and total scattering strength
remain the responsibility of the physical model / renderer interface.
