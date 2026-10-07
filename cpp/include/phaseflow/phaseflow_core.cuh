#ifndef PHASEFLOW_CORE_CUH
#define PHASEFLOW_CORE_CUH

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define PHASEFLOW_HD __host__ __device__
#else
#define PHASEFLOW_HD
#endif

// Shared scalar implementation. No CUDA allocation, launch, or OptiX API calls
// are made here. The host validates descriptors before constructing ModelView.
// CUDA compilation and GPU execution require a separate downstream validation.
namespace phaseflow {

inline constexpr double pi = 3.141592653589793238462643383279502884;

struct Config {
    double wavelength_min_nm;
    double wavelength_max_nm;
    double g_limit;
    double min_bin_width;
    double min_bin_height;
    double min_derivative;
    std::uint32_t one_blob_bins;
    std::uint32_t num_bins;
    std::uint32_t include_g_context;
};

struct DenseLayer {
    std::uint32_t input_size;
    std::uint32_t output_size;
    std::uint64_t weight_offset;
    std::uint64_t bias_offset;
};

struct Network {
    std::uint32_t first_layer;
    std::uint32_t num_layers;
    std::uint32_t input_size;
    std::uint32_t output_size;
};

struct Coupling {
    std::uint32_t network_index;
    std::uint32_t retained_index;
};

struct ModelView {
    Config config;
    const float* parameters;
    const DenseLayer* dense_layers;
    const Network* networks;
    const Coupling* couplings;
    std::uint32_t coupling_count;
    std::uint32_t workspace_width;

    PHASEFLOW_HD std::uint32_t encoded_size() const {
        return 2U + 2U * config.one_blob_bins;
    }
    PHASEFLOW_HD std::uint32_t context_size() const {
        return encoded_size() + config.include_g_context;
    }
    PHASEFLOW_HD std::size_t workspace_scalars() const {
        return static_cast<std::size_t>(context_size()) + 2U * workspace_width;
    }
};

template <typename Scalar> struct Vec3 { Scalar x, y, z; };
template <typename Scalar> struct Evaluation {
    Scalar log_pdf, pdf, g;
    bool valid;
};
template <typename Scalar> struct Sample {
    Vec3<Scalar> direction;
    Evaluation<Scalar> evaluation;
};
template <typename Scalar> struct ScalarTransform { Scalar value, log_det; };

template <typename Scalar>
PHASEFLOW_HD inline Scalar clamp(Scalar x, Scalar lo, Scalar hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}
template <typename Scalar>
PHASEFLOW_HD inline Scalar softplus(Scalar x) {
    return (x > Scalar(0) ? x : Scalar(0)) + ::log1p(::exp(-::fabs(x)));
}
template <typename Scalar>
PHASEFLOW_HD inline bool finite(Scalar x) {
    return x == x && x <= std::numeric_limits<Scalar>::max()
        && x >= -std::numeric_limits<Scalar>::max();
}
template <typename Scalar>
PHASEFLOW_HD inline Evaluation<Scalar> invalid_evaluation() {
    return {-std::numeric_limits<Scalar>::infinity(), Scalar(0), Scalar(0), false};
}

// Conditions: wavelength in nm, incident_cosine = particle_axis dot incoming
// propagation direction. No upper/lower particle symmetry is assumed.
template <typename Scalar>
PHASEFLOW_HD inline bool encode_conditions(
    const ModelView& m, Scalar wavelength_nm, Scalar incident_cosine, Scalar* out
) {
    if (!finite(wavelength_nm) || !finite(incident_cosine)
        || wavelength_nm < Scalar(m.config.wavelength_min_nm)
        || wavelength_nm > Scalar(m.config.wavelength_max_nm)
        || incident_cosine < Scalar(-1) || incident_cosine > Scalar(1)) return false;
    const Scalar s[2] = {
        (wavelength_nm - Scalar(m.config.wavelength_min_nm))
            / Scalar(m.config.wavelength_max_nm - m.config.wavelength_min_nm),
        (incident_cosine + Scalar(1)) / Scalar(2)
    };
    out[0] = s[0];
    out[1] = s[1];
    const auto bins = m.config.one_blob_bins;
    if (bins == 0) return true;
    const Scalar inv_sqrt_two = Scalar(0.707106781186547524400844362104849039);
    for (std::uint32_t dim = 0; dim < 2; ++dim) {
        for (std::uint32_t k = 0; k < bins; ++k) {
            // sigma=1/bins. The Gaussian mass outside [0,1] is intentionally
            // omitted; the retained bin masses are not renormalized.
            const Scalar lower = (Scalar(k) - Scalar(bins) * s[dim]) * inv_sqrt_two;
            const Scalar upper = (Scalar(k + 1) - Scalar(bins) * s[dim]) * inv_sqrt_two;
            out[2 + dim * bins + k] = Scalar(0.5) * (::erf(upper) - ::erf(lower));
        }
    }
    return true;
}

template <typename Scalar>
PHASEFLOW_HD inline const Scalar* network_forward(
    const ModelView& m, std::uint32_t index, const Scalar* input,
    Scalar* scratch_a, Scalar* scratch_b
) {
    const Network net = m.networks[index];
    const Scalar* current = input;
    for (std::uint32_t i = 0; i < net.num_layers; ++i) {
        const DenseLayer layer = m.dense_layers[net.first_layer + i];
        Scalar* output = current == scratch_a ? scratch_b : scratch_a;
        for (std::uint32_t row = 0; row < layer.output_size; ++row) {
            Scalar value = Scalar(m.parameters[layer.bias_offset + row]);
            const auto offset = layer.weight_offset
                + static_cast<std::uint64_t>(row) * layer.input_size;
            for (std::uint32_t col = 0; col < layer.input_size; ++col)
                value += Scalar(m.parameters[offset + col]) * current[col];
            output[row] = i + 1 < net.num_layers && value < Scalar(0) ? Scalar(0) : value;
        }
        current = output;
    }
    return current;
}

template <typename Scalar>
PHASEFLOW_HD inline bool prepare_context(
    const ModelView& m, Scalar wavelength_nm, Scalar incident_cosine,
    Scalar* workspace, Scalar& g
) {
    if (!encode_conditions(m, wavelength_nm, incident_cosine, workspace)) return false;
    Scalar* a = workspace + m.context_size();
    Scalar* b = a + m.workspace_width;
    const Scalar raw = network_forward(m, 0, workspace, a, b)[0];
    g = Scalar(m.config.g_limit) * ::tanh(raw);
    if (m.config.include_g_context) workspace[m.encoded_size()] = g;
    return finite(g);
}

// HG is defined against solid angle, with positive g favoring mu=+1.
template <typename Scalar>
PHASEFLOW_HD inline Scalar hg_log_pdf(Scalar mu, Scalar g) {
    const Scalar a = Scalar(1) - g;
    const Scalar b = Scalar(1) + g;
    const Scalar squared = g >= Scalar(0)
        ? a * a + Scalar(2) * g * (Scalar(1) - mu)
        : b * b - Scalar(2) * g * (Scalar(1) + mu);
    return ::log1p(-g) + ::log1p(g) - ::log(Scalar(4) * Scalar(pi))
        - Scalar(1.5) * ::log(squared);
}
template <typename Scalar>
PHASEFLOW_HD inline Scalar hg_cdf(Scalar mu, Scalar g) {
    if (mu <= Scalar(-1)) return Scalar(0);
    if (mu >= Scalar(1)) return Scalar(1);
    const Scalar a = Scalar(1) - g;
    const Scalar b = Scalar(1) + g;
    const Scalar squared = g >= Scalar(0)
        ? a * a + Scalar(2) * g * (Scalar(1) - mu)
        : b * b - Scalar(2) * g * (Scalar(1) + mu);
    const Scalar s = ::sqrt(squared);
    const Scalar lower = a * (Scalar(1) + mu) / (s * (b + s));
    const Scalar upper = b * (Scalar(1) - mu) / (s * (a + s));
    return clamp(lower <= Scalar(0.5) ? lower : Scalar(1) - upper, Scalar(0), Scalar(1));
}
template <typename Scalar>
PHASEFLOW_HD inline Scalar hg_icdf(Scalar u, Scalar g) {
    if (u <= Scalar(0)) return Scalar(-1);
    if (u >= Scalar(1)) return Scalar(1);
    const Scalar a = Scalar(1) - g;
    const Scalar b = Scalar(1) + g;
    const Scalar v = Scalar(1) - u;
    const Scalar denominator = a * v + b * u;
    const Scalar bd = b / denominator;
    const Scalar ad = a / denominator;
    const Scalar one_plus_mu = Scalar(2) * u * bd * bd * (a * v + u);
    const Scalar one_minus_mu = Scalar(2) * v * ad * ad * (v + b * u);
    return one_plus_mu <= Scalar(1) ? one_plus_mu - Scalar(1) : Scalar(1) - one_minus_mu;
}

template <typename Scalar>
PHASEFLOW_HD inline ScalarTransform<Scalar> rqs(
    Scalar input, const Scalar* logits, const Config& config, bool inverse,
    Scalar azimuth_strength = Scalar(1)
) {
    const std::uint32_t bins = config.num_bins;
    Scalar maximum_w = logits[0], maximum_h = logits[bins];
    for (std::uint32_t k = 1; k < bins; ++k) {
        if (logits[k] > maximum_w) maximum_w = logits[k];
        if (logits[bins + k] > maximum_h) maximum_h = logits[bins + k];
    }
    Scalar sum_w = 0, sum_h = 0;
    for (std::uint32_t k = 0; k < bins; ++k) {
        sum_w += ::exp(logits[k] - maximum_w);
        sum_h += ::exp(logits[bins + k] - maximum_h);
    }
    // At axial incidence azimuth updates become identity. Blend constrained
    // positive masses and slopes, never the logits or the transformed values.
    const Scalar identity_fraction = Scalar(1) - azimuth_strength;
    const Scalar width_floor = azimuth_strength * Scalar(config.min_bin_width)
        + identity_fraction / Scalar(bins);
    const Scalar height_floor = azimuth_strength * Scalar(config.min_bin_height)
        + identity_fraction / Scalar(bins);
    const Scalar width_factor = azimuth_strength
        * (Scalar(1) - Scalar(bins) * Scalar(config.min_bin_width));
    const Scalar height_factor = azimuth_strength
        * (Scalar(1) - Scalar(bins) * Scalar(config.min_bin_height));
    const Scalar maximum_width_mass = width_floor + width_factor / sum_w;
    const Scalar maximum_height_mass = height_floor + height_factor / sum_h;
    Scalar width_mass_sum = 0, height_mass_sum = 0;
    for (std::uint32_t k = 0; k < bins; ++k) {
        width_mass_sum += (width_floor + width_factor * ::exp(logits[k] - maximum_w) / sum_w)
            / maximum_width_mass;
        height_mass_sum += (height_floor + height_factor * ::exp(logits[bins + k] - maximum_h) / sum_h)
            / maximum_height_mass;
    }
    Scalar x0 = 0, y0 = 0, x1 = 1, y1 = 1;
    std::uint32_t bin = 0;
    for (; bin < bins; ++bin) {
        x1 = bin + 1 == bins ? Scalar(1) : x0
            + ((width_floor + width_factor * ::exp(logits[bin] - maximum_w) / sum_w)
                / maximum_width_mass) / width_mass_sum;
        y1 = bin + 1 == bins ? Scalar(1) : y0
            + ((height_floor + height_factor * ::exp(logits[bins + bin] - maximum_h) / sum_h)
                / maximum_height_mass) / height_mass_sum;
        if (input < (inverse ? y1 : x1) || bin + 1 == bins) break;
        x0 = x1;
        y0 = y1;
    }
    const Scalar width = x1 - x0, height = y1 - y0;
    Scalar slope = height / width;
    Scalar d0 = azimuth_strength
        * (Scalar(config.min_derivative) + softplus(logits[2 * bins + bin])) + identity_fraction;
    Scalar d1 = azimuth_strength
        * (Scalar(config.min_derivative) + softplus(logits[2 * bins + bin + 1])) + identity_fraction;
    const Scalar derivative_maximum = d0 > d1 ? d0 : d1;
    const Scalar slope_scale = slope > derivative_maximum ? slope : derivative_maximum;
    slope /= slope_scale;
    d0 /= slope_scale;
    d1 /= slope_scale;
    Scalar t;
    if (inverse) {
        const Scalar p = clamp((input - y0) / height, Scalar(0), Scalar(1));
        const Scalar b = d0 * (Scalar(1) - p) + p * (Scalar(2) * slope - d1);
        const Scalar a = slope - b;
        const Scalar root = ::hypot(
            d0 * (Scalar(1) - p) - d1 * p,
            Scalar(2) * slope * ::sqrt(p * (Scalar(1) - p))
        );
        // b<0 implies a>0. The b>=0 branch also handles a=0 exactly.
        t = b >= Scalar(0) ? Scalar(2) * slope * p / (b + root)
            : (root - b) / (Scalar(2) * a);
        t = clamp(t, Scalar(0), Scalar(1));
    } else t = clamp((input - x0) / width, Scalar(0), Scalar(1));
    const Scalar omt = Scalar(1) - t;
    const Scalar denominator = slope + ((d0 - slope) + (d1 - slope)) * t * omt;
    const Scalar numerator_derivative = d1 * t * t + Scalar(2) * slope * t * omt + d0 * omt * omt;
    const Scalar log_derivative = ::log(slope_scale) + Scalar(2) * ::log(slope)
        + ::log(numerator_derivative) - Scalar(2) * ::log(denominator);
    Scalar output;
    if (inverse) output = x0 + t * width;
    else {
        const Scalar fraction = t * (slope * t + d0 * omt) / denominator;
        const Scalar remaining = omt * (slope * omt + d1 * t) / denominator;
        output = fraction <= Scalar(0.5) ? y0 + height * fraction
            : y0 + height - height * remaining;
    }
    if (input == Scalar(0) || input == Scalar(1)) output = input;
    return {output, inverse ? -log_derivative : log_derivative};
}

template <typename Scalar>
PHASEFLOW_HD inline Scalar square_transform(
    const ModelView& m, Scalar point[2], Scalar* workspace, bool inverse
) {
    Scalar* a = workspace + m.context_size();
    Scalar* b = a + m.workspace_width;
    Scalar log_det = 0;
    const Scalar incident_cosine = Scalar(2) * workspace[1] - Scalar(1);
    const Scalar axial_strength = ::sqrt(
        (Scalar(1) - incident_cosine) * (Scalar(1) + incident_cosine)
    );
    for (std::uint32_t step = 0; step < m.coupling_count; ++step) {
        const auto index = inverse ? m.coupling_count - 1 - step : step;
        const Coupling layer = m.couplings[index];
        a[0] = layer.retained_index == 1
            ? Scalar(0.5) + axial_strength * (point[1] - Scalar(0.5)) : point[0];
        for (std::uint32_t j = 0; j < m.context_size(); ++j) a[j + 1] = workspace[j];
        const Scalar* logits = network_forward(m, layer.network_index, a, a, b);
        const std::uint32_t changed = 1U - layer.retained_index;
        const auto transformed = rqs(point[changed], logits, m.config, inverse,
            layer.retained_index == 0 ? axial_strength : Scalar(1));
        point[changed] = transformed.value;
        log_det += transformed.log_det;
    }
    return log_det;
}

template <typename Scalar>
PHASEFLOW_HD inline Evaluation<Scalar> evaluate_local(
    const ModelView& m, Scalar wavelength_nm, Scalar incident_cosine,
    Vec3<Scalar> direction, Scalar* workspace
) {
    Scalar g;
    if (!prepare_context(m, wavelength_nm, incident_cosine, workspace, g)
        || !finite(direction.x) || !finite(direction.y) || !finite(direction.z))
        return invalid_evaluation<Scalar>();
    const Scalar norm = ::sqrt(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
    if (!finite(norm) || norm == Scalar(0) || ::fabs(norm - Scalar(1)) > Scalar(2e-5))
        return invalid_evaluation<Scalar>();
    const Scalar mu = clamp(direction.z / norm, Scalar(-1), Scalar(1));
    const Scalar phi = direction.x == Scalar(0) && direction.y == Scalar(0)
        ? Scalar(0) : ::atan2(direction.y / norm, direction.x / norm);
    Scalar point[2] = {hg_cdf(mu, g), ::fabs(phi) / Scalar(pi)};
    const Scalar log_square = square_transform(m, point, workspace, false);
    const Scalar log_pdf = hg_log_pdf(mu, g) + log_square;
    const Scalar pdf = ::exp(log_pdf);
    return {log_pdf, pdf, g, finite(log_pdf) && finite(pdf)};
}

template <typename Scalar>
PHASEFLOW_HD inline Sample<Scalar> sample_local(
    const ModelView& m, Scalar wavelength_nm, Scalar incident_cosine,
    Scalar u0, Scalar u1, Scalar* workspace
) {
    Scalar g;
    if (!finite(u0) || !finite(u1) || u0 <= Scalar(0) || u0 >= Scalar(1)
        || u1 < Scalar(0) || u1 >= Scalar(1)
        || !prepare_context(m, wavelength_nm, incident_cosine, workspace, g))
        return {{Scalar(0), Scalar(0), Scalar(0)}, invalid_evaluation<Scalar>()};
    const Scalar sign = u1 < Scalar(0.5) ? Scalar(-1) : Scalar(1);
    Scalar point[2] = {u0, u1 < Scalar(0.5) ? Scalar(2) * u1 : Scalar(2) * u1 - Scalar(1)};
    const Scalar inverse_log_det = square_transform(m, point, workspace, true);
    // An interior uniform and a strictly monotone spline cannot produce a
    // radial endpoint in exact arithmetic. Do not disguise rounding collapse
    // as a sample with a valid surface density, and do not retry or clamp.
    if (!finite(point[0]) || point[0] <= Scalar(0) || point[0] >= Scalar(1)
        || !finite(point[1]) || !finite(inverse_log_det))
        return {{Scalar(0), Scalar(0), Scalar(0)}, invalid_evaluation<Scalar>()};
    const Scalar mu = hg_icdf(point[0], g);
    if (!finite(mu) || mu <= Scalar(-1) || mu >= Scalar(1))
        return {{Scalar(0), Scalar(0), Scalar(0)}, invalid_evaluation<Scalar>()};
    const Scalar phi = sign * Scalar(pi) * point[1];
    const Scalar radius = ::sqrt((Scalar(1) - mu) * (Scalar(1) + mu));
    const Vec3<Scalar> direction = {radius * ::cos(phi), radius * ::sin(phi), mu};
    const Scalar log_pdf = hg_log_pdf(mu, g) - inverse_log_det;
    const Scalar pdf = ::exp(log_pdf);
    return {direction, {log_pdf, pdf, g, finite(log_pdf) && finite(pdf)}};
}

} // namespace phaseflow

#undef PHASEFLOW_HD
#endif
