#include "phaseflow/model.hpp"

#include <bit>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>

namespace phaseflow {
namespace {
constexpr std::uint64_t max_file_bytes = 512ULL * 1024 * 1024;
constexpr std::uint32_t max_width = 65536, max_networks = 4096, max_dense_layers = 16384;
constexpr std::uint32_t max_bins = 4096;

[[noreturn]] void invalid(const std::string& message) {
    throw std::runtime_error("Invalid phaseflow model: " + message);
}

class Reader {
public:
    explicit Reader(const std::vector<unsigned char>& bytes) : bytes_(bytes) {}
    std::uint32_t u32() {
        need(4);
        std::uint32_t value = 0;
        for (unsigned i = 0; i < 4; ++i) value |= std::uint32_t(bytes_[position_++]) << (8 * i);
        return value;
    }
    std::uint64_t u64() {
        need(8);
        std::uint64_t value = 0;
        for (unsigned i = 0; i < 8; ++i) value |= std::uint64_t(bytes_[position_++]) << (8 * i);
        return value;
    }
    float f32() { return std::bit_cast<float>(u32()); }
    double f64() { return std::bit_cast<double>(u64()); }
    void magic() {
        need(8);
        constexpr char expected[] = "PHFLOW01";
        for (unsigned i = 0; i < 8; ++i)
            if (bytes_[position_++] != static_cast<unsigned char>(expected[i])) invalid("magic");
    }
    std::size_t position() const { return position_; }
private:
    void need(std::size_t count) const {
        if (count > bytes_.size() - position_) invalid("truncated file");
    }
    const std::vector<unsigned char>& bytes_;
    std::size_t position_ = 0;
};
} // namespace

Model Model::load(const std::filesystem::path& path) {
    static_assert(sizeof(float) == 4 && sizeof(double) == 8);
    static_assert(std::numeric_limits<float>::is_iec559 && std::numeric_limits<double>::is_iec559);
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Cannot open phaseflow model: " + path.string());
    const auto end = input.tellg();
    if (end < 0 || static_cast<std::uint64_t>(end) < 120
        || static_cast<std::uint64_t>(end) > max_file_bytes) invalid("file size");
    std::vector<unsigned char> bytes(static_cast<std::size_t>(end));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input) invalid("read failed");
    Reader reader(bytes);
    reader.magic();
    if (reader.u32() != 1) invalid("unsupported version");
    if (reader.u32() != 0x01020304U) invalid("byte-order sentinel");
    if (reader.u64() != bytes.size()) invalid("declared size differs from actual size");
    const std::uint64_t parameter_count = reader.u64();
    const auto dense_count = reader.u32(), network_count = reader.u32(), coupling_count = reader.u32();
    Model model;
    model.config_.one_blob_bins = reader.u32();
    model.config_.num_bins = reader.u32();
    model.config_.include_g_context = reader.u32();
    if (reader.u32() != 1) invalid("only ReLU activation is supported in version 1");
    model.workspace_width_ = reader.u32();
    if (reader.u32() != 0 || reader.u32() != 0) invalid("reserved header field");
    auto& config = model.config_;
    config.wavelength_min_nm = reader.f64();
    config.wavelength_max_nm = reader.f64();
    config.g_limit = reader.f64();
    config.min_bin_width = reader.f64();
    config.min_bin_height = reader.f64();
    config.min_derivative = reader.f64();
    if (!std::isfinite(config.wavelength_min_nm) || !std::isfinite(config.wavelength_max_nm)
        || !std::isfinite(config.g_limit) || !std::isfinite(config.min_bin_width)
        || !std::isfinite(config.min_bin_height) || !std::isfinite(config.min_derivative))
        invalid("nonfinite configuration");
    if (!(config.wavelength_min_nm > 0 && config.wavelength_max_nm > config.wavelength_min_nm))
        invalid("wavelength bounds");
    if (!(config.g_limit > 0 && config.g_limit < 1)) invalid("g_limit");
    if (config.one_blob_bins > max_bins || config.num_bins == 0 || config.num_bins > max_bins
        || config.include_g_context > 1) invalid("encoding or spline dimensions");
    if (!(config.min_bin_width > 0 && config.num_bins * config.min_bin_width < 1
        && config.min_bin_height > 0 && config.num_bins * config.min_bin_height < 1
        && config.min_derivative > 0)) invalid("spline minima");
    if (network_count == 0 || network_count > max_networks || coupling_count >= max_networks
        || network_count != coupling_count + 1 || dense_count < network_count
        || dense_count > max_dense_layers || model.workspace_width_ == 0
        || model.workspace_width_ > max_width || parameter_count > max_file_bytes / 4)
        invalid("descriptor counts");
    const std::uint64_t expected_size = 120ULL + 16ULL * network_count
        + 24ULL * dense_count + 8ULL * coupling_count + 4ULL * parameter_count;
    if (expected_size != bytes.size()) invalid("descriptor and parameter sizes");
    model.networks_.reserve(network_count);
    model.dense_layers_.reserve(dense_count);
    model.couplings_.reserve(coupling_count);
    model.parameters_.reserve(static_cast<std::size_t>(parameter_count));
    for (std::uint32_t i = 0; i < network_count; ++i)
        model.networks_.push_back({reader.u32(), reader.u32(), reader.u32(), reader.u32()});
    for (std::uint32_t i = 0; i < dense_count; ++i)
        model.dense_layers_.push_back({reader.u32(), reader.u32(), reader.u64(), reader.u64()});
    for (std::uint32_t i = 0; i < coupling_count; ++i)
        model.couplings_.push_back({reader.u32(), reader.u32()});
    const auto encoded_size = 2U + 2U * config.one_blob_bins;
    const auto context_size = encoded_size + config.include_g_context;
    std::uint32_t actual_width = encoded_size > context_size + 1 ? encoded_size : context_size + 1;
    std::uint32_t layer_cursor = 0;
    std::uint64_t parameter_cursor = 0;
    for (std::uint32_t index = 0; index < network_count; ++index) {
        const auto& network = model.networks_[index];
        const auto expected_input = index == 0 ? encoded_size : context_size + 1;
        const auto expected_output = index == 0 ? 1U : 3U * config.num_bins + 1;
        if (network.first_layer != layer_cursor || network.num_layers == 0
            || network.num_layers > dense_count - layer_cursor
            || network.input_size != expected_input || network.output_size != expected_output)
            invalid("network descriptor");
        auto width = expected_input;
        for (std::uint32_t j = 0; j < network.num_layers; ++j) {
            const auto& layer = model.dense_layers_[layer_cursor++];
            if (layer.input_size != width || layer.input_size == 0 || layer.input_size > max_width
                || layer.output_size == 0 || layer.output_size > max_width) invalid("dense dimensions");
            const auto weights = std::uint64_t(layer.input_size) * layer.output_size;
            if (layer.weight_offset != parameter_cursor || weights > parameter_count - parameter_cursor)
                invalid("weight offset");
            parameter_cursor += weights;
            if (layer.bias_offset != parameter_cursor || layer.output_size > parameter_count - parameter_cursor)
                invalid("bias offset");
            parameter_cursor += layer.output_size;
            if (layer.input_size > actual_width) actual_width = layer.input_size;
            if (layer.output_size > actual_width) actual_width = layer.output_size;
            width = layer.output_size;
        }
        if (width != expected_output) invalid("network output dimension");
    }
    if (layer_cursor != dense_count || parameter_cursor != parameter_count
        || actual_width != model.workspace_width_) invalid("noncanonical packing or workspace size");
    for (std::uint32_t i = 0; i < coupling_count; ++i)
        if (model.couplings_[i].network_index != i + 1 || model.couplings_[i].retained_index > 1)
            invalid("coupling descriptor");
    for (std::uint64_t i = 0; i < parameter_count; ++i) {
        const float value = reader.f32();
        if (!std::isfinite(value)) invalid("nonfinite parameter");
        model.parameters_.push_back(value);
    }
    if (reader.position() != bytes.size()) invalid("trailing bytes");
    return model;
}

ModelView Model::view() const noexcept {
    return {config_, parameters_.data(), dense_layers_.data(), networks_.data(), couplings_.data(),
        static_cast<std::uint32_t>(couplings_.size()), workspace_width_};
}

std::vector<double> Model::encode(double wavelength_nm, double incident_cosine) const {
    const auto m = view();
    std::vector<double> features(m.encoded_size());
    if (!encode_conditions(m, wavelength_nm, incident_cosine, features.data()))
        throw std::invalid_argument("Conditions must be finite and inside the trained domain");
    return features;
}

double Model::hg_g(double wavelength_nm, double incident_cosine) const {
    const auto m = view();
    std::vector<double> workspace(m.workspace_scalars());
    double g;
    if (!prepare_context(m, wavelength_nm, incident_cosine, workspace.data(), g))
        throw std::invalid_argument("Conditions must be finite and inside the trained domain");
    return g;
}

Evaluation<double> Model::evaluate(double wavelength_nm, double incident_cosine, Vec3<double> direction) const {
    const auto m = view();
    std::vector<double> workspace(m.workspace_scalars());
    return evaluate_local(m, wavelength_nm, incident_cosine, direction, workspace.data());
}

Sample<double> Model::sample(double wavelength_nm, double incident_cosine, double u0, double u1) const {
    const auto m = view();
    std::vector<double> workspace(m.workspace_scalars());
    return sample_local(m, wavelength_nm, incident_cosine, u0, u1, workspace.data());
}

} // namespace phaseflow
