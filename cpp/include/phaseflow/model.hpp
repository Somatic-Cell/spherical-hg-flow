#ifndef PHASEFLOW_MODEL_HPP
#define PHASEFLOW_MODEL_HPP

#include "phaseflow_core.cuh"

#include <filesystem>
#include <span>
#include <vector>

namespace phaseflow {

// CPU reference owner and validated binary loader. Weights are stored as FP32;
// all arithmetic in these convenience methods uses double precision.
class Model {
public:
    static Model load(const std::filesystem::path& path);
    [[nodiscard]] ModelView view() const noexcept;
    [[nodiscard]] const Config& config() const noexcept { return config_; }
    [[nodiscard]] std::span<const float> parameters() const noexcept { return parameters_; }
    [[nodiscard]] std::span<const DenseLayer> dense_layers() const noexcept { return dense_layers_; }
    [[nodiscard]] std::span<const Network> networks() const noexcept { return networks_; }
    [[nodiscard]] std::span<const Coupling> couplings() const noexcept { return couplings_; }
    [[nodiscard]] std::vector<double> encode(double wavelength_nm, double incident_cosine) const;
    [[nodiscard]] double hg_g(double wavelength_nm, double incident_cosine) const;
    [[nodiscard]] Evaluation<double> evaluate(double wavelength_nm, double incident_cosine, Vec3<double> direction) const;
    [[nodiscard]] Sample<double> sample(double wavelength_nm, double incident_cosine, double u0, double u1) const;

private:
    Config config_{};
    std::vector<float> parameters_;
    std::vector<DenseLayer> dense_layers_;
    std::vector<Network> networks_;
    std::vector<Coupling> couplings_;
    std::uint32_t workspace_width_{};
};

} // namespace phaseflow
#endif
