#include "phaseflow/model.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
void finish_input(std::istringstream& row) {
    if (!row) throw std::invalid_argument("Missing or invalid numeric input");
    row >> std::ws;
    if (!row.eof()) throw std::invalid_argument("Unexpected trailing input");
}
void print_eval(const phaseflow::Evaluation<double>& result) {
    if (!result.valid) throw std::invalid_argument("Invalid query or nonfinite inference result");
    std::cout << result.log_pdf << ' ' << result.pdf << ' ' << result.g;
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: phaseflow_cli model.pflow\n"
                     "stdin: g|encode lambda_nm incident_cosine\n"
                     "       eval lambda_nm incident_cosine x y z\n"
                     "       sample lambda_nm incident_cosine u0 u1\n";
        return 2;
    }
    try {
        const auto model = phaseflow::Model::load(argv[1]);
        std::cout << std::setprecision(17);
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            std::istringstream row(line);
            std::string operation;
            double wavelength, cosine;
            if (!(row >> operation >> wavelength >> cosine))
                throw std::invalid_argument("Each input row requires operation, wavelength, and cosine");
            if (operation == "g") {
                finish_input(row);
                std::cout << model.hg_g(wavelength, cosine);
            } else if (operation == "encode") {
                finish_input(row);
                const auto features = model.encode(wavelength, cosine);
                for (std::size_t i = 0; i < features.size(); ++i)
                    std::cout << (i == 0 ? "" : " ") << features[i];
            } else if (operation == "eval") {
                phaseflow::Vec3<double> direction;
                row >> direction.x >> direction.y >> direction.z;
                finish_input(row);
                print_eval(model.evaluate(wavelength, cosine, direction));
            } else if (operation == "sample") {
                double u0, u1;
                row >> u0 >> u1;
                finish_input(row);
                const auto result = model.sample(wavelength, cosine, u0, u1);
                if (!result.evaluation.valid) throw std::invalid_argument("Invalid sample query");
                std::cout << result.direction.x << ' ' << result.direction.y << ' ' << result.direction.z << ' ';
                print_eval(result.evaluation);
            } else throw std::invalid_argument("Unknown operation: " + operation);
            std::cout << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
    return 0;
}
