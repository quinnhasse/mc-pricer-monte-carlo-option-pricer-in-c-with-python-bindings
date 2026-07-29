#pragma once

#include <cmath>
#include <random>
#include <vector>

namespace mc {

/// Parameters shared across all GBM-based pricers.
struct GBMParams {
    double S0;    ///< Initial spot price
    double K;     ///< Strike
    double r;     ///< Risk-free rate (continuously compounded)
    double q;     ///< Continuous dividend yield
    double sigma; ///< Volatility
    double T;     ///< Time to expiry (years)
    int    steps; ///< Number of time steps per path
};

/// Generates a single GBM price path.
///
/// Uses the exact log-normal update:
///   S(t+dt) = S(t) * exp((r - q - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
/// where Z ~ N(0,1).
///
/// @param params  GBM parameters
/// @param rng     Seeded Mersenne Twister engine
/// @param normal  Standard normal distribution
/// @return        Vector of length (steps+1) — index 0 is S0
inline std::vector<double> generate_path(
    const GBMParams& params,
    std::mt19937_64& rng,
    std::normal_distribution<double>& normal)
{
    const double dt     = params.T / params.steps;
    const double drift  = (params.r - params.q - 0.5 * params.sigma * params.sigma) * dt;
    const double vol_dt = params.sigma * std::sqrt(dt);

    std::vector<double> path(params.steps + 1);
    path[0] = params.S0;
    for (int i = 1; i <= params.steps; ++i) {
        path[i] = path[i - 1] * std::exp(drift + vol_dt * normal(rng));
    }
    return path;
}

/// Generates a pair of antithetic GBM paths.
///
/// The second path uses negated normals, which reduces variance roughly
/// by half for monotone payoffs.
///
/// @return  Pair {path, antithetic_path}, each of length (steps+1)
inline std::pair<std::vector<double>, std::vector<double>>
generate_antithetic_paths(
    const GBMParams& params,
    std::mt19937_64& rng,
    std::normal_distribution<double>& normal)
{
    const double dt     = params.T / params.steps;
    const double drift  = (params.r - params.q - 0.5 * params.sigma * params.sigma) * dt;
    const double vol_dt = params.sigma * std::sqrt(dt);

    std::vector<double> path(params.steps + 1);
    std::vector<double> anti(params.steps + 1);
    path[0] = anti[0] = params.S0;

    for (int i = 1; i <= params.steps; ++i) {
        const double z = normal(rng);
        path[i] = path[i - 1] * std::exp(drift + vol_dt * z);
        anti[i] = anti[i - 1] * std::exp(drift - vol_dt * z);
    }
    return {path, anti};
}

} // namespace mc
