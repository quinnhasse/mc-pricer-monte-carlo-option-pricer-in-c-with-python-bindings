#pragma once

#include "mc/gbm.hpp"
#include "mc/payoffs.hpp"

#include <cmath>
#include <functional>
#include <random>
#include <vector>

#ifdef MC_OPENMP_ENABLED
#  include <omp.h>
#endif

namespace mc {

/// Result returned by MCEngine::price().
struct PriceResult {
    double price;       ///< Discounted expected payoff (option price)
    double stderr_est;  ///< Monte Carlo standard error
    long   paths;       ///< Number of paths used
};

/// General-purpose Monte Carlo pricer for GBM-driven payoffs.
///
/// Supports antithetic variates (halves variance for monotone payoffs).
/// When compiled with MC_OPENMP_ENABLED, path generation is parallelised
/// across threads using a per-thread RNG seeded from a master seed.
///
/// Usage:
/// @code
///   mc::MCEngine engine;
///   mc::EuropeanCall payoff{100.0};
///   mc::GBMParams p{100, 100, 0.05, 0.0, 0.2, 1.0, 50};
///   auto result = engine.price(payoff, p, 1'000'000, true, 42);
/// @endcode
class MCEngine {
public:
    /// Price an option via Monte Carlo.
    ///
    /// @param payoff      Callable: f(path) -> double. Any of the types in payoffs.hpp.
    /// @param params      GBM parameters (spot, strike, rate, vol, expiry, steps).
    /// @param n_paths     Number of Monte Carlo paths (or path-pairs with antithetic).
    /// @param antithetic  Use antithetic variates variance reduction.
    /// @param seed        RNG seed (default 0 = non-deterministic per thread).
    /// @return            PriceResult with price, stderr, path count.
    template <typename Payoff>
    PriceResult price(
        const Payoff&    payoff,
        const GBMParams& params,
        long             n_paths,
        bool             antithetic = true,
        uint64_t         seed       = 42) const
    {
        const double discount = std::exp(-params.r * params.T);

        double sum  = 0.0;
        double sum2 = 0.0;

#ifdef MC_OPENMP_ENABLED
        const int n_threads = omp_get_max_threads();
        std::vector<double> partial_sum(n_threads, 0.0);
        std::vector<double> partial_sum2(n_threads, 0.0);

        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            std::mt19937_64 rng(seed + static_cast<uint64_t>(tid) * 1000007ULL);
            std::normal_distribution<double> normal;

            #pragma omp for schedule(static)
            for (long i = 0; i < n_paths; ++i) {
                double v = 0.0;
                if (antithetic) {
                    auto [path, anti] = generate_antithetic_paths(params, rng, normal);
                    v = 0.5 * (payoff(path) + payoff(anti));
                } else {
                    auto path = generate_path(params, rng, normal);
                    v = payoff(path);
                }
                const double pv = discount * v;
                partial_sum[tid]  += pv;
                partial_sum2[tid] += pv * pv;
            }
        }

        for (int t = 0; t < n_threads; ++t) {
            sum  += partial_sum[t];
            sum2 += partial_sum2[t];
        }
#else
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> normal;

        for (long i = 0; i < n_paths; ++i) {
            double v = 0.0;
            if (antithetic) {
                auto [path, anti] = generate_antithetic_paths(params, rng, normal);
                v = 0.5 * (payoff(path) + payoff(anti));
            } else {
                auto path = generate_path(params, rng, normal);
                v = payoff(path);
            }
            const double pv = discount * v;
            sum  += pv;
            sum2 += pv * pv;
        }
#endif

        const double mean   = sum / n_paths;
        const double var    = (sum2 / n_paths) - mean * mean;
        const double stderr = std::sqrt(std::max(var, 0.0) / n_paths);

        return {mean, stderr, n_paths};
    }
};

} // namespace mc
