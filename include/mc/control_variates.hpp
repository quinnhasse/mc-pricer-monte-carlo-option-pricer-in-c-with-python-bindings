#pragma once

#include "mc/gbm.hpp"
#include "mc/payoffs.hpp"
#include "mc/black_scholes.hpp"

#include <cmath>
#include <random>
#include <vector>

#ifdef MC_OPENMP_ENABLED
#  include <omp.h>
#endif

namespace mc {

/// Result of a control-variate MC run.
struct CVPriceResult {
    double price;       ///< CV-adjusted option price
    double stderr_est;  ///< Estimated standard error after adjustment
    double beta;        ///< Optimal beta coefficient used
    long   paths;
};

/// Prices an arithmetic-average Asian option using the geometric average
/// as a control variate.
///
/// The geometric Asian has a known closed-form (Rogers & Shi, 1995):
///   sigma_g = sigma * sqrt((2*n+1) / (6*(n+1)))  (with n time steps)
///   r_g     = 0.5 * (r - q - 0.5*sigma^2 + r_g_adj)
///
/// A pilot run of `pilot_paths` paths estimates the optimal beta;
/// the main run of `n_paths` applies the adjustment.
///
/// @param params     GBM parameters
/// @param right      OptionRight::Call or OptionRight::Put
/// @param n_paths    Main simulation path count
/// @param pilot_paths  Paths for beta estimation (default 10000)
/// @param antithetic   Use antithetic variates within each run
/// @param seed       RNG master seed
CVPriceResult price_asian_cv(
    const GBMParams& params,
    OptionRight      right,
    long             n_paths,
    long             pilot_paths = 10000,
    bool             antithetic  = true,
    uint64_t         seed        = 42);

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

/// Closed-form price for a geometric-average Asian call or put.
///
/// Derived from the fact that the geometric average of a GBM is itself
/// log-normal. Adjusted vol and rate follow from the moment-matching
/// formula for the product of correlated log-normals.
inline double geometric_asian_closed_form(
    const GBMParams& p,
    OptionRight       right)
{
    const int n    = p.steps;
    // Adjusted sigma and rate for the geometric average
    const double sigma_g = p.sigma * std::sqrt((2.0 * n + 1.0) / (6.0 * (n + 1.0)));
    const double mu      = p.r - p.q - 0.5 * p.sigma * p.sigma;
    const double r_g     = 0.5 * (mu + p.sigma * sigma_g * (2.0 * n + 1.0) / (2.0 * n));
    // Re-use Black-Scholes on the adjusted vol/rate
    BSParams bsp;
    bsp.S = p.S0;
    bsp.K = p.K;
    bsp.r = r_g;
    bsp.q = p.q;
    bsp.sigma = sigma_g;
    bsp.T = p.T;
    if (right == OptionRight::Call) return bs_call(bsp);
    return bs_put(bsp);
}

inline CVPriceResult price_asian_cv(
    const GBMParams& params,
    OptionRight      right,
    long             n_paths,
    long             pilot_paths,
    bool             antithetic,
    uint64_t         seed)
{
    const double discount    = std::exp(-params.r * params.T);
    const double geo_cf_price = geometric_asian_closed_form(params, right);

    ArithmeticAsianCall arith_call{params.K};
    ArithmeticAsianPut  arith_put {params.K};
    GeometricAsianCall  geo_call  {params.K};
    GeometricAsianPut   geo_put   {params.K};

    auto arith_payoff = [&](const std::vector<double>& path) -> double {
        return right == OptionRight::Call ? arith_call(path) : arith_put(path);
    };
    auto geo_payoff = [&](const std::vector<double>& path) -> double {
        return right == OptionRight::Call ? geo_call(path) : geo_put(path);
    };

    // --- Pilot run to estimate optimal beta ---
    std::mt19937_64 rng_pilot(seed ^ 0xDEADBEEFULL);
    std::normal_distribution<double> normal;

    std::vector<double> y_pilot(pilot_paths), c_pilot(pilot_paths);
    for (long i = 0; i < pilot_paths; ++i) {
        if (antithetic) {
            auto [path, anti] = generate_antithetic_paths(params, rng_pilot, normal);
            y_pilot[i] = discount * 0.5 * (arith_payoff(path) + arith_payoff(anti));
            c_pilot[i] = discount * 0.5 * (geo_payoff(path)   + geo_payoff(anti));
        } else {
            auto path = generate_path(params, rng_pilot, normal);
            y_pilot[i] = discount * arith_payoff(path);
            c_pilot[i] = discount * geo_payoff(path);
        }
    }

    // beta = Cov(Y, C) / Var(C)
    double mean_y = 0.0, mean_c = 0.0;
    for (long i = 0; i < pilot_paths; ++i) {
        mean_y += y_pilot[i];
        mean_c += c_pilot[i];
    }
    mean_y /= pilot_paths;
    mean_c /= pilot_paths;

    double cov_yc = 0.0, var_c = 0.0;
    for (long i = 0; i < pilot_paths; ++i) {
        cov_yc += (y_pilot[i] - mean_y) * (c_pilot[i] - mean_c);
        var_c  += (c_pilot[i] - mean_c) * (c_pilot[i] - mean_c);
    }
    const double beta = (var_c > 1e-15) ? cov_yc / var_c : 1.0;

    // --- Main run with CV adjustment ---
    double sum = 0.0, sum2 = 0.0;

    std::mt19937_64 rng_main(seed);
    for (long i = 0; i < n_paths; ++i) {
        double y = 0.0, c = 0.0;
        if (antithetic) {
            auto [path, anti] = generate_antithetic_paths(params, rng_main, normal);
            y = discount * 0.5 * (arith_payoff(path) + arith_payoff(anti));
            c = discount * 0.5 * (geo_payoff(path)   + geo_payoff(anti));
        } else {
            auto path = generate_path(params, rng_main, normal);
            y = discount * arith_payoff(path);
            c = discount * geo_payoff(path);
        }
        const double z = y - beta * (c - geo_cf_price);
        sum  += z;
        sum2 += z * z;
    }

    const double mean   = sum / n_paths;
    const double var    = (sum2 / n_paths) - mean * mean;
    const double stderr = std::sqrt(std::max(var, 0.0) / n_paths);

    return {mean, stderr, beta, n_paths};
}

} // namespace mc
