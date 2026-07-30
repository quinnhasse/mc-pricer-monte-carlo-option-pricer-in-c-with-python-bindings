#pragma once

#include <cmath>
#include <stdexcept>

namespace mc {

/// Parameters for the Black-Scholes closed-form formulas.
struct BSParams {
    double S;     ///< Current spot price
    double K;     ///< Strike price
    double r;     ///< Risk-free rate (continuous compounding)
    double q;     ///< Continuous dividend yield
    double sigma; ///< Implied/flat volatility
    double T;     ///< Time to expiry (years)
};

namespace detail {

/// Standard normal CDF via complementary error function.
inline double norm_cdf(double x) noexcept {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

/// Standard normal PDF.
inline double norm_pdf(double x) noexcept {
    constexpr double inv_sqrt_2pi = 0.3989422804014327;
    return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

} // namespace detail

/// Compute d1 for the Black-Scholes formula.
inline double bs_d1(const BSParams& p) {
    return (std::log(p.S / p.K) + (p.r - p.q + 0.5 * p.sigma * p.sigma) * p.T)
           / (p.sigma * std::sqrt(p.T));
}

/// Compute d2 for the Black-Scholes formula.
inline double bs_d2(const BSParams& p) {
    return bs_d1(p) - p.sigma * std::sqrt(p.T);
}

/// Black-Scholes European call price.
///
/// C = S * exp(-q*T) * N(d1) - K * exp(-r*T) * N(d2)
inline double bs_call(const BSParams& p) {
    const double d1 = bs_d1(p);
    const double d2 = d1 - p.sigma * std::sqrt(p.T);
    return p.S * std::exp(-p.q * p.T) * detail::norm_cdf(d1)
         - p.K * std::exp(-p.r * p.T) * detail::norm_cdf(d2);
}

/// Black-Scholes European put price.
///
/// P = K * exp(-r*T) * N(-d2) - S * exp(-q*T) * N(-d1)
inline double bs_put(const BSParams& p) {
    const double d1 = bs_d1(p);
    const double d2 = d1 - p.sigma * std::sqrt(p.T);
    return p.K * std::exp(-p.r * p.T) * detail::norm_cdf(-d2)
         - p.S * std::exp(-p.q * p.T) * detail::norm_cdf(-d1);
}

/// European call delta: dC/dS
inline double bs_call_delta(const BSParams& p) {
    return std::exp(-p.q * p.T) * detail::norm_cdf(bs_d1(p));
}

/// European put delta: dP/dS
inline double bs_put_delta(const BSParams& p) {
    return -std::exp(-p.q * p.T) * detail::norm_cdf(-bs_d1(p));
}

/// Vega (same for calls and puts): dV/dsigma
inline double bs_vega(const BSParams& p) {
    return p.S * std::exp(-p.q * p.T) * detail::norm_pdf(bs_d1(p)) * std::sqrt(p.T);
}

} // namespace mc
