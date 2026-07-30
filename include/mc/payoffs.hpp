#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace mc {

// ---------------------------------------------------------------------------
// European payoffs
// ---------------------------------------------------------------------------

/// European call payoff: max(S_T - K, 0)
struct EuropeanCall {
    double K;
    double operator()(const std::vector<double>& path) const noexcept {
        return std::max(path.back() - K, 0.0);
    }
};

/// European put payoff: max(K - S_T, 0)
struct EuropeanPut {
    double K;
    double operator()(const std::vector<double>& path) const noexcept {
        return std::max(K - path.back(), 0.0);
    }
};

// ---------------------------------------------------------------------------
// Barrier option types
// ---------------------------------------------------------------------------

enum class BarrierType {
    UpAndOut,    ///< Knocked out if S ever crosses H from below
    UpAndIn,     ///< Active only if S ever crosses H from below
    DownAndOut,  ///< Knocked out if S ever crosses H from above
    DownAndIn,   ///< Active only if S ever crosses H from above
};

enum class OptionRight { Call, Put };

/// Barrier option payoff (discrete monitoring).
///
/// Monitors the barrier at each path step. Returns the vanilla payoff
/// if the barrier condition is met; zero otherwise.
struct BarrierOption {
    double      K;
    double      H;          ///< Barrier level
    BarrierType barrier;
    OptionRight right;

    double operator()(const std::vector<double>& path) const noexcept {
        bool triggered = false;
        for (double s : path) {
            if (barrier == BarrierType::UpAndOut || barrier == BarrierType::UpAndIn) {
                if (s >= H) { triggered = true; break; }
            } else {
                if (s <= H) { triggered = true; break; }
            }
        }

        bool alive = false;
        if (barrier == BarrierType::UpAndOut || barrier == BarrierType::DownAndOut) {
            alive = !triggered;
        } else {
            alive = triggered;
        }

        if (!alive) return 0.0;

        const double ST = path.back();
        if (right == OptionRight::Call) return std::max(ST - K, 0.0);
        return std::max(K - ST, 0.0);
    }
};

// ---------------------------------------------------------------------------
// Asian payoffs
// ---------------------------------------------------------------------------

/// Arithmetic-average Asian call: max(A - K, 0)
struct ArithmeticAsianCall {
    double K;
    int    skip_first{1}; ///< Skip the initial S0 when computing the average

    double operator()(const std::vector<double>& path) const noexcept {
        if (path.size() <= static_cast<std::size_t>(skip_first)) return 0.0;
        double sum = 0.0;
        const int n = static_cast<int>(path.size()) - skip_first;
        for (int i = skip_first; i < static_cast<int>(path.size()); ++i)
            sum += path[i];
        const double avg = sum / n;
        return std::max(avg - K, 0.0);
    }
};

/// Arithmetic-average Asian put: max(K - A, 0)
struct ArithmeticAsianPut {
    double K;
    int    skip_first{1};

    double operator()(const std::vector<double>& path) const noexcept {
        if (path.size() <= static_cast<std::size_t>(skip_first)) return 0.0;
        double sum = 0.0;
        const int n = static_cast<int>(path.size()) - skip_first;
        for (int i = skip_first; i < static_cast<int>(path.size()); ++i)
            sum += path[i];
        const double avg = sum / n;
        return std::max(K - avg, 0.0);
    }
};

/// Geometric-average Asian call: max(G - K, 0)
///
/// Has a Black-Scholes-style closed form, making it a useful
/// control variate for arithmetic Asian pricing.
struct GeometricAsianCall {
    double K;
    int    skip_first{1};

    double operator()(const std::vector<double>& path) const noexcept {
        if (path.size() <= static_cast<std::size_t>(skip_first)) return 0.0;
        const int n = static_cast<int>(path.size()) - skip_first;
        double log_sum = 0.0;
        for (int i = skip_first; i < static_cast<int>(path.size()); ++i)
            log_sum += std::log(path[i]);
        const double geo = std::exp(log_sum / n);
        return std::max(geo - K, 0.0);
    }
};

/// Geometric-average Asian put: max(K - G, 0)
struct GeometricAsianPut {
    double K;
    int    skip_first{1};

    double operator()(const std::vector<double>& path) const noexcept {
        if (path.size() <= static_cast<std::size_t>(skip_first)) return 0.0;
        const int n = static_cast<int>(path.size()) - skip_first;
        double log_sum = 0.0;
        for (int i = skip_first; i < static_cast<int>(path.size()); ++i)
            log_sum += std::log(path[i]);
        const double geo = std::exp(log_sum / n);
        return std::max(K - geo, 0.0);
    }
};

} // namespace mc
