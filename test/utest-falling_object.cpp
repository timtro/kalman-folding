#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <catch/catch.hpp>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>

using Eigen::Matrix;

/* Preludes:
 * In [1], Beckman introduces the static Kalman filter in a series of
 * four preludes, the fourth being an implementation of a static Kalman filter.
 * (Static meaning that model states do not vary with the independent variable)
 * Those preludes are covered in `utest-linear_least_squares.cpp`.
 *
 * In [2], Beckman generalizes to the non-static case, where the
 * model includes a control input term in addition to the drift term. Beckman's
 * exhibition centres on a textbook example from Zarchan and Musoff [3].
 *
 * This series of test cases explores Beckman's implementation. It's important
 * to keep in mind that this and previously explored implementations suffer
 * numerical stability and efficiency issues. For example, the use of
 * matrix-inversion. Better numerical hygine will be practised in another
 * iteration of the problem.
 *
 * [1] Brian Beckman, Kalman Folding-Part 1. (2016)
 * [2] Brian Beckman, Kalman Folding 2: Tracking and System Dynamics. (2016)
 * [3] Zarchan and Musoff, Fundamentals of Kalman Filtering: A Practical
 *     Approach. 4th Ed. Ch 4.
 */

std::seed_seq seed{1, 2, 3, 4, 5};

constexpr double g = 32.174;      // ft / sec^2
constexpr double dt = 0.1;        // sec
constexpr double duration = 57.5; // sec
constexpr size_t numsamples = duration / dt;
constexpr double hInit = 400'000; // ft
constexpr double vInit = -6000;   // ft / sec
constexpr double accelInit = -g;  // ft / sec^2

constexpr double radarNoiseSigma = 1E3;                                  // ft
constexpr double radarNoiseVariance = radarNoiseSigma * radarNoiseSigma; // ft^2

constexpr size_t n = 2; // number of state variables, h and dhdt.
constexpr size_t b = 1; // number of output variables, just x
constexpr size_t m = 1; // number of control variables, just acceleration.

using State = Matrix<double, n, 1>;
using Measurement = Matrix<double, b, 1>;
using Control = Matrix<double, m, 1>;
using RowState = Matrix<double, 1, n>;
using Matrix_nxn = Matrix<double, n, n>;
using Matrix_bxn = Matrix<double, b, n>;
using Matrix_bxb = Matrix<double, b, b>;
using Matrix_nxm = Matrix<double, n, m>;

using StateTimeSeries = std::array<std::pair<double, State>, numsamples>;
//                                           ^time

using MeasurementTimeSeries =
    std::array<std::pair<double, Measurement>, numsamples>;
//                       ^time

using Estimate = std::pair<State, Matrix_nxn>;
//                         ^x   , ^P

using Observation =
    std::tuple<Matrix_nxn, Matrix_nxn, Matrix_nxm, Control, RowState,
               // ^Xi       , ^Phi      , ^Gamma    , ^u     , ^A
               Measurement>;
//             ^z

const StateTimeSeries trueData = []() {
  StateTimeSeries result;
  for (unsigned k = 0; k < numsamples; ++k) {
    auto t = k * dt;
    State next;
    next << hInit + vInit * t + .5 * accelInit * t * t, vInit + accelInit * t;
    result[k] = {t, next};
  }
  return result;
}();

// Ζ (Zeta) -- Measurement noise:
const Matrix_bxb Zeta = []() {
  Matrix_bxb mat;
  mat << radarNoiseVariance;
  return mat;
}();

// P0 -- Starting covariance for x
const auto P0 = Matrix_nxn::Identity() * 9999999999999;

// Φ (Phi) -- Estimate transition matrix (AKA propagator matrix, AKA
// fundamental matrix).
const Matrix_nxn Phi = []() {
  Matrix_nxn mat;
  mat << 1.f, dt, 0.f, 1.f;
  return mat;
}();

// Propagator for the control input's contribution to the IVP:
//   x[k+1] = Φ x[k] + Γ u[k]
const Matrix_nxm Gamma = []() {
  Matrix_nxm mat;
  mat << dt * dt / 2, dt;
  return mat;
}();

// Ξ (Xi) --
const Matrix_nxn Xi = []() {
  Matrix_nxn mat;
  mat << dt * dt * dt / 3, dt * dt / 2, dt * dt / 2, dt;
  mat *= 100;
  return mat;
}();

// A -- The output matrix. This one selects position from the state vector:
//     z = A x = [ 1  0 ] [ h dh/dt ]^T
const RowState A = []() {
  RowState mat;
  mat << 1, 0;
  return mat;
}();

// Control input, gravity's pull.
const Control u = []() {
  Control mat;
  mat << -g;
  return mat;
}();

// The Kalman fold function. When folded over a recursive structure containing
// measurements, one obtains the series of estimated states.
auto kalman = [](Matrix_bxb Z) {
  return [&Z](Estimate s, Observation o) -> Estimate {
    // with…
    const auto [x, P] = s;
    const auto [Xi, Phi, Gamma, u, A, z] = o;
    const auto x2 = Phi * x + Gamma * u;
    const auto P2 = Xi + Phi * P * Phi.transpose();
    const auto D = Z + A * P2 * A.transpose();
    const auto K = P2 * A.transpose() * D.inverse();
    return {x2 + K * (z - A * x2), P2 - K * D * K.transpose()};
  };
};

auto kalman_fold = [](auto f, auto seed, MeasurementTimeSeries data) {
  auto accumulator = seed;
  std::vector<decltype(accumulator)> allAccumulated;
  static_assert(trueData.size() == data.size());
  for (const auto &datum : data) {
    Observation obs = {Xi, Phi, Gamma, u, A, Measurement{datum.second}};
    accumulator = f(accumulator, obs);
    allAccumulated.push_back(accumulator);
  };
  return allAccumulated;
};

TEST_CASE("") {

  std::mt19937 rndEngine(seed);
  std::normal_distribution<> gaussDist{0, radarNoiseSigma};

  const MeasurementTimeSeries measuredData = [&gaussDist, &rndEngine]() {
    MeasurementTimeSeries result;
    for (unsigned k = 0; k < trueData.size(); ++k) {
      result[k].first = trueData[k].first; // time
      result[k].second(0) = trueData[k].second(0) + gaussDist(rndEngine);
    }
    return result;
  }();

  auto final = kalman_fold(kalman(Zeta), Estimate{{0, 0}, P0}, measuredData);
}
