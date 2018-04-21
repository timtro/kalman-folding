#include "boost/hana/functional/curry.hpp"
#include "range/v3/all.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <catch/catch.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>

using Eigen::Matrix;
using namespace ranges;
using boost::hana::curry;

/* Preludes:
 * In [1], Beckman introduces the static Kalman filter in a series of
 * four preludes, the fourth being an implementation of a static Kalman
 * filter. (Static meaning that model states do not vary with the
 * independent variable) Those preludes are covered in
 * `utest-linear_least_squares.cpp`.
 *
 * In [2], Beckman generalizes to the non-static case, where the
 * model includes a control input term in addition to the drift term.
 * Beckman's exhibition centres on a textbook example from Zarchan and
 * Musoff [3].
 *
 * This series of test cases explores Beckman's implementation. It's
 * important to keep in mind that this and previously explored
 * implementations suffer numerical stability and efficiency issues. For
 * example, the use of matrix-inversion. Better numerical hygine will be
 * practised in another iteration of the problem.
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

using State = Matrix<double, n, 1>;       // Contains a distance and a velocity.
using Measurement = Matrix<double, b, 1>; // 1x1, contains just a distance.
using Control = Matrix<double, m, 1>;     // 1x1, the acceleration.
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
// The observation includes the model and control. Since the model and control
// input is constant for this example, you'll see that the kalman_fold function
// takes the measurement from a list and packs it in the tuple with constant
// matrices.

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

// Φ (Phi) -- State transition matrix (AKA propagator matrix
// and fundamental matrix).
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

// Ξ (Xi) -- Not sure what to call this, but it's part of the evolution of the
// estimate covariance.
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

// The foldable Kalman function. When folded over a recursive structure
// containing `Measurement`s, one obtains the series of estimated states. (That
// assumes you get a series out of the fold. A traditional fold will yield only
// it's most up to date rockoning at the end of the fold)
constexpr auto kalman = [](Matrix_bxb Z) {
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

constexpr auto kalman_fold = [](auto f, auto x0, MeasurementTimeSeries data) {
  auto accumulator = x0;
  std::vector<decltype(accumulator)> xs;
  xs.reserve(data.size());

  static_assert(trueData.size() == data.size());

  for (const auto &datum : data) {
    Observation obs = {Xi, Phi, Gamma, u, A, Measurement{datum.second}};
    accumulator = f(accumulator, obs);
    xs.push_back(accumulator);
  };

  return xs;
};

TEST_CASE("Tracking a falling object with a (simulated) noisy radar, the "
          "Kalman filtered signal ...") {

  std::mt19937 rndEngine(seed);
  std::normal_distribution<> gaussDist{0, radarNoiseSigma};

  const auto measuredData = [&gaussDist,
                             &rndEngine]() -> MeasurementTimeSeries {
    MeasurementTimeSeries result;
    for (unsigned k = 0; k < trueData.size(); ++k) {
      result[k].first = trueData[k].first; // time
      result[k].second(0) = trueData[k].second(0) + gaussDist(rndEngine);
    }
    return result;
  }();

  const auto estimationSignal =
      kalman_fold(kalman(Zeta), Estimate{{0, 0}, P0}, measuredData);

  // clang-format off
  const auto estimationResidual =
      view::zip(trueData, estimationSignal) 
      | view::transform([](const auto &truthAndEstimate) {
        // truth : (time, State)
        // estimate : (State, Matrix_nxn) = (State, P)
        const auto &[truth, estimate] = truthAndEstimate;
        State residual;

        residual << truth.second(0) - estimate.first(0),
            truth.second(1) - estimate.first(1);

        return residual;
      });
  // clang-format on

  SECTION("...  covariance should decrease monotonically since the measurement "
          "variance is constant.") {

    constexpr std::pair seed{false, std::numeric_limits<double>::max()};

    // The extracter function is used to get at whichever part of the `record`
    // we want to check for monotonicity.
    //  Record:
    //  first     second
    // (State) (Covariance)
    //    *        * *
    //    *        * *
    //
    constexpr auto covar_element =
        curry<3>([](size_t j, size_t k, Estimate e) { return e.second(j, k); });

    constexpr auto foldable_is_decreaseing =
        curry<3>([](const auto extractor, const auto &flagAndPrev,
                    const auto &record) -> std::pair<bool, double> {
          // with ...
          const auto &[flag, prevData] = flagAndPrev;
          const auto data = extractor(record);

          return flag ? flagAndPrev : std::pair{data < prevData, data};
        });

    const bool heightVarianceIsMonotoniclyDecreasing =
        accumulate(estimationSignal, seed,
                   foldable_is_decreaseing(covar_element(0, 0)))
            .first;

    REQUIRE(heightVarianceIsMonotoniclyDecreasing);

    const bool speedVarianceIsMonotoniclyDecreasing =
        accumulate(estimationSignal, seed,
                   foldable_is_decreaseing(covar_element(1, 1)))
            .first;

    REQUIRE(speedVarianceIsMonotoniclyDecreasing);
  }

  SECTION("... remain in the 90% confidence tube at least 90% of the time.") {

    assert(estimationSignal.size() == estimationResidual.size());

    constexpr auto count_if_out_of_tube = [](int counter,
                                             const auto &estimateAndResidual) {
      const auto &[estimate, residual] = estimateAndResidual;
      if (residual(0) * residual(0) > 1.65 * 1.65 * estimate.second(0, 0))
        return ++counter;
      else
        return counter;
    };

    auto outOfTubeCount =
        accumulate(view::zip(estimationSignal, estimationResidual), 0,
                   count_if_out_of_tube);

    REQUIRE(outOfTubeCount <= trueData.size() * 0.1);
  }
}
