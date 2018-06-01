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
#include "boost/hana/functional/curry.hpp"
#include "range/v3/all.hpp"

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
 * This series of test cases explores Beckman's implementation in [2]. It's
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

/* Notation:
 * - [[·]]: Semantic brackets. In this context, can be read roughly as "type
 *     of ·". Eg., [[time]] = double.
 *
 * - [·]: "List-of-·".
 *
 * - "foldable": Means a function has an appropriate signauture to be used as a
 *     binary operation on a fold: (A, B) → B or A → B → B.
 *
 * - N×M: an N-by-M matrix. In this program, N×M = Eigen::Matrix<double, N, M>.
 *
 */

constexpr double g = 32.174;       // ft / sec^2
constexpr double dt = 0.1;         // sec
constexpr double duration = 57.5;  // sec
constexpr size_t numsamples = duration / dt;
constexpr double hInit = 400'000;  // ft
constexpr double vInit = -6000;    // ft / sec
constexpr double accelInit = -g;   // ft / sec^2

constexpr double radarNoiseSigma = 1E3;  // ft
constexpr double radarNoiseVariance =
    radarNoiseSigma * radarNoiseSigma;  // ft^2

constexpr size_t n = 2;  // number of state variables, h and dhdt.
constexpr size_t b = 1;  // number of output variables, just h.
constexpr size_t m = 1;  // number of control variables, just acceleration.

using Mnxn = Matrix<double, n, n>;
using Mnx1 = Matrix<double, n, 1>;
using M1xn = Matrix<double, 1, n>;
using Mbxn = Matrix<double, b, n>;
using Mbx1 = Matrix<double, b, 1>;
using Mbxb = Matrix<double, b, b>;
using Mnxm = Matrix<double, n, m>;
using Mmx1 = Matrix<double, m, 1>;

// x : State, u : Control, z : Measurement
//   x = /   h(t)  \     u = ( -g )     z = ( h(t) + Noise )
//       \ dhdt(t) / '               ,
//
// d/dt x = F x + G u    // Dynamics model
//      z = A x + Noise  // Relationship between state and measurement.
//
//  where
//    F = / 0 1 \    G = / 0 \    A = ( 1 0 ).
//        \ 0 0 / '      \ 1 / '

using State = Mnx1;  // Contains a distance and a velocity.
using TimeState = std::pair<double, State>;
using Measurement = Mbx1;  // 1x1, contains just a distance.
using Control = Mmx1;      // 1x1, the acceleration.
using RowState = M1xn;
using StateTimeSeries = std::array<TimeState, numsamples>;
using Estimate = std::pair<State, Mnxn>;  // (State, [[P]])
// Observation = ([[Xi]], [[Phi]], [[Gamma]], [[u]], [[A]], [[z]])
//    = (n×n, n×n, n×m, m×1, 1×n, b×1)
using Observation =
    std::tuple<Mnxn, Mnxn, Mnxm, Control, RowState, Measurement>;
// The observation includes the model and control. Since the model and control
// input is constant for this example, you'll see that the kalman_fold function
// takes the measurement from a list and packs it in the tuple with constant
// matrices.

const std::vector<double> ts = view::ints(0, static_cast<int>(numsamples - 1))
                               | view::transform([](int x) { return x * dt; });

const std::vector<State> groundTruth =
    ts | view::transform([](double t) {
      State next;
      next << hInit + vInit * t + .5 * accelInit * t * t, vInit + accelInit * t;
      return next;
    });

// Ζ (Zeta) — Measurement noise:
const Mbxb Zeta = []() {
  Mbxb mat;
  mat << radarNoiseVariance;
  return mat;
}();

// P0 — Starting covariance for x
const auto P0 = Mnxn::Identity() * 9999999999999;

// Φ (Phi) — State transition matrix (AKA propagator matrix
// and fundamental matrix).
const Mnxn Phi = []() {
  Mnxn mat;
  mat << 1.f, dt, 0.f, 1.f;
  return mat;
}();

// Propagator for the control input's contribution to the IVP:
//   x[k+1] = Φ x[k] + Γ u[k]
const Mnxm Gamma = []() {
  Mnxm mat;
  mat << dt * dt / 2, dt;
  return mat;
}();

// Ξ (Xi) — Not sure what to call this, but it's part of the evolution of the
// estimate covariance.
const Mnxn Xi = []() {
  Mnxn mat;
  mat << dt * dt * dt / 3, dt * dt / 2, dt * dt / 2, dt;
  mat *= 100;
  return mat;
}();

// A — The output matrix. This one selects position from the state vector:
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

// The foldable Kalman function, when folded over a recursive structure
// containing `Measurement`s, yields the series of estimated states. (That
// assumes you get a series out of the fold. A traditional fold will yield only
// it's most up to date rockoning at the end of the fold)
//
// The `kalman` function takes a Zeta value and returns a foldable Kalman
// filter.
constexpr auto kalman = [](Mbxb Z) {
  // (Estimate, Observation) → Estimate
  //   = ( (State, [[P]]), ([[Xi]], [[Phi]], [[Gamma]], [[u]], [[A]], [[z]]) )
  //        → (State, n×n)
  //   = ( (n×1, n×n), (n×n, n×n, n×m, m×1, 1×n, b×1) ) → (n×1, n×n)
  //   = ( (2×1, 2×2), (2×2, 2×2, 2×1, 1×1, 1×2, 1×1) ) → (2×1, 2×2)
  return [&Z](Estimate s, Observation o) -> Estimate {
    // with …
    const auto [x, P] = s;
    const auto [Xi, Phi, Gamma, u, A, z] = o;
    const auto x2 = Phi * x + Gamma * u;
    const auto P2 = Xi + Phi * P * Phi.transpose();
    const auto D = Z + A * P2 * A.transpose();
    const auto K = P2 * A.transpose() * D.inverse();

    return {x2 + K * (z - A * x2), P2 - K * D * K.transpose()};
  };
};

// Normally, fold : ( (A, B) → B, B, [A] ) → B
// But we want intermediats, so we want a list of B:
//  kalman_fold : ( (A, B) → B, B, [A] ) → [B]
//    where
//      A = Measurement = b×1 = 1×1
//      B = Estimate = (State, n×n) = (2×1, 2×2)
// We use vectors as lists, but this is generalized upon later.
constexpr auto kalman_fold =
    [](auto f, auto x0,
       const std::vector<Measurement> &data) -> std::vector<Estimate> {
  auto accumulator = x0;
  std::vector<decltype(accumulator)> xs;
  xs.reserve(data.size());

  for (const auto &datum : data) {
    Observation obs = {Xi, Phi, Gamma, u, A, datum};
    accumulator = f(accumulator, obs);
    xs.push_back(accumulator);
  };

  return xs;
};

TEST_CASE(
    "Tracking a falling object with a (simulated) noisy radar, the "
    "Kalman filtered signal …") {
  std::seed_seq seed{1, 2, 3, 4, 5};
  std::mt19937 rndEngine(seed);
  std::normal_distribution<> gaussDist{0, radarNoiseSigma};

  // Reminder:
  //   groundTruth : std::vector<State>
  //      = std::vector<2×1>
  const std::vector<Measurement> measuredData =
      groundTruth
      | view::transform([&gaussDist, &rndEngine](State x) -> Measurement {
          return Measurement{x(0) + gaussDist(rndEngine)};
        });

  const auto estimationSignal =
      kalman_fold(kalman(Zeta), Estimate{{0, 0}, P0}, measuredData);

  // estimationResidual : [ State ]
  const auto estimationResidual =
      view::zip(groundTruth, estimationSignal)
      | view::transform([](const auto &truthAndEstimate) -> State {
          // NB — estmate : (State, n×n) = (State, P)
          const auto &[truth, estimate] = truthAndEstimate;
          State residual;
          residual << truth(0) - estimate.first(0),
              truth(1) - estimate.first(1);

          return residual;
        });

  SECTION(
      "…  covariance should decrease monotonically since the measurement "
      "variance is constant.") {
    constexpr std::pair seed{false, std::numeric_limits<double>::max()};

    // This function extracts an element from the covariance matrix in a
    // Measurement.
    //  Measurement = (State, [[P]])
    //      = ( State, 2×1 )
    //        / * \   / * * \
    //        \ * /'  \ * * /
    constexpr auto covar_element =
        curry<3>([](size_t j, size_t k, Estimate e) { return e.second(j, k); });

    // foldable_is_decreasing :
    //    ( (A → double), (bool, double), A ) → (bool, double)
    constexpr auto foldable_is_decreasing =
        curry<3>([](const auto extractor, const auto &flagAndPrev,
                    const auto &record) -> std::pair<bool, double> {
          // with …
          const auto &[flag, prevData] = flagAndPrev;
          const auto data = extractor(record);

          return flag ? flagAndPrev : std::pair{data < prevData, data};
        });

    const bool heightVarianceIsMonotoniclyDecreasing =
        accumulate(estimationSignal, seed,
                   foldable_is_decreasing(covar_element(0, 0)))
            .first;

    REQUIRE(heightVarianceIsMonotoniclyDecreasing);

    const bool speedVarianceIsMonotoniclyDecreasing =
        accumulate(estimationSignal, seed,
                   foldable_is_decreasing(covar_element(1, 1)))
            .first;

    REQUIRE(speedVarianceIsMonotoniclyDecreasing);
  }

  SECTION("… remain in the 90% confidence tube at least 90% of the time.") {
    assert(estimationSignal.size() == estimationResidual.size());

    // foldable_count_if_out_of_tube : ( int, (Estimate, State) ) → int
    //    = (int, ((2×1, 2×2), 2x1) → int
    constexpr auto foldable_count_if_out_of_tube =
        [](int counter, const auto &estimateAndResidual) {
          // with …
          const auto &[estimate, residual] = estimateAndResidual;
          const double heightResidualSqr = residual(0) * residual(0);
          const double conf90 = 1.65 * 1.65;  // The sigma for 90% confidence.

          if (heightResidualSqr > conf90 * estimate.second(0, 0))
            return ++counter;
          else
            return counter;
        };

    auto outOfTubeCount =
        accumulate(view::zip(estimationSignal, estimationResidual), 0,
                   foldable_count_if_out_of_tube);

    REQUIRE(outOfTubeCount <= groundTruth.size() * 0.1);
  }
}
