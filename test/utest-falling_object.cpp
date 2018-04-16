#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <catch/catch.hpp>
#include <iostream>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <vector>

#include "../include/matplotlibcpp.h"
namespace plt = matplotlibcpp;

using Eigen::Matrix;
using Eigen::Matrix2d;
using Eigen::RowVector2d;
using Eigen::Vector2d;
using Matrix1d = Eigen::Matrix<double, 1, 1>;

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

TEST_CASE("") {
  constexpr double dt = 0.1;
  constexpr double duration = 57.5;
  constexpr size_t numsamples = duration / dt;
  constexpr double hInit = 400'000; // ft
  constexpr double vInit = -6000;  // ft / sec
  constexpr double g = 32.174;     // ft / sec^2
  constexpr double accelInit = -g;     // ft / sec^2

  using SampleDataSet = std::array<std::pair<double, double>, numsamples>;
  //                       x         P
  using State = std::pair<Vector2d, Matrix2d>;
  using Observation =
      //         Xi        Phi       Gamma     u         A            z
      std::tuple<Matrix2d, Matrix2d, Vector2d, Matrix1d, RowVector2d, Matrix1d>;
  // In this example, most of the `Observation` is constant, but that's not
  // generally true. You'll see that we implement our own fold to avoid bulding
  // a massive block of data with repeated content.

  const SampleDataSet trueData = []() {
    SampleDataSet result;
    for (unsigned k = 0; k < numsamples; ++k) {
      auto t = k * dt;
      result[k] = {t, hInit + vInit * t + .5 * accelInit * t * t};
    }
    return result;
  }();

  auto kalman_static = [](Matrix1d Z) {
    return [&Z](State s, Observation o) -> State {
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

  // Ζ (Zeta) -- Measurement noise:
  const Matrix1d Zeta{100}; 

  // P0 -- Starting covariance for x
  const auto P0 = Matrix2d::Identity() * 9999999999999;

  // Φ (Phi) -- State transition matrix (AKA propagator matrix, AKA fundamental
  // matrix).
  const Matrix2d Phi = [dt]() {
    Matrix2d m;
    m << 1.f, dt, 0.f, 1.f;
    return m;
  }();

  // Propagator for the control input's contribution to the IVP:
  //   x[k+1] = Φ x[k] + Γ u[k]
  const Vector2d Gamma(dt * dt / 2, dt);

  // Ξ (Xi) --  
  const Matrix2d Xi = [dt]() {
    Matrix2d m;
    m << dt * dt * dt / 3, dt * dt / 2, dt * dt / 2, dt;
    m *= 100;
    return m;
  }();

  // A -- The output matrix. This one selects position from the state vector:
  //     z = A x = [ 1  0 ] [ h dh/dt ]^T
  const RowVector2d A(1, 0);

  // Control input, gravity's pull.
  const Matrix1d u(-g);

  auto kalman_fold = [Xi, Phi, Gamma, u, A](auto f, auto seed,
                                            SampleDataSet data) {
    auto accumulator = seed;
    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>,
               std::vector<Matrix2d>>
        plotdata;
    for (auto datum : data) {
      Observation obs = {Xi, Phi, Gamma, u, A, Matrix1d{datum.second}};
      accumulator = f(accumulator, obs);
      std::get<0>(plotdata).push_back(datum.first);
      std::get<1>(plotdata).push_back(datum.second);
      std::get<2>(plotdata).push_back(accumulator.first(0));
      std::get<3>(plotdata).push_back(accumulator.second);
    }
    return plotdata;
  };

  auto [time, trueHeight, hs, Ps] =
      kalman_fold(kalman_static(Zeta), State{{0, 0}, P0}, trueData);
  plt::plot(time, trueHeight);
  plt::plot(time, hs);
  plt::show();
}
