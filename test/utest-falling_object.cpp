#include <Eigen/Dense>
#include <algorithm>
#include <catch/catch.hpp>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

using Eigen::Matrix;
using Eigen::Matrix2d;
using Eigen::Vector2d;
using Eigen::RowVector2d;
using Matrix1d = Eigen::Matrix<double, 1, 1>;

/* Preludes:
 * In [1], Beckman introduces the static Kalman filter in a series of
 * four preludes, the fourth being an implementation of a static Kalman filter.
 * (Static meaning that model states do not vary with the independent variable)
 * Those preludes are covered in `utest-linear_least_squares.cpp`.
 *
 * In [2], Beckman generalizes to the non-static case, where the
 * model includes a control input term in addition to the drift term. Beckman's
 * exhibition centres on a textbook example from Zarchan and Musoff [3] 
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

  //                             Xi        Phi       Gamma
  using Observation = std::tuple<Matrix2d, Matrix2d, Vector2d,
  //                              u         A         z
                                  Matrix1d, Matrix2d, Matrix1d>;
  //                       x            P
  using State = std::tuple<RowVector2d, Matrix2d>;

  auto cume = [](Matrix1d Z) {
    return [&Z](State s, Observation o) -> State {
      // with…
      const auto [x, P] = s;
      const auto [Xi, Phi, Gamma, u, A, z] = o;
      const auto D = Z + A * P * A.transpose();
      const auto K = P * A.transpose() * D.inverse();
      return {x + K * (z - A * x), P - K * D * K.transpose()};
    };
  };
}
