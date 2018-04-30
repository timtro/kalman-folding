#include <Eigen/Dense>
#include <algorithm>
#include <catch/catch.hpp>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

using Eigen::Matrix;
using Eigen::Matrix4d;
using Eigen::RowVector4d;
using Eigen::Vector4d;
using Matrix1d = Eigen::Matrix<double, 1, 1>;

/* Preludes:
 * In [Beckman2016], Beckman introduces the static Kalman filter in a series of
 * four preludes, the fourth being an implementation of a static Kalman filter.
 * (Static meaning that model states do not vary with the independent variable)
 *
 * The following series of unit tests reproduce those preludes in C++, and
 * results are compared to the provided data.
 *
 * [Beckman2016] Brian Beckman, Kalman Folding-Part 1. (2016)
 */

auto bulk_average = [](auto xs) {
  return std::accumulate(cbegin(xs), cend(xs), 0.f) / xs.size();
};

// Prelude 2
TEST_CASE("Functional folding for mean (0-order/1-state filter)…",
          "[prelude2]") {
  // NOT A REAL UNIT TEST: No external code dependencies.

  using State = std::pair<double, unsigned int>;
  auto cume = [](State xANDn, double z) -> State {
    // with…
    const auto [x, n] = xANDn;
    const double K = 1.f / (n + 1);

    return {x + K * (z - x), n + 1};
  };

  SECTION("… a vector") {
    REQUIRE(cume({0, 0}, 1.414).second == 1);
    REQUIRE(cume({0, 0}, 1.414).first == Approx(1.414));

    auto data = std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const auto [average, count] =
        std::accumulate(cbegin(data), cend(data), State{0, 0}, cume);
    REQUIRE(count == data.size());
    REQUIRE(average == Approx(bulk_average(data)));
  }
}

// Prelude 3
TEST_CASE("Functional folding for mean and variance on…", "[prelude3]") {
  // NOT A REAL UNIT TEST: No external code dependencies.

  using State = std::tuple<double, double, unsigned int>;
  auto cume = [](State varANDxANDn, double z) -> State {
    // with…
    const auto [var, x, n] = varANDxANDn;
    const auto KforMean = 1. / (n + 1);
    const auto KforVariance = n * KforMean;
    const auto residual = z - x;
    const auto x2 = x + KforMean * residual;
    const auto ssr2 = (n - 1) * var + KforVariance * residual * residual;

    return {ssr2 / std::max<unsigned>(n, 1), x2, n + 1};
  };

  SECTION("… a vector") {
    auto data = std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const auto [variance, average, count] =
        std::accumulate(cbegin(data), cend(data), State{0, 0, 0}, cume);
    REQUIRE(count == data.size());
    REQUIRE(variance == Approx(9.1667));
    REQUIRE(average == Approx(bulk_average(data)));
  }
}

// Prelude 4
TEST_CASE("Static Kalman fold…", "[prelude4]") {
  // The oracle's polynomial.
  const auto coefficients = Vector4d{-5, -4, 9, -3};

  auto f = [&coefficients](double x) -> double {
    return coefficients.transpose() * Vector4d{x * x * x, x * x, x, 1};
  };

  SECTION("… with polynomial f such that …") {
    // Values obtained from Octave:
    // > polyval([-5, -4, 9, -3], [-2, -1, 0, 1, 2])
    //  ans =
    //        3  -11   -3   -3  -41

    REQUIRE(f(-2.f) == Approx(3.f));
    REQUIRE(f(-1.f) == Approx(-11.f));
    REQUIRE(f(0.f) == Approx(-3.f));
    REQUIRE(f(1.f) == Approx(-3.f));
    REQUIRE(f(2.f) == Approx(-41.f));
  }

  using State = std::tuple<Vector4d, Matrix4d>;
  using Observation = std::tuple<RowVector4d, Matrix1d>;

  // in Part 2, Beckman calls this `kalmanStatic`, which is a much better name.
  // Because it curries in a Zeta first, it's not exactly an accumulator, it's a
  // function that returns an accumulator.
  //
  auto cume = [](Matrix1d Z) {
    return [&Z](State s, Observation o) -> State {
      // with…
      const auto [A, z] = o;
      const auto [x, P] = s;
      const auto D = Z + A * P * A.transpose();
      const auto K = P * A.transpose() * D.inverse();
      return {x + K * (z - A * x), P - K * D * K.transpose()};
    };
  };

  SECTION("… given Beckman's data, should produce Beckman's estimates") {
    const auto P0 = Matrix4d::Identity() * 1000.f;
    const auto data =
        std::vector<Observation>{{{1.f, 0.f, 0.f, 0.f}, Matrix1d{-2.28442}},
                                 {{1.f, 1.f, 1.f, 1.f}, Matrix1d{-4.83168}},
                                 {{1.f, -1.f, 1.f, -1.f}, Matrix1d{-10.4601}},
                                 {{1.f, -2.f, 4.f, -8.f}, Matrix1d{1.40488}},
                                 {{1.f, 2.f, 4.f, 8.f}, Matrix1d{-40.8079}}};
    const auto Zeta = Matrix1d{1.f};

    const auto [estimatedCoefficients, estimatedCovariance] = std::accumulate(
        cbegin(data), cend(data), State{{0, 0, 0, 0}, P0}, cume(Zeta));

    // Beckman provides output corresponding to the input `data`.
    REQUIRE(estimatedCoefficients(0) == Approx(-2.97423));
    REQUIRE(estimatedCoefficients(1) == Approx(7.2624));
    REQUIRE(estimatedCoefficients(2) == Approx(-4.21051));
    REQUIRE(estimatedCoefficients(3) == Approx(-4.45378));

    // Just test diagonal elements of the covariance.
    REQUIRE(estimatedCovariance(0, 0) == Approx(0.485458));
    REQUIRE(estimatedCovariance(1, 1) == Approx(.901908));
    REQUIRE(estimatedCovariance(2, 2) == Approx(0.0714031));
    REQUIRE(estimatedCovariance(3, 3) == Approx(0.0693839));
  }
}
