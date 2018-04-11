#include "utest-linear_least_squares.hpp"
#include <algorithm>
#include <catch/catch.hpp>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

/* Preludes:
 * In [Beckman2016], Beckman introduces the static Kalman filter in a series of
 * four preludes, the fourth being an implementation of an actual Kalman filter.
 * This series of unit tests employ those recurrant relations and folding to
 * reproduce the results from his Wolfram code.
 */

auto bulk_average = [](auto xs) {
  return std::accumulate(cbegin(xs), cend(xs), 0.f) / xs.size();
};

// Prelude 2
TEST_CASE("Functional folding for mean on…", "[prelude2]") {
  // NOT A REAL UNIT TEST: No external code dependencies.

  using State = std::pair<double, unsigned>;
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

// The oracle's polynomial.
double f(double x) { return -5 * x * x * x - 4 * x * x + 9 * x - 3; }

TEST_CASE("Test polynomial") {
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
