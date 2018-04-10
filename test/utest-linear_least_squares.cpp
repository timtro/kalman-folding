#include "utest-linear_least_squares.hpp"
#include <catch/catch.hpp>

// The oracle's polynomial from Beckman (2017) Kalman Folding Pt. 1
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
