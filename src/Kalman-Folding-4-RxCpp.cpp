#include <functional>
#include <numeric>
#include <string>

#include <Eigen/Dense>
#include <catch/catch.hpp>
#include <rxcpp/rx.hpp>

#include <iostream>

#include "../include/util/util-frp.hpp"
#include "../include/util/util.hpp"

#ifdef PLOT
#include "../include/plotting/gnuplot-iostream.h"
#include "../include/plotting/plot-helpers.hpp"
#endif

using Eigen::Matrix;
using Eigen::Matrix4d;
using Eigen::RowVector4d;
using Eigen::Vector4d;
using Matrix1d = Eigen::Matrix<double, 1, 1>;

TEST_CASE("Starting from the linear least squares on vectors…") {
  // The oracle's polynomial.
  const auto coefficients = Vector4d{-3, 9, -4, -5};

  const auto f = [&coefficients](double x) -> double {
    return coefficients.transpose() * Vector4d{1, x, x * x, x * x * x};
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

  auto cume = [](Matrix1d Z) {
    return [Z](State s, Observation o) -> State {
      // with…
      const auto [A, z] = o;
      const auto [x, P] = s;
      const auto D = Z + A * P * A.transpose();
      const auto K = P * A.transpose() * D.inverse();
      return {x + K * (z - A * x), P - K * D * K.transpose()};
    };
  };

  SECTION(
      "… we repeat the computation on vectors and perform the identical "
      "computation on FRP behaviours (sodium streams), requiring identical "
      "results.") {
    const auto P0 = Matrix4d::Identity() * 1000.f;

    const std::vector<std::pair<double, double>> dataPoints = {{0., -2.28442},
                                                               {1., -4.83168},
                                                               {-1., -10.4601},
                                                               {-2., 1.40488},
                                                               {2., -40.8079}};
    const auto data = [&dataPoints]() {
      std::vector<Observation> result(dataPoints.size());
      for (auto& each : dataPoints) {
        auto& [t, val] = each;
        result.push_back({{1., t, t * t, t * t * t}, Matrix1d{val}});
      }
      return result;
    }();

    const auto Zeta = Matrix1d{1.f};
    auto seed = State{{0, 0, 0, 0}, P0};

    // First, repeat the calculation on vectors (a static data structure).
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

    // Now, repeat the calculation using RxCpp.
    const auto observations = rxcpp::observable<>::iterate(data);
    const auto stateEstimates = observations.scan(seed, cume(Zeta));

    std::vector<State> stateEstimateRecord;

    const auto values = stateEstimates.subscribe(
        [&stateEstimateRecord](auto x) { stateEstimateRecord.push_back(x); });

    REQUIRE(stateEstimateRecord.size() == data.size());

    const auto [frpEstimatedCoefficients, frpEstimatedCovariance] =
        stateEstimateRecord.back();

    REQUIRE(frpEstimatedCoefficients(0) == estimatedCoefficients(0));
    REQUIRE(frpEstimatedCoefficients(1) == estimatedCoefficients(1));
    REQUIRE(frpEstimatedCoefficients(2) == estimatedCoefficients(2));
    REQUIRE(frpEstimatedCoefficients(3) == estimatedCoefficients(3));

    REQUIRE(frpEstimatedCovariance(0, 0) == estimatedCovariance(0, 0));
    REQUIRE(frpEstimatedCovariance(1, 1) == estimatedCovariance(1, 1));
    REQUIRE(frpEstimatedCovariance(2, 2) == estimatedCovariance(2, 2));
    REQUIRE(frpEstimatedCovariance(3, 3) == estimatedCovariance(3, 3));

#ifdef PLOT
    Gnuplot gp;
    gp << poly_to_string("f", coefficients) << '\n';
    gp << poly_to_string("g", frpEstimatedCoefficients) << '\n';
    gp << poly_covar_tube_to_string("h", "i", frpEstimatedCoefficients,
                                    frpEstimatedCovariance)
       << '\n';
    gp << "set xr [-2:2]\n set yr [-45:5]\n";
    gp << "plot '+' using 1:(h($1)):(i($1)) title \"One standard deviation.\" "
          "with filledcurves closed fc rgb '#6699FF55', "
          "f(x) title \"Oracle\\\'s polynomial\" w l ls 1, "
          "g(x) title \"Estimated polynomial\" w l ls 3, '-' w p\n";
    gp.send1d(dataPoints);
    gp << "pause mouse key\nwhile (MOUSE_CHAR ne 'q') { pause mouse key;}\n";
#endif
  }
}
