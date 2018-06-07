#include <sodium/sodium.h>
#include <Eigen/Dense>
#include <catch/catch.hpp>
#include <functional>
#include <numeric>
#include <string>

#include <iostream>

using Eigen::Matrix;
using Eigen::Matrix4d;
using Eigen::RowVector4d;
using Eigen::Vector4d;
using Matrix1d = Eigen::Matrix<double, 1, 1>;

TEST_CASE("Starting from the linear least squares on vectors…") {
  // The oracle's polynomial.
  const auto coefficients = Vector4d{-5, -4, 9, -3};

  const auto f = [&coefficients](double x) -> double {
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

  // Same as cume, but structured for use with Sodium's accum().
  // I.e., arguments reversed (thus the name ucme), and wrapped in a
  // std::function.
  auto ucme = [](Matrix1d Z) {
    return std::function<State(const Observation&, const State&)>(
        [&Z](const Observation& o, const State& s) -> State {
          // with…
          const auto [A, z] = o;
          const auto [x, P] = s;
          const auto D = Z + A * P * A.transpose();
          const auto K = P * A.transpose() * D.inverse();
          return {x + K * (z - A * x), P - K * D * K.transpose()};
        });
  };

  SECTION(
      "… we repeat the computation on vectors and perform the identical "
      "computation on FRP behaviours (sodium streams), requiring identical "
      "results.") {
    const auto P0 = Matrix4d::Identity() * 1000.f;
    const auto data =
        std::vector<Observation>{{{1.f, 0.f, 0.f, 0.f}, Matrix1d{-2.28442}},
                                 {{1.f, 1.f, 1.f, 1.f}, Matrix1d{-4.83168}},
                                 {{1.f, -1.f, 1.f, -1.f}, Matrix1d{-10.4601}},
                                 {{1.f, -2.f, 4.f, -8.f}, Matrix1d{1.40488}},
                                 {{1.f, 2.f, 4.f, 8.f}, Matrix1d{-40.8079}}};
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

    // Now, repeat the calculation using FRP.
    sodium::stream_sink<Observation> observationStream;
    const auto stateEstimateStream = observationStream.accum(seed, ucme(Zeta));
    std::shared_ptr<std::vector<State>> reifiedFRPOutput{
        new std::vector<State>()};
    const auto unlisten = stateEstimateStream.listen(
        [reifiedFRPOutput](State s) { reifiedFRPOutput->push_back(s); });

    for (const auto& each : data) {
      sodium::transaction trans;
      observationStream.send(each);
    }

    unlisten();

    REQUIRE(reifiedFRPOutput->size() == data.size() + 1 /*+1 for seed*/);

    const auto [frpEstimatedCoefficients, frpEstimatedCovariance] =
        reifiedFRPOutput->back();

    REQUIRE(frpEstimatedCoefficients(0) == estimatedCoefficients(0));
    REQUIRE(frpEstimatedCoefficients(1) == estimatedCoefficients(1));
    REQUIRE(frpEstimatedCoefficients(2) == estimatedCoefficients(2));
    REQUIRE(frpEstimatedCoefficients(3) == estimatedCoefficients(3));

    REQUIRE(frpEstimatedCovariance(0, 0) == estimatedCovariance(0, 0));
    REQUIRE(frpEstimatedCovariance(1, 1) == estimatedCovariance(1, 1));
    REQUIRE(frpEstimatedCovariance(2, 2) == estimatedCovariance(2, 2));
    REQUIRE(frpEstimatedCovariance(3, 3) == estimatedCovariance(3, 3));
  }
}
