#pragma once

#include <string>

#include <Eigen/Dense>

#include "../util/util.hpp"
#include "gnuplot-iostream.h"

using Eigen::Vector4d;
using Eigen::Matrix4d;

// clang-format off
auto poly_to_string(std::string symbol, Vector4d coeff){
  std::ostringstream poly;
  poly  << symbol << "(x) = ("
        << coeff(0) << ") + ("
        << coeff(1) << " * x) + ("
        << coeff(2) << " * x**2) + ("
        << coeff(3) << " * x**3)";
  return poly.str();
}
// clang-format on

auto poly_covar_tube_to_string(std::string ubndSymbol, std::string lbndSymbol,
                          Vector4d coeff, Matrix4d cov) {
  assert(cov(0, 0) >= 0.);  // Sane covariances are always positive.
  assert(cov(1, 1) >= 0.);
  assert(cov(2, 2) >= 0.);
  assert(cov(3, 3) >= 0.);
  // with…
  auto ubndCoeff = Vector4d{coeff(0) * (1 + std::sqrt(cov(0, 0))),
                            coeff(1) * (1 + std::sqrt(cov(1, 1))),
                            coeff(2) * (1 + std::sqrt(cov(2, 2))),
                            coeff(3) * (1 + std::sqrt(cov(3, 3)))};
  auto ubndPoly = poly_to_string(ubndSymbol, ubndCoeff);
  auto lbndCoeff = Vector4d{coeff(0) * (1 - std::sqrt(cov(0, 0))),
                            coeff(1) * (1 - std::sqrt(cov(1, 1))),
                            coeff(2) * (1 - std::sqrt(cov(2, 2))),
                            coeff(3) * (1 - std::sqrt(cov(3, 3)))};
  auto lbndPoly = poly_to_string(lbndSymbol, lbndCoeff);

  return ubndPoly + '\n' + lbndPoly;
}
