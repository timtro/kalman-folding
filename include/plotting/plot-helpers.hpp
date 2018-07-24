#pragma once

#include <string>

#include <boost/format.hpp>

#include "gnuplot-iostream.h"
#include "../util/util.hpp"

template <typename Data, typename F>
void plot_with_tube(std::string title, const Data &data, F ref_func,
                    double margin) {
  const std::string tubeColour = "#6699FF55";

  Gnuplot gp;
  gp << "set title '" << title << "'\n"
     << "plot '-' u 1:2:3 title 'Acceptable margin: analytical ±"
     << boost::format("%.3f") % margin << "' w filledcu fs solid fc rgb '"
     << tubeColour
     << "', '-' u 1:2 "
        "title 'Test result' w l\n";

  auto range = util::fmap(
      [&](auto x) {
        // with…
        const double t = x.first;
        const double analyt = ref_func(t);

        return std::make_tuple(t, analyt + margin, analyt - margin);
      },
      data);

  gp.send1d(range);
  gp.send1d(data);
  gp << "pause mouse key\nwhile (MOUSE_CHAR ne 'q') { pause mouse "
        "key; }\n";
}
