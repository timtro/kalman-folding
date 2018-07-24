#pragma once

#include <vector>

#include <sodium/sodium.h>

namespace chrono = std::chrono;

namespace util {

  template <typename T>
  auto make_listener(const sodium::stream<T> &s) {
    auto sRecord = std::make_shared<std::vector<T>>();
    const auto s_unlisten =
        s.listen([sRecord](auto x) { sRecord->push_back(x); });
    return std::make_pair(s_unlisten, sRecord);
  }

  template <typename T>
  auto make_listener(const sodium::cell<T> &c) {
    auto cRecord = std::make_shared<std::vector<T>>();
    const auto s_unlisten =
        c.listen([cRecord](auto x) { cRecord->push_back(x); });
    return std::make_pair(s_unlisten, cRecord);
  }

}  // namespace util
