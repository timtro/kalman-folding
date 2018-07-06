#pragma once

#include <vector>

#include <sodium/sodium.h>

namespace util {

  template <typename T, typename A, typename F>
  auto fmap(F f, const std::vector<T, A> &v) {
    using MapedToType = std::invoke_result_t<F, decltype(v.at(0))>;
    std::vector<MapedToType> result;
    result.reserve(v.size());

    for (const auto &each : v) result.push_back(f(each));

    return result;
  }

  template <typename T>
  auto make_listener(const sodium::stream<T> &s) {
    auto sRecord = std::make_shared<std::vector<T>>();
    const auto s_unlisten =
        s.listen([sRecord](auto x) { sRecord->push_back(x); });
    return std::make_pair(s_unlisten, sRecord);
  }

  template <typename T>
  auto make_listener(const sodium::cell<T> &s) {
    auto sRecord = std::make_shared<std::vector<T>>();
    const auto s_unlisten =
        s.listen([sRecord](auto x) { sRecord->push_back(x); });
    return std::make_pair(s_unlisten, sRecord);
  }

}  // namespace util
