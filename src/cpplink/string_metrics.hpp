// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <string_view>

namespace cpplink {

// Edit distance, abandoned as soon as it is known to exceed `max_distance`, which
// is returned as `max_distance + 1`. Callers only ever ask "is this within k", so
// computing the true distance beyond k is wasted work on billions of pairs.
int BoundedLevenshtein(std::string_view a, std::string_view b, int max_distance);

// Jaro similarity in [0, 1].
double Jaro(std::string_view a, std::string_view b);

// Jaro-Winkler: Jaro with a bonus for a shared prefix, applied only above
// `boost_threshold` as in the original formulation.
double JaroWinkler(std::string_view a, std::string_view b, double prefix_scale = 0.1,
                   double boost_threshold = 0.7);

// Jaro-Winkler against a threshold, rejecting on length alone where that is
// already decisive. Since matches <= min(|a|,|b|) and transpositions >= 0, Jaro is
// bounded above by (2 + min/max)/3, and the prefix bonus adds at most 0.4(1 - jaro).
// The bound is exact, so this never disagrees with JaroWinkler(a, b) >= threshold.
bool JaroWinklerAtLeast(std::string_view a, std::string_view b, double threshold);

// Great-circle distance in kilometres.
double HaversineKm(double lat_a, double lon_a, double lat_b, double lon_b);

}  // namespace cpplink
