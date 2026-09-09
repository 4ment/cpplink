// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/string_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string_view>
#include <utility>
#include <vector>

namespace cpplink {
namespace {

// Rows are indexed by the shorter string, so this bound is on the shorter side.
constexpr size_t kStackRow = 256;

// The longest pattern Myers' bit-vector algorithm handles in one word. Longer
// values fall back to the banded DP below; on name and identifier columns they
// essentially do not occur.
constexpr size_t kMyersWidth = 64;

// The original formulation's prefix weight and the similarity it starts rewarding
// at. Named because the screen inside the metric has to apply exactly the metric,
// not an algebraic rearrangement of it.
constexpr double kPrefixScale = 0.1;
constexpr double kBoostThreshold = 0.7;

double ToRadians(double degrees) { return degrees * 3.14159265358979323846 / 180.0; }

// The prefix bonus, over a Jaro similarity already in hand. Non-decreasing in
// `jaro` on both sides of the boost threshold -- and non-decreasing in floating
// point too, since every step of it is -- which is what lets an upper bound on the
// similarity stand in for an upper bound on the whole metric.
double ApplyWinkler(std::string_view a, std::string_view b, double jaro,
                    double prefix_scale, double boost_threshold) {
    if (jaro < boost_threshold) return jaro;
    size_t prefix = 0;
    const size_t limit = std::min<size_t>({a.size(), b.size(), 4});
    while (prefix < limit && a[prefix] == b[prefix]) ++prefix;
    return jaro + static_cast<double>(prefix) * prefix_scale * (1.0 - jaro);
}

// The one instruction, asked for by name. `std::bitset<64>::count()` does not
// lower to it here -- it leaves a call to the generic bit-iterator count in the
// binary, which a profile of the scoring loop finds among the hot leaves.
int PopCount(uint64_t value) { return __builtin_popcountll(value); }

// Myers' bit-vector edit distance (1999). The DP's column of vertical deltas is
// carried in two words -- vp for +1, vn for -1 -- so a whole column costs a dozen
// integer ops instead of one pass per cell. O(|text|) word operations against the
// banded DP's O(|pattern| . |text|) cells.
int MyersDistance(std::string_view pattern, std::string_view text, int max_distance) {
    const size_t m = pattern.size();
    // Cleared again at the end, only for the characters this call set, so the cost
    // stays proportional to the pattern and not to the alphabet.
    thread_local uint64_t peq[256] = {0};
    for (size_t i = 0; i < m; ++i) {
        peq[static_cast<unsigned char>(pattern[i])] |= uint64_t{1} << i;
    }

    uint64_t vp = ~uint64_t{0};
    uint64_t vn = 0;
    int score = static_cast<int>(m);
    const uint64_t top = uint64_t{1} << (m - 1);
    size_t remaining = text.size();
    for (const char ch : text) {
        const uint64_t eq = peq[static_cast<unsigned char>(ch)];
        const uint64_t xv = eq | vn;
        const uint64_t xh = (((eq & vp) + vp) ^ vp) | eq;
        uint64_t ph = vn | ~(xh | vp);
        uint64_t mh = vp & xh;
        if (ph & top) ++score;
        if (mh & top) --score;
        ph = (ph << 1) | 1;
        mh <<= 1;
        vp = mh | ~(xv | ph);
        vn = ph & xv;
        --remaining;
        // Each remaining column moves the score by at most one, so once the floor
        // is past the bound the answer can only be "further than k".
        if (score - static_cast<int>(remaining) > max_distance) {
            score = max_distance + 1;
            break;
        }
    }

    for (size_t i = 0; i < m; ++i) peq[static_cast<unsigned char>(pattern[i])] = 0;
    return score > max_distance ? max_distance + 1 : score;
}

// The classic match window: max(|a|, |b|) / 2 - 1, floored at zero.
size_t MatchReach(size_t n, size_t m) {
    const size_t half = std::max<size_t>(n, m) / 2;
    return half > 0 ? half - 1 : 0;
}

// Jaro over two values that both fit a 64-bit word, which on the columns this
// runs against is every value: names and identifiers are not 65 characters long.
//
// The two match sets are carried in one word each rather than in two byte arrays,
// which is what the whole win is. It buys three things at once. The arrays no
// longer have to be cleared, so a call costs no memset -- two of them, on a
// function a profile of the scoring loop puts at well over half of all CPU. The
// inner scan over the window becomes a mask and a count-trailing-zeros, because
// the positions of each byte of `b` are indexed once up front exactly as Myers'
// algorithm indexes its pattern. And the transposition pass walks the two match
// sets bit by bit in step, so it needs no search for the next matched position.
//
// The greedy choice is unchanged -- the first unmatched equal position in the
// window, scanning up -- so this returns bit-for-bit what the byte-array path
// below returns.
//
// `screen` is a Jaro-Winkler threshold the caller is testing against, or zero to
// ask for the exact similarity. Once the match count is known it caps the
// similarity on its own, because a transposition can only ever cost bits; where
// the whole metric applied to that cap already falls short of the threshold, the
// cap is returned instead of finishing. The caller then reaches the same verdict
// from it that the exact value would have given, because the metric is
// non-decreasing in the similarity and the cap is never below the truth -- and
// because the test made here is the caller's own test, not an algebraic
// rearrangement of it that floating point could round the other way.
double JaroWords(std::string_view a, std::string_view b, double screen) {
    const size_t n = a.size();
    const size_t m = b.size();
    const size_t reach = MatchReach(n, m);

    // Cleared again at the end, only for the characters this call set, so the cost
    // stays proportional to the value and not to the alphabet.
    thread_local uint64_t positions[256] = {0};
    for (size_t j = 0; j < m; ++j) {
        positions[static_cast<unsigned char>(b[j])] |= uint64_t{1} << j;
    }

    uint64_t taken = 0;    // matched positions of b
    uint64_t claimed = 0;  // matched positions of a
    size_t matches = 0;
    for (size_t i = 0; i < n; ++i) {
        const size_t lo = i > reach ? i - reach : 0;
        // `lo` only ever rises, so once the window starts past the end of `b`
        // no later position can match either.
        if (lo >= m) break;
        const size_t hi = std::min(i + reach + 1, m);
        const size_t span = hi - lo;
        const uint64_t window = (span >= 64 ? ~uint64_t{0} : ((uint64_t{1} << span) - 1))
                                << lo;
        const uint64_t candidates =
            positions[static_cast<unsigned char>(a[i])] & window & ~taken;
        if (candidates == 0) continue;
        taken |= candidates & (~candidates + 1);  // the lowest set bit: first match
        claimed |= uint64_t{1} << i;
        ++matches;
    }

    for (size_t j = 0; j < m; ++j) positions[static_cast<unsigned char>(b[j])] = 0;
    if (matches == 0) return 0.0;

    const double matched = static_cast<double>(matches);
    // Transpositions only ever lower it, so the count alone caps the similarity.
    const double best =
        (matched / static_cast<double>(n) + matched / static_cast<double>(m) + 1.0) / 3.0;
    // Zero is the caller asking for the exact value, and `levels` is such a caller:
    // it bins the similarity rather than testing it, so it must not be charged the
    // prefix scan of a screen it did not ask for.
    if (screen > 0.0 &&
        ApplyWinkler(a, b, best, kPrefixScale, kBoostThreshold) < screen) {
        return best;
    }

    // Both words hold `matches` bits, so clearing the lowest of each in step pairs
    // the k-th matched character of `a` with the k-th of `b`, which is what a
    // transposition is counted on.
    size_t transpositions = 0;
    uint64_t left = claimed;
    uint64_t right = taken;
    while (left != 0) {
        if (a[__builtin_ctzll(left)] != b[__builtin_ctzll(right)]) ++transpositions;
        left &= left - 1;
        right &= right - 1;
    }
    transpositions /= 2;

    // Spelled exactly as the byte-array path below spells it: the two are one
    // metric, and a level decision can sit on the last bit of it.
    return (matched / static_cast<double>(n) + matched / static_cast<double>(m) +
            (matched - static_cast<double>(transpositions)) / matched) /
           3.0;
}

}  // namespace

int BoundedLevenshtein(std::string_view a, std::string_view b, int max_distance) {
    if (max_distance < 0) return 1;
    if (a.size() > b.size()) std::swap(a, b);
    const size_t n = a.size();
    const size_t m = b.size();

    // A length gap alone already exceeds the bound: the commonest early exit.
    if (m - n > static_cast<size_t>(max_distance)) return max_distance + 1;
    if (n == 0) return static_cast<int>(m);
    // The bit-parallel path carries a whole DP column in two words. It needs the
    // shorter side to fit one word, which on the columns this runs against it
    // always does; the banded DP stays for the values that do not.
    if (n <= kMyersWidth) return MyersDistance(a, b, max_distance);

    int stack_prev[kStackRow + 1];
    int stack_cur[kStackRow + 1];
    std::vector<int> heap_prev;
    std::vector<int> heap_cur;
    int* prev = stack_prev;
    int* cur = stack_cur;
    if (n > kStackRow) {
        heap_prev.resize(n + 1);
        heap_cur.resize(n + 1);
        prev = heap_prev.data();
        cur = heap_cur.data();
    }

    for (size_t i = 0; i <= n; ++i) prev[i] = static_cast<int>(i);
    for (size_t j = 1; j <= m; ++j) {
        cur[0] = static_cast<int>(j);
        int row_min = cur[0];
        for (size_t i = 1; i <= n; ++i) {
            const int substitution = prev[i - 1] + (a[i - 1] == b[j - 1] ? 0 : 1);
            cur[i] = std::min({prev[i] + 1, cur[i - 1] + 1, substitution});
            row_min = std::min(row_min, cur[i]);
        }
        // Every remaining row can only grow, so the bound can never be met again.
        if (row_min > max_distance) return max_distance + 1;
        std::swap(prev, cur);
    }
    return prev[n] > max_distance ? max_distance + 1 : prev[n];
}

namespace {

// Jaro, with the screen `JaroWords` documents. The byte-array path takes no
// screen: it runs only on values longer than a word, which the columns this serves
// do not hold, so there is nothing there worth the second spelling of the metric.
double JaroScreened(std::string_view a, std::string_view b, double screen) {
    if (a.empty() && b.empty()) return 1.0;
    if (a.empty() || b.empty()) return 0.0;

    const size_t n = a.size();
    const size_t m = b.size();
    // Both sides fit one word on every column this is asked about; the byte-array
    // path below stays for the values that do not.
    if (n <= kMyersWidth && m <= kMyersWidth) return JaroWords(a, b, screen);
    const size_t reach = MatchReach(n, m);

    char stack_a[kStackRow];
    char stack_b[kStackRow];
    std::vector<char> heap_a;
    std::vector<char> heap_b;
    char* matched_a = stack_a;
    char* matched_b = stack_b;
    if (n > kStackRow || m > kStackRow) {
        heap_a.assign(n, 0);
        heap_b.assign(m, 0);
        matched_a = heap_a.data();
        matched_b = heap_b.data();
    } else {
        std::memset(stack_a, 0, n);
        std::memset(stack_b, 0, m);
    }

    size_t matches = 0;
    for (size_t i = 0; i < n; ++i) {
        const size_t lo = i > reach ? i - reach : 0;
        const size_t hi = std::min(i + reach + 1, m);
        for (size_t j = lo; j < hi; ++j) {
            if (matched_b[j] != 0 || a[i] != b[j]) continue;
            matched_a[i] = 1;
            matched_b[j] = 1;
            ++matches;
            break;
        }
    }
    if (matches == 0) return 0.0;

    size_t transpositions = 0;
    size_t k = 0;
    for (size_t i = 0; i < n; ++i) {
        if (matched_a[i] == 0) continue;
        while (matched_b[k] == 0) ++k;
        if (a[i] != b[k]) ++transpositions;
        ++k;
    }
    transpositions /= 2;

    const double matched = static_cast<double>(matches);
    return (matched / static_cast<double>(n) + matched / static_cast<double>(m) +
            (matched - static_cast<double>(transpositions)) / matched) /
           3.0;
}

}  // namespace

double Jaro(std::string_view a, std::string_view b) { return JaroScreened(a, b, 0.0); }

double JaroWinkler(std::string_view a, std::string_view b, double prefix_scale,
                   double boost_threshold) {
    return ApplyWinkler(a, b, Jaro(a, b), prefix_scale, boost_threshold);
}

double JaroWinklerUpperBound(uint64_t mask_a, uint32_t len_a, uint64_t mask_b,
                             uint32_t len_b) {
    if (len_a == 0 || len_b == 0) return len_a == len_b ? 1.0 : 0.0;
    // Every bit set in one mask and clear in the other names at least one
    // character of that string with no counterpart at all in the other.
    const int deficit_a = PopCount(mask_a & ~mask_b);
    const int deficit_b = PopCount(mask_b & ~mask_a);
    const int reach_a = static_cast<int>(len_a) - deficit_a;
    const int reach_b = static_cast<int>(len_b) - deficit_b;
    const int matches = std::min(reach_a, reach_b);
    if (matches <= 0) return 0.0;
    const double m = static_cast<double>(matches);
    const double jaro =
        (m / static_cast<double>(len_a) + m / static_cast<double>(len_b) + 1.0) / 3.0;
    return 0.4 + 0.6 * jaro;
}

int LevenshteinLowerBound(uint64_t mask_a, uint32_t len_a, uint64_t mask_b,
                          uint32_t len_b) {
    const int gap = std::abs(static_cast<int>(len_a) - static_cast<int>(len_b));
    const int deficit = std::max(PopCount(mask_a & ~mask_b), PopCount(mask_b & ~mask_a));
    return std::max(gap, deficit);
}

bool JaroWinklerAtLeast(std::string_view a, std::string_view b, double threshold) {
    const size_t shorter = std::min(a.size(), b.size());
    const size_t longer = std::max(a.size(), b.size());
    if (longer == 0) return threshold <= 1.0;
    const double ratio = static_cast<double>(shorter) / static_cast<double>(longer);
    const double jaro_bound = (2.0 + ratio) / 3.0;
    if (0.4 + 0.6 * jaro_bound < threshold) return false;
    // The threshold goes down into the metric, so the match count alone can end the
    // call before a transposition is counted. The count is what the signature bound
    // could only bound, since a presence mask knows nothing of position, of
    // multiplicity, or of the match window. Where it does end the call the value
    // that comes back is an over-estimate the metric has already been shown to
    // score below the threshold, so the same test decides both cases and there is
    // nothing here that can disagree with `JaroWinkler(a, b) >= threshold`.
    const double jaro = JaroScreened(a, b, threshold);
    return ApplyWinkler(a, b, jaro, kPrefixScale, kBoostThreshold) >= threshold;
}

double HaversineKm(double lat_a, double lon_a, double lat_b, double lon_b) {
    constexpr double kEarthRadiusKm =
        kKmPerDegreeLatitude * 180.0 / 3.14159265358979323846;
    const double delta_lat = ToRadians(lat_b - lat_a);
    const double delta_lon = ToRadians(lon_b - lon_a);
    const double sin_lat = std::sin(delta_lat / 2.0);
    const double sin_lon = std::sin(delta_lon / 2.0);
    const double h = sin_lat * sin_lat + std::cos(ToRadians(lat_a)) *
                                             std::cos(ToRadians(lat_b)) * sin_lon *
                                             sin_lon;
    return 2.0 * kEarthRadiusKm * std::asin(std::min(1.0, std::sqrt(h)));
}

}  // namespace cpplink
