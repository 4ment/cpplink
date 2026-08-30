// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/dictionary.hpp"

#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

TEST(DictionaryTest, EqualValuesShareAnId) {
    cpplink::Dictionary dict;
    const uint32_t a = dict.Intern("smith");
    const uint32_t b = dict.Intern("jones");
    EXPECT_EQ(dict.Intern("smith"), a);
    EXPECT_NE(a, b);
    EXPECT_EQ(dict.Size(), 2u);
}

TEST(DictionaryTest, IdsAreDenseAndValuesRecoverable) {
    cpplink::Dictionary dict;
    const std::vector<std::string> values = {"alpha", "beta", "gamma", "delta"};
    for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ(dict.Intern(values[i]), static_cast<uint32_t>(i));
    }
    for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ(dict.Value(static_cast<uint32_t>(i)), values[i]);
    }
}

// Values are stored in chunks so that the string_views held by the index stay
// valid as the arena grows. Interning past a chunk boundary must not corrupt
// values that were stored earlier.
TEST(DictionaryTest, ValuesSurviveArenaGrowth) {
    cpplink::Dictionary dict;
    std::vector<uint32_t> ids;
    for (int i = 0; i < 60000; ++i) {
        ids.push_back(dict.Intern("value_" + std::to_string(i)));
    }
    EXPECT_EQ(dict.Size(), 60000u);
    for (int i = 0; i < 60000; ++i) {
        EXPECT_EQ(dict.Value(ids[i]), "value_" + std::to_string(i));
        EXPECT_EQ(dict.Intern("value_" + std::to_string(i)), ids[i]);
    }
}

TEST(DictionaryTest, HandlesValuesLargerThanAChunk) {
    cpplink::Dictionary dict;
    const std::string huge(3u << 20, 'x');
    const uint32_t id = dict.Intern(huge);
    EXPECT_EQ(dict.Value(id), huge);
    EXPECT_EQ(dict.Intern(huge), id);
}

TEST(DictionaryTest, EmptyStringIsAValueNotANull) {
    cpplink::Dictionary dict;
    const uint32_t id = dict.Intern("");
    EXPECT_NE(id, cpplink::kNullId);
    EXPECT_EQ(dict.Value(id), "");
}

TEST(DictionaryTest, ReleasingTheIndexKeepsValuesReadable) {
    cpplink::Dictionary dict;
    const uint32_t id = dict.Intern("keep me");
    const uint64_t before = dict.BytesUsed();
    dict.ReleaseIndex();
    EXPECT_EQ(dict.Value(id), "keep me");
    EXPECT_LT(dict.BytesUsed(), before);
}

}  // namespace
