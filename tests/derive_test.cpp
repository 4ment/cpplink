// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/derive.hpp"

#include <memory>
#include <set>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "cpplink/blocking.hpp"
#include "cpplink/comparison.hpp"
#include "cpplink/estimate.hpp"
#include "cpplink/model.hpp"
#include "cpplink/profile.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"

namespace {

std::string Derived(const std::vector<cpplink::Transform>& chain,
                    const std::string& value) {
    std::string out;
    cpplink::ApplyTransforms(chain, value, &out);
    return out;
}

std::string DerivedDate(const std::vector<cpplink::Transform>& chain, int32_t days) {
    std::string out;
    cpplink::ApplyDateTransforms(chain, days, &out);
    return out;
}

// Soundex is checked against the reference keys the algorithm is usually stated
// with, because the two rules it is easy to get wrong -- h and w being
// transparent, a vowel not being -- are exactly what those examples separate.
TEST(DeriveTest, SoundexMatchesTheReferenceKeys) {
    const std::vector<cpplink::Transform> chain = {cpplink::Transform::kSoundex};
    EXPECT_EQ(Derived(chain, "Robert"), "R163");
    EXPECT_EQ(Derived(chain, "Rupert"), "R163");
    EXPECT_EQ(Derived(chain, "Rubin"), "R150");
    EXPECT_EQ(Derived(chain, "Ashcraft"), "A261");  // h is transparent, not a break
    EXPECT_EQ(Derived(chain, "Ashcroft"), "A261");
    EXPECT_EQ(Derived(chain, "Tymczak"), "T522");  // a vowel does break a repeat
    EXPECT_EQ(Derived(chain, "Pfister"), "P236");
    EXPECT_EQ(Derived(chain, "Honeyman"), "H555");
    EXPECT_EQ(Derived(chain, "Jackson"), "J250");
    EXPECT_EQ(Derived(chain, "Washington"), "W252");
    // Case and surrounding punctuation do not reach the key.
    EXPECT_EQ(Derived(chain, "o'brien"), Derived(chain, "OBrien"));
    // Nothing to key on is a missing value rather than an empty one.
    EXPECT_EQ(Derived(chain, "12345"), "");
    EXPECT_EQ(Derived(chain, ""), "");
}

TEST(DeriveTest, NormalizeKeepsSeparatorsAsSpaces) {
    const std::vector<cpplink::Transform> chain = {cpplink::Transform::kNormalize};
    EXPECT_EQ(Derived(chain, "  O'BRIEN-Smith  "), "o brien smith");
    EXPECT_EQ(Derived(chain, "Name-12"), Derived(chain, "name_12"));
    EXPECT_EQ(Derived(chain, "SW1A 1AA"), "sw1a 1aa");
    EXPECT_EQ(Derived(chain, "--,--"), "");
    // Bytes above ASCII survive: lowercasing them needs a locale this project
    // does not carry, and dropping them would erase the name rather than key it.
    EXPECT_EQ(Derived(chain, "Ren\xC3\xA9"), "ren\xC3\xA9");
}

TEST(DeriveTest, SortedTokensMakeOrderIrrelevant) {
    const std::vector<cpplink::Transform> chain = {cpplink::Transform::kSortedTokens};
    EXPECT_EQ(Derived(chain, "john smith"), "john smith");
    EXPECT_EQ(Derived(chain, "smith john"), "john smith");
    EXPECT_EQ(Derived(chain, "  smith   john "), "john smith");
}

// The composition that motivates a chain at all: a two-token name written either
// way round, in either case, with either separator, is one key.
TEST(DeriveTest, AChainAppliesInOrder) {
    const std::vector<cpplink::Transform> chain = {cpplink::Transform::kNormalize,
                                                   cpplink::Transform::kSortedTokens};
    EXPECT_EQ(Derived(chain, "SMITH, John"), "john smith");
    EXPECT_EQ(Derived(chain, "John Smith"), "john smith");
    EXPECT_EQ(Derived(chain, "john-smith"), "john smith");
}

TEST(DeriveTest, DatePartsSplitTheCivilDate) {
    // 1987-03-17 is 6284 days after the epoch, and the rest are the cases a
    // civil-date conversion is wrong about when it is wrong: both kinds of
    // century, and dates before the epoch, where the era arithmetic changes sign.
    EXPECT_EQ(DerivedDate({cpplink::Transform::kYear}, 6284), "1987");
    EXPECT_EQ(DerivedDate({cpplink::Transform::kMonth}, 6284), "03");
    EXPECT_EQ(DerivedDate({cpplink::Transform::kDay}, 6284), "17");
    EXPECT_EQ(DerivedDate({cpplink::Transform::kYearMonth}, 6284), "1987-03");
    EXPECT_EQ(DerivedDate({cpplink::Transform::kYearMonth}, 0), "1970-01");
    EXPECT_EQ(DerivedDate({cpplink::Transform::kYearMonth}, 11016), "2000-02");
    EXPECT_EQ(DerivedDate({cpplink::Transform::kDay}, 11016), "29");  // a leap day
    EXPECT_EQ(DerivedDate({cpplink::Transform::kDay}, 19782), "29");
    EXPECT_EQ(DerivedDate({cpplink::Transform::kYearMonth}, 47482), "2100-01");
    EXPECT_EQ(DerivedDate({cpplink::Transform::kYearMonth}, -25508), "1900-03");
    EXPECT_EQ(DerivedDate({cpplink::Transform::kDay}, -1), "31");
    EXPECT_EQ(DerivedDate({cpplink::Transform::kYear}, -1), "1969");
    // A missing date derives nothing, which the store reads as a missing value.
    EXPECT_EQ(DerivedDate({cpplink::Transform::kYear}, cpplink::kNullDate), "");
}

// ---------------------------------------------------------------------------
// The schema half: what a derivation may say, and what it may not.

cpplink::Schema Parse(const std::string& json) {
    cpplink::Schema schema;
    std::string error;
    EXPECT_TRUE(cpplink::ParseSchema(json, &schema, &error)) << error;
    return schema;
}

std::string Refuse(const std::string& json) {
    cpplink::Schema schema;
    std::string error;
    EXPECT_FALSE(cpplink::ParseSchema(json, &schema, &error)) << "was accepted";
    return error;
}

TEST(DeriveSchemaTest, ADerivedColumnTakesItsTypeFromItsTransforms) {
    const cpplink::Schema schema = Parse(R"({
      "columns": [
        {"name": "surname", "type": "string"},
        {"name": "dob", "type": "date"},
        {"name": "surname_key", "derive": {"from": "surname", "transform": "soundex"}},
        {"name": "birth_year", "derive": {"from": "dob", "transform": "year"}}
      ]
    })");
    ASSERT_EQ(schema.columns.size(), 4u);
    EXPECT_FALSE(schema.columns[0].IsDerived());
    EXPECT_TRUE(schema.columns[2].IsDerived());
    EXPECT_EQ(schema.columns[2].derive.from, "surname");
    EXPECT_EQ(schema.columns[2].type, cpplink::ColumnType::kString);
    // A date part is a string, so it is interned and its exact level is an
    // integer equality like any other.
    EXPECT_EQ(schema.columns[3].type, cpplink::ColumnType::kString);
    EXPECT_EQ(schema.columns[2].derive.Describe(), "surname -> soundex");
}

// A derivation names a column, not a position, so the order of the two is free.
TEST(DeriveSchemaTest, ADerivationMayNameAColumnDeclaredLater) {
    const cpplink::Schema schema = Parse(R"({
      "columns": [
        {"name": "surname_key", "derive": {"from": "surname", "transform": "soundex"}},
        {"name": "surname", "type": "string"}
      ]
    })");
    EXPECT_TRUE(schema.columns[0].IsDerived());
}

TEST(DeriveSchemaTest, RefusesADerivationItCannotRun) {
    EXPECT_NE(Refuse(R"({"columns": [
        {"name": "a", "type": "string"},
        {"name": "b", "derive": {"from": "a", "transform": "metaphone"}}]})")
                  .find("unknown transform"),
              std::string::npos);
    EXPECT_NE(Refuse(R"({"columns": [
        {"name": "b", "derive": {"from": "missing", "transform": "soundex"}}]})")
                  .find("not declared"),
              std::string::npos);
    EXPECT_NE(Refuse(R"({"columns": [
        {"name": "b", "derive": {"from": "b", "transform": "soundex"}}]})")
                  .find("derives from itself"),
              std::string::npos);
    // A date transform on a string, and a string transform on a date: both are
    // caught before a file is opened rather than on the first row.
    EXPECT_NE(Refuse(R"({"columns": [
        {"name": "a", "type": "string"},
        {"name": "b", "derive": {"from": "a", "transform": "year"}}]})")
                  .find("which it cannot read"),
              std::string::npos);
    EXPECT_NE(Refuse(R"({"columns": [
        {"name": "a", "type": "date"},
        {"name": "b", "derive": {"from": "a", "transform": "soundex"}}]})")
                  .find("which it cannot read"),
              std::string::npos);
    // The same check catches the second transform of a chain.
    EXPECT_NE(Refuse(R"({"columns": [
        {"name": "a", "type": "string"},
        {"name": "b", "derive": {"from": "a", "transform": ["soundex", "year"]}}]})")
                  .find("which it cannot read"),
              std::string::npos);
    // Composition belongs in the transform list, not in a chain of columns.
    EXPECT_NE(Refuse(R"({"columns": [
        {"name": "a", "type": "string"},
        {"name": "b", "derive": {"from": "a", "transform": "normalize"}},
        {"name": "c", "derive": {"from": "b", "transform": "soundex"}}]})")
                  .find("itself derived"),
              std::string::npos);
    // And a declared type the transforms do not produce is a contradiction.
    EXPECT_NE(Refuse(R"({"columns": [
        {"name": "a", "type": "date"},
        {"name": "b", "type": "date", "derive": {"from": "a", "transform": "year"}}]})")
                  .find("but its transforms produce"),
              std::string::npos);
}

TEST(DeriveSchemaTest, SameSourceReadsTheDeclaredDependency) {
    const cpplink::Schema schema = Parse(R"({
      "columns": [
        {"name": "surname", "type": "string"},
        {"name": "town", "type": "string"},
        {"name": "surname_key", "derive": {"from": "surname", "transform": "soundex"}},
        {"name": "surname_flat", "derive": {"from": "surname", "transform": "normalize"}}
      ]
    })");
    EXPECT_TRUE(SameSource(schema, "surname", "surname_key"));
    EXPECT_TRUE(SameSource(schema, "surname_key", "surname"));
    // Two keys off the same column are the same evidence as much as either is.
    EXPECT_TRUE(SameSource(schema, "surname_key", "surname_flat"));
    EXPECT_FALSE(SameSource(schema, "surname", "town"));
    EXPECT_FALSE(SameSource(schema, "surname_key", "town"));
}

// ---------------------------------------------------------------------------
// The store half: a derived column is built once per distinct value, and is a
// column like any other by the time anything downstream sees it.

constexpr uint64_t kRecords = 4000;
constexpr uint64_t kPairs = 200;
constexpr uint32_t kValues = 1000;
constexpr uint32_t kTowns = 50;
constexpr uint32_t kJobs = 40;
constexpr uint32_t kExtras = 60;

uint32_t Draw(uint64_t row, uint64_t salt, uint32_t values) {
    uint64_t hash = row * 0x9E3779B97F4A7C15ull + salt;
    hash ^= hash >> 29;
    hash *= 0xBF58476D1CE4E5B9ull;
    hash ^= hash >> 32;
    return static_cast<uint32_t>(hash % values);
}

bool Agrees(uint64_t pair, uint64_t salt, uint32_t percent) {
    return Draw(pair, salt, 100) < percent;
}

// `name` is written one of two ways -- "Name-12" or "name_12" -- and `name_key`
// normalises the two together. Every planted pair is written the other way round
// from its partner, so the raw column agrees on none of them and the derived
// column agrees on all of them: this is the shape a derived column exists for.
const char* const kSchemaJson = R"({
  "unique_id": "id",
  "columns": [
    {"name": "name", "type": "string"},
    {"name": "name_key", "derive": {"from": "name", "transform": "normalize"}},
    {"name": "town", "type": "string"},
    {"name": "job", "type": "string"},
    {"name": "extra", "type": "string"}
  ],
  "comparisons": [
    {"name": "name", "columns": ["name"],
     "levels": [{"type": "null"}, {"type": "exact"}, {"type": "else"}]},
    {"name": "name_key", "columns": ["name_key"],
     "levels": [{"type": "null"}, {"type": "exact"}, {"type": "else"}]},
    {"name": "town", "columns": ["town"],
     "levels": [{"type": "null"}, {"type": "exact"}, {"type": "else"}]},
    {"name": "job", "columns": ["job"],
     "levels": [{"type": "null"}, {"type": "exact"}, {"type": "else"}]},
    {"name": "extra", "columns": ["extra"],
     "levels": [{"type": "null"}, {"type": "exact"}, {"type": "else"}]}
  ],
  "blocking": [
    {"type": "exact_value", "column": "name_key"},
    {"type": "exact_value", "column": "town"}
  ]
})";

class DeriveFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        std::string error;
        ASSERT_TRUE(cpplink::ParseSchema(kSchemaJson, &schema_, &error)) << error;
        store_ = std::make_unique<cpplink::RecordStore>(schema_);

        std::vector<uint32_t> value(kRecords);
        std::vector<uint32_t> town(kRecords);
        std::vector<uint32_t> job(kRecords);
        std::vector<uint32_t> extra(kRecords);
        for (uint64_t row = 0; row < kRecords; ++row) {
            value[row] = Draw(row, 11, kValues);
            town[row] = Draw(row, 22, kTowns);
            job[row] = Draw(row, 33, kJobs);
            extra[row] = Draw(row, 44, kExtras);
        }
        for (uint64_t i = 0; i < kPairs; ++i) {
            const uint64_t a = 2 * i;
            const uint64_t b = 2 * i + 1;
            value[b] = value[a];
            town[b] = Agrees(i, 55, 90) ? town[a] : (town[a] + 1) % kTowns;
            job[b] = Agrees(i, 66, 70) ? job[a] : (job[a] + 1) % kJobs;
            extra[b] = Agrees(i, 77, 60) ? extra[a] : (extra[a] + 1) % kExtras;
        }

        auto& name = std::get<cpplink::StringColumn>(store_->mutable_column(0));
        auto& towns = std::get<cpplink::StringColumn>(store_->mutable_column(2));
        auto& jobs = std::get<cpplink::StringColumn>(store_->mutable_column(3));
        auto& extras = std::get<cpplink::StringColumn>(store_->mutable_column(4));
        for (uint64_t row = 0; row < kRecords; ++row) {
            const std::string digits = std::to_string(value[row]);
            const std::string written =
                row % 2 == 0 ? "Name-" + digits : "name_" + digits;
            name.ids.push_back(name.dict.Intern(written));
            towns.ids.push_back(towns.dict.Intern("t" + std::to_string(town[row])));
            jobs.ids.push_back(jobs.dict.Intern("j" + std::to_string(job[row])));
            extras.ids.push_back(extras.dict.Intern("e" + std::to_string(extra[row])));
            store_->mutable_ids().Append("r" + std::to_string(row));
        }
        store_->set_num_records(kRecords);
        store_->Finalize();
    }

    const cpplink::StringColumn& Column(size_t index) const {
        return std::get<cpplink::StringColumn>(store_->column(index));
    }

    cpplink::Schema schema_;
    std::unique_ptr<cpplink::RecordStore> store_;
};

// The point of the whole feature in one assertion: the raw column agrees on none
// of the planted pairs and the key agrees on all of them.
TEST_F(DeriveFixture, TheKeyAgreesWhereTheRawColumnDoesNot) {
    const cpplink::StringColumn& name = Column(0);
    const cpplink::StringColumn& key = Column(1);
    ASSERT_EQ(key.ids.size(), kRecords);
    for (uint64_t i = 0; i < kPairs; ++i) {
        const uint64_t a = 2 * i;
        const uint64_t b = 2 * i + 1;
        EXPECT_NE(name.ids[a], name.ids[b]) << "row " << a;
        EXPECT_EQ(key.ids[a], key.ids[b]) << "row " << a;
    }
    // The derived dictionary is exactly the set of normalised values: no key the
    // rows do not hold, and none of them twice.
    std::set<std::string> raw;
    std::set<std::string> keys;
    for (uint64_t row = 0; row < kRecords; ++row) {
        const std::string value(name.dict.Value(name.ids[row]));
        raw.insert(value);
        keys.insert(Derived({cpplink::Transform::kNormalize}, value));
        EXPECT_EQ(key.dict.Value(key.ids[row]),
                  Derived({cpplink::Transform::kNormalize}, value));
    }
    EXPECT_EQ(name.dict.Size(), raw.size());
    EXPECT_EQ(key.dict.Size(), keys.size());
    EXPECT_LT(keys.size(), raw.size());
}

// Finalize counts it like anything else, which is what makes it available to
// blocking, to term-frequency adjustment and to the profile with no further
// plumbing.
TEST_F(DeriveFixture, TheDerivedColumnCarriesTermFrequencies) {
    const cpplink::StringColumn& key = Column(1);
    ASSERT_EQ(key.tf.size(), key.dict.Size());
    uint64_t counted = 0;
    for (const uint32_t count : key.tf) counted += count;
    EXPECT_EQ(counted, kRecords);
    EXPECT_EQ(store_->DistinctValues(1), key.dict.Size());
    EXPECT_LT(store_->DistinctValues(1), store_->DistinctValues(0));
    EXPECT_EQ(store_->NullCount(1), 0u);
}

// A value that derives to nothing is missing, not empty. Without this every row
// whose name is punctuation would share one key and block together.
TEST(DeriveTest, AnEmptyDerivationIsANullValue) {
    const char* const json = R"({
      "columns": [
        {"name": "name", "type": "string"},
        {"name": "key", "derive": {"from": "name", "transform": "soundex"}}
      ]
    })";
    cpplink::Schema schema;
    std::string error;
    ASSERT_TRUE(cpplink::ParseSchema(json, &schema, &error)) << error;
    cpplink::RecordStore store(schema);
    auto& name = std::get<cpplink::StringColumn>(store.mutable_column(0));
    name.ids.push_back(name.dict.Intern("Smith"));
    name.ids.push_back(name.dict.Intern("12345"));  // derives to nothing
    name.ids.push_back(cpplink::kNullId);           // missing to begin with
    name.ids.push_back(name.dict.Intern("Smyth"));
    store.set_num_records(4);
    store.Finalize();

    const auto& key = std::get<cpplink::StringColumn>(store.column(1));
    ASSERT_EQ(key.ids.size(), 4u);
    EXPECT_EQ(key.ids[1], cpplink::kNullId);
    EXPECT_EQ(key.ids[2], cpplink::kNullId);
    EXPECT_EQ(key.ids[0], key.ids[3]);  // Smith and Smyth are one Soundex key
    EXPECT_EQ(store.NullCount(1), 2u);
    EXPECT_EQ(store.DistinctValues(1), 1u);
}

// ---------------------------------------------------------------------------
// The estimation half: a derived column and its source are one piece of evidence.

// Why the tie cannot be left to the pairwise pass. u is the rate two non-matching
// rows collide, and the joint of a column with a key derived from it is rare
// enough that the file's own duplicates swamp it, so the measurement is refused
// exactly where it would have mattered. The derivation is declared, so it does
// not need measuring.
TEST_F(DeriveFixture, ThePairwisePassCannotResolveTheDerivedPair) {
    cpplink::ProfileOptions options;
    options.anchors = false;
    options.threads = 1;
    const cpplink::ProfileReport report =
        BuildProfile(*store_, cpplink::PairMode::kAll, options);
    ASSERT_TRUE(report.walked);
    bool seen = false;
    for (const cpplink::ColumnPairProfile& pair : report.pairs) {
        const bool here = (pair.left_name == "name" && pair.right_name == "name_key") ||
                          (pair.left_name == "name_key" && pair.right_name == "name");
        if (!here) continue;
        seen = true;
        EXPECT_FALSE(pair.resolved);
        EXPECT_LT(pair.containment, 0.90);
    }
    EXPECT_TRUE(seen);
}

// So the session that blocks on the key holds out the column it came from, and a
// column with no such relation is held out of nothing.
TEST_F(DeriveFixture, ASessionHoldsOutTheColumnItsKeyCameFrom) {
    cpplink::ComparisonSet comparisons;
    cpplink::BlockingPlan plan;
    std::string error;
    ASSERT_TRUE(comparisons.Bind(schema_, *store_, &error)) << error;
    ASSERT_TRUE(plan.Build(schema_, *store_, cpplink::PairMode::kAll, &error)) << error;

    cpplink::EstimateOptions options;
    options.threads = 1;
    cpplink::Model model;
    cpplink::EstimateReport report;
    ASSERT_TRUE(Estimate(*store_, comparisons, plan, options, &model, &report, &error))
        << error;

    bool seen = false;
    for (const cpplink::SessionReport& session : report.sessions) {
        if (session.column == "name_key") {
            seen = true;
            ASSERT_EQ(session.excluded.size(), 1u);
            EXPECT_EQ(session.excluded[0], "name_key");
            ASSERT_EQ(session.tied.size(), 1u);
            EXPECT_EQ(session.tied[0], "name");
        } else if (session.column == "town") {
            EXPECT_TRUE(session.tied.empty()) << "tied out " << session.tied.front();
        }
    }
    EXPECT_TRUE(seen) << "no session blocked on the derived column";
}

}  // namespace
