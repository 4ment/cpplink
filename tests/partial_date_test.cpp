// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

// The partial_date level: dates of birth written DDMMYYYY with 00 for an unknown
// day or month, one per record or a list of aliases per record. Each level reads
// one date of each record whole, so the parts of different aliases never combine
// into a date neither record holds, and two partial dates are never a full match.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "cpplink/comparison.hpp"
#include "cpplink/init.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"

namespace {

constexpr const char* kSchema = R"({
  "unique_id": "id",
  "columns": [
    {"name": "dobs", "type": "string_list"},
    {"name": "dob", "type": "string"}
  ],
  "comparisons": [
    {"name": "dobs", "columns": ["dobs"], "levels": [
      {"type": "null"},
      {"type": "partial_date", "match": "full"},
      {"type": "partial_date", "match": "typo"},
      {"type": "partial_date", "match": "compatible_month"},
      {"type": "partial_date", "match": "compatible_year"},
      {"type": "else"}]},
    {"name": "dob", "columns": ["dob"], "levels": [
      {"type": "null"},
      {"type": "partial_date", "match": "full"},
      {"type": "partial_date", "match": "typo"},
      {"type": "partial_date", "match": "compatible_month"},
      {"type": "partial_date", "match": "compatible_year"},
      {"type": "else"}]}
  ]
})";

constexpr uint8_t kNull = 0;
constexpr uint8_t kFull = 1;
constexpr uint8_t kTypo = 2;
constexpr uint8_t kMonth = 3;
constexpr uint8_t kYear = 4;
constexpr uint8_t kElse = 5;

// A store whose rows are the given lists of dates; the scalar column holds each
// row's first date, or nothing for an empty list.
class Dates {
   public:
    explicit Dates(const std::vector<std::vector<std::string>>& rows) {
        std::string error;
        EXPECT_TRUE(cpplink::ParseSchema(kSchema, &schema_, &error)) << error;
        store_ = std::make_unique<cpplink::RecordStore>(schema_);
        auto& lists = std::get<cpplink::StringListColumn>(store_->mutable_column(0));
        auto& scalar = std::get<cpplink::StringColumn>(store_->mutable_column(1));
        lists.offsets.push_back(0);
        for (size_t row = 0; row < rows.size(); ++row) {
            std::vector<uint32_t> cell;
            for (const std::string& date : rows[row])
                cell.push_back(lists.dict.Intern(date));
            std::sort(cell.begin(), cell.end());
            cell.erase(std::unique(cell.begin(), cell.end()), cell.end());
            lists.ids.insert(lists.ids.end(), cell.begin(), cell.end());
            lists.offsets.push_back(lists.ids.size());
            scalar.ids.push_back(rows[row].empty()
                                     ? cpplink::kNullId
                                     : scalar.dict.Intern(rows[row].front()));
            store_->mutable_ids().Append("r" + std::to_string(row));
        }
        store_->set_num_records(rows.size());
        store_->Finalize();
        bound_ = comparisons_.Bind(schema_, *store_, &error_);
    }

    bool bound() const { return bound_; }
    const std::string& error() const { return error_; }
    uint8_t List(uint64_t a, uint64_t b) const {
        return comparisons_.EvaluateOne(0, a, b);
    }
    uint8_t Scalar(uint64_t a, uint64_t b) const {
        return comparisons_.EvaluateOne(1, a, b);
    }
    const cpplink::ComparisonSet& comparisons() const { return comparisons_; }

   private:
    cpplink::Schema schema_;
    std::unique_ptr<cpplink::RecordStore> store_;
    cpplink::ComparisonSet comparisons_;
    bool bound_ = false;
    std::string error_;
};

// One date against one date, through both the list and the scalar column, in both
// orders: the level is symmetric and must not depend on which side is which.
uint8_t Pair(const std::string& left, const std::string& right) {
    Dates dates({{left}, {right}});
    EXPECT_TRUE(dates.bound()) << dates.error();
    const uint8_t level = dates.List(0, 1);
    EXPECT_EQ(dates.List(1, 0), level) << left << " " << right;
    EXPECT_EQ(dates.Scalar(0, 1), level) << left << " " << right;
    EXPECT_EQ(dates.Scalar(1, 0), level) << left << " " << right;
    return level;
}

TEST(PartialDate, FullNeedsTwoCompleteIdenticalDates) {
    EXPECT_EQ(Pair("25111978", "25111978"), kFull);
    // Two identical partial dates share what they know and nothing more.
    EXPECT_EQ(Pair("00001978", "00001978"), kYear);
    EXPECT_EQ(Pair("00111978", "00111978"), kMonth);
    EXPECT_EQ(Pair("25001978", "25001978"), kYear);
}

TEST(PartialDate, TypoIsOneKeyingErrorBetweenCompleteDates) {
    EXPECT_EQ(Pair("25111978", "25111979"), kTypo);  // one digit
    EXPECT_EQ(Pair("25111978", "26111978"), kTypo);
    EXPECT_EQ(Pair("12111978", "21111978"), kTypo);  // adjacent digits swapped
    EXPECT_EQ(Pair("25111978", "25111987"), kTypo);
    EXPECT_EQ(Pair("05111978", "11051978"), kTypo);  // day and month swapped
    // Two digits wrong, or two swapped that are not adjacent, is not one mistake.
    EXPECT_EQ(Pair("25111978", "25121979"), kElse);
    EXPECT_EQ(Pair("25111978", "15121978"), kElse);
    EXPECT_EQ(Pair("21111978", "11112978"), kElse);
    // A typo needs both dates complete.
    EXPECT_EQ(Pair("00111978", "00111979"), kElse);
}

TEST(PartialDate, CompatibleReadsOnlyThePartsBothSidesKnow) {
    EXPECT_EQ(Pair("00001978", "25111978"), kYear);
    EXPECT_EQ(Pair("00111978", "25111978"), kMonth);
    EXPECT_EQ(Pair("25001978", "25111978"), kYear);
    EXPECT_EQ(Pair("00111978", "25001978"), kYear);
    EXPECT_EQ(Pair("00111978", "00121978"), kElse);  // the months contradict
    EXPECT_EQ(Pair("25001978", "24111978"), kElse);  // the days contradict
    EXPECT_EQ(Pair("00001978", "00001979"), kElse);  // the years do
    // The same year with different known parts is a disagreement, not a level.
    EXPECT_EQ(Pair("25111978", "03021978"), kElse);
}

TEST(PartialDate, BestAliasPairWinsAndAliasesNeverCombine) {
    Dates dates({
        {"25111978", "03021965"},  // 0
        {"00001965", "11112000"},  // 1: shares 1965 with row 0's second alias
        {"25111978"},              // 2
        {"25021965", "01111978"},  // 3: holds every part of 25111978, in two dates
        {"01011990", "25111978"},  // 4: holds row 2's date as its second alias
        {"25111979", "00111978"},  // 5: a typo of row 2 and a compatible month
        {},                        // 6: no dates at all
    });
    ASSERT_TRUE(dates.bound()) << dates.error();
    EXPECT_EQ(dates.List(0, 1), kYear);
    EXPECT_EQ(dates.List(2, 3), kElse);
    EXPECT_EQ(dates.List(3, 2), kElse);
    EXPECT_EQ(dates.List(2, 4), kFull);
    EXPECT_EQ(dates.List(2, 5), kTypo);  // the stronger of the two pairs
    EXPECT_EQ(dates.List(0, 6), kNull);
    EXPECT_EQ(dates.Scalar(0, 6), kNull);
}

TEST(PartialDate, ThePossibleTestIsThePredicate) {
    Dates dates({{"25111978", "00001965"},
                 {"00001978"},
                 {"25111979"},
                 {"00111978", "12121912"},
                 {"21121912"},
                 {}});
    ASSERT_TRUE(dates.bound()) << dates.error();
    for (uint64_t a = 0; a < 6; ++a) {
        for (uint64_t b = 0; b < 6; ++b) {
            for (size_t c = 0; c < 2; ++c) {
                const uint8_t level = dates.comparisons().EvaluateOne(c, a, b);
                // Every level above the one that fired is impossible, and that
                // one is possible: the bound is exact for this level type.
                for (size_t l = 1; l < level; ++l) {
                    EXPECT_FALSE(dates.comparisons().LevelPossible(c, l, a, b));
                }
                EXPECT_TRUE(dates.comparisons().LevelPossible(c, level, a, b));
            }
        }
    }
}

TEST(PartialDate, ValueThatIsNotADateFailsTheBind) {
    const std::vector<std::pair<std::string, std::string>> bad = {
        {"2511978", "7 characters"}, {"251119780", "9 characters"},
        {"25-11-78", "not a digit"}, {"25110000", "year is 0000"},
        {"25131978", "month is 13"}, {"32111978", "day is 32"},
    };
    for (const auto& [value, reason] : bad) {
        Dates dates({{"25111978"}, {value}});
        EXPECT_FALSE(dates.bound()) << value;
        EXPECT_NE(dates.error().find(value), std::string::npos) << dates.error();
        EXPECT_NE(dates.error().find(reason), std::string::npos) << dates.error();
    }
}

TEST(PartialDate, SchemaChecksTheMatchAndTheColumn) {
    cpplink::Schema schema;
    std::string error;
    const std::string columns =
        R"("columns": [{"name": "dob", "type": "string"},
                       {"name": "other", "type": "string"},
                       {"name": "day", "type": "date"}], )";
    auto parse = [&](const std::string& comparison) {
        error.clear();
        return cpplink::ParseSchema(
            "{" + columns + R"("comparisons": [)" + comparison + "]}", &schema, &error);
    };
    EXPECT_FALSE(parse(R"({"columns": ["dob"], "levels": [
        {"type": "partial_date"}, {"type": "else"}]})"));
    EXPECT_NE(error.find("needs a \"match\""), std::string::npos) << error;
    EXPECT_FALSE(parse(R"({"columns": ["dob"], "levels": [
        {"type": "partial_date", "match": "close"}, {"type": "else"}]})"));
    EXPECT_FALSE(parse(R"({"columns": ["dob"], "levels": [
        {"type": "exact", "match": "full"}, {"type": "else"}]})"));
    EXPECT_NE(error.find("only partial_date takes"), std::string::npos) << error;
    EXPECT_FALSE(parse(R"({"columns": ["day"], "levels": [
        {"type": "partial_date", "match": "full"}, {"type": "else"}]})"));
    EXPECT_FALSE(parse(R"({"columns": ["dob", "other"], "levels": [
        {"type": "partial_date", "match": "full"}, {"type": "else"}]})"));
    EXPECT_TRUE(parse(R"({"columns": ["dob"], "levels": [
        {"type": "partial_date", "match": "compatible_month"}, {"type": "else"}]})"))
        << error;
    EXPECT_EQ(schema.comparisons[0].levels[0].Describe(), "date compatible month");
}

TEST(PartialDate, SchemaWriterKeepsTheMatch) {
    cpplink::Schema schema;
    std::string error;
    ASSERT_TRUE(cpplink::ParseSchema(kSchema, &schema, &error)) << error;
    const std::string written = cpplink::SchemaToJson(schema);
    cpplink::Schema reread;
    ASSERT_TRUE(cpplink::ParseSchema(written, &reread, &error)) << error << written;
    ASSERT_EQ(reread.comparisons.size(), 2u);
    for (size_t l = 0; l < schema.comparisons[0].levels.size(); ++l) {
        EXPECT_EQ(reread.comparisons[0].levels[l].type,
                  schema.comparisons[0].levels[l].type);
        EXPECT_EQ(reread.comparisons[0].levels[l].date_match,
                  schema.comparisons[0].levels[l].date_match);
    }
    EXPECT_EQ(written.find("threshold"), std::string::npos) << written;
}

}  // namespace
