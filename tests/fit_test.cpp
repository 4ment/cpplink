// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/fit.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "cpplink/app.hpp"
#include "cpplink/blocking.hpp"
#include "cpplink/comparison.hpp"
#include "cpplink/estimate.hpp"
#include "cpplink/histogram.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/sample_data.hpp"
#include "cpplink/schema.hpp"
#include "tests/process_id.hpp"

namespace {

constexpr uint64_t kRecords = 4000;
constexpr uint64_t kPlanted = 400;  // duplicate pairs at rows (2i, 2i + 1)

uint32_t Draw(uint64_t row, uint64_t salt, uint32_t pool) {
    uint64_t value = row * 0x9E3779B97F4A7C15ull + salt;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return static_cast<uint32_t>((value ^ (value >> 31)) % pool);
}

// The interaction fixture's shape: `surname_copy` is `surname` under another
// name, so the two agree together always and an independent mixture cannot
// predict their joint; `city` and `postcode` are drawn on their own coins.
const char* const kSchemaJson = R"({
  "columns": [
    {"name": "email", "type": "string"},
    {"name": "surname", "type": "string"},
    {"name": "surname_copy", "type": "string"},
    {"name": "city", "type": "string"},
    {"name": "postcode", "type": "string"}
  ],
  "comparisons": [
    {"name": "email", "columns": ["email"],
     "levels": [{"type": "null"}, {"type": "exact"}, {"type": "else"}]},
    {"name": "surname", "columns": ["surname"], "term_frequency": true,
     "levels": [{"type": "null"}, {"type": "exact"}, {"type": "else"}]},
    {"name": "surname_copy", "columns": ["surname_copy"],
     "levels": [{"type": "null"}, {"type": "exact"}, {"type": "else"}]},
    {"name": "city", "columns": ["city"],
     "levels": [{"type": "exact"}, {"type": "else"}]},
    {"name": "postcode", "columns": ["postcode"],
     "levels": [{"type": "exact"}, {"type": "else"}]}
  ],
  "blocking": [
    {"type": "exact_value", "column": "email"},
    {"type": "exact_value", "column": "city"},
    {"type": "exact_value", "column": "postcode"}
  ]
})";

class FitFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        std::string error;
        ASSERT_TRUE(cpplink::ParseSchema(kSchemaJson, &schema_, &error)) << error;
        store_ = std::make_unique<cpplink::RecordStore>(schema_);

        auto& email = std::get<cpplink::StringColumn>(store_->mutable_column(0));
        auto& surname = std::get<cpplink::StringColumn>(store_->mutable_column(1));
        auto& copy = std::get<cpplink::StringColumn>(store_->mutable_column(2));
        auto& city = std::get<cpplink::StringColumn>(store_->mutable_column(3));
        auto& postcode = std::get<cpplink::StringColumn>(store_->mutable_column(4));

        std::vector<uint32_t> surnames;
        std::vector<uint32_t> copies;
        for (uint32_t i = 0; i < 300; ++i) {
            const std::string name = "name" + std::to_string(i);
            surnames.push_back(surname.dict.Intern(name));
            copies.push_back(copy.dict.Intern(name));
        }
        std::vector<uint32_t> cities;
        for (uint32_t i = 0; i < 12; ++i) {
            cities.push_back(city.dict.Intern("city" + std::to_string(i)));
        }
        std::vector<uint32_t> postcodes;
        for (uint32_t i = 0; i < 40; ++i) {
            postcodes.push_back(postcode.dict.Intern("pc" + std::to_string(i)));
        }

        email.ids.assign(kRecords, 0);
        surname.ids.assign(kRecords, 0);
        copy.ids.assign(kRecords, 0);
        city.ids.assign(kRecords, 0);
        postcode.ids.assign(kRecords, 0);
        const auto write = [&](uint64_t row, uint32_t name, uint32_t town,
                               uint32_t code) {
            surname.ids[row] = surnames[name];
            copy.ids[row] = copies[name];
            city.ids[row] = cities[town];
            postcode.ids[row] = postcodes[code];
        };
        for (uint64_t i = 0; i < kPlanted; ++i) {
            const uint64_t a = 2 * i;
            const uint64_t b = 2 * i + 1;
            const uint32_t shared = email.dict.Intern("dup" + std::to_string(i));
            email.ids[a] = shared;
            email.ids[b] = shared;
            const uint32_t name = static_cast<uint32_t>(i % 300);
            write(a, name, static_cast<uint32_t>(i % 12), static_cast<uint32_t>(i % 40));
            write(b, i % 4 == 0 ? static_cast<uint32_t>((i + 7) % 300) : name,
                  static_cast<uint32_t>(i % 3 == 0 ? (i + 1) % 12 : i % 12),
                  static_cast<uint32_t>(i % 5 == 0 ? (i + 3) % 40 : i % 40));
        }
        for (uint64_t row = 2 * kPlanted; row < kRecords; ++row) {
            email.ids[row] = email.dict.Intern("solo" + std::to_string(row));
            write(row, Draw(row, 1, 300), Draw(row, 2, 12), Draw(row, 3, 40));
        }

        store_->set_num_records(kRecords);
        store_->Finalize();
        ASSERT_TRUE(plan_.Build(schema_, *store_, &error)) << error;
        ASSERT_TRUE(comparisons_.Bind(schema_, *store_, &error)) << error;

        options_.u_sample = 400000;
        options_.threads = 2;
        options_.seed = 11;
    }

    bool Run() {
        std::string error;
        const bool ok = cpplink::Estimate(*store_, comparisons_, plan_, options_, &model_,
                                          &report_, &error);
        if (!ok) message_ = error;
        return ok;
    }

    const cpplink::SessionReport& Session(const std::string& column) const {
        for (const cpplink::SessionReport& session : report_.sessions) {
            if (session.column == column) return session;
        }
        ADD_FAILURE() << "no session " << column;
        return report_.sessions.front();
    }

    static const cpplink::PairResidual& Residual(const cpplink::SessionFit& fit,
                                                 const std::string& left,
                                                 const std::string& right) {
        for (const cpplink::PairResidual& residual : fit.residuals) {
            if (residual.left == left && residual.right == right) return residual;
        }
        ADD_FAILURE() << "no residual " << left << " x " << right;
        return fit.residuals.front();
    }

    cpplink::Schema schema_;
    std::unique_ptr<cpplink::RecordStore> store_;
    cpplink::BlockingPlan plan_;
    cpplink::ComparisonSet comparisons_;
    cpplink::EstimateOptions options_;
    cpplink::Model model_;
    cpplink::EstimateReport report_;
    std::string message_;
};

// The whole point: a column written out twice is the pair the independent
// mixture is worst at, and every session that leaves both free says so from its
// own histogram with no notion of what the columns hold.
TEST_F(FitFixture, TheDuplicatedColumnIsTheWorstResidual) {
    ASSERT_TRUE(Run()) << message_;
    size_t judged = 0;
    for (const cpplink::SessionReport& session : report_.sessions) {
        if (!session.fit.measured) continue;
        ASSERT_FALSE(session.fit.residuals.empty()) << session.column;
        ++judged;
        const cpplink::PairResidual& worst = session.fit.residuals.front();
        EXPECT_EQ(worst.left, "surname") << session.column;
        EXPECT_EQ(worst.right, "surname_copy") << session.column;
        EXPECT_GT(worst.bits, 0.005) << session.column;
        // And every pair drawn on its own coins reads as what it is.
        for (size_t i = 1; i < session.fit.residuals.size(); ++i) {
            EXPECT_LT(session.fit.residuals[i].bits, worst.bits / 4.0)
                << session.column << " " << session.fit.residuals[i].left << " x "
                << session.fit.residuals[i].right;
        }
    }
    EXPECT_EQ(judged, 3u);
}

// A residual over one pair can never exceed the deviance over the whole table it
// is a margin of, and the whole-table reading sits above what N pairs over K
// patterns would show under an exact model.
TEST_F(FitFixture, ResidualsAreBoundedByTheWholeTable) {
    ASSERT_TRUE(Run()) << message_;
    for (const cpplink::SessionReport& session : report_.sessions) {
        if (!session.fit.measured) continue;
        EXPECT_GT(session.fit.deviance, 0.0);
        EXPECT_GE(session.fit.bits, session.fit.floor_bits);
        for (const cpplink::PairResidual& residual : session.fit.residuals) {
            EXPECT_LE(residual.g2, session.fit.deviance + 1e-6) << residual.left;
            EXPECT_NEAR(residual.bits,
                        residual.g2 / (2.0 * static_cast<double>(session.fit.pairs) *
                                       std::log(2.0)),
                        1e-12);
        }
    }
}

// The term the interaction fit admits on the duplicated pair takes that pair's
// residual away and lowers the whole-table deviance with it, and a pair the term
// does not touch keeps exactly the residual it had.
TEST_F(FitFixture, AnAdmittedTermTakesItsPairsResidualAway) {
    options_.interactions.enabled = true;
    options_.interactions.max_terms = 1;
    ASSERT_TRUE(Run()) << message_;
    ASSERT_EQ(model_.interactions.size(), 1u);
    ASSERT_EQ(model_.interactions.front().left, "surname");
    size_t corrected = 0;
    for (const cpplink::SessionReport& session : report_.sessions) {
        if (!session.fit.corrected) continue;
        ++corrected;
        EXPECT_LT(session.fit.corrected_deviance, session.fit.deviance) << session.column;
        const cpplink::PairResidual& fitted =
            Residual(session.fit, "surname", "surname_copy");
        ASSERT_TRUE(fitted.corrected);
        EXPECT_LT(fitted.corrected_bits, fitted.bits / 4.0) << session.column;
        for (const cpplink::PairResidual& residual : session.fit.residuals) {
            const bool touched =
                residual.left == "surname" || residual.left == "surname_copy" ||
                residual.right == "surname" || residual.right == "surname_copy";
            EXPECT_EQ(residual.corrected, touched)
                << residual.left << " x " << residual.right;
            // A term on one pair leaves the others' margins exactly alone, so a
            // touched pair that is not the fitted one keeps its residual.
            if (touched && &residual != &fitted) {
                EXPECT_NEAR(residual.corrected_bits, residual.bits, 1e-3)
                    << residual.left << " x " << residual.right;
            }
        }
    }
    EXPECT_EQ(corrected, 3u);
}

// The report prints the fit beside the session it belongs to, and says what the
// numbers are.
TEST_F(FitFixture, TheReportPrintsTheFit) {
    ASSERT_TRUE(Run()) << message_;
    std::ostringstream compact;
    cpplink::PrintEstimateReport(report_, compact);
    const std::string text = compact.str();
    EXPECT_NE(text.find("  fit        G^2 "), std::string::npos);
    EXPECT_NE(text.find("sampling floor"), std::string::npos);
    EXPECT_NE(text.find("  worst pair surname x surname_copy at "), std::string::npos);
    EXPECT_EQ(text.find("  residuals  "), std::string::npos);
    EXPECT_NE(text.find("fit is each session's mixture"), std::string::npos);
    EXPECT_NE(text.find("--report <file>"), std::string::npos);

    // The full report lists every pair of every session.
    std::ostringstream full;
    cpplink::PrintEstimateReport(report_, full, cpplink::ReportDetail::kFull);
    size_t rows = 0;
    for (const cpplink::SessionReport& session : report_.sessions) {
        rows += session.fit.residuals.size();
    }
    EXPECT_GT(rows, 0u);
    // Every residual row starts on its own line with the pair, and no other line
    // of the report puts a comparison name at that column.
    size_t listed = 0;
    for (size_t at = full.str().find("\n             city   "); at != std::string::npos;
         at = full.str().find("\n             city   ", at + 1)) {
        ++listed;
    }
    size_t city_pairs = 0;
    for (const cpplink::SessionReport& session : report_.sessions) {
        for (const cpplink::PairResidual& residual : session.fit.residuals) {
            if (residual.left == "city") ++city_pairs;
        }
    }
    EXPECT_EQ(listed, city_pairs);
    EXPECT_EQ(full.str().find("  worst pair "), std::string::npos);
}

// --report writes the full report beside the compact one on the terminal.
TEST(FitReport, ReportOptionWritesTheFullReport) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() /
                                      ("cpplink_fit_report_" + cpplink_test::ProcessId());
    std::filesystem::create_directories(dir);
    const std::string parquet = (dir / "sample.parquet").string();
    cpplink::SampleOptions sample;
    sample.rows = 3000;
    sample.duplicate_rate = 0.1;
    std::string error;
    ASSERT_TRUE(cpplink::WriteSampleParquet(parquet, sample, &error)) << error;
    const std::string schema_path =
        std::string(CPPLINK_SOURCE_DIR) + "/examples/sample_schema.json";
    const std::string report_path = (dir / "estimate.txt").string();
    std::ostringstream out;
    std::ostringstream err;
    const int code = cpplink::Run({"estimate", "--schema", schema_path, "--report",
                                   report_path, "--threads", "2", parquet},
                                  out, err);
    ASSERT_EQ(code, 0) << err.str();
    EXPECT_NE(out.str().find("  worst pair "), std::string::npos);
    EXPECT_EQ(out.str().find("  residuals  "), std::string::npos);
    EXPECT_NE(out.str().find("Wrote the full report to "), std::string::npos);
    std::ifstream file(report_path);
    const std::string written((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
    EXPECT_NE(written.find("  residuals  "), std::string::npos);
    EXPECT_EQ(written.find("  worst pair "), std::string::npos);
    EXPECT_NE(written.find("Weight is log2(m/u)"), std::string::npos);
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

// MeasureFit on a histogram the mixture generated exactly: the deviance is the
// sampling floor and nothing more, and a planted association on one pair puts
// that pair at the top and nowhere else.
TEST_F(FitFixture, AnExactMixtureReadsNoMisfit) {
    const size_t count = comparisons_.Size();
    // city and postcode are the two binary comparisons; the rest are held out so
    // the table is 2 x 2 and can be written down.
    std::vector<bool> excluded(count, true);
    size_t city = 0;
    size_t postcode = 0;
    for (size_t c = 0; c < count; ++c) {
        if (comparisons_.at(c).spec->name == "city") city = c;
        if (comparisons_.at(c).spec->name == "postcode") postcode = c;
    }
    excluded[city] = false;
    excluded[postcode] = false;
    std::vector<std::vector<double>> m(count);
    std::vector<std::vector<double>> u(count);
    for (size_t c = 0; c < count; ++c) {
        const size_t levels = comparisons_.at(c).spec->levels.size();
        m[c].assign(levels, 1.0 / static_cast<double>(levels));
        u[c].assign(levels, 1.0 / static_cast<double>(levels));
    }
    m[city] = {0.9, 0.1};
    u[city] = {0.1, 0.9};
    m[postcode] = {0.8, 0.2};
    u[postcode] = {0.05, 0.95};
    const double lambda = 0.2;
    const double pairs = 1000000.0;

    // Pack the four cells exactly as the mixture predicts them.
    const auto pattern = [&](uint8_t x, uint8_t y) {
        uint32_t gamma = 0;
        gamma |= static_cast<uint32_t>(x) << comparisons_.at(city).shift;
        gamma |= static_cast<uint32_t>(y) << comparisons_.at(postcode).shift;
        return gamma;
    };
    std::vector<cpplink::PatternCount> exact;
    for (uint8_t x = 0; x < 2; ++x) {
        for (uint8_t y = 0; y < 2; ++y) {
            const double probability = lambda * m[city][x] * m[postcode][y] +
                                       (1.0 - lambda) * u[city][x] * u[postcode][y];
            exact.push_back({pattern(x, y),
                             static_cast<uint64_t>(std::llround(pairs * probability))});
        }
    }
    const cpplink::SessionFit clean = MeasureFit(comparisons_, exact, m, u, lambda,
                                                 excluded, {}, cpplink::FitOptions());
    ASSERT_TRUE(clean.measured) << clean.refusal;
    EXPECT_EQ(clean.patterns, 4u);
    EXPECT_LT(clean.bits, 1e-9);
    EXPECT_EQ(clean.degrees, 0u);  // 3 cells free against lambda and two m
    ASSERT_EQ(clean.residuals.size(), 1u);
    EXPECT_LT(clean.residuals.front().bits, 1e-9);
    EXPECT_EQ(clean.residuals.front().degrees, 1u);

    // Move a tenth of the pairs from the off-diagonal onto (agree, agree): an
    // association the independent mixture cannot produce.
    std::vector<cpplink::PatternCount> skewed = exact;
    const uint64_t moved = skewed[1].count / 10;
    skewed[1].count -= moved;
    skewed[0].count += moved;
    const cpplink::SessionFit bent = MeasureFit(comparisons_, skewed, m, u, lambda,
                                                excluded, {}, cpplink::FitOptions());
    EXPECT_GT(bent.bits, 0.001);
    EXPECT_NEAR(bent.residuals.front().g2, bent.deviance, 1e-6);
    EXPECT_LT(bent.residuals.front().p_value, 1e-6);
}

}  // namespace
