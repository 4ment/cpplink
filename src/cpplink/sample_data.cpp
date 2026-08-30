// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/sample_data.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

namespace cpplink {
namespace {

constexpr int32_t kEarliestDob = -7305;  // 1950-01-01
constexpr int32_t kLatestDob = 13514;    // 2007-01-01

std::mt19937_64 SeededFor(uint64_t seed, uint64_t index) {
    // splitmix64, so neighbouring indices give unrelated streams.
    uint64_t z = index + seed * 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return std::mt19937_64(z ^ (z >> 31));
}

std::string MakeToken(std::mt19937_64* rng, int min_syllables, int max_syllables) {
    static const char* kOnsets[] = {
        "b", "br", "c",  "ch", "d", "f",  "g",  "gr", "h",  "j",  "k", "l", "m",  "n",
        "p", "pr", "qu", "r",  "s", "sh", "st", "t",  "th", "tr", "v", "w", "wh", "z"};
    static const char* kNuclei[] = {"a",  "e",  "i",  "o",  "u",  "ai",
                                    "ea", "ee", "ie", "oa", "oo", "ou"};
    static const char* kCodas[] = {"",  "",  "n",  "r",  "s",  "l",  "t",
                                   "d", "m", "ck", "ng", "st", "ll", "rd"};
    std::uniform_int_distribution<int> syllables(min_syllables, max_syllables);
    std::uniform_int_distribution<size_t> onset(0, std::size(kOnsets) - 1);
    std::uniform_int_distribution<size_t> nucleus(0, std::size(kNuclei) - 1);
    std::uniform_int_distribution<size_t> coda(0, std::size(kCodas) - 1);

    std::string word;
    const int count = syllables(*rng);
    for (int i = 0; i < count; ++i) {
        word += kOnsets[onset(*rng)];
        word += kNuclei[nucleus(*rng)];
        word += kCodas[coda(*rng)];
    }
    return word;
}

// A vocabulary sampled with a power law, which is how names and postcodes really
// behave: a few very common values and a long tail of rare ones.
class Vocabulary {
   public:
    Vocabulary(uint64_t seed, size_t size, int min_syllables, int max_syllables,
               double exponent)
        : exponent_(exponent) {
        std::mt19937_64 rng = SeededFor(seed, 0);
        words_.reserve(size);
        for (size_t i = 0; i < size; ++i) {
            words_.push_back(MakeToken(&rng, min_syllables, max_syllables));
        }
    }

    const std::string& Sample(std::mt19937_64* rng) const {
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        const double u = uniform(*rng);
        const size_t rank = static_cast<size_t>(std::pow(u, exponent_) *
                                                static_cast<double>(words_.size()));
        return words_[std::min(rank, words_.size() - 1)];
    }

   private:
    std::vector<std::string> words_;
    double exponent_;
};

struct SampleRecord {
    std::string first_name;
    std::string last_name;
    int32_t dob = 0;
    std::string email;
    std::string phone;
    std::string postcode;
    double latitude = 0.0;
    double longitude = 0.0;
    std::vector<std::string> address_tokens;
};

class Generator {
   public:
    explicit Generator(uint64_t seed)
        : seed_(seed),
          first_names_(seed + 11, 4000, 1, 2, 2.2),
          last_names_(seed + 23, 120000, 2, 3, 1.9),
          street_words_(seed + 37, 30000, 1, 3, 1.6) {}

    SampleRecord Make(uint64_t index) const {
        std::mt19937_64 rng = SeededFor(seed_, index);
        SampleRecord record;
        record.first_name = first_names_.Sample(&rng);
        record.last_name = last_names_.Sample(&rng);

        std::uniform_int_distribution<int32_t> dob(kEarliestDob, kLatestDob);
        record.dob = dob(rng);

        // Email and phone are near-unique, which is what drives dictionary size.
        std::uniform_int_distribution<int> suffix(0, 9999);
        record.email = record.first_name + "." + record.last_name +
                       std::to_string(suffix(rng)) + "@example.com";
        std::uniform_int_distribution<int64_t> phone(400000000LL, 499999999LL);
        record.phone = "0" + std::to_string(phone(rng));

        std::uniform_int_distribution<int> postcode(1000, 9999);
        record.postcode = std::to_string(postcode(rng));

        std::uniform_real_distribution<double> lat(-43.6, -10.7);
        std::uniform_real_distribution<double> lon(113.2, 153.6);
        record.latitude = lat(rng);
        record.longitude = lon(rng);

        std::uniform_int_distribution<int> token_count(3, 6);
        const int tokens = token_count(rng);
        for (int i = 0; i < tokens; ++i) {
            record.address_tokens.push_back(street_words_.Sample(&rng));
        }
        return record;
    }

    // A plausible duplicate: typos, a dropped field, jittered coordinates.
    SampleRecord Corrupt(SampleRecord record, std::mt19937_64* rng) const {
        std::uniform_real_distribution<double> chance(0.0, 1.0);
        if (chance(*rng) < 0.45) Typo(&record.last_name, rng);
        if (chance(*rng) < 0.30) Typo(&record.first_name, rng);
        if (chance(*rng) < 0.35) record.email.clear();
        if (chance(*rng) < 0.25) record.phone.clear();
        if (chance(*rng) < 0.15) record.postcode.clear();
        if (chance(*rng) < 0.20) {
            std::uniform_int_distribution<int> days(-2, 2);
            record.dob += days(*rng);
        }
        std::normal_distribution<double> jitter(0.0, 0.002);
        record.latitude += jitter(*rng);
        record.longitude += jitter(*rng);
        if (chance(*rng) < 0.40 && !record.address_tokens.empty()) {
            record.address_tokens.pop_back();
        }
        return record;
    }

   private:
    static void Typo(std::string* word, std::mt19937_64* rng) {
        if (word->size() < 3) return;
        std::uniform_int_distribution<size_t> position(0, word->size() - 2);
        std::swap((*word)[position(*rng)], (*word)[position(*rng) + 1]);
    }

    uint64_t seed_;
    Vocabulary first_names_;
    Vocabulary last_names_;
    Vocabulary street_words_;
};

std::shared_ptr<arrow::Schema> MakeArrowSchema() {
    return arrow::schema({
        arrow::field("id", arrow::utf8()),
        arrow::field("first_name", arrow::utf8()),
        arrow::field("last_name", arrow::utf8()),
        arrow::field("dob", arrow::date32()),
        arrow::field("email", arrow::utf8()),
        arrow::field("phone", arrow::utf8()),
        arrow::field("postcode", arrow::utf8()),
        arrow::field("latitude", arrow::float64()),
        arrow::field("longitude", arrow::float64()),
        arrow::field("address_tokens", arrow::list(arrow::utf8())),
    });
}

}  // namespace

bool WriteSampleParquet(const std::string& path, const SampleOptions& options,
                        std::string* error) {
    auto sink_result = arrow::io::FileOutputStream::Open(path);
    if (!sink_result.ok()) {
        *error = "cannot create " + path + ": " + sink_result.status().message();
        return false;
    }

    const auto schema = MakeArrowSchema();
    auto props = parquet::WriterProperties::Builder()
                     .compression(parquet::Compression::SNAPPY)
                     ->build();
    auto writer_result = parquet::arrow::FileWriter::Open(
        *schema, arrow::default_memory_pool(), *sink_result, props);
    if (!writer_result.ok()) {
        *error = "cannot open parquet writer: " + writer_result.status().message();
        return false;
    }
    std::unique_ptr<parquet::arrow::FileWriter> writer = std::move(*writer_result);

    std::ofstream truth;
    if (!options.truth_path.empty()) {
        truth.open(options.truth_path);
        if (!truth) {
            *error = "cannot create " + options.truth_path;
            return false;
        }
        truth << "id_a,id_b\n";
    }

    const Generator generator(options.seed);
    std::mt19937_64 planner = SeededFor(options.seed, 0xD1FFull);
    std::uniform_real_distribution<double> chance(0.0, 1.0);

    auto* pool = arrow::default_memory_pool();
    uint64_t written = 0;
    while (written < options.rows) {
        const uint64_t batch =
            std::min<uint64_t>(options.row_group_size, options.rows - written);

        arrow::StringBuilder id(pool), first(pool), last(pool), email(pool), phone(pool),
            postcode(pool);
        arrow::Date32Builder dob(pool);
        arrow::DoubleBuilder latitude(pool), longitude(pool);
        auto token_values = std::make_shared<arrow::StringBuilder>(pool);
        arrow::ListBuilder tokens(pool, token_values);

        for (uint64_t i = 0; i < batch; ++i) {
            const uint64_t index = written + i;
            SampleRecord record = generator.Make(index);
            std::string identifier = "r" + std::to_string(index);

            // Duplicates copy an earlier row, which is regenerated from its index.
            if (index > 0 && chance(planner) < options.duplicate_rate) {
                std::uniform_int_distribution<uint64_t> pick(0, index - 1);
                const uint64_t original = pick(planner);
                record = generator.Corrupt(generator.Make(original), &planner);
                if (truth.is_open()) {
                    truth << "r" << original << ",r" << index << "\n";
                }
            }

            auto status = id.Append(identifier);
            status &= first.Append(record.first_name);
            status &= last.Append(record.last_name);
            status &= dob.Append(record.dob);
            status &=
                record.email.empty() ? email.AppendNull() : email.Append(record.email);
            status &=
                record.phone.empty() ? phone.AppendNull() : phone.Append(record.phone);
            status &= record.postcode.empty() ? postcode.AppendNull()
                                              : postcode.Append(record.postcode);
            status &= latitude.Append(record.latitude);
            status &= longitude.Append(record.longitude);
            status &= tokens.Append();
            for (const std::string& token : record.address_tokens) {
                status &= token_values->Append(token);
            }
            if (!status.ok()) {
                *error =
                    "building row " + std::to_string(index) + ": " + status.message();
                return false;
            }
        }

        std::vector<std::shared_ptr<arrow::Array>> arrays(10);
        arrow::Status status = id.Finish(&arrays[0]);
        status &= first.Finish(&arrays[1]);
        status &= last.Finish(&arrays[2]);
        status &= dob.Finish(&arrays[3]);
        status &= email.Finish(&arrays[4]);
        status &= phone.Finish(&arrays[5]);
        status &= postcode.Finish(&arrays[6]);
        status &= latitude.Finish(&arrays[7]);
        status &= longitude.Finish(&arrays[8]);
        status &= tokens.Finish(&arrays[9]);
        if (!status.ok()) {
            *error = "finishing a batch: " + status.message();
            return false;
        }

        const auto table =
            arrow::Table::Make(schema, arrays, static_cast<int64_t>(batch));
        status = writer->WriteTable(*table, static_cast<int64_t>(batch));
        if (!status.ok()) {
            *error = "writing a row group: " + status.message();
            return false;
        }
        written += batch;
    }

    const arrow::Status closed = writer->Close();
    if (!closed.ok()) {
        *error = "closing " + path + ": " + closed.message();
        return false;
    }
    return true;
}

}  // namespace cpplink
