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

#include "cpplink/arrow_export.hpp"
#include "cpplink/parquet_io.hpp"

namespace cpplink {
namespace {

constexpr int32_t kEarliestDob = -7305;  // 1950-01-01
constexpr int32_t kLatestDob = 13514;    // 2007-01-01

// The domains an address is drawn from, skewed so the first takes about half the
// rows the way one provider does. A duplicate that keeps its username and moves
// domain is the case the email comparison's username level exists for, and one
// no benchmark dataset carried until this one did.
constexpr const char* kDomains[] = {"example.com",     "mail.example.com",
                                    "example.net",     "example.org",
                                    "post.example.io", "inbox.example.co"};
constexpr size_t kDomainCount = std::size(kDomains);
constexpr double kDomainSkew = 2.5;

// A missing value one row in thirty, and the two values otherwise even.
constexpr double kGenderNullRate = 0.03;

size_t DrawDomain(std::mt19937_64* rng) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const size_t rank = static_cast<size_t>(std::pow(uniform(*rng), kDomainSkew) *
                                            static_cast<double>(kDomainCount));
    return std::min(rank, kDomainCount - 1);
}

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
    int gender = -1;  // -1 missing, else 0 for F or 1 for M: a two-valued string
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

        std::uniform_real_distribution<double> chance(0.0, 1.0);
        const double gender = chance(rng);
        if (gender >= kGenderNullRate) {
            record.gender = gender < 0.5 + kGenderNullRate / 2 ? 1 : 0;
        }

        // Email and phone are near-unique, which is what drives dictionary size.
        // The username is what carries that; the domain is a handful of values.
        std::uniform_int_distribution<int> suffix(0, 9999);
        record.email = record.first_name + "." + record.last_name +
                       std::to_string(suffix(rng)) + "@" + kDomains[DrawDomain(&rng)];
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

    // A plausible duplicate: typos, a dropped field, jittered coordinates, and an
    // address that moved provider.
    SampleRecord Corrupt(SampleRecord record, std::mt19937_64* rng) const {
        std::uniform_real_distribution<double> chance(0.0, 1.0);
        if (chance(*rng) < 0.45) Typo(&record.last_name, rng);
        if (chance(*rng) < 0.30) Typo(&record.first_name, rng);
        // Dropped 35% of the time as before; of the addresses kept, one in four
        // keeps its username under a different domain, which is a pair the exact
        // address level cannot see and the username level can.
        const double email_fate = chance(*rng);
        if (email_fate < 0.35) {
            record.email.clear();
        } else if (email_fate < 0.35 + 0.25 * 0.65) {
            MoveDomain(&record.email, rng);
        }
        if (chance(*rng) < 0.25) record.phone.clear();
        // A flag is rarely wrong and sometimes not filled in.
        if (record.gender >= 0) {
            const double gender_fate = chance(*rng);
            if (gender_fate < 0.05) {
                record.gender = -1;
            } else if (gender_fate < 0.07) {
                record.gender = 1 - record.gender;
            }
        }
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

    // Keeps the username and draws a different domain, redrawing on the one it
    // already has so the address really does change.
    static void MoveDomain(std::string* email, std::mt19937_64* rng) {
        const size_t at = email->find('@');
        if (at == std::string::npos) return;
        const std::string current = email->substr(at + 1);
        std::string domain = current;
        while (domain == current) domain = kDomains[DrawDomain(rng)];
        *email = email->substr(0, at + 1) + domain;
    }

    uint64_t seed_;
    Vocabulary first_names_;
    Vocabulary last_names_;
    Vocabulary street_words_;
};

// The eleven columns one output file accumulates rows into, as C Data batches,
// and the writer they are flushed to. Routing a row to one of two files is then
// a choice of sink rather than a second copy of the append code.
struct RowSink {
    RowSink() {
        builder.AddColumn("id", ExportType::kString);
        builder.AddColumn("first_name", ExportType::kString);
        builder.AddColumn("last_name", ExportType::kString);
        builder.AddColumn("gender", ExportType::kString);
        builder.AddColumn("dob", ExportType::kDate32);
        builder.AddColumn("email", ExportType::kString);
        builder.AddColumn("phone", ExportType::kString);
        builder.AddColumn("postcode", ExportType::kString);
        builder.AddColumn("latitude", ExportType::kDouble);
        builder.AddColumn("longitude", ExportType::kDouble);
        builder.AddColumn("address_tokens", ExportType::kStringList);
    }

    bool Open(const std::string& path, std::string* error) {
        this->path = path;
        ArrowSchema schema;
        builder.ExportSchema(&schema);
        const bool ok = writer.Open(path, schema, error);
        schema.release(&schema);
        writing = ok;
        return ok;
    }

    void Append(const SampleRecord& record, const std::string& identifier) {
        builder.AppendString(0, identifier);
        builder.AppendString(1, record.first_name);
        builder.AppendString(2, record.last_name);
        if (record.gender < 0) {
            builder.AppendNull(3);
        } else {
            builder.AppendString(3, record.gender == 1 ? "M" : "F");
        }
        builder.AppendDate32(4, record.dob);
        if (record.email.empty()) {
            builder.AppendNull(5);
        } else {
            builder.AppendString(5, record.email);
        }
        if (record.phone.empty()) {
            builder.AppendNull(6);
        } else {
            builder.AppendString(6, record.phone);
        }
        if (record.postcode.empty()) {
            builder.AppendNull(7);
        } else {
            builder.AppendString(7, record.postcode);
        }
        builder.AppendDouble(8, record.latitude);
        builder.AppendDouble(9, record.longitude);
        builder.AppendStringList(10, record.address_tokens);
    }

    uint64_t Pending() const { return static_cast<uint64_t>(builder.Rows()); }

    // A sink with no file keeps every row in its builder.
    bool Flush(std::string* error) {
        if (!writing || builder.Rows() == 0) return true;
        ArrowArray batch;
        if (!builder.ExportBatch(&batch, error)) return false;
        return writer.Write(&batch, error);
    }

    bool Close(std::string* error) { return !writing || writer.Close(error); }

    BatchBuilder builder;
    std::string path;
    bool writing = false;
    ParquetWriter writer;
};

// Plants the rows into the sinks: originals into the first, every duplicate into
// one of the later ones where there are any. The one generator behind both the
// files and the in-memory tables.
bool GenerateInto(const SampleOptions& options,
                  std::vector<std::unique_ptr<RowSink>>* sinks_ptr, std::string* error) {
    std::vector<std::unique_ptr<RowSink>>& sinks = *sinks_ptr;
    const bool linking = sinks.size() > 1;
    // Never asked when there is no later file, but constructed either way, and a
    // distribution over an empty range is undefined.
    std::uniform_int_distribution<size_t> which_link(
        1, std::max<size_t>(1, sinks.size() - 1));

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

    // Which original record each row is ultimately a copy of. A duplicate may pick
    // a row that is itself a duplicate, and corrupting *that row's index* would
    // regenerate a record the file never contained -- the two rows would share
    // nothing, and the truth file would still claim they were a pair. Following
    // the chain to its base keeps every recorded pair genuinely similar and lets
    // clusters larger than two arise honestly. Four bytes a row: 80 MB at 20M.
    std::vector<uint32_t> base_of(options.rows);

    for (uint64_t index = 0; index < options.rows; ++index) {
        SampleRecord record = generator.Make(index);
        const std::string identifier = "r" + std::to_string(index);

        // Duplicates copy an earlier row by corrupting the record that row is
        // itself a copy of, so a chain of duplicates stays a cluster of genuinely
        // similar records rather than a chain of unrelated ones.
        base_of[index] = static_cast<uint32_t>(index);
        bool duplicate = false;
        uint64_t original = 0;
        uint64_t base = index;
        if (index > 0 && chance(planner) < options.duplicate_rate) {
            std::uniform_int_distribution<uint64_t> pick(0, index - 1);
            original = pick(planner);
            base = base_of[original];
            base_of[index] = static_cast<uint32_t>(base);
            record = generator.Corrupt(generator.Make(base), &planner);
            duplicate = true;
        }
        if (duplicate && truth.is_open()) {
            // Splitting the files puts every duplicate in the second one, so the
            // pair is recorded against the chain's base -- which is always an
            // original, and so always in the first file. Without the split the
            // planted relationship is the one to record, and it is the row that was
            // actually picked.
            const uint64_t other = linking ? base : original;
            truth << "r" << other << ",r" << index << "\n";
        }

        // A duplicate lands in one of the later files, drawn from the planner so
        // the file a row lands in is as reproducible as the row.
        RowSink& sink = *sinks[linking && duplicate ? which_link(planner) : 0];
        sink.Append(record, identifier);
        if (sink.Pending() >= static_cast<uint64_t>(options.row_group_size) &&
            !sink.Flush(error)) {
            return false;
        }
    }

    for (const std::unique_ptr<RowSink>& sink : sinks) {
        if (!sink->Flush(error)) return false;
    }
    return true;
}

}  // namespace

bool WriteSampleParquet(const std::string& path, const SampleOptions& options,
                        std::string* error) {
    // One sink per output file. With more than one file every planted duplicate is
    // routed to one of the later ones, so the files are exactly the link fixture
    // the cross-dataset path needs: every recorded pair crosses them.
    std::vector<std::unique_ptr<RowSink>> sinks;
    sinks.push_back(std::make_unique<RowSink>());
    if (!sinks.back()->Open(path, error)) return false;
    for (const std::string& link_path : options.link_paths) {
        sinks.push_back(std::make_unique<RowSink>());
        if (!sinks.back()->Open(link_path, error)) return false;
    }
    if (!GenerateInto(options, &sinks, error)) return false;
    for (const std::unique_ptr<RowSink>& sink : sinks) {
        if (!sink->Close(error)) return false;
    }
    return true;
}

bool GenerateSampleTables(const SampleOptions& options, std::vector<BatchBuilder>* tables,
                          std::string* error) {
    std::vector<std::unique_ptr<RowSink>> sinks;
    sinks.push_back(std::make_unique<RowSink>());
    for (size_t i = 0; i < options.link_paths.size(); ++i) {
        sinks.push_back(std::make_unique<RowSink>());
    }
    if (!GenerateInto(options, &sinks, error)) return false;
    tables->clear();
    for (std::unique_ptr<RowSink>& sink : sinks)
        tables->push_back(std::move(sink->builder));
    return true;
}

}  // namespace cpplink
