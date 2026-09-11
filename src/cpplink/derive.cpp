// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/derive.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace cpplink {
namespace {

bool IsSpace(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

// Lowercase, with every byte that is neither a letter nor a digit becoming a
// single space; runs collapse and the ends are trimmed.
//
// A separator becomes a space rather than vanishing, which is what lets
// "sorted_tokens" run after this one and still see tokens. Bytes above ASCII are
// kept as they are: lowercasing them needs a locale this project does not carry,
// and dropping them would erase most of a name rather than normalise it.
void Normalize(std::string_view value, std::string* out) {
    bool separated = false;
    for (const char c : value) {
        const unsigned char byte = static_cast<unsigned char>(c);
        const bool keep = byte >= 0x80 || std::isalnum(byte) != 0;
        if (!keep) {
            separated = true;
            continue;
        }
        if (separated && !out->empty()) out->push_back(' ');
        separated = false;
        out->push_back(byte >= 0x80 ? c : static_cast<char>(std::tolower(byte)));
    }
}

// Whitespace-separated tokens, sorted and rejoined with one space, which makes
// "smith john" and "john smith" the same key.
void SortedTokens(std::string_view value, std::string* out) {
    std::vector<std::string_view> tokens;
    size_t i = 0;
    while (i < value.size()) {
        while (i < value.size() && IsSpace(value[i])) ++i;
        const size_t begin = i;
        while (i < value.size() && !IsSpace(value[i])) ++i;
        if (i > begin) tokens.push_back(value.substr(begin, i - begin));
    }
    std::sort(tokens.begin(), tokens.end());
    for (const std::string_view token : tokens) {
        if (!out->empty()) out->push_back(' ');
        out->append(token.data(), token.size());
    }
}

char SoundexCode(char letter) {
    switch (letter) {
        case 'b':
        case 'f':
        case 'p':
        case 'v':
            return '1';
        case 'c':
        case 'g':
        case 'j':
        case 'k':
        case 'q':
        case 's':
        case 'x':
        case 'z':
            return '2';
        case 'd':
        case 't':
            return '3';
        case 'l':
            return '4';
        case 'm':
        case 'n':
            return '5';
        case 'r':
            return '6';
        default:
            return '0';
    }
}

// The American Soundex: the first letter, then one code per consonant, padded to
// four characters. Two letters sharing a code are written once, unless a vowel
// stands between them -- h and w are transparent and do not separate them, which
// is the rule a naive implementation gets wrong and the reason "Ashcraft" keys as
// A261 rather than A226.
void Soundex(std::string_view value, std::string* out) {
    constexpr size_t kKeyLength = 4;
    size_t i = 0;
    while (i < value.size() && std::isalpha(static_cast<unsigned char>(value[i])) == 0) {
        ++i;
    }
    if (i == value.size()) return;  // no letters, so nothing to key on
    const unsigned char head = static_cast<unsigned char>(value[i]);
    const char first = static_cast<char>(std::tolower(head));
    out->push_back(static_cast<char>(std::toupper(head)));
    char previous = SoundexCode(first);
    for (++i; i < value.size() && out->size() < kKeyLength; ++i) {
        const unsigned char byte = static_cast<unsigned char>(value[i]);
        if (std::isalpha(byte) == 0) continue;
        const char letter = static_cast<char>(std::tolower(byte));
        const char code = SoundexCode(letter);
        if (code != '0' && code != previous) out->push_back(code);
        if (letter != 'h' && letter != 'w') previous = code;
    }
    out->append(kKeyLength - out->size(), '0');
}

// The two halves of an address, split the way splink's email comparison splits
// it: the username is the run before the first "@", or the whole value when there
// is none, and the domain is the run after the last "@", or nothing. Neither
// touches case or punctuation, so a chain can add "normalize" where it wants it.
void EmailUsername(std::string_view value, std::string* out) {
    const size_t at = value.find('@');
    const std::string_view head =
        at == std::string_view::npos ? value : value.substr(0, at);
    out->append(head.data(), head.size());
}

void EmailDomain(std::string_view value, std::string* out) {
    const size_t at = value.rfind('@');
    if (at == std::string_view::npos) return;
    const std::string_view tail = value.substr(at + 1);
    out->append(tail.data(), tail.size());
}

// Howard Hinnant's civil_from_days: days since 1970-01-01 to a proleptic
// Gregorian date, with no library dependency and no time zone to be wrong about.
void CivilFromDays(int32_t days, int64_t* year, int64_t* month, int64_t* day) {
    const int64_t z = static_cast<int64_t>(days) + 719468;
    const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const int64_t doe = z - era * 146097;
    const int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const int64_t mp = (5 * doy + 2) / 153;
    *day = doy - (153 * mp + 2) / 5 + 1;
    *month = mp + (mp < 10 ? 3 : -9);
    *year = yoe + era * 400 + (*month <= 2 ? 1 : 0);
}

void AppendPadded(int64_t value, std::string* out) {
    if (value < 10) out->push_back('0');
    out->append(std::to_string(value));
}

void ApplyStringTransform(Transform transform, std::string_view value, std::string* out) {
    switch (transform) {
        case Transform::kNormalize:
            Normalize(value, out);
            return;
        case Transform::kSortedTokens:
            SortedTokens(value, out);
            return;
        case Transform::kSoundex:
            Soundex(value, out);
            return;
        case Transform::kEmailUsername:
            EmailUsername(value, out);
            return;
        case Transform::kEmailDomain:
            EmailDomain(value, out);
            return;
        default:
            // A date transform never reaches a string: the chain is type-checked
            // against the source column before a file is opened.
            return;
    }
}

// The rest of a chain, in place. One scratch string is reused across the whole
// chain, and the chain itself runs once per distinct value rather than per row.
void ApplyChain(const std::vector<Transform>& transforms, size_t from,
                std::string* value) {
    std::string next;
    for (size_t i = from; i < transforms.size(); ++i) {
        next.clear();
        ApplyStringTransform(transforms[i], *value, &next);
        value->swap(next);
    }
}

void ApplyDateTransform(Transform transform, int32_t days, std::string* out) {
    if (days == kNullDate) return;
    int64_t year = 0;
    int64_t month = 0;
    int64_t day = 0;
    CivilFromDays(days, &year, &month, &day);
    switch (transform) {
        case Transform::kYear:
            out->append(std::to_string(year));
            return;
        case Transform::kMonth:
            AppendPadded(month, out);
            return;
        case Transform::kDay:
            AppendPadded(day, out);
            return;
        case Transform::kYearMonth:
            out->append(std::to_string(year));
            out->push_back('-');
            AppendPadded(month, out);
            return;
        default:
            return;
    }
}

// Where the derived column reads from. Parse time has already established that
// the name is one of the store's columns.
size_t SourceIndex(const Schema& schema, const std::string& name) {
    for (size_t i = 0; i < schema.columns.size(); ++i) {
        if (schema.columns[i].name == name) return i;
    }
    return schema.columns.size();
}

void DeriveFromStrings(const StringColumn& source, const std::vector<Transform>& chain,
                       uint64_t records, StringColumn* target) {
    // Once per distinct value rather than once per row, which is what interning
    // buys here: the derived dictionary is built from the source dictionary and
    // the rows are a gather through the map between them.
    std::vector<uint32_t> mapped(source.dict.Size(), kNullId);
    std::string value;
    for (uint32_t v = 0; v < source.dict.Size(); ++v) {
        ApplyTransforms(chain, source.dict.Value(v), &value);
        if (!value.empty()) mapped[v] = target->dict.Intern(value);
    }
    const uint64_t rows = std::min<uint64_t>(records, source.ids.size());
    for (uint64_t row = 0; row < rows; ++row) {
        const uint32_t id = source.ids[row];
        if (id != kNullId) target->ids[row] = mapped[id];
    }
}

void DeriveFromDates(const DateColumn& source, const std::vector<Transform>& chain,
                     uint64_t records, StringColumn* target) {
    const uint64_t rows = std::min<uint64_t>(records, source.values.size());
    bool any = false;
    int32_t low = 0;
    int32_t high = 0;
    for (uint64_t row = 0; row < rows; ++row) {
        const int32_t days = source.values[row];
        if (days == kNullDate) continue;
        low = any ? std::min(low, days) : days;
        high = any ? std::max(high, days) : days;
        any = true;
    }
    if (!any) return;

    // A date column is dense and narrow -- a century is 36,500 values -- so the
    // chain is memoized over the observed range and the rows become the same
    // gather the string path does. A range too wide to hold is computed row by
    // row instead, which is correct and merely slower.
    constexpr uint64_t kMaxMemo = 1u << 22;
    const uint64_t span = static_cast<uint64_t>(static_cast<int64_t>(high) - low) + 1;
    std::string value;
    if (span > kMaxMemo) {
        for (uint64_t row = 0; row < rows; ++row) {
            const int32_t days = source.values[row];
            if (days == kNullDate) continue;
            ApplyDateTransforms(chain, days, &value);
            if (!value.empty()) target->ids[row] = target->dict.Intern(value);
        }
        return;
    }
    std::vector<uint32_t> mapped(span, kNullId);
    std::vector<uint8_t> seen(span, 0);
    for (uint64_t row = 0; row < rows; ++row) {
        const int32_t days = source.values[row];
        if (days == kNullDate) continue;
        const uint64_t at = static_cast<uint64_t>(static_cast<int64_t>(days) - low);
        if (seen[at] == 0) {
            seen[at] = 1;
            ApplyDateTransforms(chain, days, &value);
            if (!value.empty()) mapped[at] = target->dict.Intern(value);
        }
        target->ids[row] = mapped[at];
    }
}

}  // namespace

void ApplyTransforms(const std::vector<Transform>& transforms, std::string_view value,
                     std::string* out) {
    out->assign(value.begin(), value.end());
    ApplyChain(transforms, 0, out);
}

void ApplyDateTransforms(const std::vector<Transform>& transforms, int32_t days,
                         std::string* out) {
    out->clear();
    if (transforms.empty()) return;
    ApplyDateTransform(transforms.front(), days, out);
    if (out->empty()) return;
    ApplyChain(transforms, 1, out);
}

void BuildDerivedColumns(RecordStore* store) {
    const Schema& schema = store->schema();
    const uint64_t records = store->NumRecords();
    for (size_t i = 0; i < schema.columns.size(); ++i) {
        const ColumnSpec& spec = schema.columns[i];
        if (!spec.IsDerived()) continue;
        const size_t source = SourceIndex(schema, spec.derive.from);
        if (source >= schema.columns.size() || source == i) continue;

        auto* target = std::get_if<StringColumn>(&store->mutable_column(i));
        if (target == nullptr) continue;
        target->ids.assign(records, kNullId);
        if (const auto* from = std::get_if<StringColumn>(&store->column(source))) {
            DeriveFromStrings(*from, spec.derive.transforms, records, target);
        } else if (const auto* dates = std::get_if<DateColumn>(&store->column(source))) {
            DeriveFromDates(*dates, spec.derive.transforms, records, target);
        }
    }
}

}  // namespace cpplink
