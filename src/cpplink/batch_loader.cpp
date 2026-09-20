// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/batch_loader.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cpplink {
namespace {

// What a format string says, reduced to the cases the store can read.
enum class Kind {
    kNull,
    kBool,
    kInt8,
    kUInt8,
    kInt16,
    kUInt16,
    kInt32,
    kUInt32,
    kInt64,
    kUInt64,
    kFloat,
    kDouble,
    kUtf8,
    kLargeUtf8,
    kUtf8View,
    kList,
    kLargeList,
    kDate32,
    kDate64,
    kTimestamp,
    kOther,
};

struct Format {
    Kind kind = Kind::kOther;
    int64_t per_day = 0;  // timestamps: units in one day
    char unit = 0;        // timestamps: s, m, u or n
    std::string_view timezone;
};

Format ParseFormat(const char* text) {
    const std::string_view f(text == nullptr ? "" : text);
    Format format;
    if (f == "n") {
        format.kind = Kind::kNull;
    } else if (f == "b") {
        format.kind = Kind::kBool;
    } else if (f == "c") {
        format.kind = Kind::kInt8;
    } else if (f == "C") {
        format.kind = Kind::kUInt8;
    } else if (f == "s") {
        format.kind = Kind::kInt16;
    } else if (f == "S") {
        format.kind = Kind::kUInt16;
    } else if (f == "i") {
        format.kind = Kind::kInt32;
    } else if (f == "I") {
        format.kind = Kind::kUInt32;
    } else if (f == "l") {
        format.kind = Kind::kInt64;
    } else if (f == "L") {
        format.kind = Kind::kUInt64;
    } else if (f == "f") {
        format.kind = Kind::kFloat;
    } else if (f == "g") {
        format.kind = Kind::kDouble;
    } else if (f == "u") {
        format.kind = Kind::kUtf8;
    } else if (f == "U") {
        format.kind = Kind::kLargeUtf8;
    } else if (f == "vu") {
        format.kind = Kind::kUtf8View;
    } else if (f == "+l") {
        format.kind = Kind::kList;
    } else if (f == "+L") {
        format.kind = Kind::kLargeList;
    } else if (f == "tdD") {
        format.kind = Kind::kDate32;
    } else if (f == "tdm") {
        format.kind = Kind::kDate64;
    } else if (f.size() >= 4 && f[0] == 't' && f[1] == 's' && f[3] == ':') {
        format.unit = f[2];
        switch (f[2]) {
            case 's':
                format.per_day = 86400LL;
                break;
            case 'm':
                format.per_day = 86400000LL;
                break;
            case 'u':
                format.per_day = 86400000000LL;
                break;
            case 'n':
                format.per_day = 86400000000000LL;
                break;
            default:
                return format;
        }
        format.kind = Kind::kTimestamp;
        format.timezone = f.substr(4);
    }
    return format;
}

bool IsInteger(Kind kind) { return kind >= Kind::kInt8 && kind <= Kind::kUInt64; }

bool BitAt(const uint8_t* bits, int64_t i) { return (bits[i >> 3] >> (i & 7)) & 1; }

// One array read at its slice: `i` below is always relative to `array->offset`.
struct ArrayView {
    const ArrowSchema* schema = nullptr;
    const ArrowArray* array = nullptr;
    Format format;

    ArrayView(const ArrowSchema* s, const ArrowArray* a)
        : schema(s), array(a), format(ParseFormat(s->format)) {}

    int64_t length() const { return array->length; }
    template <typename T>
    const T* Buffer(int64_t index) const {
        return static_cast<const T*>(array->buffers[index]);
    }
    bool IsNull(int64_t i) const {
        if (array->null_count == 0 || array->n_buffers < 1) return false;
        const uint8_t* validity = Buffer<uint8_t>(0);
        return validity != nullptr && !BitAt(validity, array->offset + i);
    }
    // The child of a list or struct, read at this array's slice where the
    // format says the parent's offset applies to it (a struct's does; a list's
    // children are addressed through the offsets instead).
    ArrayView Child(int64_t index) const {
        return ArrayView(schema->children[index], array->children[index]);
    }
};

// A shallow copy of an array re-sliced to `[begin, begin + length)` of its own
// values. The copy owns nothing and is never released; it exists so a struct's
// children and a list's elements can be read through the same view.
struct Slice {
    ArrowArray array;
    ArrayView view;

    Slice(const ArrayView& of, int64_t begin, int64_t length)
        : array(*of.array), view(of.schema, &array) {
        array.offset = of.array->offset + begin;
        array.length = length;
    }
};

std::string KindName(const Format& format) {
    switch (format.kind) {
        case Kind::kNull:
            return "null";
        case Kind::kBool:
            return "bool";
        case Kind::kInt8:
            return "int8";
        case Kind::kUInt8:
            return "uint8";
        case Kind::kInt16:
            return "int16";
        case Kind::kUInt16:
            return "uint16";
        case Kind::kInt32:
            return "int32";
        case Kind::kUInt32:
            return "uint32";
        case Kind::kInt64:
            return "int64";
        case Kind::kUInt64:
            return "uint64";
        case Kind::kFloat:
            return "float";
        case Kind::kDouble:
            return "double";
        case Kind::kUtf8:
            return "string";
        case Kind::kLargeUtf8:
            return "large_string";
        case Kind::kUtf8View:
            return "string_view";
        case Kind::kList:
            return "list";
        case Kind::kLargeList:
            return "large_list";
        case Kind::kDate32:
            return "date32[day]";
        case Kind::kDate64:
            return "date64[ms]";
        case Kind::kTimestamp: {
            std::string name = "timestamp[";
            name += format.unit == 's'   ? "s"
                    : format.unit == 'm' ? "ms"
                    : format.unit == 'u' ? "us"
                                         : "ns";
            if (!format.timezone.empty()) {
                name += ", tz=";
                name += format.timezone;
            }
            return name + "]";
        }
        case Kind::kOther:
            break;
    }
    return "";
}

// Calls `fn(i, index)` for every non-null row of an integer array, whatever its
// width or sign, or returns false where the array is not an integer at all.
template <typename Fn>
bool WithIntegers(const ArrayView& view, Fn&& fn) {
    auto walk = [&](auto tag) {
        using T = decltype(tag);
        const T* values = view.Buffer<T>(1);
        for (int64_t i = 0; i < view.length(); ++i) {
            if (!view.IsNull(i)) fn(i, values[view.array->offset + i]);
        }
    };
    switch (view.format.kind) {
        case Kind::kInt8:
            walk(int8_t{});
            return true;
        case Kind::kUInt8:
            walk(uint8_t{});
            return true;
        case Kind::kInt16:
            walk(int16_t{});
            return true;
        case Kind::kUInt16:
            walk(uint16_t{});
            return true;
        case Kind::kInt32:
            walk(int32_t{});
            return true;
        case Kind::kUInt32:
            walk(uint32_t{});
            return true;
        case Kind::kInt64:
            walk(int64_t{});
            return true;
        case Kind::kUInt64:
            walk(uint64_t{});
            return true;
        default:
            return false;
    }
}

// `fn(value)` for row `i` of an integer array, whatever its width or sign.
template <typename Fn>
void WithIntegerAt(const ArrayView& view, int64_t i, Fn&& fn) {
    const int64_t at = view.array->offset + i;
    switch (view.format.kind) {
        case Kind::kInt8:
            fn(view.Buffer<int8_t>(1)[at]);
            break;
        case Kind::kUInt8:
            fn(view.Buffer<uint8_t>(1)[at]);
            break;
        case Kind::kInt16:
            fn(view.Buffer<int16_t>(1)[at]);
            break;
        case Kind::kUInt16:
            fn(view.Buffer<uint16_t>(1)[at]);
            break;
        case Kind::kInt32:
            fn(view.Buffer<int32_t>(1)[at]);
            break;
        case Kind::kUInt32:
            fn(view.Buffer<uint32_t>(1)[at]);
            break;
        case Kind::kInt64:
            fn(view.Buffer<int64_t>(1)[at]);
            break;
        case Kind::kUInt64:
            fn(view.Buffer<uint64_t>(1)[at]);
            break;
        default:
            break;
    }
}

// The bytes of value `i` of a variable-length string array, by its offsets.
template <typename Offset>
std::string_view StringAt(const ArrayView& view, int64_t i) {
    const Offset* offsets = view.Buffer<Offset>(1) + view.array->offset;
    const char* data = view.Buffer<char>(2);
    return std::string_view(data + offsets[i],
                            static_cast<size_t>(offsets[i + 1] - offsets[i]));
}

// A string view is 16 bytes: the length, then either the bytes inline or a
// prefix, the index of a data buffer and an offset into it.
std::string_view StringViewAt(const ArrayView& view, int64_t i) {
    const uint8_t* entry = view.Buffer<uint8_t>(1) + 16 * (view.array->offset + i);
    int32_t length = 0;
    std::memcpy(&length, entry, sizeof(length));
    if (length <= 12)
        return std::string_view(reinterpret_cast<const char*>(entry + 4), length);
    int32_t buffer = 0;
    int32_t offset = 0;
    std::memcpy(&buffer, entry + 8, sizeof(buffer));
    std::memcpy(&offset, entry + 12, sizeof(offset));
    return std::string_view(view.Buffer<char>(2 + buffer) + offset, length);
}

template <typename Visit>
bool ForEachText(const ArrayView& view, Visit&& visit, std::string* error);

// The dictionary's values as owned text, so an integer dictionary spells its
// digits once rather than per row.
bool DictionaryValues(const ArrayView& view, std::vector<std::string>* values,
                      std::vector<bool>* valid, std::string* error) {
    const ArrayView dictionary(view.schema->dictionary, view.array->dictionary);
    return ForEachText(
        dictionary,
        [&](std::string_view value, bool is_null) {
            values->emplace_back(value);
            valid->push_back(!is_null);
        },
        error);
}

// Visits every value of a text-like column in order as `visit(value, is_null)`.
// A frame rarely holds the types the schema names: pandas and polars hand over
// `large_string` or `string_view` for text, and a postcode, a year or an id that
// was numeric arrives as an integer. Both are the string the user meant, so an
// integer is visited as its decimal digits and interned like any other value,
// which is what makes "2000" in one input and 2000 in another the same id.
template <typename Visit>
bool ForEachText(const ArrayView& view, Visit&& visit, std::string* error) {
    if (view.schema->dictionary != nullptr) {
        std::vector<std::string> values;
        std::vector<bool> valid;
        if (!DictionaryValues(view, &values, &valid, error)) return false;
        int64_t next = 0;
        const bool integer = WithIntegers(view, [&](int64_t i, auto index) {
            for (; next < i; ++next) visit(std::string_view(), true);
            next = i + 1;
            const auto at = static_cast<size_t>(index);
            visit(std::string_view(values[at]), !valid[at]);
        });
        if (!integer) {
            *error = "dictionary indices are not integers";
            return false;
        }
        for (; next < view.length(); ++next) visit(std::string_view(), true);
        return true;
    }
    switch (view.format.kind) {
        case Kind::kUtf8:
        case Kind::kLargeUtf8:
        case Kind::kUtf8View:
            for (int64_t i = 0; i < view.length(); ++i) {
                if (view.IsNull(i)) {
                    visit(std::string_view(), true);
                } else if (view.format.kind == Kind::kUtf8) {
                    visit(StringAt<int32_t>(view, i), false);
                } else if (view.format.kind == Kind::kLargeUtf8) {
                    visit(StringAt<int64_t>(view, i), false);
                } else {
                    visit(StringViewAt(view, i), false);
                }
            }
            return true;
        case Kind::kNull:
            for (int64_t i = 0; i < view.length(); ++i) visit(std::string_view(), true);
            return true;
        default:
            break;
    }
    int64_t next = 0;
    char digits[24];  // -9223372036854775808 is 20 characters
    const bool integer = WithIntegers(view, [&](int64_t i, auto value) {
        for (; next < i; ++next) visit(std::string_view(), true);
        next = i + 1;
        const auto result = std::to_chars(digits, digits + sizeof(digits), value);
        visit(std::string_view(digits, result.ptr - digits), false);
    });
    if (integer) {
        for (; next < view.length(); ++next) visit(std::string_view(), true);
        return true;
    }
    *error = "expected a string or integer column, found " + ArrowTypeName(*view.schema);
    return false;
}

bool AppendStrings(const ArrayView& view, StringColumn* column, std::string* error) {
    if (view.schema->dictionary != nullptr) {
        // A category column is the store's own shape: each distinct value is
        // interned once, on the first row that uses it, and the rows are a remap
        // of the indices. A category no row holds is not interned, so the
        // dictionary is what the rows say, as it is when the same column is read
        // from a file.
        std::vector<std::string> values;
        std::vector<bool> valid;
        if (!DictionaryValues(view, &values, &valid, error)) return false;
        std::vector<uint32_t> mapped(values.size(), kNullId);
        std::vector<bool> seen(values.size(), false);
        const size_t first = column->ids.size();
        column->ids.resize(first + static_cast<size_t>(view.length()), kNullId);
        const bool integer = WithIntegers(view, [&](int64_t i, auto index) {
            const auto at = static_cast<size_t>(index);
            if (!seen[at]) {
                seen[at] = true;
                if (valid[at]) mapped[at] = column->dict.Intern(values[at]);
            }
            column->ids[first + i] = mapped[at];
        });
        if (!integer) {
            *error = "dictionary indices are not integers";
            return false;
        }
        return true;
    }
    return ForEachText(
        view,
        [&](std::string_view value, bool is_null) {
            column->ids.push_back(is_null ? kNullId : column->dict.Intern(value));
        },
        error);
}

// Slices the interned elements of a list array into one sorted, deduplicated cell
// per row, whether the offsets are 32-bit (`list`) or 64-bit (`large_list`).
template <typename Offset>
void AppendListRows(const ArrayView& view, const std::vector<uint32_t>& elements,
                    Offset first, StringListColumn* column) {
    const Offset* offsets = view.Buffer<Offset>(1) + view.array->offset;
    std::vector<uint32_t> row;
    for (int64_t i = 0; i < view.length(); ++i) {
        row.clear();
        if (!view.IsNull(i)) {
            for (Offset j = offsets[i] - first; j < offsets[i + 1] - first; ++j) {
                if (elements[j] != kNullId) row.push_back(elements[j]);
            }
        }
        // Sorted and deduplicated here so overlap is a linear merge later.
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
        column->ids.insert(column->ids.end(), row.begin(), row.end());
        column->offsets.push_back(column->ids.size());
    }
}

template <typename Offset>
bool AppendLists(const ArrayView& view, StringListColumn* column, std::string* error) {
    // The elements are interned first, as one flat run over the child array, so a
    // list of integers goes through the same visitor a scalar column does; the rows
    // are then cut out of that run by the list's own offsets. Only the elements
    // this slice reaches are interned, or a sliced frame would leave values in the
    // dictionary that no row holds.
    const Offset* offsets = view.Buffer<Offset>(1) + view.array->offset;
    const Offset first = view.length() > 0 ? offsets[0] : 0;
    const Offset last = view.length() > 0 ? offsets[view.length()] : 0;
    const Slice elements_slice(view.Child(0), first, last - first);
    std::vector<uint32_t> elements;
    elements.reserve(static_cast<size_t>(last - first));
    std::string element_error;
    const bool ok = ForEachText(
        elements_slice.view,
        [&](std::string_view value, bool is_null) {
            elements.push_back(is_null ? kNullId : column->dict.Intern(value));
        },
        &element_error);
    if (!ok) {
        *error = "expected a list of strings or integers, found a list of " +
                 ArrowTypeName(*view.schema->children[0]);
        return false;
    }
    if (column->offsets.empty()) column->offsets.push_back(0);
    AppendListRows<Offset>(view, elements, first, column);
    return true;
}

bool AppendStringLists(const ArrayView& view, StringListColumn* column,
                       std::string* error) {
    switch (view.format.kind) {
        case Kind::kList:
            return AppendLists<int32_t>(view, column, error);
        case Kind::kLargeList:
            return AppendLists<int64_t>(view, column, error);
        case Kind::kNull:
            if (column->offsets.empty()) column->offsets.push_back(0);
            for (int64_t i = 0; i < view.length(); ++i) {
                column->offsets.push_back(column->ids.size());
            }
            return true;
        default:
            *error = "expected a list column, found " + ArrowTypeName(*view.schema);
            return false;
    }
}

// Days since the Unix epoch, whatever the column's temporal unit.
bool AppendDates(const ArrayView& view, DateColumn* column, std::string* error) {
    const int64_t offset = view.array->offset;
    switch (view.format.kind) {
        case Kind::kDate32: {
            const int32_t* values = view.Buffer<int32_t>(1) + offset;
            for (int64_t i = 0; i < view.length(); ++i) {
                column->values.push_back(view.IsNull(i) ? kNullDate : values[i]);
            }
            return true;
        }
        case Kind::kDate64:
        case Kind::kTimestamp: {
            const int64_t per_day =
                view.format.kind == Kind::kDate64 ? 86400000LL : view.format.per_day;
            const int64_t* values = view.Buffer<int64_t>(1) + offset;
            for (int64_t i = 0; i < view.length(); ++i) {
                column->values.push_back(view.IsNull(i)
                                             ? kNullDate
                                             : static_cast<int32_t>(values[i] / per_day));
            }
            return true;
        }
        case Kind::kNull:
            column->values.insert(column->values.end(), view.length(), kNullDate);
            return true;
        default:
            *error = "expected a date or timestamp column, found " +
                     ArrowTypeName(*view.schema);
            return false;
    }
}

bool AppendDoubles(const ArrayView& view, DoubleColumn* column, std::string* error) {
    const double kMissing = std::numeric_limits<double>::quiet_NaN();
    const int64_t offset = view.array->offset;
    switch (view.format.kind) {
        case Kind::kDouble: {
            const double* values = view.Buffer<double>(1) + offset;
            for (int64_t i = 0; i < view.length(); ++i) {
                column->values.push_back(view.IsNull(i) ? kMissing : values[i]);
            }
            return true;
        }
        case Kind::kFloat: {
            const float* values = view.Buffer<float>(1) + offset;
            for (int64_t i = 0; i < view.length(); ++i) {
                column->values.push_back(view.IsNull(i) ? kMissing : values[i]);
            }
            return true;
        }
        case Kind::kNull:
            column->values.insert(column->values.end(), view.length(), kMissing);
            return true;
        default:
            *error =
                "expected a floating point column, found " + ArrowTypeName(*view.schema);
            return false;
    }
}

bool AppendBooleans(const ArrayView& view, BooleanColumn* column, std::string* error) {
    switch (view.format.kind) {
        case Kind::kBool: {
            const uint8_t* bits = view.Buffer<uint8_t>(1);
            for (int64_t i = 0; i < view.length(); ++i) {
                column->values.push_back(view.IsNull(i) ? kNullBoolean
                                         : BitAt(bits, view.array->offset + i)
                                             ? int8_t{1}
                                             : int8_t{0});
            }
            return true;
        }
        case Kind::kNull:
            column->values.insert(column->values.end(), view.length(), kNullBoolean);
            return true;
        default:
            *error = "expected a boolean column, found " + ArrowTypeName(*view.schema);
            return false;
    }
}

bool AppendIds(const ArrayView& view, IdColumn* column, std::string* error) {
    std::string text_error;
    const bool ok = ForEachText(
        view, [&](std::string_view value, bool) { column->Append(value); }, &text_error);
    if (ok) return true;
    *error = "expected a string or integer unique_id column, found " +
             ArrowTypeName(*view.schema);
    return false;
}

bool AppendColumn(const ArrayView& view, ColumnType type, Column* column,
                  std::string* error) {
    switch (type) {
        case ColumnType::kString:
            return AppendStrings(view, &std::get<StringColumn>(*column), error);
        case ColumnType::kStringList:
            return AppendStringLists(view, &std::get<StringListColumn>(*column), error);
        case ColumnType::kDate:
            return AppendDates(view, &std::get<DateColumn>(*column), error);
        case ColumnType::kBoolean:
            return AppendBooleans(view, &std::get<BooleanColumn>(*column), error);
        case ColumnType::kDouble:
            return AppendDoubles(view, &std::get<DoubleColumn>(*column), error);
    }
    return false;
}

// The types the store does not read, named as Arrow prints them where the
// spelling is short, and by their format string otherwise.
std::string OtherTypeName(const ArrowSchema& field) {
    const std::string_view f(field.format == nullptr ? "" : field.format);
    if (f == "z") return "binary";
    if (f == "Z") return "large_binary";
    if (f == "vz") return "binary_view";
    if (f == "e") return "halffloat";
    if (f == "+s") return "struct";
    if (f == "+m") return "map";
    if (f.size() > 2 && f[0] == 'd' && f[1] == ':')
        return "decimal(" + std::string(f.substr(2)) + ")";
    if (f.size() > 2 && f[0] == 'w' && f[1] == ':')
        return "fixed_size_binary[" + std::string(f.substr(2)) + "]";
    if (f == "tts") return "time32[s]";
    if (f == "ttm") return "time32[ms]";
    if (f == "ttu") return "time64[us]";
    if (f == "ttn") return "time64[ns]";
    return std::string(f);
}

}  // namespace

int FieldIndex(const ArrowSchema& schema, std::string_view name) {
    for (int64_t i = 0; i < schema.n_children; ++i) {
        const char* child = schema.children[i]->name;
        if (child != nullptr && name == child) return static_cast<int>(i);
    }
    return -1;
}

bool ColumnTypeOf(const ArrowSchema& field, ColumnType* out) {
    if (field.dictionary != nullptr) return ColumnTypeOf(*field.dictionary, out);
    const Format format = ParseFormat(field.format);
    if (IsInteger(format.kind)) {
        *out = ColumnType::kString;
        return true;
    }
    switch (format.kind) {
        case Kind::kUtf8:
        case Kind::kLargeUtf8:
        case Kind::kUtf8View:
            *out = ColumnType::kString;
            return true;
        case Kind::kList:
        case Kind::kLargeList:
            *out = ColumnType::kStringList;
            return true;
        case Kind::kDate32:
        case Kind::kDate64:
        case Kind::kTimestamp:
            *out = ColumnType::kDate;
            return true;
        case Kind::kBool:
            *out = ColumnType::kBoolean;
            return true;
        case Kind::kFloat:
        case Kind::kDouble:
            *out = ColumnType::kDouble;
            return true;
        default:
            return false;
    }
}

std::string ArrowTypeName(const ArrowSchema& field) {
    const Format format = ParseFormat(field.format);
    if (field.dictionary != nullptr) {
        return "dictionary<values=" + ArrowTypeName(*field.dictionary) +
               ", indices=" + KindName(format) + ", ordered=" +
               ((field.flags & ARROW_FLAG_DICTIONARY_ORDERED) ? "1" : "0") + ">";
    }
    if (format.kind == Kind::kList || format.kind == Kind::kLargeList) {
        const ArrowSchema& child = *field.children[0];
        const std::string name = child.name == nullptr ? "item" : child.name;
        return KindName(format) + "<" + name + ": " + ArrowTypeName(child) + ">";
    }
    if (format.kind == Kind::kOther) return OtherTypeName(field);
    return KindName(format);
}

bool ResolveBatchColumns(const ArrowSchema& schema, const Schema& spec,
                         BatchColumns* columns, std::string* error) {
    columns->id = -1;
    columns->children.assign(spec.columns.size(), -1);
    if (!spec.unique_id.empty()) {
        columns->id = FieldIndex(schema, spec.unique_id);
        if (columns->id < 0) {
            *error = "unique_id column \"" + spec.unique_id + "\" is not in the input";
            return false;
        }
    }
    // A derived column is not among them: it is computed from another column when
    // the store is finalized, and the input is not expected to hold it.
    for (size_t i = 0; i < spec.columns.size(); ++i) {
        if (spec.columns[i].IsDerived()) continue;
        columns->children[i] = FieldIndex(schema, spec.columns[i].name);
        if (columns->children[i] < 0) {
            *error = "column \"" + spec.columns[i].name + "\" is not in the input";
            return false;
        }
    }
    return true;
}

bool ResolveColumnTypesFrom(const ArrowSchema& schema, const std::string& source,
                            Schema* spec, std::string* error) {
    for (ColumnSpec& column : spec->columns) {
        if (column.type_declared || column.IsDerived()) continue;
        const int index = FieldIndex(schema, column.name);
        if (index < 0) {
            *error = "column \"" + column.name + "\" is not in " + source;
            return false;
        }
        const ArrowSchema& field = *schema.children[index];
        if (!ColumnTypeOf(field, &column.type)) {
            *error = "column \"" + column.name + "\" is " + ArrowTypeName(field) +
                     " in " + source + ", which cpplink cannot read; cast it to a string";
            return false;
        }
    }
    return CheckTypes(spec, error);
}

bool AppendRecordBatch(const ArrowSchema& schema, const ArrowArray& batch,
                       const Schema& spec, RecordStore* store, std::string* error) {
    if (batch.n_children != schema.n_children) {
        *error = "the batch holds " + std::to_string(batch.n_children) +
                 " columns where its schema names " + std::to_string(schema.n_children);
        return false;
    }
    BatchColumns columns;
    if (!ResolveBatchColumns(schema, spec, &columns, error)) return false;

    // A record batch is a struct array, and a struct's offset applies to its
    // children: each is read at the batch's slice, whatever its own length says.
    const ArrayView root(&schema, &batch);
    auto column_view = [&](int child) {
        return Slice(root.Child(child), batch.offset, batch.length);
    };
    for (int64_t i = 0; i < batch.n_children; ++i) {
        if (batch.children[i]->length < batch.offset + batch.length) {
            *error = std::string("column \"") + schema.children[i]->name + "\" holds " +
                     std::to_string(batch.children[i]->length) +
                     " rows where the batch has " + std::to_string(batch.length);
            return false;
        }
    }

    if (columns.id >= 0) {
        const Slice ids = column_view(columns.id);
        if (!AppendIds(ids.view, &store->mutable_ids(), error)) {
            *error = spec.unique_id + ": " + *error;
            return false;
        }
    }
    for (size_t i = 0; i < spec.columns.size(); ++i) {
        if (columns.children[i] < 0) continue;
        const Slice column = column_view(columns.children[i]);
        if (!AppendColumn(column.view, spec.columns[i].type, &store->mutable_column(i),
                          error)) {
            *error = spec.columns[i].name + ": " + *error;
            return false;
        }
    }
    return true;
}

bool AppendArrayStream(ArrowArrayStream* stream, const Schema& spec, RecordStore* store,
                       uint64_t* rows, int* batches, std::string* error) {
    auto fail = [&](const std::string& what, int code) {
        const char* detail =
            stream->get_last_error == nullptr ? nullptr : stream->get_last_error(stream);
        *error =
            what + ": " + (detail == nullptr ? "error " + std::to_string(code) : detail);
        if (stream->release != nullptr) stream->release(stream);
        return false;
    };
    ArrowSchema schema;
    schema.release = nullptr;
    if (const int code = stream->get_schema(stream, &schema); code != 0) {
        return fail("cannot read the stream's schema", code);
    }
    bool ok = true;
    while (ok) {
        ArrowArray batch;
        batch.release = nullptr;
        if (const int code = stream->get_next(stream, &batch); code != 0) {
            schema.release(&schema);
            return fail("cannot read the next batch", code);
        }
        if (batch.release == nullptr) break;  // the stream has ended
        ok = AppendRecordBatch(schema, batch, spec, store, error);
        if (ok) {
            *rows += static_cast<uint64_t>(batch.length);
            ++*batches;
        }
        batch.release(&batch);
    }
    schema.release(&schema);
    if (stream->release != nullptr) stream->release(stream);
    return ok;
}

bool LoadStreams(const std::vector<InputStream>& inputs, const Schema& spec,
                 RecordStore* store, LoadStats* stats, std::string* error) {
    const auto started = std::chrono::steady_clock::now();
    if (inputs.empty()) {
        *error = "no input was given";
        return false;
    }
    // Boundaries are recorded as they are crossed, so a dataset is the row range
    // between two of them and nothing per row has to be stored.
    std::vector<uint64_t> starts;
    std::vector<uint64_t> per_input;
    std::vector<std::string> names;
    starts.push_back(0);
    uint64_t total = 0;
    int batches = 0;
    bool ok = true;
    for (size_t i = 0; i < inputs.size(); ++i) {
        uint64_t rows = 0;
        int count = 0;
        if (ok) {
            ok = AppendArrayStream(inputs[i].stream, spec, store, &rows, &count, error);
            // With several inputs the message has to say which one failed; a
            // column missing from the second reads the same as one missing from
            // the first otherwise.
            if (!ok && inputs.size() > 1) *error = inputs[i].name + ": " + *error;
        } else if (inputs[i].stream->release != nullptr) {
            inputs[i].stream->release(inputs[i].stream);
        }
        total += rows;
        batches += count;
        per_input.push_back(rows);
        starts.push_back(total);
        names.push_back(inputs[i].name);
    }
    if (!ok) return false;

    store->set_num_records(total);
    // One dataset is the absence of a boundary, not a boundary at each end: the
    // dedup path then pays no lookup at all.
    if (inputs.size() > 1) {
        store->set_datasets(std::move(starts));
        store->set_dataset_names(std::move(names));
    }
    store->Finalize();

    if (stats != nullptr) {
        stats->rows = total;
        stats->row_groups = batches;
        stats->dataset_rows = std::move(per_input);
        stats->seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
                .count();
    }
    return true;
}

struct TextReader::Impl {
    ArrowArray sliced;
    ArrayView view;
    std::vector<std::string> values;  // a dictionary's, decoded once
    std::vector<bool> valid;
    Impl(const ArrowSchema* schema, const ArrowArray* array)
        : sliced(*array), view(schema, &sliced) {}
};

TextReader::TextReader() = default;
TextReader::~TextReader() = default;

bool TextReader::Bind(const ArrowSchema& schema, const ArrowArray& batch, int column,
                      std::string* error) {
    impl_ = std::make_unique<Impl>(schema.children[column], batch.children[column]);
    impl_->sliced.offset += batch.offset;
    impl_->sliced.length = batch.length;
    const Format& format = impl_->view.format;
    if (impl_->view.schema->dictionary != nullptr) {
        if (!IsInteger(format.kind)) {
            *error = "dictionary indices are not integers";
            return false;
        }
        return DictionaryValues(impl_->view, &impl_->values, &impl_->valid, error);
    }
    if (format.kind == Kind::kUtf8 || format.kind == Kind::kLargeUtf8 ||
        format.kind == Kind::kUtf8View || format.kind == Kind::kNull ||
        IsInteger(format.kind)) {
        return true;
    }
    *error = "expected a string or integer column, found " + ArrowTypeName(schema);
    return false;
}

bool TextReader::At(int64_t row, std::string_view* out, char (*scratch)[24]) const {
    const ArrayView& view = impl_->view;
    if (view.IsNull(row)) return false;
    if (view.schema->dictionary != nullptr) {
        int64_t index = 0;
        WithIntegerAt(view, row,
                      [&](auto value) { index = static_cast<int64_t>(value); });
        if (!impl_->valid[static_cast<size_t>(index)]) return false;
        *out = impl_->values[static_cast<size_t>(index)];
        return true;
    }
    switch (view.format.kind) {
        case Kind::kUtf8:
            *out = StringAt<int32_t>(view, row);
            return true;
        case Kind::kLargeUtf8:
            *out = StringAt<int64_t>(view, row);
            return true;
        case Kind::kUtf8View:
            *out = StringViewAt(view, row);
            return true;
        case Kind::kNull:
            return false;
        default:
            break;
    }
    WithIntegerAt(view, row, [&](auto value) {
        const auto result = std::to_chars(*scratch, *scratch + sizeof(*scratch), value);
        *out = std::string_view(*scratch, result.ptr - *scratch);
    });
    return true;
}

struct NumberReader::Impl {
    ArrowArray sliced;
    ArrayView view;
    Impl(const ArrowSchema* schema, const ArrowArray* array)
        : sliced(*array), view(schema, &sliced) {}
};

NumberReader::NumberReader() = default;
NumberReader::~NumberReader() = default;

bool NumberReader::Bind(const ArrowSchema& schema, const ArrowArray& batch, int column,
                        std::string* error) {
    impl_ = std::make_unique<Impl>(schema.children[column], batch.children[column]);
    impl_->sliced.offset += batch.offset;
    impl_->sliced.length = batch.length;
    const Kind kind = impl_->view.format.kind;
    if (kind == Kind::kDouble || kind == Kind::kFloat || kind == Kind::kNull ||
        IsInteger(kind)) {
        return true;
    }
    *error = "expected a numeric column, found " + ArrowTypeName(schema);
    return false;
}

bool NumberReader::At(int64_t row, double* out) const {
    const ArrayView& view = impl_->view;
    if (view.IsNull(row)) return false;
    const int64_t at = view.array->offset + row;
    switch (view.format.kind) {
        case Kind::kDouble:
            *out = view.Buffer<double>(1)[at];
            return true;
        case Kind::kFloat:
            *out = view.Buffer<float>(1)[at];
            return true;
        case Kind::kNull:
            return false;
        default:
            break;
    }
    WithIntegerAt(view, row, [&](auto value) { *out = static_cast<double>(value); });
    return true;
}

}  // namespace cpplink
