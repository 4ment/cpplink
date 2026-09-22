// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

// The Arrow C++ side of `parquet_io.hpp`, and the only file in the library that
// includes an Arrow header. Compiled under CPPLINK_WITH_ARROW=ON; the stub takes
// its place otherwise.

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <arrow/api.h>
#include <arrow/c/bridge.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

#include "cpplink/parquet_io.hpp"

namespace cpplink {
namespace {

bool OpenReader(const std::string& path,
                std::unique_ptr<parquet::arrow::FileReader>* reader,
                std::shared_ptr<arrow::Schema>* schema, std::string* error) {
    auto input = arrow::io::ReadableFile::Open(path);
    if (!input.ok()) {
        *error = "cannot open " + path + ": " + input.status().message();
        return false;
    }
    parquet::arrow::FileReaderBuilder builder;
    arrow::Status status = builder.Open(*input);
    if (!status.ok()) {
        *error = "cannot read parquet metadata: " + status.message();
        return false;
    }
    auto built = builder.Build();
    if (!built.ok()) {
        *error = "cannot open parquet reader: " + built.status().message();
        return false;
    }
    *reader = std::move(*built);
    status = (*reader)->GetSchema(schema);
    if (!status.ok()) {
        *error = "cannot read parquet schema: " + status.message();
        return false;
    }
    return true;
}

}  // namespace

bool ParquetSupported() { return true; }

bool ReadParquetSchema(const std::string& path, ArrowSchema* out, std::string* error) {
    std::unique_ptr<parquet::arrow::FileReader> reader;
    std::shared_ptr<arrow::Schema> schema;
    if (!OpenReader(path, &reader, &schema, error)) return false;
    const arrow::Status status = arrow::ExportSchema(*schema, out);
    if (!status.ok()) {
        *error = "cannot export parquet schema: " + status.message();
        return false;
    }
    return true;
}

bool OpenParquetStream(const std::string& path, const std::vector<std::string>& columns,
                       ArrowArrayStream* out, std::string* error) {
    std::unique_ptr<parquet::arrow::FileReader> reader;
    std::shared_ptr<arrow::Schema> schema;
    if (!OpenReader(path, &reader, &schema, error)) return false;
    std::vector<int> indices;
    for (const std::string& name : columns) {
        const int index = schema->GetFieldIndex(name);
        if (index < 0) {
            *error = "column \"" + name + "\" is not in " + path;
            return false;
        }
        indices.push_back(index);
    }
    if (columns.empty()) {
        for (int i = 0; i < schema->num_fields(); ++i) indices.push_back(i);
    }
    // One row group at a time, read whole and handed out as its batches, so a
    // group's buffers are released before the next is read and the two
    // representations are never both resident for the whole file. The reader
    // owns the file, so releasing the stream closes it.
    class RowGroups : public arrow::RecordBatchReader {
       public:
        RowGroups(std::unique_ptr<parquet::arrow::FileReader> file,
                  std::shared_ptr<arrow::Schema> schema, std::vector<int> indices)
            : file_(std::move(file)),
              schema_(std::move(schema)),
              indices_(std::move(indices)) {}
        std::shared_ptr<arrow::Schema> schema() const override { return schema_; }
        arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override {
            while (true) {
                if (batches_) {
                    ARROW_RETURN_NOT_OK(batches_->ReadNext(batch));
                    if (*batch != nullptr) return arrow::Status::OK();
                    batches_.reset();
                    table_.reset();
                }
                if (group_ >= file_->num_row_groups()) {
                    batch->reset();
                    return arrow::Status::OK();
                }
                ARROW_ASSIGN_OR_RAISE(table_, file_->ReadRowGroup(group_++, indices_));
                batches_ = std::make_unique<arrow::TableBatchReader>(*table_);
            }
        }

       private:
        std::unique_ptr<parquet::arrow::FileReader> file_;
        std::shared_ptr<arrow::Schema> schema_;
        std::vector<int> indices_;
        int group_ = 0;
        std::shared_ptr<arrow::Table> table_;
        std::unique_ptr<arrow::TableBatchReader> batches_;
    };
    // The schema of the columns read, in the order they were asked for.
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (const int index : indices) fields.push_back(schema->field(index));
    auto rows = std::make_shared<RowGroups>(std::move(reader), arrow::schema(fields),
                                            std::move(indices));
    const arrow::Status status = arrow::ExportRecordBatchReader(std::move(rows), out);
    if (!status.ok()) {
        *error = "cannot export " + path + ": " + status.message();
        return false;
    }
    return true;
}

struct ParquetWriter::Impl {
    std::string path;
    std::shared_ptr<arrow::Schema> schema;
    std::unique_ptr<parquet::arrow::FileWriter> writer;
};

ParquetWriter::ParquetWriter() : impl_(std::make_unique<Impl>()) {}
ParquetWriter::~ParquetWriter() = default;

bool ParquetWriter::Open(const std::string& path, const ArrowSchema& schema,
                         std::string* error) {
    impl_->path = path;
    // Importing consumes the struct, so a copy is imported and the caller's is
    // left for the caller to release.
    ArrowSchema copy = schema;
    copy.release = [](ArrowSchema* s) { s->release = nullptr; };
    auto imported = arrow::ImportSchema(&copy);
    if (!imported.ok()) {
        *error =
            "cannot import the schema for " + path + ": " + imported.status().message();
        return false;
    }
    impl_->schema = std::move(*imported);
    auto sink = arrow::io::FileOutputStream::Open(path);
    if (!sink.ok()) {
        *error = "cannot create " + path + ": " + sink.status().message();
        return false;
    }
    auto props = parquet::WriterProperties::Builder()
                     .compression(parquet::Compression::SNAPPY)
                     ->build();
    auto writer = parquet::arrow::FileWriter::Open(
        *impl_->schema, arrow::default_memory_pool(), *sink, props);
    if (!writer.ok()) {
        *error =
            "cannot open parquet writer for " + path + ": " + writer.status().message();
        return false;
    }
    impl_->writer = std::move(*writer);
    return true;
}

bool ParquetWriter::Write(ArrowArray* batch, std::string* error) {
    auto imported = arrow::ImportRecordBatch(batch, impl_->schema);
    if (!imported.ok()) {
        if (batch->release != nullptr) batch->release(batch);
        *error = "cannot import a batch for " + impl_->path + ": " +
                 imported.status().message();
        return false;
    }
    const auto& rows = *imported;
    if (rows->num_rows() == 0) return true;
    // Each batch is one row group: `WriteTable` closes the group where
    // `WriteRecordBatch` would buffer batches into one until a million rows.
    auto table = arrow::Table::FromRecordBatches(impl_->schema, {rows});
    if (!table.ok()) {
        *error =
            "cannot table a batch for " + impl_->path + ": " + table.status().message();
        return false;
    }
    const arrow::Status status = impl_->writer->WriteTable(**table, rows->num_rows());
    if (!status.ok()) {
        *error = "writing a row group to " + impl_->path + ": " + status.message();
        return false;
    }
    return true;
}

bool ParquetWriter::Close(std::string* error) {
    if (!impl_->writer) return true;
    const arrow::Status closed = impl_->writer->Close();
    impl_->writer.reset();
    if (!closed.ok()) {
        *error = "closing " + impl_->path + ": " + closed.message();
        return false;
    }
    return true;
}

}  // namespace cpplink
