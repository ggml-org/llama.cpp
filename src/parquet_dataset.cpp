#ifdef LLAMA_PARQUET
#include "parquet_dataset.h"
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>
#include "llama-impl.h"

std::vector<llama_token> load_parquet_dataset(const std::string &path, const std::string &column) {
    arrow::MemoryPool *pool = arrow::default_memory_pool();
    std::shared_ptr<arrow::io::RandomAccessFile> infile;
    PARQUET_ASSIGN_OR_THROW(infile, arrow::io::ReadableFile::Open(path));
    arrow::Result<std::unique_ptr<parquet::arrow::FileReader>> reader_raw;
    PARQUET_ASSIGN_OR_THROW(reader_raw, parquet::arrow::OpenFile(infile, pool));

    std::unique_ptr<parquet::arrow::FileReader> reader = std::move(reader_raw.ValueUnsafe());
    std::shared_ptr<arrow::Table> table;
    PARQUET_THROW_NOT_OK(reader->ReadTable(&table));

    auto field = table->schema()->GetFieldByName(column);
    if (!field || !field->type()->Equals(arrow::list(arrow::int32()))) {
        LLAMA_LOG_ERROR("Parquet column '%s' missing or not list<int32>", column.c_str());
        return {};
    }

    auto col = table->GetColumnByName(column);
    std::vector<llama_token> tokens;
    for (int chunk = 0; chunk < col->num_chunks(); ++chunk) {
        auto list_arr = std::static_pointer_cast<arrow::ListArray>(col->chunk(chunk));
        auto values_arr = std::static_pointer_cast<arrow::Int32Array>(list_arr->values());
        // get raw offsets (int32_t or int64_t based on ListArray template)
        const auto *offsets = list_arr->raw_value_offsets();
        // offsets length = list_arr->length() + 1
        int64_t values_length = values_arr->length();
        for (int64_t i = 0; i < list_arr->length(); ++i) {
            int64_t start = offsets[i];
            int64_t end   = offsets[i + 1];
            // Clamp end
            if (start < 0) start = 0;
            if (end > values_length) end = values_length;
            for (int64_t j = start; j < end; ++j) {
                tokens.push_back(static_cast<llama_token>(values_arr->Value(j)));
            }
        }
    }
    return tokens;
}
#endif // LLAMA_PARQUET
