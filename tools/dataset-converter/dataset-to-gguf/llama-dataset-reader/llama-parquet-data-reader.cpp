#ifdef LLAMA_PARQUET
#include "llama-parquet-data-reader.h"

#include <algorithm>  // For std::min
#include <iostream>   // For std::cerr

// Constructor
llama_parquet_dataset_reader::llama_parquet_dataset_reader(const struct llama_model * model, int32_t max_seq_len,
                                                           bool pre_tokenized, const std::string & text_column_name,
                                                           const std::string & tokens_column_name) :
    model_(model),
    max_seq_len_(max_seq_len),
    pre_tokenized_(pre_tokenized),
    current_row_group_index_(0),  // Initialize row group index
    current_row_in_table_(0),
    current_column_index_(-1),    // Initialize to -1, will be set in open
    text_column_name_(text_column_name),
    tokens_column_name_(tokens_column_name) {}

// Destructor
llama_parquet_dataset_reader::~llama_parquet_dataset_reader() {
    close();
    m_file_path.clear();  // Clear the stored path only on destruction
}

// Opens the Parquet file for reading.
bool llama_parquet_dataset_reader::open(const std::string & path) {
    // Close any previously open file
    // Note: m_file_path is NOT cleared here, it's preserved for reset()
    close();

    m_file_path = path;  // Store the file path for reset()

    // Open the Parquet file
    arrow::Status status = arrow::io::ReadableFile::Open(path).Value(&input_file_);
    if (!status.ok()) {
        std::cerr << "Error (llama_parquet_dataset_reader::open): Failed to open Parquet file '" << path
                  << "': " << status.ToString() << std::endl;
        return false;
    }

    // Create a Parquet reader using parquet::arrow::OpenFile
    arrow::Result<std::unique_ptr<parquet::arrow::FileReader>> reader_raw =
        parquet::arrow::OpenFile(input_file_, arrow::default_memory_pool());

    if (!reader_raw.ok()) {
        std::cerr << "Error (llama_parquet_dataset_reader::open): Failed to create Parquet file reader for '" << path
                  << "': " << reader_raw.status().ToString() << std::endl;
        close();
        return false;
    }
    parquet_reader_ = std::move(reader_raw.ValueUnsafe());

    // Get the schema to determine the correct column index
    std::shared_ptr<arrow::Schema> schema;
    status = parquet_reader_->GetSchema(&schema);  // Corrected: Use GetSchema and pass by address
    if (!status.ok() || schema == nullptr) {
        std::cerr << "Error (llama_parquet_dataset_reader::open): Failed to get schema from Parquet file: "
                  << status.ToString() << std::endl;
        close();
        return false;
    }

    // Determine the column index based on pre_tokenized_ flag
    if (pre_tokenized_) {
        current_column_index_ = schema->GetFieldIndex(tokens_column_name_);  // Use configurable name
        if (current_column_index_ == -1) {
            std::cerr << "Error (llama_parquet_dataset_reader::open): Pre-tokenized mode selected, but column '"
                      << tokens_column_name_ << "' not found in Parquet schema." << std::endl;
            close();
            return false;
        }
        // Validate column type: should be List<Int32>
        if (schema->field(current_column_index_)->type()->id() != arrow::Type::LIST) {
            std::cerr << "Error (llama_parquet_dataset_reader::open): Column '" << tokens_column_name_
                      << "' is not of LIST type as expected for pre-tokenized data. Actual type: "
                      << schema->field(current_column_index_)->type()->ToString() << std::endl;
            close();
            return false;
        }
        auto list_type = std::static_pointer_cast<arrow::ListType>(schema->field(current_column_index_)->type());
        if (list_type->value_type()->id() != arrow::Type::INT32) {
            std::cerr << "Error (llama_parquet_dataset_reader::open): List items in column '" << tokens_column_name_
                      << "' are not of INT32 type as expected. Actual value type: "
                      << list_type->value_type()->ToString() << std::endl;
            close();
            return false;
        }

    } else {
        current_column_index_ = schema->GetFieldIndex(text_column_name_);  // Use configurable name
        if (current_column_index_ == -1) {
            std::cerr << "Error (llama_parquet_dataset_reader::open): Raw text mode selected, but column '"
                      << text_column_name_ << "' not found in Parquet schema." << std::endl;
            close();
            return false;
        }
        // Validate column type: should be String
        if (schema->field(current_column_index_)->type()->id() != arrow::Type::STRING) {
            std::cerr << "Error (llama_parquet_dataset_reader::open): Column '" << text_column_name_
                      << "' is not of STRING type as expected for raw text. Actual type: "
                      << schema->field(current_column_index_)->type()->ToString() << std::endl;
            close();
            return false;
        }
    }

    // Initialize row group index
    current_row_group_index_ = 0;
    // Read the first batch (row group)
    return llama_parquet_dataset_reader_get_next_batch();
}

// Reads the next sequence of tokens from the Parquet file.
bool llama_parquet_dataset_reader::read_next_sequence(std::vector<llama_token> & tokens) {
    tokens.clear();

    // If current_table_ is null or we've processed all rows in the current batch, get the next batch (row group)
    if (!current_table_ || current_row_in_table_ >= current_table_->num_rows()) {
        if (!llama_parquet_dataset_reader_get_next_batch()) {
            return false;  // No more batches/row groups or error getting next batch
        }
    }

    if (!current_table_ || current_table_->num_rows() == 0) {
        return false;  // Should not happen if get_next_batch was successful, but as a safeguard
    }

    // Assuming single chunk for simplicity. For multi-chunk columns, you'd iterate through chunks.
    // When reading a column from a row group, it typically returns a single chunk.
    std::shared_ptr<arrow::Array> column_array =
        current_table_->column(0)->chunk(0);  // column(0) because we read only one column into current_table_

    if (pre_tokenized_) {
        // Pre-tokenized data: read List<Int32> array
        auto list_array  = std::static_pointer_cast<arrow::ListArray>(column_array);
        auto value_array = std::static_pointer_cast<arrow::Int32Array>(list_array->values());

        if (list_array->IsNull(current_row_in_table_)) {
            // Handle null list (empty sequence)
            current_row_in_table_++;
            return true;
        }

        int32_t start_offset      = list_array->value_offset(current_row_in_table_);
        int32_t end_offset        = list_array->value_offset(current_row_in_table_ + 1);
        int32_t num_tokens_in_row = end_offset - start_offset;

        tokens.reserve(std::min((int32_t) max_seq_len_, num_tokens_in_row));
        for (int32_t i = 0; i < num_tokens_in_row && i < max_seq_len_; ++i) {
            tokens.push_back(static_cast<llama_token>(value_array->Value(start_offset + i)));
        }

    } else {
        // Raw text data: read String array and tokenize
        if (!model_) {
            std::cerr << "Error (llama_parquet_dataset_reader::read_next_sequence): Llama model not provided for "
                         "tokenization of raw text."
                      << std::endl;
            return false;
        }

        auto string_array = std::static_pointer_cast<arrow::StringArray>(column_array);
        if (string_array->IsNull(current_row_in_table_)) {
            // Handle null string (empty sequence)
            current_row_in_table_++;
            return true;
        }

        std::string text = string_array->GetString(current_row_in_table_);
        std::vector<llama_token> tokens_buffer(max_seq_len_);  // Use a temporary buffer for tokenization

        int n_tokens = llama_tokenize(llama_model_get_vocab(model_), text.c_str(), text.length(), tokens_buffer.data(),
                                      max_seq_len_, false, true);
        if (n_tokens < 0) {
            std::cerr << "Error (llama_parquet_dataset_reader::read_next_sequence): Tokenization failed for text: '"
                      << text << "'" << std::endl;
            current_row_in_table_++;
            return true;  // Return true with empty tokens to continue processing
        }
        tokens.assign(tokens_buffer.begin(), tokens_buffer.begin() + n_tokens);
    }

    current_row_in_table_++;
    return true;
}

// Closes the Parquet file.
void llama_parquet_dataset_reader::close() {
    parquet_reader_.reset();
    current_row_group_reader_.reset();  // Reset row group reader
    current_table_.reset();
    chunked_array_.reset();             // Reset chunked array
    if (input_file_) {
        arrow::Status status = input_file_->Close();
        if (!status.ok()) {
            std::cerr << "Warning (llama_parquet_dataset_reader::close): Failed to close Arrow file: "
                      << status.ToString() << std::endl;
        }
    }
    input_file_.reset();
    current_row_group_index_ = 0;  // Reset row group index
    current_row_in_table_    = 0;
    current_column_index_    = -1;
    // m_file_path is NOT cleared here. It's preserved for reset()
}

// Resets the reader to the beginning of the Parquet file.
bool llama_parquet_dataset_reader::reset() {
    if (m_file_path.empty()) {  // Check if path is stored
        std::cerr << "Error (llama_parquet_dataset_reader::reset): Cannot reset, file path was not stored."
                  << std::endl;
        return false;
    }
    // Re-open the file and re-initialize the reader
    return open(m_file_path);  // Use the stored path
}

// Private helper to get the next batch of data (now a row group)
bool llama_parquet_dataset_reader::llama_parquet_dataset_reader_get_next_batch() {
    current_table_.reset();     // Clear previous table
    current_row_in_table_ = 0;  // Reset row index for new table
    chunked_array_.reset();     // Reset chunked array for new batch

    if (!parquet_reader_) {
        std::cerr << "Error (llama_parquet_dataset_reader::llama_parquet_dataset_reader_get_next_batch): Parquet "
                     "reader is not initialized."
                  << std::endl;
        return false;
    }

    if (current_row_group_index_ >= parquet_reader_->num_row_groups()) {
        return false;  // No more row groups
    }

    // Get the reader for the current row group
    current_row_group_reader_ = parquet_reader_->RowGroup(current_row_group_index_);
    if (!current_row_group_reader_) {
        std::cerr << "Error (llama_parquet_dataset_reader::llama_parquet_dataset_reader_get_next_batch): Failed to get "
                     "row group reader for index "
                  << current_row_group_index_ << std::endl;
        return false;
    }

    // Get the ColumnChunkReader for the specific column
    std::shared_ptr<parquet::arrow::ColumnChunkReader> column_chunk_reader =
        current_row_group_reader_->Column(current_column_index_);
    if (!column_chunk_reader) {
        std::cerr << "Error (llama_parquet_dataset_reader::llama_parquet_dataset_reader_get_next_batch): Failed to get "
                     "column chunk reader for column "
                  << current_column_index_ << " in row group " << current_row_group_index_ << std::endl;
        return false;
    }

    // Read the column data into a ChunkedArray
    arrow::Status status = column_chunk_reader->Read(&chunked_array_);  // Use member variable
    if (!status.ok()) {
        std::cerr << "Error (llama_parquet_dataset_reader::llama_parquet_dataset_reader_get_next_batch): Failed to "
                     "read column "
                  << current_column_index_ << " from row group " << current_row_group_index_ << ": "
                  << status.ToString() << std::endl;
        return false;
    }

    // Get the schema from the parquet_reader_ to construct the table
    std::shared_ptr<arrow::Schema> schema;
    status = parquet_reader_->GetSchema(&schema);
    if (!status.ok() || schema == nullptr) {
        std::cerr << "Error (llama_parquet_dataset_reader::llama_parquet_dataset_reader_get_next_batch): Failed to get "
                     "schema from Parquet reader for column "
                  << current_column_index_ << std::endl;
        return false;
    }

    // Get the field for the current column index
    std::shared_ptr<arrow::Field> column_field = schema->field(current_column_index_);
    if (column_field == nullptr) {
        std::cerr << "Error (llama_parquet_dataset_reader::llama_parquet_dataset_reader_get_next_batch): Column field "
                     "is null for index "
                  << current_column_index_ << std::endl;
        return false;
    }

    current_table_ = arrow::Table::Make(arrow::schema({ column_field }),  // Create a schema with just this column
                                        { chunked_array_ }                // Pass the chunked array as the column data
    );

    if (!current_table_ || current_table_->num_rows() == 0) {
        return false;  // No data in this row group
    }

    current_row_group_index_++;  // Move to the next row group for the next call
    return true;
}

// Method to get the total number of sequences in the dataset.
// For Parquet files, this will be the number of rows obtained from metadata.
uint64_t llama_parquet_dataset_reader::total_sequences() const {
    if (!parquet_reader_) {
        std::cerr << "Error (llama_parquet_dataset_reader::total_sequences): Parquet reader is not initialized."
                  << std::endl;
        return 0;
    }
    // Total number of rows in the Parquet file
    return parquet_reader_->parquet_reader()->metadata()->num_rows();
}
#endif
