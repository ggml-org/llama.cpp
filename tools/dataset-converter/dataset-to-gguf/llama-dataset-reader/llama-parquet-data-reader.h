#pragma once
#ifdef LLAMA_PARQUET
#include "llama.h"  // For llama_token

// Include necessary Apache Arrow and Parquet headers
// You will need to link against these libraries (e.g., -larrow -lparquet)
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

#include <memory>  // For std::unique_ptr
#include <string>
#include <vector>

#include "llama-dataset-reader.h"

// Implementation of DatasetReader for reading Parquet files.
// This class will handle reading tokenized sequences from a Parquet file.
struct llama_parquet_dataset_reader : public llama_dataset_reader {
    // Constructor.
    // model: Pointer to the llama model for tokenization (can be nullptr if data is pre-tokenized).
    // max_seq_len: Maximum sequence length for truncation.
    // pre_tokenized: If true, input data is already tokenized (token IDs in a numeric column).
    // text_column_name: Name of the column containing raw text data.
    // tokens_column_name: Name of the column containing pre-tokenized data (list<int32>).
    llama_parquet_dataset_reader(const struct llama_model * model, int32_t max_seq_len, bool pre_tokenized,
                                 const std::string & dataset_column_name);

    // Destructor.
    ~llama_parquet_dataset_reader();

    // Opens the Parquet file for reading.
    // path: Path to the Parquet file.
    // Returns true if the source is successfully opened, otherwise false.
    bool open(const std::string & path) override;

    // Reads the next sequence of tokens from the Parquet file.
    // tokens: Vector where the read tokens will be stored.
    // Returns true if a sequence is successfully read, otherwise false (including end of file).
    bool read_next_sequence(std::vector<llama_token> & tokens) override;

    // Closes the Parquet file.
    void close() override;

    // Resets the reader to the beginning of the Parquet file.
    // Returns true if reset is successful, otherwise false.
    bool reset() override;

    // Method to get the total number of sequences in the dataset.
    // For Parquet files, this will be the number of rows obtained from metadata.
    uint64_t total_sequences() const override;

  private:
    const struct llama_model * model_;                            // Llama model for tokenization (if needed)
    int32_t                    max_seq_len_;                      // Maximum sequence length
    bool                       pre_tokenized_;                    // Flag for pre-tokenized data

    std::shared_ptr<arrow::io::ReadableFile>    input_file_;      // Arrow file handle
    std::unique_ptr<parquet::arrow::FileReader> parquet_reader_;  // Parquet reader
    std::shared_ptr<arrow::Table>               current_table_;   // Current table batch being processed
    std::shared_ptr<arrow::ChunkedArray>        chunked_array_;   // Member to store the chunked array

    int                                             current_row_group_index_;   // Current row group index
    std::shared_ptr<parquet::arrow::RowGroupReader> current_row_group_reader_;  // Reader for the current row group

    int64_t     current_row_in_table_;  // Current row index within the current_table_
    int         current_column_index_;  // Index of the column containing text/tokens
    std::string m_file_path;            // Path to the Parquet file

    std::string dataset_column_name_;      // Configurable name for column

    // Private helper to get the next batch of data (now a row group)
    bool llama_parquet_dataset_reader_get_next_batch();
};
#endif
