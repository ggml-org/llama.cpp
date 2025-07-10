#pragma once

#include <string>   // For std::string
#include <vector>   // For std::vector

#include "llama.h"  // For llama_token

// Abstract base class for reading dataset.
// Defines the interface that all concrete readers must implement.
struct llama_dataset_reader {
    // Virtual destructor for correct deletion of derived classes.
    virtual ~llama_dataset_reader() = default;

    // Method to open the data source.
    // path: path to the file or other data source identifier.
    // Returns true if the source is successfully opened, otherwise false.
    virtual bool open(const std::string & path) = 0;

    // Method to read the next sequence of tokens.
    // tokens: vector where the read tokens will be stored.
    // Returns true if a sequence is successfully read, otherwise false (including end of file).
    virtual bool read_next_sequence(std::vector<llama_token> & tokens) = 0;

    // Method to close the data source.
    virtual void close() = 0;

    // Method to reset the reader to the beginning of the data source.
    // Used for the second pass over the data.
    virtual bool reset() = 0;

    // Method to get the total number of sequences in the dataset.
    // Can be implemented differently for various data source types.
    // Returns 0 if the count is unknown or not applicable.
    virtual uint64_t total_sequences() const = 0;
};
