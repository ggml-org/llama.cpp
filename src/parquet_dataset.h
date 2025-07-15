#ifndef PARQUET_DATASET_H
#define PARQUET_DATASET_H
#include <string>
#include <vector>
#include "llama.h"

#ifdef LLAMA_PARQUET
std::vector<llama_token> load_parquet_dataset(const std::string &path, const std::string &column);
#endif
#endif  //
