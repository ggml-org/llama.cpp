#pragma once

#include <string>
#include <vector>
#include <memory>


int sp_init(const std::string& model);
int sp_encode(const std::string& str,std::vector<int32_t>& token_ids);
