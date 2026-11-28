#pragma once

#include <cstddef>
#include <filesystem>
#include <map>
#include <string>

std::string read_ini_file(const std::filesystem::path & path);
std::string ini_version_from_file(const std::filesystem::path & path);
std::map<std::string, std::map<std::string, std::string>> parse_ini_from_file_v1(const std::filesystem::path & path);
size_t ini_line_number(const std::string & contents, size_t offset);
void warn_legacy_ini(const std::string & contents, const std::filesystem::path & path);
