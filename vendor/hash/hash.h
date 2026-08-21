#pragma once

// C++ wrapper for the vendored hash functions

#include <cstddef>
#include <string>

// returns the SHA-256 digest as a lowercase hex string
std::string hash_sha256_hex(const void * data, size_t len);
