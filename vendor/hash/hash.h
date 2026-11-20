#pragma once

// C++ wrapper for the vendored hash functions

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

// returns the raw SHA-256 digest
std::array<uint8_t, 32> hash_sha256(const void * data, size_t len);

// returns the SHA-256 digest as a lowercase hex string
std::string hash_sha256_hex(const void * data, size_t len);
