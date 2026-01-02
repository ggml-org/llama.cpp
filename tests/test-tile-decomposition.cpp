#include <cassert>
#include <cstdio>
#include <vector>

// Inline the helpers for testing (will be in common.hpp)
inline int tile_count(int blocks) {
    int count = 0;
    while (blocks > 0) {
        int tile_size = 1;
        while (tile_size * 2 <= blocks && tile_size < 32) {
            tile_size *= 2;
        }
        count++;
        blocks -= tile_size;
    }
    return count;
}

inline int tile_size_at(int blocks, int tile_idx) {
    int idx = 0;
    while (blocks > 0) {
        int tile_size = 1;
        while (tile_size * 2 <= blocks && tile_size < 32) {
            tile_size *= 2;
        }
        if (idx == tile_idx) return tile_size;
        blocks -= tile_size;
        idx++;
    }
    return 0;
}

inline int tile_offset_at(int blocks, int tile_idx) {
    int idx = 0, offset = 0;
    while (blocks > 0 && idx < tile_idx) {
        int tile_size = 1;
        while (tile_size * 2 <= blocks && tile_size < 32) {
            tile_size *= 2;
        }
        offset += tile_size;
        blocks -= tile_size;
        idx++;
    }
    return offset;
}

int main() {
    // Test 16 blocks: single tile of 16
    assert(tile_count(16) == 1);
    assert(tile_size_at(16, 0) == 16);
    assert(tile_offset_at(16, 0) == 0);
    printf("PASS: 16 blocks = 1x16\n");

    // Test 56 blocks: 32 + 16 + 8 (Mistral FFN case)
    assert(tile_count(56) == 3);
    assert(tile_size_at(56, 0) == 32);
    assert(tile_size_at(56, 1) == 16);
    assert(tile_size_at(56, 2) == 8);
    assert(tile_offset_at(56, 0) == 0);
    assert(tile_offset_at(56, 1) == 32);
    assert(tile_offset_at(56, 2) == 48);
    printf("PASS: 56 blocks = 32+16+8\n");

    // Test 125 blocks: 32+32+32+16+8+4+1 = 7 tiles (max tile size is 32)
    assert(tile_count(125) == 7);
    assert(tile_size_at(125, 0) == 32);
    assert(tile_size_at(125, 1) == 32);
    assert(tile_size_at(125, 2) == 32);
    assert(tile_size_at(125, 3) == 16);
    assert(tile_size_at(125, 4) == 8);
    assert(tile_size_at(125, 5) == 4);
    assert(tile_size_at(125, 6) == 1);
    assert(tile_offset_at(125, 0) == 0);
    assert(tile_offset_at(125, 1) == 32);
    assert(tile_offset_at(125, 2) == 64);
    assert(tile_offset_at(125, 3) == 96);
    assert(tile_offset_at(125, 4) == 112);
    assert(tile_offset_at(125, 5) == 120);
    assert(tile_offset_at(125, 6) == 124);
    printf("PASS: 125 blocks = 32+32+32+16+8+4+1\n");

    // Test 32 blocks: single full tile
    assert(tile_count(32) == 1);
    assert(tile_size_at(32, 0) == 32);
    printf("PASS: 32 blocks = 1x32\n");

    // Test edge cases
    assert(tile_count(1) == 1);
    assert(tile_size_at(1, 0) == 1);
    printf("PASS: 1 block = 1x1\n");

    assert(tile_count(33) == 2);
    assert(tile_size_at(33, 0) == 32);
    assert(tile_size_at(33, 1) == 1);
    printf("PASS: 33 blocks = 32+1\n");

    printf("\nAll tile decomposition tests passed!\n");
    return 0;
}
