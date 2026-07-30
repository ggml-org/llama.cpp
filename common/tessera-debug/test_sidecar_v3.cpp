#include "tessera-sidecar-v3.h"

#include <cmath>
#include <cstdio>
#include <cstring>

static const char * TEST_PATH = "/tmp/test_sidecar.tdqt";

static int write_test_file(const char * path) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        return -1;
    }

    const int64_t  rows      = 4;
    const int64_t  cols      = 8;
    const uint32_t version   = 3;
    const uint32_t dtype     = 0;
    const float    threshold = 6.0f;
    const int64_t  total     = 3;

    fwrite("TDQT", 1, 4, f);
    fwrite(&version,   sizeof(version),   1, f);
    fwrite(&rows,      sizeof(rows),      1, f);
    fwrite(&cols,      sizeof(cols),      1, f);
    fwrite(&dtype,     sizeof(dtype),     1, f);
    fwrite(&threshold, sizeof(threshold), 1, f);
    fwrite(&total,     sizeof(total),     1, f);

    const int32_t counts[4] = { 1, 0, 2, 0 };
    fwrite(counts, sizeof(int32_t), 4, f);

    for (int r = 0; r < 4; r++) {
        uint64_t timing   = (uint64_t)(r + 1) * 1000;
        uint32_t kid      = (uint32_t)(r + 10);
        uint32_t dc       = 1;
        uint64_t reserved = 0;
        fwrite(&timing,   sizeof(timing),   1, f);
        fwrite(&kid,      sizeof(kid),      1, f);
        fwrite(&dc,       sizeof(dc),       1, f);
        fwrite(&reserved, sizeof(reserved), 1, f);
    }

    float data[32];
    for (int i = 0; i < 32; i++) {
        data[i] = (float) i * 0.5f;
    }
    fwrite(data, sizeof(float), 32, f);

    fclose(f);
    return 0;
}

static int check_sha256_test_vector() {
    // SHA-256("abc") = ba7816bf...
    const char * abc_path = "/tmp/test_sidecar_abc.bin";
    FILE * f = fopen(abc_path, "wb");
    if (!f) {
        return -1;
    }
    fwrite("abc", 1, 3, f);
    fclose(f);

    uint8_t hash[32];
    ts_sidecar_v3_sha256(abc_path, hash);

    static const uint8_t expect[32] = {
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
        0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
        0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
        0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
    };
    if (memcmp(hash, expect, 32) != 0) {
        fprintf(stderr, "FAIL: sha256 test vector mismatch\n");
        return -1;
    }
    return 0;
}

int main() {
    if (write_test_file(TEST_PATH) != 0) {
        fprintf(stderr, "FAIL: could not write test file\n");
        return 1;
    }

    // header-only read
    ts_sidecar_v3_header hdr;
    if (ts_sidecar_v3_read_header(TEST_PATH, &hdr) != 0) {
        fprintf(stderr, "FAIL: read_header\n");
        return 1;
    }
    if (hdr.version != 3 || hdr.rows != 4 || hdr.cols != 8 ||
        hdr.dtype != 0 || hdr.outlier_threshold != 6.0f ||
        hdr.outlier_count_total != 3) {
        fprintf(stderr, "FAIL: header field mismatch\n");
        return 1;
    }

    // full read
    ts_sidecar_v3 sc;
    std::string err;
    if (ts_sidecar_v3_read(TEST_PATH, &sc, &err) != 0) {
        fprintf(stderr, "FAIL: read: %s\n", err.c_str());
        return 1;
    }

    if (sc.header.rows != 4 || sc.header.cols != 8 ||
        sc.header.outlier_count_total != 3) {
        fprintf(stderr, "FAIL: header dims\n");
        return 1;
    }

    const int32_t expect_counts[4] = { 1, 0, 2, 0 };
    for (int r = 0; r < 4; r++) {
        if (sc.row_outlier_counts[r] != expect_counts[r]) {
            fprintf(stderr, "FAIL: outlier count row %d: got %d, want %d\n",
                    r, sc.row_outlier_counts[r], expect_counts[r]);
            return 1;
        }
    }

    for (int r = 0; r < 4; r++) {
        const auto & m = sc.row_meta[r];
        if (m.timing_ns != (uint64_t)(r + 1) * 1000 ||
            m.kernel_id != (uint32_t)(r + 10) ||
            m.dispatch_count != 1 ||
            m.reserved != 0) {
            fprintf(stderr, "FAIL: row meta %d\n", r);
            return 1;
        }
    }

    for (int i = 0; i < 32; i++) {
        float expect = (float) i * 0.5f;
        if (sc.data[i] != expect) {
            fprintf(stderr, "FAIL: data[%d] = %f, want %f\n", i, sc.data[i], expect);
            return 1;
        }
    }

    // SHA-256 of the sidecar file: must be non-zero
    uint8_t hash[32];
    ts_sidecar_v3_sha256(TEST_PATH, hash);
    bool nonzero = false;
    for (int i = 0; i < 32; i++) {
        if (hash[i] != 0) {
            nonzero = true;
            break;
        }
    }
    if (!nonzero) {
        fprintf(stderr, "FAIL: sha256 is all zeros\n");
        return 1;
    }

    printf("sha256: ");
    for (int i = 0; i < 32; i++) {
        printf("%02x", hash[i]);
    }
    printf("\n");

    // SHA-256 known-answer test
    if (check_sha256_test_vector() != 0) {
        return 1;
    }

    printf("PASS\n");
    return 0;
}
