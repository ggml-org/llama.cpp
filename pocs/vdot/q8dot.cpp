#include <cstdio>
#include <type_traits>
#include <vector>
#include <random>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <cstring>
#include <array>
#include <type_traits>

#include <ggml.h>
#include <ggml-cpu.h>

constexpr int kVecSize = 1 << 16;

// Copy-pasted from ggml.c
#define QK4_0 32
typedef struct {
    float   d;          // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(float) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK4_1 32
typedef struct {
    float   d;          // delta
    float   m;          // min
    uint8_t qs[QK4_1 / 2];  // nibbles / quants
} block_q4_1;
static_assert(sizeof(block_q4_1) == sizeof(float) * 2 + QK4_1 / 2, "wrong q4_1 block size/padding");

// Copy-pasted from ggml.c
#define QK8_0 32
typedef struct {
    float   d;          // delta
    float   s;          // d * sum(qs[i])
    int8_t  qs[QK8_0];  // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == 2*sizeof(float) + QK8_0, "wrong q8_0 block size/padding");

static_assert(QK4_1 == QK8_0, "QK4_1 and QK8_0 must be the same");
static_assert(QK4_0 == QK8_0, "QK4_0 and QK8_0 must be the same");

template <typename T>
static void fillQ4blocks(std::vector<T>& blocks, std::mt19937& rndm) {
    for (auto& b : blocks) {
        b.d = 1;
        for (int i=0; i<QK4_1/2; ++i) {
            uint8_t v1 = rndm() >> 28;
            uint8_t v2 = rndm() >> 28;
            b.qs[i] = v1 | (v2 << 4);
        }
    }
}

static void fillQ80blocks(std::vector<block_q8_0>& blocks, std::mt19937& rndm) {
    for (auto& b : blocks) {
        b.d = 1;
        int sum = 0;
        for (int i=0; i<QK8_0; ++i) {
            b.qs[i] = (rndm() >> 24) - 128;
            sum += b.qs[i];
        }
        b.s = b.d * sum;
    }
}

static float simpleDot(const block_q4_0& x, const block_q8_0& y) {
    int s1 = 0; //, s2 = 0;
    for (int i=0; i<QK4_1/2; i+=2) {
        int v1 = x.qs[i+0] & 0xf;
        int v2 = x.qs[i+0] >> 4;
        int v3 = x.qs[i+1] & 0xf;
        int v4 = x.qs[i+1] >> 4;
        int j = 2*i;
        s1 += v1*y.qs[j] + v2*y.qs[j+1] + v3*y.qs[j+2] + v4*y.qs[j+3];
        //s2 += y.qs[j] + y.qs[j+1] + y.qs[j+2] + y.qs[j+3];
    }
    return y.d * x.d * s1 - 8 * x.d * y.s;
    //return y.d * x.d * (s1 - 8 * s2);
}

static float simpleDot(const block_q4_1& x, const block_q8_0& y) {
    int s1 = 0; //, s2 = 0;
    for (int i=0; i<QK4_1/2; i+=2) {
        int v1 = x.qs[i+0] & 0xf;
        int v2 = x.qs[i+0] >> 4;
        int v3 = x.qs[i+1] & 0xf;
        int v4 = x.qs[i+1] >> 4;
        int j = 2*i;
        s1 += v1*y.qs[j] + v2*y.qs[j+1] + v3*y.qs[j+2] + v4*y.qs[j+3];
        //s2 += y.qs[j] + y.qs[j+1] + y.qs[j+2] + y.qs[j+3];
    }
    return y.d * x.d * s1 + y.s * x.m;
    //return y.d * (x.d * s1 + x.m * s2);
}

struct Stat {
    double sum = 0, sumt = 0, sumt2 = 0, maxt = 0;
    int nloop = 0;
    void addResult(double s, double t) {
        sum += s;
        sumt += t; sumt2 += t*t; maxt = std::max(maxt, t);
        ++nloop;
    }
    void reportResult(const char* title) const {
        if (nloop < 1) {
            printf("%s(%s): no result\n",__func__,title);
            return;
        }
        printf("============ %s\n",title);
        printf("<dot> = %g\n",sum/nloop);
        auto t = sumt/nloop, dt = sumt2/nloop - t*t;
        if (dt > 0) dt = sqrt(dt);
        printf("<time> = %g +/- %g us. Max. time = %g us.\n",t,dt,maxt);
    }
};


int main(int argc, char** argv) {

    int nloop = argc > 1 ? atoi(argv[1]) : 10;
    int type  = argc > 2 ? atoi(argv[2]) : 1;

    std::mt19937 rndm(1234);

    std::vector<block_q4_1> x41;
    std::vector<block_q4_0> x40;
    std::vector<block_q8_0> y(kVecSize);
    if (type == 0) x40.resize(kVecSize);
    else {
        x41.resize(kVecSize);
        for (auto& b : x41) b.m = 1;
    }

    auto ggml_type = type == 0 ? GGML_TYPE_Q4_0 : GGML_TYPE_Q4_1;

    const auto * funcs = ggml_get_type_traits_cpu(ggml_type);

    Stat simple, ggml;

    for (int iloop=0; iloop<nloop; ++iloop) {

        if (type == 0) fillQ4blocks(x40, rndm);
        else fillQ4blocks(x41, rndm);
        fillQ80blocks(y, rndm);

        auto t1 = std::chrono::high_resolution_clock::now();
        double s = 0;
        if (type == 0) for (int i=0; i<kVecSize; ++i) s += simpleDot(x40[i], y[i]);
        else for (int i=0; i<kVecSize; ++i) s += simpleDot(x41[i], y[i]);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto t = 1e-3*std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
        if (iloop > 3) simple.addResult(s, t);

        t1 = std::chrono::high_resolution_clock::now();
        float fs;
        if (type == 0) funcs->vec_dot(kVecSize * QK4_1, &fs, 0, x40.data(), 0, y.data(), 0, 1);
        else funcs->vec_dot(kVecSize * QK4_1, &fs, 0, x41.data(), 0, y.data(), 0, 1);
        t2 = std::chrono::high_resolution_clock::now();
        t = 1e-3*std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
        if (iloop > 3) ggml.addResult(fs, t);

    }

    // Report the time (and the average of the dot products so the compiler does not come up with the idea
    // of optimizing away the function calls after figuring that the result is not used).
    simple.reportResult("Simple");
    ggml.reportResult("ggml");
    return 0;
}
