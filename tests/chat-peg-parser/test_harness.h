#pragma once

#include "log.h"

#include <chrono>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

struct testing {
    std::vector<std::string> stack;
    int tests = 0;
    int assertions = 0;
    int failures = 0;
    int unnamed = 0;
    int exceptions = 0;

    std::string indent() {
        return std::string((stack.size() - 1) * 2, ' ');
    }

    template <typename F>
    void run_with_exceptions(F &&f, const char *ctx) {
        try {
            f();
        } catch (const std::exception &e) {
            ++failures;
            ++exceptions;
            LOG_ERR("%sUNHANDLED EXCEPTION (%s): %s\n", indent().c_str(), ctx, e.what());
        } catch (...) {
            ++failures;
            ++exceptions;
            LOG_ERR("%sUNHANDLED EXCEPTION (%s): unknown\n", indent().c_str(), ctx);
        }
    }

    void print_result(const std::string &label, const std::string &name, int new_failures, int new_assertions, const std::string &extra = "") {
        std::string ind = indent();
        std::string status = (new_failures == 0) ? "ok" : (std::to_string(new_failures) + " failed of");
        std::string extra_str = extra.empty() ? "" : (", " + extra);

        if (new_failures == 0) {
            LOG_INF("%s%s: %s [ok, %d assertion(s)%s]\n", ind.c_str(), label.c_str(), name.c_str(), new_assertions, extra_str.c_str());
        } else {
            LOG_ERR("%s%s: %s [%d failed of %d assertion(s)%s]\n", ind.c_str(), label.c_str(), name.c_str(), new_failures, new_assertions, extra_str.c_str());
        }
    }

    // Named test
    template <typename F>
    void test(const std::string &name, F f) {
        ++tests;
        stack.push_back(name);

        LOG_INF("%sBEGIN: %s\n", indent().c_str(), name.c_str());

        int before_failures = failures;
        int before_assertions = assertions;

        run_with_exceptions([&] { f(*this); }, "test");

        print_result("END", name,
            failures - before_failures,
            assertions - before_assertions);

        stack.pop_back();
    }

    // Unnamed test
    template <typename F>
    void test(F f) {
        test("test #" + std::to_string(++unnamed), f);
    }

    // Named benchmark
    template <typename F>
    void bench(const std::string &name, F f, int iterations = 100) {
        ++tests;
        stack.push_back(name);

        LOG_INF("%sBEGIN BENCH: %s\n", indent().c_str(), name.c_str());

        int before_failures = failures;
        int before_assertions = assertions;

        using clock = std::chrono::high_resolution_clock;

        std::chrono::microseconds duration(0);

        run_with_exceptions([&] {
            for (auto i = 0; i < iterations; i++) {
                auto start = clock::now();
                f();
                duration += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - start);
            }
        }, "bench");

        auto avg_elapsed = duration.count() / iterations;
        auto avg_elapsed_s = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count() / iterations;
        auto rate = (avg_elapsed_s > 0.0) ? (1.0 / avg_elapsed_s) : 0.0;

        print_result("END BENCH", name,
            failures - before_failures,
            assertions - before_assertions,
            std::to_string(iterations) + " iteration(s), " +
            "avg elapsed " + std::to_string(avg_elapsed) +
            " us (" + std::to_string(rate) + " /s)");

        stack.pop_back();
    }

    // Unnamed benchmark
    template <typename F>
    void bench(F f, int iterations = 100) {
        bench("bench #" + std::to_string(++unnamed), f, iterations);
    }

    // Assertions
    bool assert_true(bool cond) {
        return assert_true("", cond);
    }

    bool assert_true(const std::string &msg, bool cond) {
        ++assertions;
        if (!cond) {
            ++failures;
            if (msg.empty()) {
                LOG_ERR("%sASSERT TRUE FAILED\n", indent().c_str());
            } else {
                LOG_ERR("%sASSERT TRUE FAILED : %s\n", indent().c_str(), msg.c_str());
            }
            return false;
        }
        return true;
    }

    template <typename A, typename B>
    bool assert_equal(const A & expected, const B & actual) {
        return assert_equal("", expected, actual);
    }

    template <typename A, typename B>
    bool assert_equal(const std::string & msg, const A & expected, const B & actual) {
        ++assertions;
        if (!(actual == expected)) {
            ++failures;
            std::string ind = indent();

            std::ostringstream exp_ss, act_ss;
            exp_ss << expected;
            act_ss << actual;

            if (msg.empty()) {
                LOG_ERR("%sASSERT EQUAL FAILED\n", ind.c_str());
            } else {
                LOG_ERR("%sASSERT EQUAL FAILED : %s\n", ind.c_str(), msg.c_str());
            }
            LOG_ERR("%s  expected: %s\n", ind.c_str(), exp_ss.str().c_str());
            LOG_ERR("%s  actual  : %s\n", ind.c_str(), act_ss.str().c_str());
            return false;
        }
        return true;
    }

    // Print summary and return an exit code
    int summary() {
        LOG_INF("\n==== TEST SUMMARY ====\n");
        LOG_INF("tests      : %d\n", tests);
        LOG_INF("assertions : %d\n", assertions);
        LOG_INF("failures   : %d\n", failures);
        LOG_INF("exceptions : %d\n", exceptions);
        LOG_INF("======================\n");
        return failures == 0 ? 0 : 1;
    }
};
