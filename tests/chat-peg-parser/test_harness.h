#pragma once

#include "log.h"
#include <chrono>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct testing {
    std::ostream &out;
    std::vector<std::string> stack;
    int tests = 0;
    int assertions = 0;
    int failures = 0;
    int unnamed = 0;
    int exceptions = 0;

    explicit testing(std::ostream &os = std::cout) : out(os) {}

    void indent() const {
        for (std::size_t i = 0; i < stack.size() - 1; ++i) {
            LOG_ERR("  ");
        }
    }

    template <typename F>
    void run_with_exceptions(F &&f, const char *ctx) {
        try {
            f();
        } catch (const std::exception &e) {
            ++failures;
            ++exceptions;
            indent();
            out << "UNHANDLED EXCEPTION (" << ctx << "): " << e.what() << "\n";
            throw e;
        } catch (...) {
            ++failures;
            ++exceptions;
            indent();
            out << "UNHANDLED EXCEPTION (" << ctx << "): unknown\n";
            throw;
        }
    }

    void print_result(const std::string &label, const std::string &name, int new_failures, int new_assertions, const std::string &extra = "") const {
        indent();
        LOG_ERR("%s: %s [", label.c_str(), name.c_str());
        if (new_failures == 0) {
            LOG_ERR("ok, ");
        } else {
            LOG_ERR("%d failed of ", new_failures);
        }
        LOG_ERR("%d assertion(s)", new_assertions);
        if (!extra.empty()) {
            LOG_ERR(", %s", extra.c_str());
        }
        LOG_ERR("]\n");
    }

    // Named test
    template <typename F>
    void test(const std::string &name, F f) {
        ++tests;
        stack.push_back(name);

        indent();
        LOG_ERR("BEGIN: %s\n", name.c_str());

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

        indent();
        out << "BEGIN BENCH: " << name << "\n";

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
            indent();
            out << "ASSERT TRUE FAILED";
            if (!msg.empty()) {
                out << " : " << msg;
            }
            out << "\n";
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
            indent();
            LOG_ERR("ASSERT EQUAL FAILED");
            if (!msg.empty()) {
                LOG_ERR(" : %s", msg.c_str());
            }
            LOG_ERR("\n");

            indent();
            LOG_ERR("  expected: %s\n", to_string_convert(expected).c_str());
            indent();
            LOG_ERR("  actual  : %s\n", to_string_convert(actual).c_str());
            return false;
        }
        return true;
    }

    // Print summary and return an exit code
    int summary() const {
        LOG_ERR("\n==== TEST SUMMARY ====\n");
        LOG_ERR("tests      : %d\n", tests);
        LOG_ERR("assertions : %d\n", assertions);
        LOG_ERR("failures   : %d\n", failures);
        LOG_ERR("exceptions : %d\n", exceptions);
        LOG_ERR("======================\n");
        return failures == 0 ? 0 : 1;
    }

private:
    template <typename T>
    std::string to_string_convert(const T & value) const {
        std::ostringstream oss;
        oss << value;
        return oss.str();
    }

};
