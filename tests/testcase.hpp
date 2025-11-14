#pragma once

#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class test_harness {
  private:
    class test_case & tc_;
    std::ostream &    error_stream_;
    std::string       test_label_;
  public:
    test_harness(test_case &tc, const std::string &test_label, std::ostream & error_stream = std::cerr);
    template <class T> bool assert_equals(const std::string &label, T expected, T actual);
};

class test_case {
    friend class test_harness;
  private:
    std::function<void(test_harness)> test_func_;
    std::string                       name_;
    int                               successes = 0, failures = 0, errors = 0;
    bool                              omit_success_msg = false;

    void inc_fail() { failures++; }

    void inc_suc() { successes++; }
  public:
    test_case(std::function<void(test_harness)> test_func, const std::string & name) :
        test_func_(std::move(test_func)),
        name_(name) {}

    bool run() {
        // clean counters on run
        successes = 0;
        failures  = 0;
        test_harness harness(*this, name_);
        // execute run with harness
        try {
            test_func_(harness);
        } catch (std::exception & e) {
            errors++;
            std::cerr << "[" << get_name() << "] error during execution:\n" << e.what() << "\n";
        }
        
        if (is_success()) {
            if (!omit_success_msg) {
                std::cerr << "[" << get_name() << "] PASSED" << '\n';
            }
            return true;
        }
        if (is_error()) {
            std::cerr << "[" << get_name() << "] ERROR" << '\n';
            return false;
        }
        std::cerr << "[" << get_name() << "] FAILED (" << successes << "/" << (successes + failures) << ")\n";
        return false;
    }

    void reset() {
        successes = 0;
        failures  = 0;
        errors    = 0;
    }

    std::string get_name() { return name_; }

    bool is_success() const { return successes > 0 && failures == 0 && errors == 0; }

    bool is_error() const { return errors > 0; }

    void set_omit_success_msg(bool omit) { this->omit_success_msg = omit; }

    bool is_omit_success_msg() const { return this->omit_success_msg; }
};

inline test_harness::test_harness(test_case & tc, const std::string & test_label, std::ostream & error_stream) :
    tc_(tc),
    error_stream_(error_stream),
    test_label_(test_label) {}

template <class T> bool test_harness::assert_equals(const std::string & label, T expected, T actual) {
    if (expected != actual) {
        error_stream_ << "[" << label << "] FAILED\n";
        error_stream_ << "Expected: " << expected << "\n";
        error_stream_ << "Actual: " << actual << "\n";
        error_stream_ << std::flush;
        tc_.inc_fail();
        return false;
    }
    if (!tc_.is_omit_success_msg()) {
        error_stream_ << "[" << test_label_ << " -> " << label << "] PASSED\n";
    }
    tc_.inc_suc();
    return true;
}

class compound_test {
  private:
    std::vector<std::unique_ptr<test_case>> test_cases_;
    std::string                             name_;
    int                                     successes_ = 0;
    int                                     failures_  = 0;
    int                                     errors_    = 0;
    std::unordered_map<std::string, int>    test_name_to_index_;

    void run_test_case(std::unique_ptr<test_case> & test_case) {
        try {
            bool result = test_case->run();

            if (result) {
                successes_++;
            } else {
                failures_++;
            }
        } catch (std::exception & e) {
            errors_++;
            std::cerr << "Error while running test " << test_case->get_name() << ":\n" << e.what() << "\n";
        }
    }

  public:
    explicit compound_test(const std::string & name) : name_(name) {}

    // Add a test case
    void add_test(const std::function<void(test_harness)> & test_func, const std::string & test_name) {
        auto test                      = std::make_unique<test_case>(test_func, test_name);
        int  index                     = test_cases_.size();
        test_name_to_index_[test_name] = index;
        test_cases_.push_back(std::move(test));
    }

    // Access test by name
    bool operator[](const std::string & test_name) {
        auto it = test_name_to_index_.find(test_name);
        if (it == test_name_to_index_.end()) {
            std::cerr << "Test case '" << test_name << "' not found in compound test '" << name_ << "'\n";
            return false;
        }
        int    index     = it->second;
        auto & test_case = test_cases_[index];
        run_test_case(test_case);
        return test_case->is_success();
    }

    // Execute all tests
    void run_all() {
        std::cerr << "Running all tests for: " << name_ << "\n";
        for (auto & test_case : test_cases_) {
            run_test_case(test_case);
        }
    }

    // Display summary
    void summary() {
        std::cerr << "\n=== Compound Test Summary: " << name_ << " ===\n";
        std::cerr << "Successes: " << successes_ << "\n";
        std::cerr << "Failures: " << failures_ << "\n";
        std::cerr << "Total: " << (successes_ + failures_) << "\n";
        if (successes_ + failures_ > 0) {
            std::cerr << "Pass Rate: " << (successes_ * 100.0 / (successes_ + failures_)) << "%\n";
        }
        std::cerr << "========================================\n";
    }

    // Provide a convenient way to run all tests
    void run_all_tests() {
        run_all();
        summary();
    }

    // Get results
    int get_successes() const { return successes_; }

    int get_failures() const { return failures_; }

    int get_total() const { return successes_ + failures_; }

    double get_pass_rate() const {
        int total = successes_ + failures_;
        return total > 0 ? (successes_ * 100.0 / total) : 0.0;
    }
};
