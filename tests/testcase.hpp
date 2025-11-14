#pragma once

#include <iostream>
#include <string>
#include <functional>
#include <utility>
#include <unordered_map>
#include <memory>
#include <vector>

class test_harness { // TODO: more prototypes?
private:
    int& successes_;
    int& failures_;
    std::ostream& error_stream_;

public:
    test_harness(int& successes, int& failures, std::ostream& error_stream = std::cerr)
        : successes_(successes), failures_(failures), error_stream_(error_stream) {}

    template<class T>
    bool assert_equals(const std::string &label, T expected, T actual) {
        if (expected != actual) {
            error_stream_ << "[" << label << "] FAILED\n";
            error_stream_ << "Expected: " << expected << "\n";
            error_stream_ << "Actual: " << actual << "\n";
            error_stream_ << std::flush;
            failures_++;
            return false;
        }
        error_stream_ << "[" << label << "] PASSED\n";
        successes_++;
        return true;
    }
};

class base_test_case {
public:
    virtual ~base_test_case() = default;
    virtual bool run() = 0;
    virtual std::string get_name() const = 0;
};

class test_case : public base_test_case {
private:
    std::function<void(test_harness)> test_func_;
    std::string name_;
    int successes = 0, failures = 0;
    test_harness harness;

public:
    test_case(std::function<void(test_harness)> test_func, const std::string& name)
        : test_func_(std::move(test_func)), name_(name),
          harness(successes, failures) {}

    bool run() override {
        // clean counters on run
        successes = 0;
        failures = 0;
        // execute run with harness
        test_func_(harness);
        std::cerr << "[" << get_name() << "] ";
        if (is_success()) {
            std::cerr << "PASSED" << '\n';
            return true;
        }
        std::cerr << "FAILED (" << successes << "/" << (successes + failures) << ")\n";
        return false;
    }

    std::string get_name() const override { return name_; }
    bool is_success() const { return successes > 0 && failures == 0; }
};

class compound_test {
private:
    std::vector<std::unique_ptr<base_test_case>> test_cases_;
    std::string name_;
    int successes_ = 0;
    int failures_ = 0;
    std::unordered_map<std::string, int> test_name_to_index_;

public:
    explicit compound_test(const std::string& name) : name_(name) {}

    // Add a test case
    void add_test(const std::function<void(test_harness)>& test_func, const std::string& test_name) {
        auto test = std::make_unique<test_case>(test_func, test_name);
        int index = test_cases_.size();
        test_name_to_index_[test_name] = index;
        test_cases_.push_back(std::move(test));
    }

    // Access test by name
    bool operator[](const std::string& test_name) {
        auto it = test_name_to_index_.find(test_name);
        if (it == test_name_to_index_.end()) {
            std::cerr << "Test case '" << test_name << "' not found in compound test '" << name_ << "'\n";
            return false;
        }

        int index = it->second;
        bool result = test_cases_[index]->run();

        if (result) {
            successes_++;
        } else {
            failures_++;
        }

        return result;
    }

    // Execute all tests
    void run_all() {
        std::cerr << "Running all tests for: " << name_ << "\n";
        for (auto& test_case : test_cases_) {
            bool result = test_case->run();
            if (result) {
                successes_++;
            } else {
                failures_++;
            }
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

    // Get results
    int get_successes() const { return successes_; }
    int get_failures() const { return failures_; }
    int get_total() const { return successes_ + failures_; }
    double get_pass_rate() const {
        int total = successes_ + failures_;
        return total > 0 ? (successes_ * 100.0 / total) : 0.0;
    }
};