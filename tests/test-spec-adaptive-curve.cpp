// Analyze request-total acceptance traces; these cannot replay per-round adaptation.
#include "nlohmann/json.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>

namespace {
constexpr size_t MIN_REQUESTS = 200;
constexpr size_t TAIL_REQUESTS = 50;
constexpr int EXIT_SKIP = 77;

struct curve_point {
    int64_t drafted;
    int64_t accepted;
};

struct curve {
    size_t requests = 0;
    std::deque<curve_point> tail;
    int64_t drafted = 0;
    int64_t accepted = 0;
};

bool parse_curve(std::istream & input, curve & out, std::string & error) {
    std::string line;
    size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.find_first_not_of(" \t\r") == std::string::npos) {
            continue;
        }
        const auto row = nlohmann::json::parse(line, nullptr, false);
        if (!row.is_object() || !row.contains("n_draft") || !row.contains("n_accepted") ||
            !row["n_draft"].is_number_integer() || !row["n_accepted"].is_number_integer() ||
            row["n_draft"] <= 0 || row["n_draft"] > std::numeric_limits<int32_t>::max() ||
            row["n_accepted"] < 0 || row["n_accepted"] > row["n_draft"]) {
            error = "line " + std::to_string(line_number) + ": invalid request draft counts";
            return false;
        }
        const curve_point point = {row["n_draft"].get<int64_t>(), row["n_accepted"].get<int64_t>()};
        out.tail.push_back(point);
        out.drafted += point.drafted;
        out.accepted += point.accepted;
        ++out.requests;
        if (out.tail.size() > TAIL_REQUESTS) {
            out.drafted -= out.tail.front().drafted;
            out.accepted -= out.tail.front().accepted;
            out.tail.pop_front();
        }
    }
    if (input.bad()) {
        error = "cannot read trace";
        return false;
    }
    return true;
}

int evaluate(const curve & adaptive, const curve & fixed) {
    if (adaptive.requests < MIN_REQUESTS || fixed.requests < MIN_REQUESTS) {
        return EXIT_SKIP;
    }
    const double adaptive_rate = double(adaptive.accepted) / adaptive.drafted;
    const double fixed_rate = double(fixed.accepted) / fixed.drafted;
    // Allow rounding at the exact 0.80 acceptance and 0.20 improvement boundaries.
    constexpr double ROUNDING_TOLERANCE = 1e-12;
    return adaptive_rate + ROUNDING_TOLERANCE >= 0.80 &&
           adaptive_rate - fixed_rate + ROUNDING_TOLERANCE >= 0.20 ? 0 : 1;
}

int self_test() {
    int failures = 0;
    auto check = [&](bool condition, const char * name) {
        if (!condition) {
            std::fprintf(stderr, "FAIL: %s\n", name);
            ++failures;
        }
    };
    for (const char * invalid : {"{", "[]", "{}", "{\"n_draft\":3,\"n_accepted\":x}",
            "{\"n_draft\":0,\"n_accepted\":0}", "{\"n_draft\":3,\"n_accepted\":4}",
            "{\"n_draft\":3,\"n_accepted\":-1}", "{\"n_draft\":3,\"n_accepted\":true}",
            "{\"n_draft\":3.0,\"n_accepted\":1}", "{\"n_draft\":18446744073709551615,\"n_accepted\":1}"}) {
        std::istringstream input(invalid);
        curve parsed;
        std::string error;
        check(!parse_curve(input, parsed, error), invalid);
    }
    auto sample = [&](int drafted, int accepted, size_t requests) {
        std::string text;
        for (size_t i = 0; i < requests; ++i) {
            text += "{\"n_accepted\":" + std::to_string(accepted) + ",\"n_draft\":" +
                    std::to_string(drafted) + "}\n";
        }
        std::istringstream input(text);
        curve parsed;
        std::string error;
        check(parse_curve(input, parsed, error), "reordered keys");
        return parsed;
    };
    const auto adaptive = sample(10, 8, MIN_REQUESTS);
    const auto fixed = sample(10, 6, MIN_REQUESTS);
    check(evaluate(adaptive, fixed) == 0, "exact acceptance and gap boundaries pass");
    check(evaluate(adaptive, sample(10, 7, MIN_REQUESTS)) == 1, "insufficient gap fails");
    check(evaluate(sample(10, 7, MIN_REQUESTS), sample(10, 4, MIN_REQUESTS)) == 1,
          "low adaptive acceptance fails");
    check(evaluate(sample(10, 8, MIN_REQUESTS - 1), fixed) == EXIT_SKIP, "insufficient data skips");
    curve weighted = sample(1, 0, MIN_REQUESTS);
    std::istringstream extra("{\"n_draft\":100,\"n_accepted\":100}\n");
    std::string error;
    check(parse_curve(extra, weighted, error) && weighted.requests == MIN_REQUESTS + 1 &&
          weighted.tail.size() == TAIL_REQUESTS && weighted.drafted == 149 && weighted.accepted == 100,
          "window is bounded and token weighted");
    std::printf("self-test: %d failures\n", failures);
    return failures ? 1 : 0;
}
} // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--self-test") == 0) {
        return self_test();
    }
    if (argc == 1) {
        std::printf("SKIP: supply --curve-file adaptive.jsonl --fixed-curve-file fixed7.jsonl\n");
        return EXIT_SKIP;
    }
    std::string adaptive_path, fixed_path;
    for (int i = 1; i < argc; ++i) {
        std::string * path = nullptr;
        if (std::strcmp(argv[i], "--curve-file") == 0) {
            path = &adaptive_path;
        } else if (std::strcmp(argv[i], "--fixed-curve-file") == 0) {
            path = &fixed_path;
        }
        if (!path || !path->empty() || i + 1 == argc) {
            std::fprintf(stderr, "invalid argument: %s\n", argv[i]);
            return 1;
        }
        *path = argv[++i];
    }
    if (adaptive_path.empty() || fixed_path.empty()) {
        std::fprintf(stderr, "both --curve-file and --fixed-curve-file are required\n");
        return 1;
    }
    curve adaptive, fixed;
    for (const auto & item : {std::make_pair(adaptive_path, &adaptive), std::make_pair(fixed_path, &fixed)}) {
        std::ifstream input(item.first);
        std::string error;
        if (!input.is_open() || !parse_curve(input, *item.second, error)) {
            std::fprintf(stderr, "%s: %s\n", item.first.c_str(), error.empty() ? "cannot open trace" : error.c_str());
            return 1;
        }
    }
    const int result = evaluate(adaptive, fixed);
    if (result == EXIT_SKIP) {
        std::printf("SKIP: adaptive=%zu fixed=%zu requests; need >= %zu each\n",
                    adaptive.requests, fixed.requests, MIN_REQUESTS);
    } else {
        std::printf("%s: last %zu requests, adaptive=%.4f fixed7=%.4f\n", result ? "FAIL" : "PASS",
                    TAIL_REQUESTS, double(adaptive.accepted) / adaptive.drafted,
                    double(fixed.accepted) / fixed.drafted);
    }
    return result;
}
