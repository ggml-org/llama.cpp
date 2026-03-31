#pragma once

#include "ggml.h"
#include "llama-quant.h"

#include <map>
#include <string>
#include <vector>

enum class recipe_condition_type {
    LAYER,        // layer<1/8 — absolute layer position from tensor name
    CATEGORY,     // category<1/8 — position within category count (e.g., 3rd of 32 attn_v tensors)
    INDEX,        // index<2 — sequential counter within category
    ARCH,         // arch=falcon, arch!=falcon
    N_EXPERT,     // n_expert>=8
    N_GQA,        // n_gqa>=4
    MODEL_TYPE,   // model_type=70B
    HAS_IMATRIX,  // has_imatrix, !has_imatrix
};

enum class recipe_comparison {
    EQ,   // = or exact match
    LT,   // <
    LTE,  // <=
    GT,   // >
    GTE,  // >=
    NEG,  // negation (!, !=)
};

struct recipe_condition {
    recipe_condition_type type;
    recipe_comparison     cmp = recipe_comparison::EQ;

    // LAYER: true = more_bits heuristic, false = fraction comparison using frac_num/frac_den
    bool more_bits = false;
    int  frac_num  = 0;
    int  frac_den  = 1;

    int         int_val = 0;   // INDEX, N_EXPERT, N_GQA
    std::string str_val;       // ARCH, MODEL_TYPE
};

// conditions are AND'd together; empty conditions = always matches
struct recipe_rule {
    std::vector<recipe_condition> conditions;
    ggml_type                     type = GGML_TYPE_COUNT;
};

struct quant_recipe {
    std::string name;
    ggml_type   default_type = GGML_TYPE_COUNT;

    // per-category rules, evaluated as if/else chain (first match wins)
    std::map<tensor_category, std::vector<recipe_rule>> categories;
};

quant_recipe recipe_parse_file(const std::string & filepath);
quant_recipe recipe_parse_string(const std::string & content);
