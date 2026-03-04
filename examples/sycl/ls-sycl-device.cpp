//
//  MIT license
//  Copyright (C) 2024 Intel Corporation
//  SPDX-License-Identifier: MIT
//


#include "ggml-sycl.h"
#include <clocale>

int main() {
    std::setlocale(LC_NUMERIC, "C");
    ggml_backend_sycl_print_sycl_devices();
    return 0;
}
