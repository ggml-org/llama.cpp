//===--------- ur_level_zero.cpp - Level Zero Adapter ---------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <climits>
#include <string.h>

#include "ur_level_zero.hpp"

// Define the static class field
std::mutex ZeCall::GlobalLock;
