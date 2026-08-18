#!/bin/bash

#  MIT license
#  Copyright (C) 2026 Intel Corporation
#  SPDX-License-Identifier: MIT

./build/bin/test-backend-ops support --output csv > docs/ops/SYCL.csv
./scripts/create_ops_docs.py

