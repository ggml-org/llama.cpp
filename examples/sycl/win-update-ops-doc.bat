@echo off

rem MIT license
rem Copyright (C) 2026 Intel Corporation
rem SPDX-License-Identifier: MIT

build\bin\test-backend-ops support --output csv > docs\ops\SYCL.csv
python scripts\create_ops_docs.py
