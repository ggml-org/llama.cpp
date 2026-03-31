# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Shared pytest fixtures for QDC on-device test runners."""

import pytest
from appium import webdriver  # ty: ignore[unresolved-import]

from utils import options


@pytest.fixture(scope="session", autouse=True)
def driver():
    return webdriver.Remote(command_executor="http://127.0.0.1:4723/wd/hub", options=options)
