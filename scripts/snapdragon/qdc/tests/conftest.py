"""Shared pytest fixtures for QDC on-device test runners."""

import os

import pytest
from appium import webdriver

from utils import options, write_qdc_log


@pytest.fixture(scope="session", autouse=True)
def driver():
    return webdriver.Remote(command_executor="http://127.0.0.1:4723/wd/hub", options=options)


def pytest_sessionfinish(session, exitstatus):
    xml_path = getattr(session.config.option, "xmlpath", None) or "results.xml"
    if os.path.exists(xml_path):
        with open(xml_path) as f:
            write_qdc_log("results.xml", f.read())
