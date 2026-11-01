"""Shared pytest fixtures for QDC on-device test runners."""

import logging
import os

import pytest
from appium import webdriver

from utils import options, write_qdc_log


@pytest.fixture(scope="session", autouse=True)
def driver():
    # Tests drive the device over `adb shell`; none use the driver. Keep the session
    # best-effort so a failed Appium helper-APK install cannot fail the whole run.
    try:
        return webdriver.Remote(command_executor="http://127.0.0.1:4723/wd/hub", options=options)
    except Exception as e:
        logging.getLogger(__name__).warning("Appium session unavailable (%s); continuing over adb", type(e).__name__)
        return None


def pytest_sessionfinish(session, exitstatus):
    xml_path = getattr(session.config.option, "xmlpath", None) or "results.xml"
    if os.path.exists(xml_path):
        with open(xml_path) as f:
            write_qdc_log("results.xml", f.read())
