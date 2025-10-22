import pytest
from utils import *

# Test the new secure slots monitoring endpoint
def test_slots_status_basic():
    global server
    res = server.make_request("GET", "/slots/status")
    assert res.status_code == 200

def test_slots_status_structure():
    global server
    res = server.make_request("GET", "/slots/status")
    data = res.body
    
    # Check response structure
    assert "slots" in data
    assert "total_slots" in data
    assert "idle_slots" in data
    assert "processing_slots" in data
    assert isinstance(data["slots"], list)

def test_slots_status_no_sensitive_data():
    """Critical security test: ensure no sensitive data leakage"""
    global server
    res = server.make_request("GET", "/slots/status")
    data = res.body
    
    for slot in data["slots"]:
        # These fields must NEVER appear in the response
        assert "prompt" not in slot
        assert "generated_text" not in slot
        assert "generated" not in slot
        assert "tokens" not in slot
        assert "cache_tokens" not in slot
