"""Egress-тест: рантайм НЕ совершает исходящих соединений (offline-инвариант).

Каркас. В CI заменить на реальный запуск gateway+llama-server под egress-deny
(ip netns / nftables) и сверку счётчиков исходящих пакетов (== 0 вне loopback).
"""
import pytest


@pytest.mark.egress
def test_runtime_has_zero_egress():
    pytest.skip("scaffold: реализовать запуск под egress-deny и проверку счётчиков")
