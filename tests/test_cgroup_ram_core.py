# FLOW-PRODUCED: cap-host-ram-on-kaggle
"""Tests for resolve_ram_cap_bytes() — pure logic, no psutil needed."""

from __future__ import annotations

import os

import pytest
from raylight.cgroup_ram import resolve_ram_cap_bytes


def _gb(n_bytes: int) -> float:
    return n_bytes / (1024**3)


def test_env_override_takes_precedence(monkeypatch):
    """When RAYLIGHT_H3_MAX_HOST_RAM_GB is set, it overrides everything."""
    monkeypatch.setenv("RAYLIGHT_H3_MAX_HOST_RAM_GB", "24.5")
    monkeypatch.setattr(
        "raylight.cgroup_ram._read_cgroup_memory_max",
        lambda: 30 * 1024**3,
    )

    cap = resolve_ram_cap_bytes()
    assert cap is not None
    assert _gb(cap) == pytest.approx(24.5)


def test_cgroup_limit_minus_margin(monkeypatch):
    """When cgroup limit is 30 GB, default cap should be 28 GB (30 - 2)."""
    monkeypatch.delenv("RAYLIGHT_H3_MAX_HOST_RAM_GB", raising=False)
    monkeypatch.setattr(
        "raylight.cgroup_ram._read_cgroup_memory_max",
        lambda: 30 * 1024**3,
    )

    cap = resolve_ram_cap_bytes()
    assert cap is not None
    assert _gb(cap) == pytest.approx(28.0)


def test_no_cgroup_no_env_returns_none(monkeypatch):
    """Without cgroup or env var, no cap should be returned."""
    monkeypatch.delenv("RAYLIGHT_H3_MAX_HOST_RAM_GB", raising=False)
    monkeypatch.setattr("raylight.cgroup_ram._read_cgroup_memory_max", lambda: None)

    assert resolve_ram_cap_bytes() is None


def test_invalid_env_value_ignored(monkeypatch):
    """An invalid env value should log a warning and fall through."""
    monkeypatch.setenv("RAYLIGHT_H3_MAX_HOST_RAM_GB", "not-a-number")
    monkeypatch.setattr(
        "raylight.cgroup_ram._read_cgroup_memory_max",
        lambda: 30 * 1024**3,
    )

    cap = resolve_ram_cap_bytes()
    # Falls through to cgroup limit
    assert cap is not None
    assert _gb(cap) == pytest.approx(28.0)


def test_cap_respects_safety_margin(monkeypatch):
    """Custom safety margin should be subtracted from the cgroup limit."""
    monkeypatch.delenv("RAYLIGHT_H3_MAX_HOST_RAM_GB", raising=False)
    monkeypatch.setattr(
        "raylight.cgroup_ram._read_cgroup_memory_max",
        lambda: 30 * 1024**3,
    )

    # 4 GB margin → 26 GB cap
    cap = resolve_ram_cap_bytes(safety_margin_gb=4.0)
    assert cap is not None
    assert _gb(cap) == pytest.approx(26.0)
