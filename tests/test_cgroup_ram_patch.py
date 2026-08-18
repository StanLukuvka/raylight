# FLOW-PRODUCED: cap-host-ram-on-kaggle
"""Tests for apply_host_ram_cap() — requires psutil."""

from __future__ import annotations

import os

import pytest

psutil = pytest.importorskip("psutil")

from raylight.cgroup_ram import apply_host_ram_cap


def _gb(n_bytes: int) -> float:
    return n_bytes / (1024**3)


def test_apply_host_ram_cap_patches_psutil(monkeypatch):
    """apply_host_ram_cap() should replace psutil.virtual_memory."""
    monkeypatch.delenv("RAYLIGHT_H3_MAX_HOST_RAM_GB", raising=False)
    monkeypatch.setattr(
        "raylight.cgroup_ram._read_cgroup_memory_max",
        lambda: 30 * 1024**3,
    )

    cap = apply_host_ram_cap()
    assert cap is not None
    # psutil.virtual_memory should now report the capped value
    result = psutil.virtual_memory()
    assert _gb(result.total) == pytest.approx(28.0, abs=0.1)


def test_apply_host_ram_cap_no_op_when_no_limit(monkeypatch):
    """If there's no cgroup limit, psutil should not be patched."""
    monkeypatch.delenv("RAYLIGHT_H3_MAX_HOST_RAM_GB", raising=False)
    monkeypatch.setattr("raylight.cgroup_ram._read_cgroup_memory_max", lambda: None)

    original = psutil.virtual_memory
    cap = apply_host_ram_cap()
    assert cap is None
    # Should be unchanged
    assert psutil.virtual_memory is original
