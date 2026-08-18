"""Cap ComfyUI's perceived host RAM on cgroup-limited environments.

ComfyUI calls ``psutil.virtual_memory()`` at import time to decide how much
host RAM it can use for pinned-memory buffers and smart offload.  On Kaggle
the kernel reports ~32 GB of *physical* RAM through psutil, but the container
is cgroup-limited to ~30 GB.  ComfyUI happily allocates pinned memory up to the
32 GB it thinks it has, the cgroup OOM-killer triggers at 30 GB, and the whole
notebook hard-crashes.

This module patches ``psutil.virtual_memory`` so that ComfyUI sees a
value that respects the cgroup limit minus a safety margin.  The cap is:

    min(physical_ram, cgroup_limit - safety_margin)

If the ``RAYLIGHT_H3_MAX_HOST_RAM_GB`` environment variable is set, its value
*overrides* everything else (useful for testing or for hosts without cgroup).

Call :func:`apply_host_ram_cap` as early as possible — before
``comfy.model_management`` is imported — from the Kaggle provisioner and from
Ray worker boot.
"""

# FLOW-PRODUCED: cap-host-ram-on-kaggle

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_CGROUP_ROOT = Path("/sys/fs/cgroup")
_DEFAULT_SAFETY_MARGIN_GB = 2.0  # leave 2 GB below the cgroup limit


def _read_cgroup_memory_max() -> int | None:
    """Return the cgroup v2 memory.max in bytes, or ``None`` if unavailable."""
    value = _read_text(_CGROUP_ROOT / "memory.max")
    if value is None or value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except (OSError, UnicodeError):
        return None


def _gb_to_bytes(gb: float) -> int:
    return int(gb * 1024 * 1024 * 1024)


def resolve_ram_cap_bytes(
    *,
    env_override_gb: float | None = None,
    safety_margin_gb: float = _DEFAULT_SAFETY_MARGIN_GB,
) -> int | None:
    """Compute the capped host RAM in bytes, or ``None`` if no cap is needed.

    Resolution order:
    1. ``RAYLIGHT_H3_MAX_HOST_RAM_GB`` env var (explicit override, any host).
    2. cgroup v2 ``memory.max`` minus ``safety_margin_gb``.
    3. ``None`` — no cgroup limit detected, no env override → don't patch.
    """
    if env_override_gb is not None:
        return _gb_to_bytes(env_override_gb)

    env_str = os.environ.get("RAYLIGHT_H3_MAX_HOST_RAM_GB")
    if env_str:
        try:
            return _gb_to_bytes(float(env_str))
        except ValueError:
            logger.warning(
                "RAYLIGHT_H3_MAX_HOST_RAM_GB=%r is not a number, ignoring", env_str
            )

    cgroup_max = _read_cgroup_memory_max()
    if cgroup_max is not None:
        cap = cgroup_max - _gb_to_bytes(safety_margin_gb)
        if cap <= 0:
            logger.warning(
                "cgroup memory.max (%d bytes) minus safety margin yields <=0; "
                "falling back to 1 GB",
                cgroup_max,
            )
            return _gb_to_bytes(1.0)
        return cap

    return None


def apply_host_ram_cap() -> int | None:
    """Patch ``psutil.virtual_memory`` to respect the cgroup RAM cap.

    Must be called **before** ``comfy.model_management`` is imported so that
    ComfyUI picks up the capped value at module init.

    Returns the cap in bytes, or ``None`` if no patch was applied.
    """
    cap_bytes = resolve_ram_cap_bytes()
    if cap_bytes is None:
        logger.info("No cgroup RAM cap needed — psutil left unpatched")
        return None

    try:
        import psutil
    except ImportError:
        logger.warning("psutil not available; cannot patch virtual_memory")
        return None

    original_virtual_memory = psutil.virtual_memory

    _CappedResult = _make_capped_result(cap_bytes, original_virtual_memory)

    def _patched_virtual_memory():
        return _CappedResult

    psutil.virtual_memory = _patched_virtual_memory  # type: ignore[assignment]

    physical_gb = _physical_ram_gb(original_virtual_memory)
    logger.info(
        "Patched psutil.virtual_memory: physical=%.1f GB → capped=%.1f GB",
        physical_gb,
        cap_bytes / (1024**3),
    )
    return cap_bytes


def _physical_ram_gb(original_fn) -> float:
    """Read the real physical RAM (in GB) from the original psutil call."""
    try:
        return original_fn().total / (1024**3)
    except Exception:
        return 0.0


def _make_capped_result(cap_bytes: int, original_fn):
    """Create a named tuple compatible with psutil.virtual_memory() output."""
    try:
        result = original_fn()
        # Reuse the same namedtuple class that psutil returns, just swap .total
        return type(result)(
            total=cap_bytes,
            available=min(result.available, cap_bytes),
            percent=result.percent,
            used=min(result.used, cap_bytes),
            free=min(result.free, cap_bytes),
            **{
                k: v
                for k, v in result._asdict().items()
                if k not in ("total", "available", "percent", "used", "free")
            },
        )
    except Exception:
        # psutil itself failed — create a minimal stand-in
        from collections import namedtuple
        sismem = namedtuple("svmem", "total available percent used free")
        return sismem(
            total=cap_bytes,
            available=cap_bytes,
            percent=0.0,
            used=0,
            free=cap_bytes,
        )
