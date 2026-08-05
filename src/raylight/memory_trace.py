"""Opt-in CUDA memory markers for MiniMax-H3 diagnostics."""

from __future__ import annotations

import importlib.metadata
import json
import os
import time

import torch

_H3_PHASE_PROFILE_ACTIVE = False


def h3_memory_trace_enabled() -> bool:
    return os.environ.get("RAYLIGHT_H3_MEMORY_TRACE", "0") == "1"


def h3_stop_after_first_forward() -> bool:
    return os.environ.get("RAYLIGHT_H3_STOP_AFTER_FIRST_FORWARD", "0") == "1"


def h3_phase_profile_enabled() -> bool:
    return os.environ.get("RAYLIGHT_H3_PHASE_PROFILE", "0") == "1"


def set_h3_phase_profile_active(active: bool) -> None:
    global _H3_PHASE_PROFILE_ACTIVE
    _H3_PHASE_PROFILE_ACTIVE = active


def h3_phase_profile_active() -> bool:
    return h3_phase_profile_enabled() and _H3_PHASE_PROFILE_ACTIVE


def h3_cuda_phase_report(phases, **fields) -> None:
    """Synchronize once and report elapsed CUDA-event timings for one bounded phase set."""
    if not h3_phase_profile_active() or not torch.cuda.is_available():
        return
    torch.cuda.synchronize()

    def package_version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "unknown"

    payload = {
        "marker": "h3_cuda_phase_profile",
        "unix": time.time(),
        "device": torch.cuda.current_device(),
        "versions": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "ray": package_version("ray"),
            "xfuser": package_version("xfuser"),
            "yunchang": package_version("yunchang"),
        },
        "phases_ms": {
            name: start.elapsed_time(end)
            for name, start, end in phases
        },
        **fields,
    }
    print("[raylight-h3-phase] " + json.dumps(payload, sort_keys=True), flush=True)


def h3_memory_snapshot(marker: str, **fields) -> None:
    if not h3_memory_trace_enabled() or not torch.cuda.is_available():
        return
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    payload = {
        "marker": marker,
        "unix": time.time(),
        "device": torch.cuda.current_device(),
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
        "free": free_bytes,
        "total": total_bytes,
        **fields,
    }
    print("[raylight-h3-memory] " + json.dumps(payload, sort_keys=True), flush=True)
