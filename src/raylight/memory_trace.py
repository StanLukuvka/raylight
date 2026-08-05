"""Opt-in CUDA memory markers for MiniMax-H3 diagnostics."""

from __future__ import annotations

import json
import os
import time

import torch


def h3_memory_trace_enabled() -> bool:
    return os.environ.get("RAYLIGHT_H3_MEMORY_TRACE", "0") == "1"


def h3_stop_after_first_forward() -> bool:
    return os.environ.get("RAYLIGHT_H3_STOP_AFTER_FIRST_FORWARD", "0") == "1"


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
