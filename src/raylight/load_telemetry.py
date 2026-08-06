"""Structured timing markers for distributed model-loading diagnostics."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Any, Iterator


def _write_event(payload: dict[str, Any]) -> dict[str, Any]:
    print("[raylight-load-phase] " + json.dumps(payload, sort_keys=True), flush=True)
    return payload


def emit_load_event(event: str, *, rank: int | None = None, **details: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "marker": "raylight_load_phase",
        "event": event,
        "unix": time.time(),
        "monotonic": time.monotonic(),
    }
    if rank is not None:
        payload["rank"] = rank
    payload.update(details)
    return _write_event(payload)


@contextmanager
def load_phase(name: str, *, rank: int | None = None, **details: Any) -> Iterator[None]:
    start = emit_load_event(f"{name}_start", rank=rank, **details)
    try:
        yield
    except BaseException as exc:
        end_monotonic = time.monotonic()
        payload = {
            "marker": "raylight_load_phase",
            "event": f"{name}_end",
            "unix": time.time(),
            "monotonic": end_monotonic,
            "elapsed_seconds": end_monotonic - start["monotonic"],
            "status": "error",
            "error_type": type(exc).__name__,
            **details,
        }
        if rank is not None:
            payload["rank"] = rank
        _write_event(payload)
        raise
    else:
        end_monotonic = time.monotonic()
        payload = {
            "marker": "raylight_load_phase",
            "event": f"{name}_end",
            "unix": time.time(),
            "monotonic": end_monotonic,
            "elapsed_seconds": end_monotonic - start["monotonic"],
            "status": "ok",
            **details,
        }
        if rank is not None:
            payload["rank"] = rank
        _write_event(payload)
