"""Structured timing markers for distributed model-loading diagnostics."""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


_PREFIX = "[raylight-load-phase] "
_SCHEMA_VERSION = 2
_CGROUP_ROOT = Path("/sys/fs/cgroup")
_PROC_STATUS_FIELDS = {
    "VmRSS": "rss_bytes",
    "VmHWM": "peak_rss_bytes",
    "RssAnon": "rss_anon_bytes",
    "RssFile": "rss_file_bytes",
    "RssShmem": "rss_shmem_bytes",
    "VmSwap": "swap_bytes",
    "Threads": "thread_count",
}
_CGROUP_STAT_FIELDS = {
    "anon": "anon_bytes",
    "file": "file_bytes",
    "shmem": "shmem_bytes",
    "kernel": "kernel_bytes",
    "pagetables": "pagetables_bytes",
    "sock": "socket_bytes",
}


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except (OSError, UnicodeError):
        return None


def _parse_size(value: str) -> int | None:
    fields = value.split()
    if not fields:
        return None
    try:
        amount = int(fields[0])
    except ValueError:
        return None
    if len(fields) > 1 and fields[1].lower() == "kb":
        amount *= 1024
    return amount


def _process_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "rss_bytes": None,
    }
    status = _read_text(Path("/proc/self/status"))
    if status is not None:
        for line in status.splitlines():
            name, separator, value = line.partition(":")
            output_name = _PROC_STATUS_FIELDS.get(name)
            if not separator or output_name is None:
                continue
            parsed = _parse_size(value)
            if parsed is not None:
                snapshot[output_name] = parsed
    try:
        with os.scandir("/proc/self/fd") as descriptors:
            snapshot["open_fd_count"] = sum(1 for _ in descriptors)
    except OSError:
        snapshot["open_fd_count"] = None
    return snapshot


def _cgroup_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for filename, output_name in (
        ("memory.current", "memory_current_bytes"),
        ("memory.peak", "memory_peak_bytes"),
        ("memory.max", "memory_max_bytes"),
    ):
        value = _read_text(_CGROUP_ROOT / filename)
        if value is None:
            continue
        if value == "max":
            snapshot[output_name] = None
        else:
            try:
                snapshot[output_name] = int(value)
            except ValueError:
                continue

    events = _read_text(_CGROUP_ROOT / "memory.events")
    if events is not None:
        for line in events.splitlines():
            fields = line.split()
            if len(fields) == 2:
                try:
                    snapshot[f"event_{fields[0]}"] = int(fields[1])
                except ValueError:
                    continue

    memory_stat = _read_text(_CGROUP_ROOT / "memory.stat")
    if memory_stat is not None:
        for line in memory_stat.splitlines():
            fields = line.split()
            if len(fields) != 2 or fields[0] not in _CGROUP_STAT_FIELDS:
                continue
            try:
                snapshot[_CGROUP_STAT_FIELDS[fields[0]]] = int(fields[1])
            except ValueError:
                continue
    return snapshot


def _cuda_snapshot() -> dict[str, Any]:
    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None) if torch is not None else None
    if cuda is None:
        return {"initialized": False}
    try:
        initialized = bool(cuda.is_initialized())
    except Exception:
        initialized = False
    if not initialized:
        return {"initialized": False}

    snapshot: dict[str, Any] = {"initialized": True}
    try:
        device = int(cuda.current_device())
        free_bytes, total_bytes = cuda.mem_get_info(device)
        properties = cuda.get_device_properties(device)
        snapshot.update(
            {
                "device": device,
                "device_name": properties.name,
                "compute_capability": [int(properties.major), int(properties.minor)],
                "free_bytes": int(free_bytes),
                "total_bytes": int(total_bytes),
                "allocated_bytes": int(cuda.memory_allocated(device)),
                "reserved_bytes": int(cuda.memory_reserved(device)),
                "max_allocated_bytes": int(cuda.max_memory_allocated(device)),
                "max_reserved_bytes": int(cuda.max_memory_reserved(device)),
            }
        )
    except Exception as exc:
        snapshot["snapshot_error_type"] = type(exc).__name__
    return snapshot


def capture_load_snapshot() -> dict[str, Any]:
    """Capture bounded host/cgroup/CUDA state without initializing CUDA."""
    return {
        "process": _process_snapshot(),
        "cgroup": _cgroup_snapshot(),
        "cuda": _cuda_snapshot(),
    }


def _new_phase_id() -> str:
    return f"{os.getpid()}-{time.monotonic_ns()}"


def _write_event(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        encoded = json.dumps(payload, sort_keys=True)
    except Exception as exc:
        fallback = {
            "marker": "raylight_load_phase",
            "schema_version": _SCHEMA_VERSION,
            "event": str(payload.get("event", "telemetry_unknown")),
            "unix": payload.get("unix"),
            "monotonic": payload.get("monotonic"),
            "status": "telemetry_error",
            "telemetry_error_type": type(exc).__name__,
        }
        try:
            print(_PREFIX + json.dumps(fallback, sort_keys=True), flush=True)
        except Exception:
            pass
        return payload
    try:
        print(_PREFIX + encoded, flush=True)
    except Exception:
        pass
    return payload


def _base_payload(
    event: str,
    *,
    rank: int | None,
    process_role: str | None,
    monotonic: float,
    details: dict[str, Any],
) -> dict[str, Any]:
    try:
        snapshot = capture_load_snapshot()
    except Exception as exc:
        snapshot = {
            "process": {"pid": os.getpid(), "ppid": os.getppid(), "rss_bytes": None},
            "cgroup": {},
            "cuda": {"initialized": False},
            "telemetry_snapshot_error_type": type(exc).__name__,
        }
    payload: dict[str, Any] = {
        **details,
        "marker": "raylight_load_phase",
        "schema_version": _SCHEMA_VERSION,
        "event": event,
        "unix": time.time(),
        "monotonic": monotonic,
        "process_role": process_role or ("ray_worker" if rank is not None else "comfy_driver"),
        **snapshot,
    }
    if rank is not None:
        payload["rank"] = rank
    return payload


def emit_load_event(
    event: str,
    *,
    rank: int | None = None,
    process_role: str | None = None,
    **details: Any,
) -> dict[str, Any]:
    return _write_event(
        _base_payload(
            event,
            rank=rank,
            process_role=process_role,
            monotonic=time.monotonic(),
            details=details,
        )
    )


@contextmanager
def load_phase(
    name: str,
    *,
    rank: int | None = None,
    process_role: str | None = None,
    **details: Any,
) -> Iterator[None]:
    phase_id = _new_phase_id()
    start = emit_load_event(
        f"{name}_start",
        rank=rank,
        process_role=process_role,
        phase_id=phase_id,
        **details,
    )
    try:
        yield
    except BaseException as exc:
        end_monotonic = time.monotonic()
        payload = _base_payload(
            f"{name}_end",
            rank=rank,
            process_role=process_role,
            monotonic=end_monotonic,
            details={
                **details,
                "phase_id": phase_id,
                "elapsed_seconds": end_monotonic - start["monotonic"],
                "status": "error",
                "error_type": type(exc).__name__,
            },
        )
        _write_event(payload)
        raise
    else:
        end_monotonic = time.monotonic()
        payload = _base_payload(
            f"{name}_end",
            rank=rank,
            process_role=process_role,
            monotonic=end_monotonic,
            details={
                **details,
                "phase_id": phase_id,
                "elapsed_seconds": end_monotonic - start["monotonic"],
                "status": "ok",
            },
        )
        _write_event(payload)
