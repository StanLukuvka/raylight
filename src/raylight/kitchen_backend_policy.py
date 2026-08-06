"""Fail-closed Comfy Kitchen INT8 backend policy for Ray workers."""

from __future__ import annotations

import importlib
import os
from typing import Any, Callable


def requested_int8_backend() -> str:
    backend = os.environ.get("RAYLIGHT_INT8_BACKEND", "eager").strip().lower()
    if backend not in {"eager", "cuda"}:
        raise ValueError(
            f"RAYLIGHT_INT8_BACKEND must be eager or cuda, got {backend!r}"
        )
    return backend


def _cuda_major(torch_module: Any) -> int:
    runtime = getattr(getattr(torch_module, "version", None), "cuda", None)
    if not runtime:
        raise RuntimeError("CUDA INT8 backend requires a CUDA PyTorch runtime")
    try:
        return int(str(runtime).split(".", 1)[0])
    except ValueError as exc:
        raise RuntimeError(f"Could not parse PyTorch CUDA runtime {runtime!r}") from exc


def initialize_worker_int8_backend(
    torch_module: Any,
    *,
    import_kitchen: Callable[[str], Any] = importlib.import_module,
) -> tuple[str, Any]:
    """Validate policy before the eager Comfy Kitchen extension import."""
    backend = requested_int8_backend()
    if backend == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("CUDA INT8 backend requires an available CUDA device")
        if (
            _cuda_major(torch_module) < 13
            and os.environ.get("RAYLIGHT_INT8_CUDA_ALLOW_UNDER_13") != "1"
        ):
            raise RuntimeError(
                "CUDA runtime below 13 requires "
                "RAYLIGHT_INT8_CUDA_ALLOW_UNDER_13=1 before importing Comfy Kitchen"
            )
        bounded_diagnostic = (
            os.environ.get("RAYLIGHT_H3_STOP_AFTER_FIRST_FORWARD") == "1"
            and os.environ.get("RAYLIGHT_H3_PHASE_PROFILE") == "1"
        )
        unbounded_approved = (
            os.environ.get("RAYLIGHT_H3_STOP_AFTER_FIRST_FORWARD") != "1"
            and os.environ.get("RAYLIGHT_INT8_CUDA_ALLOW_UNBOUNDED") == "1"
        )
        if not bounded_diagnostic and not unbounded_approved:
            raise RuntimeError(
                "CUDA INT8 worker requires bounded stop and phase profile controls "
                "or RAYLIGHT_INT8_CUDA_ALLOW_UNBOUNDED=1"
            )
        if str(torch_module.version.cuda) != "12.8":
            raise RuntimeError(
                f"CUDA INT8 diagnostic requires CUDA runtime 12.8, got {torch_module.version.cuda!r}"
            )
        capability = tuple(torch_module.cuda.get_device_capability())
        if capability != (7, 5):
            raise RuntimeError(
                f"CUDA INT8 diagnostic requires Tesla T4 / SM75, got SM{capability[0]}{capability[1]}"
            )

    kitchen = import_kitchen("comfy_kitchen")
    status = kitchen.list_backends().get(backend, {})
    if not status.get("available") or (backend == "eager" and status.get("disabled")):
        raise RuntimeError(f"Requested Comfy Kitchen backend {backend!r} is unavailable: {status}")
    return backend, kitchen
