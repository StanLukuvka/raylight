"""Small orchestration helpers for bounded-memory model loading."""

from collections.abc import Callable, Iterable
from typing import TypeVar


Worker = TypeVar("Worker")
Pending = TypeVar("Pending")


def load_workers_sequentially(
    workers: Iterable[Worker],
    *,
    start: Callable[[Worker], Pending],
    wait: Callable[[Pending], object],
) -> None:
    """Start and finish one worker load before starting the next.

    Distributed quantized model construction and local FSDP shard
    materialization can create large transient host allocations. Ray's usual
    fan-out starts those allocations on every worker simultaneously. The
    worker's load operation must therefore finish checkpoint mapping, local
    shard materialization, and full-state release before it returns. This
    helper intentionally trades setup latency for a bounded aggregate peak.
    """
    for worker in workers:
        wait(start(worker))
