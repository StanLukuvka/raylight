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

    Distributed quantized model construction can create large transient host
    allocations. Ray's usual fan-out starts those allocations on every worker
    simultaneously. This helper intentionally trades setup latency for a lower
    aggregate peak without changing the later collective FSDP materialization.
    """
    for worker in workers:
        wait(start(worker))
