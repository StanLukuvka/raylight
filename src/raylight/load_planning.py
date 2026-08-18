"""Small orchestration helpers for model loading."""

from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager, nullcontext
from typing import TypeVar

Worker = TypeVar("Worker")
Pending = TypeVar("Pending")


def load_workers_sequentially(
    workers: Iterable[Worker],
    *,
    start: Callable[[Worker], Pending],
    wait: Callable[[Pending], object],
    phase: Callable[[int], AbstractContextManager[object]] | None = None,
) -> None:
    """Start and finish one worker load before starting the next.

    Distributed quantized model construction and local FSDP shard
    materialization can create large transient host allocations. Ray's usual
    fan-out starts those allocations on every worker simultaneously. The
    worker's load operation must therefore finish checkpoint mapping, local
    shard materialization, and full-state release before it returns. This
    helper intentionally trades setup latency for a bounded aggregate peak.

    Kept as the default for quantized loads.  Parallel fan-out is available
    via ``RAYLIGHT_PARALLEL_QUANT_LOAD=1`` for hosts with confirmed headroom;
    call :func:`load_workers_fanout` to select it instead.
    """
    for worker_index, worker in enumerate(workers):
        phase_context = phase(worker_index) if phase is not None else nullcontext()
        with phase_context:
            wait(start(worker))


def load_workers_fanout(
    workers: Iterable[Worker],
    *,
    start: Callable[[Worker], Pending],
    wait: Callable[[Pending], object],
) -> None:
    """Start every worker load, then collect every result (parallel fan-out).

    Each rank maps the checkpoint and materializes its own quantized shards
    onto its own GPU.  Opt-in only: on cgroup-limited hosts with thin host
    headroom the concurrent transient peak can thrash (see docs/19).  Callers
    select this helper behind ``RAYLIGHT_PARALLEL_QUANT_LOAD=1``.

    ``wait`` is called once per started worker immediately (Ray queues the
    remote call and blocks on result readiness), which preserves per-worker
    error attribution while letting both actors work concurrently.
    """
    workers = list(workers)
    futures = [(start(worker), index) for index, worker in enumerate(workers)]
    for future, _index in futures:
        wait(future)
