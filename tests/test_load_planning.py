"""Tests for load orchestration helpers."""

from contextlib import contextmanager

from raylight.load_planning import load_workers_fanout, load_workers_sequentially


@contextmanager
def _no_phase(_index):
    yield


def test_fanout_starts_all_workers_before_waiting():
    started: list[str] = []
    events: list[str] = []

    def start(worker):
        events.append(f"start:{worker}")
        return worker

    def wait(pending):
        events.append(f"wait:{pending}")

    load_workers_fanout(["a", "b"], start=start, wait=wait)

    assert events == ["start:a", "start:b", "wait:a", "wait:b"]


def test_sequential_waits_between_workers():
    started: list[str] = []
    events: list[str] = []

    def start(worker):
        events.append(f"start:{worker}")
        return worker

    def wait(pending):
        events.append(f"wait:{pending}")

    load_workers_sequentially(
        ["a", "b"],
        start=start,
        wait=wait,
        phase=_no_phase,
    )

    assert events == ["start:a", "wait:a", "start:b", "wait:b"]
