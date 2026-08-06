from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "src" / "raylight" / "load_telemetry.py"
NODES_PATH = Path(__file__).parents[1] / "src" / "raylight" / "nodes.py"
WORKER_PATH = Path(__file__).parents[1] / "src" / "raylight" / "distributed_worker" / "ray_worker.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("raylight_load_telemetry", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_emit_load_event_writes_parseable_ranked_timestamp(capsys) -> None:
    module = _load_module()
    module.time.time = lambda: 1234.5
    module.time.monotonic = lambda: 99.25

    payload = module.emit_load_event("fsdp_materialized", rank=1, checkpoint="h3.safetensors")

    line = capsys.readouterr().out.strip()
    assert line.startswith("[raylight-load-phase] ")
    assert json.loads(line.removeprefix("[raylight-load-phase] ")) == {
        "checkpoint": "h3.safetensors",
        "event": "fsdp_materialized",
        "marker": "raylight_load_phase",
        "monotonic": 99.25,
        "rank": 1,
        "unix": 1234.5,
    }
    assert payload["event"] == "fsdp_materialized"


def test_load_phase_emits_duration_and_error_status(capsys) -> None:
    module = _load_module()
    monotonic_values = iter((10.0, 12.5, 20.0, 21.0))
    module.time.time = lambda: 1234.5
    module.time.monotonic = lambda: next(monotonic_values)

    with module.load_phase("ray_init"):
        pass

    try:
        with module.load_phase("rank_load", rank=1):
            raise ValueError("bad checkpoint")
    except ValueError:
        pass

    events = [
        json.loads(line.removeprefix("[raylight-load-phase] "))
        for line in capsys.readouterr().out.splitlines()
    ]
    assert [(event["event"], event.get("status")) for event in events] == [
        ("ray_init_start", None),
        ("ray_init_end", "ok"),
        ("rank_load_start", None),
        ("rank_load_end", "error"),
    ]
    assert events[1]["elapsed_seconds"] == 2.5
    assert events[3]["elapsed_seconds"] == 1.0
    assert events[3]["error_type"] == "ValueError"


def test_loading_boundaries_emit_structured_phase_events() -> None:
    nodes = NODES_PATH.read_text()
    worker = WORKER_PATH.read_text()

    for marker in (
        'load_phase("initializer_conditioning_release")',
        'load_phase("ray_init")',
        'load_phase("nccl_probe")',
        'emit_load_event("nccl_probe_skipped")',
        'load_phase("actor_spawn")',
        'load_phase("loader_conditioning_release")',
    ):
        assert marker in nodes

    for marker in (
        'load_phase("checkpoint_mapping", rank=self.local_rank)',
        'load_phase("fsdp_materialization", rank=self.local_rank)',
    ):
        assert marker in worker
