from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "src" / "raylight" / "load_telemetry.py"
NODES_PATH = Path(__file__).parents[1] / "src" / "raylight" / "nodes.py"
WORKER_PATH = Path(__file__).parents[1] / "src" / "raylight" / "distributed_worker" / "ray_worker.py"
SAMPLER_PATH = Path(__file__).parents[1] / "src" / "raylight" / "comfy_extra_dist" / "nodes_custom_sampler.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("raylight_load_telemetry", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_emit_load_event_writes_parseable_ranked_timestamp(capsys, monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module.time, "time", lambda: 1234.5)
    monkeypatch.setattr(module.time, "monotonic", lambda: 99.25)
    setattr(
        module,
        "capture_load_snapshot",
        lambda: {
            "process": {"pid": 123, "ppid": 45, "rss_bytes": 4096},
            "cgroup": {"memory_current_bytes": 8192},
            "cuda": {"initialized": False},
        },
    )

    payload = module.emit_load_event(
        "fsdp_materialized",
        rank=1,
        process_role="ray_worker",
        checkpoint="h3.safetensors",
    )

    line = capsys.readouterr().out.strip()
    assert line.startswith("[raylight-load-phase] ")
    assert json.loads(line.removeprefix("[raylight-load-phase] ")) == {
        "checkpoint": "h3.safetensors",
        "cgroup": {"memory_current_bytes": 8192},
        "cuda": {"initialized": False},
        "event": "fsdp_materialized",
        "marker": "raylight_load_phase",
        "monotonic": 99.25,
        "process": {"pid": 123, "ppid": 45, "rss_bytes": 4096},
        "process_role": "ray_worker",
        "rank": 1,
        "schema_version": 2,
        "unix": 1234.5,
    }
    assert payload["event"] == "fsdp_materialized"


def test_load_phase_emits_duration_and_error_status(capsys, monkeypatch) -> None:
    module = _load_module()
    monotonic_values = iter((10.0, 12.5, 20.0, 21.0))
    monkeypatch.setattr(module.time, "time", lambda: 1234.5)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic_values))
    setattr(
        module,
        "capture_load_snapshot",
        lambda: {
            "process": {"pid": 123},
            "cgroup": {},
            "cuda": {"initialized": False},
        },
    )
    phase_ids = iter(("phase-ok", "phase-error"))
    setattr(module, "_new_phase_id", lambda: next(phase_ids))

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
    assert [event["phase_id"] for event in events] == [
        "phase-ok",
        "phase-ok",
        "phase-error",
        "phase-error",
    ]


def test_capture_load_snapshot_is_bounded_and_machine_parseable() -> None:
    module = _load_module()

    snapshot = module.capture_load_snapshot()

    assert set(snapshot) == {"process", "cgroup", "cuda"}
    assert snapshot["process"]["pid"] > 0
    assert snapshot["process"]["ppid"] >= 0
    assert snapshot["process"]["rss_bytes"] is None or snapshot["process"]["rss_bytes"] >= 0
    assert snapshot["cuda"]["initialized"] is False


def test_telemetry_output_failure_never_prevents_phase_body(monkeypatch) -> None:
    module = _load_module()
    body_ran = False

    def broken_print(*args, **kwargs):
        raise BrokenPipeError("diagnostic stream closed")

    monkeypatch.setattr("builtins.print", broken_print)
    with module.load_phase("must_run"):
        body_ran = True

    assert body_ran is True


def test_telemetry_failure_never_masks_original_exception(monkeypatch) -> None:
    module = _load_module()

    def broken_print(*args, **kwargs):
        raise BrokenPipeError("diagnostic stream closed")

    monkeypatch.setattr("builtins.print", broken_print)
    try:
        with module.load_phase("must_fail"):
            raise ValueError("original workload failure")
    except ValueError as exc:
        assert str(exc) == "original workload failure"
    else:
        raise AssertionError("original workload exception was swallowed")


def test_nonserializable_event_details_are_fail_open(monkeypatch) -> None:
    module = _load_module()
    output = []
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: output.append(args[0]))

    payload = module.emit_load_event("nonserializable", value=object())

    assert payload["event"] == "nonserializable"
    assert len(output) == 1
    fallback = json.loads(output[0].removeprefix("[raylight-load-phase] "))
    assert fallback["event"] == "nonserializable"
    assert fallback["status"] == "telemetry_error"
    assert fallback["telemetry_error_type"] == "TypeError"


def test_snapshot_failure_emits_minimal_fallback_without_aborting(capsys) -> None:
    module = _load_module()

    def broken_snapshot():
        raise OSError("proc unavailable")

    setattr(module, "capture_load_snapshot", broken_snapshot)
    payload = module.emit_load_event("snapshot_failed")

    line = capsys.readouterr().out.strip()
    emitted = json.loads(line.removeprefix("[raylight-load-phase] "))
    assert emitted["event"] == "snapshot_failed"
    assert emitted["telemetry_snapshot_error_type"] == "OSError"
    assert emitted["cuda"] == {"initialized": False}
    assert payload["event"] == "snapshot_failed"


def test_loading_boundaries_emit_structured_phase_events() -> None:
    nodes = NODES_PATH.read_text()
    worker = WORKER_PATH.read_text()
    sampler = SAMPLER_PATH.read_text()

    def assert_calls(source: str, function: str, names: tuple[str, ...]) -> None:
        for name in names:
            assert re.search(rf'{function}\(\s*"{name}"', source), f"missing {function}({name!r})"

    assert_calls(
        nodes,
        "(?:emit_load_event|load_phase)",
        (
            "initializer_plan",
            "ray_pre_init_cleanup",
            "initializer_conditioning_release",
            "ray_init",
            "nccl_probe",
            "nccl_probe_skipped",
            "actor_spawn",
            "loader_conditioning_release",
            "worker_configuration",
            "checkpoint_classification",
            "worker_model_load",
            "worker_load_rpc",
            "worker_runtime_patch",
            "model_ready",
            "worker_vram_release",
            "worker_shutdown",
        ),
    )
    assert_calls(
        worker,
        "(?:emit_load_event|load_phase)",
        (
            "worker_load_request",
            "worker_model_reused",
            "checkpoint_mapping",
            "checkpoint_reclaim",
            "state_dict_handoff",
            "fsdp_materialization",
            "worker_model_ready",
            "sampling_vram_release",
            "process_group_destroy",
        ),
    )
    assert "def _telemetry_memory_snapshot(self):" in worker
    assert "worker_snapshot=self.get_memory_snapshot()" not in worker
    assert_calls(
        sampler,
        "(?:emit_load_event|load_phase)",
        (
            "sampling_prepare",
            "sampling_dispatch",
            "sampling_complete",
            "worker_vram_release",
        ),
    )
