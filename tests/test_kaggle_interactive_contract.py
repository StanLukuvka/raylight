from __future__ import annotations

import ast
import json
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, cast


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "kaggle_h3_interactive.py"


def _source() -> str:
    return SCRIPT.read_text()


def test_interactive_provisioner_compiles() -> None:
    compile(_source(), str(SCRIPT), "exec")


def test_phase_timing_summary_extracts_ranked_schema_v2_events(tmp_path: Path) -> None:
    tree = ast.parse(_source())
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "_phase_timing_summary"
    )
    namespace = {"json": json}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SCRIPT), "exec"), namespace)
    log = tmp_path / "comfy0.log"
    log.write_text(
        "noise\n"
        "(RayWorker pid=1) [raylight-load-phase] "
        '{"event":"rank_load_end","schema_version":2,"rank":0,'
        '"status":"ok","elapsed_seconds":12.5,"unix":100.0}\n'
        "[raylight-load-phase] "
        '{"event":"process_group_destroy_end","schema_version":2,"rank":1,'
        '"status":"ok","elapsed_seconds":0.75,"unix":120.0}\n'
    )

    summary_function = cast(Callable[[Path], dict], namespace["_phase_timing_summary"])
    summary = summary_function(log)

    assert summary["event_count"] == 2
    assert summary["first_event_unix"] == 100.0
    assert summary["last_event_unix"] == 120.0
    assert summary["phases"] == [
        {
            "event": "rank_load_end",
            "rank": 0,
            "status": "ok",
            "elapsed_seconds": 12.5,
            "unix": 100.0,
        },
        {
            "event": "process_group_destroy_end",
            "rank": 1,
            "status": "ok",
            "elapsed_seconds": 0.75,
            "unix": 120.0,
        },
    ]


def test_phase_timing_summary_bounds_lines_and_fields(tmp_path: Path) -> None:
    tree = ast.parse(_source())
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "_phase_timing_summary"
    )
    namespace = {"json": json}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SCRIPT), "exec"), namespace)
    summary_function = cast(Callable[[Path], dict], namespace["_phase_timing_summary"])
    log = tmp_path / "comfy0.log"
    log.write_text(
        "[raylight-load-phase] "
        + json.dumps({"event": "oversized", "elapsed_seconds": 1, "detail": "x" * 70_000})
        + "\n[raylight-load-phase] "
        + json.dumps({
            "event": "sampling_end",
            "status": "ok",
            "error_type": "y" * 1_000,
            "elapsed_seconds": 3.5,
            "unix": 5.0,
        })
        + "\n"
    )

    summary = summary_function(log)

    assert summary["event_count"] == 1
    assert summary["oversized_line_count"] == 1
    assert summary["field_truncated_count"] == 1
    assert len(summary["phases"][0]["error_type"]) == 512


def test_phase_timing_summary_is_fail_open_on_reader_failure() -> None:
    tree = ast.parse(_source())
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "_phase_timing_summary"
    )
    namespace = {"json": json}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SCRIPT), "exec"), namespace)
    summary_function = cast(Callable[[object], dict], namespace["_phase_timing_summary"])

    class BrokenLog:
        name = "comfy0.log"

        @staticmethod
        def is_file():
            return True

        @staticmethod
        def open(*args, **kwargs):
            raise MemoryError("diagnostic reader exhausted memory")

    summary = summary_function(BrokenLog())

    assert summary["event_count"] == 0
    assert summary["summary_error_type"] == "MemoryError"
    assert summary["phases"] == []


def test_diagnostics_archive_is_self_describing_and_excludes_video() -> None:
    source = _source()
    tree = ast.parse(source)
    export = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "export_memory_diagnostics"
    )
    manifest_writer = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "_write_run_manifest"
    )
    export_source = ast.get_source_segment(source, export)
    writer_source = ast.get_source_segment(source, manifest_writer)
    assert export_source is not None
    assert writer_source is not None
    assert "h3-run-manifest.json" in export_source
    assert "raylight-phase-summary.json" in export_source
    assert '"video_included": False' in writer_source
    assert "torch.__version__" in writer_source
    assert "torch.version.cuda" in writer_source
    assert "gpu_identities" in writer_source
    assert "temporary_path.replace(manifest_path)" in writer_source
    assert ".mp4" not in export_source
    assert ".webm" not in export_source
    assert ".mov" not in export_source
    assert ".mkv" not in export_source


def test_diagnostics_archive_survives_phase_summary_failure(tmp_path: Path) -> None:
    source = _source()
    tree = ast.parse(source)
    export = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "export_memory_diagnostics"
    )
    work_dir = tmp_path / "work"
    app_root = tmp_path / "app"
    work_dir.mkdir()
    app_root.mkdir()
    (work_dir / "comfy0.log").write_text("complete log\n")
    (work_dir / "memory_snapshots.jsonl").write_text('{"memory":1}\n')
    (work_dir / "generated.mp4").write_bytes(b"video")
    startup_manifest = {
        "schema_version": 1,
        "run_started_unix": 123.0,
        "video_included": False,
        "runtime": {
            "cuda": "12.8",
            "pytorch": "2.10.0+cu128",
            "comfy_kitchen": "0.2.26",
            "visible_gpu_count": 2,
        },
        "gpu_identities": ["GPU 0: Tesla T4", "GPU 1: Tesla T4"],
        "distributed": {
            "shutdown_after_sampling": True,
            "sequential_rank_materialization": True,
        },
    }
    manifest_text = json.dumps(startup_manifest, sort_keys=True) + "\n"
    (work_dir / "h3-run-manifest.json").write_text(manifest_text)
    output = tmp_path / "h3-diagnostics.zip"

    def mapped_path(value):
        if str(value) == "/kaggle/working/h3-diagnostics.zip":
            return output
        return Path(value)

    def fail_summary(_log):
        raise MemoryError("summary failed")

    namespace = {
        "Path": mapped_path,
        "APP_ROOT": str(app_root),
        "json": json,
        "time": time,
        "zipfile": zipfile,
        "_phase_timing_summary": fail_summary,
        "WORK_DIR": str(work_dir),
        "ACTIVE_PROFILE": "full-run",
        "RAYLIGHT_COMMIT": "a" * 40,
        "COMFY_COMMIT": "b" * 40,
        "H3_WIDTH": 736,
        "H3_HEIGHT": 416,
        "H3_LENGTH": 124,
        "INT8_ACCUMULATOR_MIB": 64,
    }
    exec(compile(ast.Module(body=[export], type_ignores=[]), str(SCRIPT), "exec"), namespace)

    result = namespace["export_memory_diagnostics"]()

    assert result == output
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("h3-run-manifest.json"))
    assert "comfy0.log" in names
    assert "memory_snapshots.jsonl" in names
    assert "h3-run-manifest.json" in names
    assert "generated.mp4" not in names
    assert manifest == startup_manifest
    assert (work_dir / "h3-run-manifest.json").read_text() == manifest_text
    assert manifest["distributed"]["shutdown_after_sampling"] is True
    assert manifest["distributed"]["sequential_rank_materialization"] is True


def test_diagnostics_archive_continues_after_candidate_stat_and_write_failures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _source()
    tree = ast.parse(source)
    export = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "export_memory_diagnostics"
    )
    work_dir = tmp_path / "work"
    app_root = tmp_path / "app"
    work_dir.mkdir()
    app_root.mkdir()
    (work_dir / "h3-run-manifest.json").write_text('{"video_included":false}\n')
    (work_dir / "memory_snapshots.jsonl").write_text('{"memory":1}\n')
    (work_dir / "memory_monitor.log").write_text("monitor survives\n")
    (work_dir / "comfy0.log").write_text("complete log survives\n")
    output = tmp_path / "h3-diagnostics.zip"

    def mapped_path(value):
        if str(value) == "/kaggle/working/h3-diagnostics.zip":
            return output
        return Path(value)

    original_is_file = Path.is_file

    def flaky_is_file(path):
        if path.name == "memory_events_last.txt":
            raise OSError("candidate disappeared during stat")
        return original_is_file(path)

    original_write = zipfile.ZipFile.write

    def flaky_write(archive, filename, arcname=None, *args, **kwargs):
        if Path(filename).name == "memory_snapshots.jsonl":
            raise PermissionError("candidate became unreadable")
        return original_write(archive, filename, arcname, *args, **kwargs)

    monkeypatch.setattr(Path, "is_file", flaky_is_file)
    monkeypatch.setattr(zipfile.ZipFile, "write", flaky_write)
    namespace = {
        "Path": mapped_path,
        "APP_ROOT": str(app_root),
        "json": json,
        "time": time,
        "zipfile": zipfile,
        "_phase_timing_summary": lambda _log: {"event_count": 0, "phases": []},
        "WORK_DIR": str(work_dir),
    }
    exec(compile(ast.Module(body=[export], type_ignores=[]), str(SCRIPT), "exec"), namespace)

    assert namespace["export_memory_diagnostics"]() == output

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        errors = json.loads(archive.read("archive-errors.json"))["errors"]
    assert "h3-run-manifest.json" in names
    assert "memory_monitor.log" in names
    assert "comfy0.log" in names
    assert "memory_snapshots.jsonl" not in names
    assert errors == [
        {"candidate": "memory_snapshots.jsonl", "error_type": "PermissionError"},
        {"candidate": "memory_events_last.txt", "error_type": "OSError"},
    ]
    assert not any(name.lower().endswith((".mp4", ".webm", ".mov", ".mkv")) for name in names)


def test_dependency_verification_is_derived_from_config() -> None:
    source = _source()
    assert "for package in EXTRA_PIP_PACKAGES" in source
    assert "actual == expected" in source
    assert "kernels==0.14.0" not in source
    assert "kernels==0.16.0" not in source


def test_workflow_is_five_second_dual_t4_profile() -> None:
    source = _source()
    assert 'if conditioning_node.get("type") != "MiniMaxH3ImageToVideo":' in source
    assert "conditioning_values[1] = h3_width" in source
    assert "conditioning_values[2] = h3_height" in source
    assert "conditioning_values[3] = h3_length" in source
    assert "conditioning_values[3] = 0.20" not in source
    assert '{"name": "shutdown_after_sampling", "type": "BOOLEAN"' in source
    assert 'sampler["widgets_values"] = [True, 1, "randomize", True]' in source


def test_workflow_rejects_unaligned_or_non_five_second_diagnostic_dimensions() -> None:
    source = _source()
    assert "h3_width <= 0" in source
    assert "h3_height <= 0" in source
    assert "h3_width % 32" in source
    assert "h3_height % 32" in source
    assert "h3_length != 124" in source


def test_workflow_releases_conditioning_before_ray_initializer() -> None:
    source = _source()
    assert '"name": "load_after", "shape": 7, "type": "CONDITIONING"' in source
    assert '"target_id": initializer["id"]' in source
    assert '"target_id": loader["id"]' in source


def test_production_workflow_skips_redundant_nccl_probe() -> None:
    tree = ast.parse(_source())
    workflow = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_make_raylight_workflow"
    )
    initializer = next(
        node for node in workflow.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "initializer" for target in node.targets)
    )
    assert isinstance(initializer.value, ast.Dict)
    widgets = next(
        value for key, value in zip(initializer.value.keys, initializer.value.values, strict=True)
        if isinstance(key, ast.Constant) and key.value == "widgets_values"
    )
    assert isinstance(widgets, ast.List)
    assert ast.literal_eval(widgets.elts[12]) is True


def test_downloaded_native_binaries_are_checksum_verified() -> None:
    source = _source()
    assert '_require_sha256(archive, FILEBROWSER_ARCHIVE_SHA256' in source
    assert '_require_sha256(binary, FILEBROWSER_BINARY_SHA256' in source
    assert '_require_sha256(binary, CLOUDFLARED_SHA256' in source


def test_fake_weight_mode_uses_sparse_exact_size_placeholders(tmp_path: Path) -> None:
    tree = ast.parse(_source())
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "_prepare_fake_model_stubs"
    )
    namespace = {
        "Path": Path,
        "APP_ROOT": str(tmp_path),
        "REQUIRED_MODELS": [
            {"folder": "vae", "name": "fake.safetensors", "expected_bytes": 1_000_000_000}
        ],
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SCRIPT), "exec"), namespace)
    root = namespace["_prepare_fake_model_stubs"]()
    stub = root / "vae" / "fake.safetensors"
    assert stub.stat().st_size == 1_000_000_000
    assert stub.stat().st_blocks * 512 < 1_000_000


def test_incomplete_checkout_inside_app_root_is_repaired(tmp_path: Path) -> None:
    tree = ast.parse(_source())
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "_checkout_pinned_repo"
    )
    destination = tmp_path / "ComfyUI"
    destination.mkdir()
    (destination / "partial-file").write_text("partial")
    commit = "a" * 40

    def fake_run(command, **_kwargs):
        if command[1] == "clone":
            destination.mkdir(parents=True, exist_ok=True)
            (destination / ".git").mkdir()
        if "rev-parse" in command:
            return SimpleNamespace(returncode=0, stdout=commit + "\n")
        return SimpleNamespace(returncode=0, stdout="")

    namespace = {
        "Path": Path,
        "APP_ROOT": str(tmp_path),
        "shutil": __import__("shutil"),
        "_run": fake_run,
        "RuntimeError": RuntimeError,
        "print": print,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SCRIPT), "exec"), namespace)
    namespace["_checkout_pinned_repo"]("https://example.invalid/repo.git", destination, commit)
    assert not (destination / "partial-file").exists()
    assert (destination / ".git").is_dir()


def test_int8_probe_action_is_isolated_from_server_startup_and_model_discovery() -> None:
    source = _source()
    probe_branch = source.index('if action == "int8-probe":')
    model_discovery = source.index("selected_models = _discover_kaggle_models()")
    server_start = source.index("        _start_memory_monitor()")
    assert probe_branch < model_discovery < server_start
    assert "kaggle_h3_int8_backend_probe.py" in source
    assert 'globals().get("H3_INT8_PROBE_ALLOW_CUDA_UNDER_13", False)' in source


def test_all_helpers_precede_main_dispatch() -> None:
    tree = ast.parse(_source())
    function_names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert "_print_storage" in function_names
    assert "_accelerator_preflight" in function_names
    assert "print_memory_diagnostics" in function_names
    main_index = function_names.index("main")
    assert function_names.index("_print_storage") < main_index
    assert function_names.index("_accelerator_preflight") < main_index
    assert function_names.index("print_memory_diagnostics") < main_index
    assert source_ends_in_main_call(tree)


def source_ends_in_main_call(tree: ast.Module) -> bool:
    last = tree.body[-1]
    return (
        isinstance(last, ast.Expr)
        and isinstance(last.value, ast.Call)
        and isinstance(last.value.func, ast.Name)
        and last.value.func.id == "main"
    )
