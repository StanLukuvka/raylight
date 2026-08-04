import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "src" / "raylight" / "memory_telemetry.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("raylight_memory_telemetry", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_reads_current_and_peak_rss_from_proc_status(tmp_path):
    module = _load_module()
    status = tmp_path / "status"
    status.write_text("Name:\tworker\nVmRSS:\t1024 kB\nVmHWM:\t3072 kB\n")

    snapshot = module.process_memory_snapshot(status)

    assert snapshot == {"rss_bytes": 1024 * 1024, "peak_rss_bytes": 3072 * 1024}


def test_missing_proc_fields_are_reported_as_none(tmp_path):
    module = _load_module()
    status = tmp_path / "status"
    status.write_text("Name:\tworker\n")

    snapshot = module.process_memory_snapshot(status)

    assert snapshot == {"rss_bytes": None, "peak_rss_bytes": None}
