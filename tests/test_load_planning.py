import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "src" / "raylight" / "load_planning.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("raylight_load_planning", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_workers_finish_loading_one_at_a_time():
    module = _load_module()
    events = []

    def start(worker):
        events.append(("start", worker))
        return f"future-{worker}"

    def wait(future):
        events.append(("finish", future))

    module.load_workers_sequentially(["rank0", "rank1"], start=start, wait=wait)

    assert events == [
        ("start", "rank0"),
        ("finish", "future-rank0"),
        ("start", "rank1"),
        ("finish", "future-rank1"),
    ]
