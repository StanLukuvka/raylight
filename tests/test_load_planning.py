import ast
import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "src" / "raylight" / "load_planning.py"
NODES_PATH = Path(__file__).parents[1] / "src" / "raylight" / "nodes.py"
RAY_WORKER_PATH = (
    Path(__file__).parents[1] / "src" / "raylight" / "distributed_worker" / "ray_worker.py"
)


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


def test_quantized_fsdp_branches_use_the_sequential_loader():
    tree = ast.parse(NODES_PATH.read_text())
    loader_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RayUNETLoader"
    )
    load_method = next(
        node
        for node in loader_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_ray_unet"
    )
    calls = [
        node
        for node in ast.walk(load_method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_workers_sequentially"
    ]

    assert len(calls) == 2


def test_quantized_worker_materializes_and_releases_full_state_before_returning():
    tree = ast.parse(RAY_WORKER_PATH.read_text())
    worker_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RayWorker"
    )
    load_method = next(
        node
        for node in worker_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_unet"
    )
    calls = {
        node.func.attr: node
        for node in ast.walk(load_method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr in {"set_state_dict", "_patch_fsdp_for_sampling"}
    }
    parents = {
        child: parent
        for parent in ast.walk(load_method)
        for child in ast.iter_child_nodes(parent)
    }

    assert calls["set_state_dict"].lineno < calls["_patch_fsdp_for_sampling"].lineno
    for call in calls.values():
        ancestors = []
        node = call
        while node in parents:
            node = parents[node]
            ancestors.append(node)
        guards = [ancestor for ancestor in ancestors if isinstance(ancestor, ast.If)]
        assert any("is_quant" in ast.dump(guard.test) for guard in guards)
