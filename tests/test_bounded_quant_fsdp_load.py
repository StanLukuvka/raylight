import ast
from pathlib import Path


MODEL_PATCHER_PATH = Path(__file__).parents[1] / "src" / "raylight" / "comfy_dist" / "model_patcher.py"


def test_quantized_fsdp_releases_checkpoint_entries_as_shards_materialize():
    tree = ast.parse(MODEL_PATCHER_PATH.read_text())
    patch_fsdp = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "patch_fsdp"
    )
    loader_call = next(
        node
        for node in ast.walk(patch_fsdp)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_from_full_model_state_dict"
    )
    release_keyword = next(
        keyword for keyword in loader_call.keywords if keyword.arg == "release_sd"
    )

    assert isinstance(release_keyword.value, ast.Constant)
    assert release_keyword.value.value is True
