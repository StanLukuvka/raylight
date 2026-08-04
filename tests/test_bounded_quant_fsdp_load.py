import ast
from pathlib import Path


ROOT = Path(__file__).parents[1]
MODEL_PATCHER_PATH = ROOT / "src" / "raylight" / "comfy_dist" / "model_patcher.py"
FSDP_UTILS_PATH = ROOT / "src" / "raylight" / "comfy_dist" / "fsdp_utils.py"


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


def test_quant_release_keeps_independent_input_scales_until_their_turn():
    tree = ast.parse(FSDP_UTILS_PATH.read_text())
    release = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_release_quant_keys"
    )
    literals = {
        node.value
        for node in ast.walk(release)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert "input_scale" not in literals
    assert "scale_input" not in literals
