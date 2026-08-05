import ast
from pathlib import Path


ROOT = Path(__file__).parents[1]
MODEL_PATCHER_PATH = ROOT / "src" / "raylight" / "comfy_dist" / "model_patcher.py"
FSDP_UTILS_PATH = ROOT / "src" / "raylight" / "comfy_dist" / "fsdp_utils.py"
INT8_PATCH_PATH = ROOT / "src" / "raylight" / "comfy_dist" / "kitchen_patches" / "int8.py"


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


def test_quantized_local_shards_do_not_pass_through_dtensor_from_local_view():
    tree = ast.parse(FSDP_UTILS_PATH.read_text())
    wrapper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_wrap_quantized_local_as_dtensor"
    )
    calls = list(ast.walk(wrapper))

    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "DTensor"
        for node in calls
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_local"
        for node in calls
    )


def test_quant_layout_handlers_cover_fsdp_state_assignment():
    tree = ast.parse(MODEL_PATCHER_PATH.read_text())
    patch_fsdp = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "patch_fsdp"
    )
    with_block = next(
        node
        for node in ast.walk(patch_fsdp)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Name)
            and item.context_expr.id == "patch_context"
            for item in node.items
        )
    )

    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_from_full_model_state_dict"
        for node in ast.walk(with_block)
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "sitepkg_ck_patches"
        for node in ast.walk(patch_fsdp)
    )


def test_eager_int8_linear_streams_matmul_rows_into_preallocated_output():
    tree = ast.parse(INT8_PATCH_PATH.read_text())
    bounded = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_bounded_eager_int8_linear"
    )
    source = ast.unparse(bounded)

    assert "output = torch.empty" in source
    assert "128 * 1024 * 1024 // (n * 4)" in source
    assert "_int8_matmul_accumulate(x_8[i:end_i], weight_t)" in source
    assert "output[i:end_i].copy_" in source

    assert "scaled_parts" not in source
    assert "torch.cat" not in source


def test_int8_patch_installs_and_restores_bounded_eager_kernel():
    source = INT8_PATCH_PATH.read_text()
    assert "eager_backend.int8_linear = _bounded_eager_int8_linear" in source
    assert "eager_backend.int8_linear = _ORIG_EAGER_INT8_LINEAR" in source
