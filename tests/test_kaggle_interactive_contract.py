from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "kaggle_h3_interactive.py"


def _source() -> str:
    return SCRIPT.read_text()


def test_interactive_provisioner_compiles() -> None:
    compile(_source(), str(SCRIPT), "exec")


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
