from __future__ import annotations

import ast
from pathlib import Path


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
