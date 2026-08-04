import ast
from pathlib import Path


NODES_PATH = Path(__file__).parents[1] / "src" / "raylight" / "nodes.py"


def _ray_unet_loader() -> ast.ClassDef:
    tree = ast.parse(NODES_PATH.read_text())
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RayUNETLoader"
    )


def test_ray_unet_loader_accepts_conditioning_dependency_before_loading_weights():
    loader = _ray_unet_loader()
    input_types = next(
        node
        for node in loader.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "INPUT_TYPES"
    )
    load_method = next(
        node
        for node in loader.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_ray_unet"
    )

    input_contract = ast.dump(input_types)
    argument_names = [argument.arg for argument in load_method.args.args]

    assert "load_after" in input_contract
    assert "CONDITIONING" in input_contract
    assert "load_after" in argument_names
