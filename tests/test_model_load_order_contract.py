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
    assert "load_after" in argument_names
    assert "load_after" in input_contract
    assert "CONDITIONING" in input_contract


def test_conditioning_barrier_collects_released_encoder_before_starting_actors():
    cls = _ray_unet_loader()
    load_method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "load_ray_unet"
    )
    collect_call = next(
        node
        for node in ast.walk(load_method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "gc"
        and node.func.attr == "collect"
    )
    actor_call = next(
        node
        for node in ast.walk(load_method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ensure_fresh_actors"
    )

    assert collect_call.lineno < actor_call.lineno
