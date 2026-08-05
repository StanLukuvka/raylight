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


def _ray_initializer() -> ast.ClassDef:
    tree = ast.parse(NODES_PATH.read_text())
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RayInitializer"
    )


def test_ray_initializer_releases_conditioning_before_ray_init():
    initializer = _ray_initializer()
    input_types = next(
        node
        for node in initializer.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "INPUT_TYPES"
    )
    spawn = next(
        node
        for node in initializer.body
        if isinstance(node, ast.FunctionDef) and node.name == "spawn_actor"
    )
    contract = ast.dump(input_types)
    assert "load_after" in contract
    assert "CONDITIONING" in contract
    assert "ray_object_store_gb" in contract

    unload_call = next(
        node
        for node in ast.walk(spawn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "unload_all_models"
    )
    ray_init_call = next(
        node
        for node in ast.walk(spawn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "ray"
        and node.func.attr == "init"
    )
    assert unload_call.lineno < ray_init_call.lineno
    default_names = [argument.arg for argument in spawn.args.args[-len(spawn.args.defaults):]]
    defaults = dict(zip(default_names, spawn.args.defaults, strict=True))
    assert ast.literal_eval(defaults["ray_object_store_gb"]) == 0.5


def test_ray_runtime_env_explicitly_propagates_h3_diagnostic_controls():
    source = NODES_PATH.read_text()
    assert "_inject_h3_diagnostic_env(runtime_env_base)" in source
    for name in (
        "RAYLIGHT_INT8_ACCUMULATOR_MIB",
        "RAYLIGHT_H3_MEMORY_TRACE",
        "RAYLIGHT_H3_STOP_AFTER_FIRST_FORWARD",
    ):
        assert name in source


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
    unload_call = next(
        node
        for node in ast.walk(load_method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "unload_all_models"
    )
    actor_call = next(
        node
        for node in ast.walk(load_method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ensure_fresh_actors"
    )

    assert unload_call.lineno < actor_call.lineno
    assert collect_call.lineno < actor_call.lineno


def test_minimax_h3_fails_closed_without_conditioning_barrier_before_starting_actors():
    cls = _ray_unet_loader()
    load_method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "load_ray_unet"
    )
    actor_call = next(
        node
        for node in ast.walk(load_method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ensure_fresh_actors"
    )
    fail_closed = next(
        node
        for node in ast.walk(load_method)
        if isinstance(node, ast.If)
        and "minimax_h3" in ast.dump(node)
        and any(isinstance(child, ast.Raise) for child in ast.walk(node))
    )

    assert fail_closed.lineno < actor_call.lineno
    assert "load_after" in ast.dump(fail_closed)
