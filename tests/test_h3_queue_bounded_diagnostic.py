from pathlib import Path


ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "tools/kaggle_h3_queue_bounded_diagnostic.py"


def test_bounded_queue_uses_exact_accepted_distributed_graph_and_two_unload_barriers():
    source = RUNNER.read_text()
    for value in (
        '"class_type": "RayInitializer"',
        '"GPU": 2',
        '"ulysses_degree": 2',
        '"ring_degree": 1',
        '"cfg_degree": 1',
        '"dp_degree": 1',
        '"FSDP": True',
        '"XFuser_attention": "TORCH_EFFICIENT"',
        '"ray_object_store_gb": 0.5',
        '"ray_dashboard_address": "None"',
        '"torch_dist_address": "127.0.0.1:29500"',
        '"class_type": "RayUNETLoader"',
        '"class_type": "XFuserSamplerCustomAdvanced"',
    ):
        assert value in source
    assert source.count('"load_after": ["conditioning", 0]') == 2


def test_bounded_queue_is_shape_configurable_and_accepts_only_the_intentional_stop():
    source = RUNNER.read_text()
    for arg in ("--width", "--height", "--length"):
        assert f'"{arg}"' in source
    assert "MiniMax-H3 diagnostic intentionally stopped after the first denoiser forward" in source
    assert '"execution_error"' in source
    assert '"completed_without_intentional_stop"' in source
    assert "60 * 20" in source
