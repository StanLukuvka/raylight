from pathlib import Path


ROOT = Path(__file__).parents[1]
TRACE = ROOT / "src/raylight/memory_trace.py"
INT8 = ROOT / "src/raylight/comfy_dist/kitchen_patches/int8.py"
XDIT = ROOT / "src/raylight/diffusion_models/minimax/xdit_context_parallel.py"


def test_memory_trace_reports_allocator_and_device_state():
    source = TRACE.read_text()
    assert 'os.environ.get("RAYLIGHT_H3_MEMORY_TRACE", "0") == "1"' in source
    assert "torch.cuda.memory_allocated()" in source
    assert "torch.cuda.memory_reserved()" in source
    assert "torch.cuda.max_memory_allocated()" in source
    assert "torch.cuda.mem_get_info()" in source
    assert 'os.environ.get("RAYLIGHT_H3_STOP_AFTER_FIRST_FORWARD", "0") == "1"' in source


def test_first_h3_forward_marks_split_blocks_and_final_gather():
    source = XDIT.read_text()
    for marker in (
        '"h3_forward_start"',
        '"h3_before_split"',
        '"h3_after_split_release"',
        '"h3_block_start"',
        '"h3_block_end"',
        '"h3_before_final_gather"',
        '"h3_forward_end"',
    ):
        assert marker in source
    assert "MiniMax-H3 diagnostic intentionally stopped after the first denoiser forward" in source


def test_initial_int8_calls_report_operator_shape_and_cap():
    source = INT8.read_text()
    assert '"int8_linear_start"' in source
    assert '"int8_output_allocated"' in source
    assert '"int8_linear_end"' in source
    for field in ("m=m", "k=x_8.shape[1]", "n=n", "accumulator_mib=accumulator_mib"):
        assert field in source
