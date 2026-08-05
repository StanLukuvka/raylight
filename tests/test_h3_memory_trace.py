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


def test_first_attention_call_reports_bounded_cuda_phase_timings():
    trace = TRACE.read_text()
    source = XDIT.read_text()
    assert 'os.environ.get("RAYLIGHT_H3_PHASE_PROFILE", "0") == "1"' in trace
    assert "def h3_cuda_phase_report(" in trace
    assert "torch.cuda.synchronize()" in trace
    for version_field in ('"torch"', '"cuda"', '"ray"', '"xfuser"', '"yunchang"'):
        assert version_field in trace
    for phase in ("qkv_projection", "rmsnorm_rope", "ulysses_attention", "attention_output_projection"):
        assert f'"{phase}"' in source
    assert "_H3_ATTN_PROFILE_DONE = True" in source
    assert "phase_start = phase_end" not in source
    assert "_H3_FORWARD_COUNT" in source
    assert "profile_this_forward = phase_profile_enabled and _H3_FORWARD_COUNT == 1" in source
    assert "if profile_this_forward and i == 0:" in source
    assert "set_h3_phase_profile_active(True)" in source
    assert "set_h3_phase_profile_active(False)" in source
    assert "def h3_phase_profile_active(" in trace


def test_initial_int8_calls_report_operator_shape_cap_and_cuda_time():
    source = INT8.read_text()
    assert '"int8_linear_start"' in source
    assert '"int8_output_allocated"' in source
    assert '"int8_linear_end"' in source
    for field in ("m=m", "k=x_8.shape[1]", "n=n", "accumulator_mib=accumulator_mib"):
        assert field in source
    for phase in ("qkv_int8", "attention_output_int8", "fc1_int8", "fc2_int8"):
        assert f'"{phase}"' in source
    assert "h3_cuda_phase_report(" in source
    assert "and not h3_phase_profile_enabled()" in source
    assert "_INT8_PHASE_CALLS" in source
