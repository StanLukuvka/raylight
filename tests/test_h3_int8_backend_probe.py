from pathlib import Path


ROOT = Path(__file__).parents[1]
PROBE = ROOT / "tools/kaggle_h3_int8_backend_probe.py"


def test_probe_covers_real_h3_linear_shapes_and_backends():
    source = PROBE.read_text()
    for name, m, k, n in (
        ("qkv", 6103, 5376, 21504),
        ("attention_out", 6103, 7168, 5376),
        ("fc1", 6103, 5376, 28672),
        ("fc2", 6103, 14336, 5376),
    ):
        assert f'("{name}", {m}, {k}, {n})' in source
    assert 'choices=("eager", "cuda")' in source
    assert "ck.use_backend(backend)" in source
    assert "install_int8_patches()" in source


def test_probe_fails_closed_for_cuda_below_13_without_explicit_override():
    source = PROBE.read_text()
    assert '"--allow-cuda-under-13"' in source
    assert "CUDA backend probe below CUDA 13 requires --allow-cuda-under-13" in source
    assert "torch.cuda.get_device_capability()" in source
    assert "ck.list_backends()" in source
    guard_pos = source.index('backend == "cuda" and _cuda_major(torch) < 13')
    import_pos = source.index("import comfy_kitchen as ck")
    enable_pos = source.index('ck.enable_backend("cuda")')
    assert guard_pos < import_pos < enable_pos


def test_probe_reports_timing_memory_and_numerical_comparison():
    source = PROBE.read_text()
    for field in (
        '"median_ms"',
        '"peak_allocated"',
        '"peak_reserved"',
        '"free_after"',
        '"max_abs_error"',
        '"mean_abs_error"',
        '"mismatch_fraction"',
    ):
        assert field in source
