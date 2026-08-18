from pathlib import Path


SMOKE = Path(__file__).parents[1] / "tools" / "kaggle_h3_full_smoke.py"


def test_full_smoke_preserves_required_h3_sampling_contract():
    source = SMOKE.read_text()

    assert '"width": 608' in source
    assert '"height": 352' in source
    assert '"length": 124' in source
    assert '"steps": 20' in source
    assert '"scheduler": "simple"' in source
    assert '"sampler_name": "res_multistep"' in source
    assert '"ulysses_degree": 2' in source
    assert '"FSDP": True' in source
    assert '"FSDP_CPU_OFFLOAD": False' in source
    assert '"XFuser_attention": "SAGE_FP16"' in source
    assert '"clear_vram_after_sampling": True' in source


def test_full_smoke_conditions_before_unet_and_decodes_both_modalities():
    source = SMOKE.read_text()

    assert '"load_after": ["conditioning", 0]' in source
    assert '"ray_object_store_gb": 0.5' in source
    assert '"class_type": "VAEDecode"' in source
    assert '"class_type": "VAEDecodeAudio"' in source
    assert '"class_type": "SaveVideo"' in source
