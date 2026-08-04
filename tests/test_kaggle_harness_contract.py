from pathlib import Path


HARNESS = Path(__file__).parents[1] / "tools" / "kaggle_h3_loader_acceptance.py"


def test_acceptance_harness_uses_an_owned_workspace():
    source = HARNESS.read_text()

    assert 'ROOT = WORK / "raylight-loader-acceptance"' in source
    assert 'COMFY = ROOT / "ComfyUI"' in source
    assert 'COMFY = WORK / "ComfyUI"' not in source
    assert "destination.resolve().relative_to(ROOT.resolve())" in source


def test_acceptance_harness_pins_and_verifies_raylight_commit():
    source = HARNESS.read_text()

    assert '"7d9219fd854b1851463479274666b9f9fd2b3eae"' in source
    assert '"fix/bounded-quant-fsdp-load"' not in source
    assert 'resolved != ref' in source
