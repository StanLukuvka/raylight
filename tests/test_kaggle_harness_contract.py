from pathlib import Path


HARNESS = Path(__file__).parents[1] / "tools" / "kaggle_h3_loader_acceptance.py"


def test_acceptance_harness_uses_an_owned_workspace():
    source = HARNESS.read_text()

    assert 'ROOT = WORK / "raylight-loader-acceptance"' in source
    assert 'COMFY = ROOT / "ComfyUI"' in source
    assert 'COMFY = WORK / "ComfyUI"' not in source
    assert "destination.resolve().relative_to(ROOT.resolve())" in source
    assert "if ROOT.is_symlink()" in source
    assert "if child.is_symlink()" in source


def test_acceptance_harness_pins_and_verifies_raylight_commit():
    source = HARNESS.read_text()

    assert '"d54addfe58d9e36993e2a010784b4b4ede1b1ef5"' in source
    assert '"fix/bounded-quant-fsdp-load"' not in source
    assert 'resolved != ref' in source
