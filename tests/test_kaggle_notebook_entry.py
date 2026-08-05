import hashlib
from pathlib import Path
from unittest.mock import patch


ENTRY_PATH = Path(__file__).parents[1] / "tools" / "kaggle_h3_notebook_entry.py"
PROVISIONER_PATH = Path(__file__).parents[1] / "tools" / "kaggle_h3_interactive.py"


def test_compact_notebook_entry_defines_complete_configuration_before_dispatch():
    provisioner = PROVISIONER_PATH.read_bytes()
    namespace = {
        "RAYLIGHT_COMMIT": "1" * 40,
        "ACTION": "status",
        "USE_FAKE_MODEL_STUBS": True,
    }

    with patch("urllib.request.urlopen") as urlopen:
        urlopen.return_value.read.return_value = provisioner
        with patch.object(hashlib, "sha256") as sha256:
            sha256.return_value.hexdigest.return_value = (
                "3c1f22716fda7183cac6a8182e98581b2e7e72a3b193b0d5e34aae76a7bcc48b"
            )
            # Stop before executing the real provisioner while retaining its source contract.
            urlopen.side_effect = RuntimeError("entry reached provisioner fetch")
            try:
                exec(  # noqa: S102
                    compile(ENTRY_PATH.read_text(), str(ENTRY_PATH), "exec"), namespace
                )
            except RuntimeError as exc:
                assert str(exc) == "entry reached provisioner fetch"

    required = {
        "APP_ROOT",
        "COMFY_COMMIT",
        "COMFY_DIR",
        "EXTRA_PIP_PACKAGES",
        "REQUIRED_MODELS",
        "RAYLIGHT_WORKFLOW_PATH",
        "OFFICIAL_T2V_WORKFLOW_URL",
    }
    assert required <= namespace.keys()
    assert namespace["ACTION"] == "status"
    assert namespace["USE_FAKE_MODEL_STUBS"] is True
    assert namespace["COMFY_EXTRA_ARGS"] == [
        "--listen",
        "0.0.0.0",
        "--cache-none",
        "--preview-method",
        "none",
    ]
    assert namespace["REQUIRED_MODELS"][0]["name"] == (
        "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    )
