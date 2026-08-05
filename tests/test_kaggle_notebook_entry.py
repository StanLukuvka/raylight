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
                "618aec351e7ba697856e29ac16186f94cb9ab786ae2d9619fa730e465d176b34"
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
    assert namespace["H3_WIDTH"] == 608
    assert namespace["H3_HEIGHT"] == 352
    assert namespace["H3_LENGTH"] == 124
    assert namespace["INT8_ACCUMULATOR_MIB"] == 128
    assert namespace["H3_MEMORY_TRACE"] is False
    assert namespace["H3_STOP_AFTER_FIRST_FORWARD"] is False
    assert namespace["H3_PHASE_PROFILE"] is False
    assert namespace["H3_INT8_PROBE_ROWS"] == 128
    assert namespace["H3_INT8_PROBE_WARMUPS"] == 1
    assert namespace["H3_INT8_PROBE_ITERATIONS"] == 3
    assert namespace["H3_INT8_PROBE_FULL_ROWS"] is False
    assert namespace["H3_INT8_PROBE_ALLOW_CUDA_UNDER_13"] is False
    assert namespace["H3_AUTO_QUEUE_DIAGNOSTIC"] is False
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


def test_notebook_entry_exports_diagnostic_memory_controls():
    source = ENTRY_PATH.read_text()
    assert 'os.environ["RAYLIGHT_INT8_ACCUMULATOR_MIB"] = str(INT8_ACCUMULATOR_MIB)' in source
    assert 'os.environ["RAYLIGHT_H3_MEMORY_TRACE"] = "1" if H3_MEMORY_TRACE else "0"' in source
    assert (
        'os.environ["RAYLIGHT_H3_STOP_AFTER_FIRST_FORWARD"] = '
        '"1" if H3_STOP_AFTER_FIRST_FORWARD else "0"'
    ) in source
    assert (
        'os.environ["RAYLIGHT_H3_PHASE_PROFILE"] = '
        '"1" if H3_PHASE_PROFILE else "0"'
    ) in source


def test_notebook_entry_can_queue_and_package_only_the_explicit_bounded_diagnostic():
    source = ENTRY_PATH.read_text()
    assert 'globals().setdefault("H3_AUTO_QUEUE_DIAGNOSTIC", False)' in source
    assert '"kaggle_h3_queue_bounded_diagnostic.py"' in source
    assert 'globals()["ACTION"] = "diagnostics"' in source
    assert 'globals()["H3_AUTO_QUEUE_DIAGNOSTIC"] = False' in source
    assert source.count('exec(compile(source, provisioner_url, "exec"), globals())') == 2
