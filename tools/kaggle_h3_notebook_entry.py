"""Compact, pinned entry point for the MiniMax-H3 Kaggle notebook."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import urllib.request


if "RAYLIGHT_COMMIT" not in globals():
    raise RuntimeError("Section 1 must define the immutable RAYLIGHT_COMMIT")
raylight_commit = globals()["RAYLIGHT_COMMIT"]
if len(raylight_commit) != 40 or any(c not in "0123456789abcdef" for c in raylight_commit):
    raise RuntimeError(f"RAYLIGHT_COMMIT must be a full lowercase Git SHA: {raylight_commit!r}")

globals().setdefault("ACTION", "start")
globals().setdefault("USE_FAKE_MODEL_STUBS", False)
globals().setdefault("VERIFY_MODEL_SHA256", False)
globals().setdefault("ENABLE_CLOUDFLARE", True)
INT8_ACCUMULATOR_MIB = globals().setdefault("INT8_ACCUMULATOR_MIB", 128)
H3_MEMORY_TRACE = globals().setdefault("H3_MEMORY_TRACE", False)
H3_STOP_AFTER_FIRST_FORWARD = globals().setdefault("H3_STOP_AFTER_FIRST_FORWARD", False)
H3_PHASE_PROFILE = globals().setdefault("H3_PHASE_PROFILE", False)
H3_INT8_PROBE_ROWS = globals().setdefault("H3_INT8_PROBE_ROWS", 128)
H3_INT8_PROBE_WARMUPS = globals().setdefault("H3_INT8_PROBE_WARMUPS", 1)
H3_INT8_PROBE_ITERATIONS = globals().setdefault("H3_INT8_PROBE_ITERATIONS", 3)
H3_INT8_PROBE_FULL_ROWS = globals().setdefault("H3_INT8_PROBE_FULL_ROWS", False)
H3_INT8_PROBE_ALLOW_CUDA_UNDER_13 = globals().setdefault(
    "H3_INT8_PROBE_ALLOW_CUDA_UNDER_13", False
)
H3_AUTO_QUEUE_DIAGNOSTIC = globals().setdefault("H3_AUTO_QUEUE_DIAGNOSTIC", False)
H3_INT8_BACKEND = str(globals().setdefault("H3_INT8_BACKEND", "eager")).strip().lower()
H3_INT8_CUDA_ALLOW_UNDER_13 = globals().setdefault(
    "H3_INT8_CUDA_ALLOW_UNDER_13", False
)
H3_INT8_CUDA_ALLOW_UNBOUNDED = globals().setdefault(
    "H3_INT8_CUDA_ALLOW_UNBOUNDED", False
)
H3_MEMORY_LIMIT_GB = globals().setdefault("H3_MEMORY_LIMIT_GB", None)
# FLOW-PRODUCED: cap-host-ram-on-kaggle
if not isinstance(H3_INT8_CUDA_ALLOW_UNDER_13, bool):
    raise TypeError("H3_INT8_CUDA_ALLOW_UNDER_13 must be bool")
if not isinstance(H3_INT8_CUDA_ALLOW_UNBOUNDED, bool):
    raise TypeError("H3_INT8_CUDA_ALLOW_UNBOUNDED must be bool")
if H3_INT8_BACKEND not in {"eager", "cuda"}:
    raise RuntimeError(f"H3_INT8_BACKEND must be eager or cuda, got {H3_INT8_BACKEND!r}")
if H3_AUTO_QUEUE_DIAGNOSTIC and not H3_STOP_AFTER_FIRST_FORWARD:
    raise RuntimeError("Auto-queued diagnostics require H3_STOP_AFTER_FIRST_FORWARD=True")
if (
    H3_INT8_BACKEND == "cuda"
    and not H3_STOP_AFTER_FIRST_FORWARD
    and not H3_INT8_CUDA_ALLOW_UNBOUNDED
):
    raise RuntimeError("CUDA INT8 backend is restricted to bounded diagnostics")
if H3_INT8_BACKEND == "cuda" and H3_STOP_AFTER_FIRST_FORWARD and not H3_PHASE_PROFILE:
    raise RuntimeError("CUDA INT8 backend requires H3_PHASE_PROFILE=True")
if H3_INT8_BACKEND == "cuda" and not H3_INT8_CUDA_ALLOW_UNDER_13:
    raise RuntimeError("CUDA INT8 on Kaggle CUDA 12.8 requires explicit under-13 override")
os.environ["RAYLIGHT_INT8_ACCUMULATOR_MIB"] = str(INT8_ACCUMULATOR_MIB)
os.environ["RAYLIGHT_H3_MEMORY_TRACE"] = "1" if H3_MEMORY_TRACE else "0"
os.environ["RAYLIGHT_H3_STOP_AFTER_FIRST_FORWARD"] = "1" if H3_STOP_AFTER_FIRST_FORWARD else "0"
os.environ["RAYLIGHT_H3_PHASE_PROFILE"] = "1" if H3_PHASE_PROFILE else "0"
os.environ["RAYLIGHT_INT8_BACKEND"] = H3_INT8_BACKEND
os.environ["RAYLIGHT_INT8_CUDA_ALLOW_UNDER_13"] = "1" if H3_INT8_CUDA_ALLOW_UNDER_13 else "0"
os.environ["RAYLIGHT_INT8_CUDA_ALLOW_UNBOUNDED"] = (
    "1" if H3_INT8_CUDA_ALLOW_UNBOUNDED else "0"
)
if H3_MEMORY_LIMIT_GB is not None:
    os.environ["RAYLIGHT_H3_MAX_HOST_RAM_GB"] = str(float(H3_MEMORY_LIMIT_GB))


def _default(name, value):
    globals().setdefault(name, value)

_default("ACTIVE_PROFILE", "minimax-h3-int8-convrot-raylight-dual-t4")
_default("WORK_DIR", "/kaggle/working")
work_dir = globals()["WORK_DIR"]
_default("APP_ROOT", f"{work_dir}/minimax-h3-raylight")
app_root = globals()["APP_ROOT"]
_default("VENV_DIR", f"{app_root}/venv")
_default("COMFY_DIR", f"{app_root}/ComfyUI")
comfy_dir = globals()["COMFY_DIR"]
_default("PYTHON_EXECUTABLE", sys.executable)
_default("RESET_INSTALL", False)
_default("FORCE_DEPENDENCY_INSTALL", False)
_default("UPDATE_REPOSITORIES", False)
_default("COMFY_REPO_URL", "https://github.com/Comfy-Org/ComfyUI.git")
_default("COMFY_COMMIT", "9a9fdb10ed144ce760d9682cb247526ea23cc525")
_default("INSTALL_RAYLIGHT", True)
_default("RAYLIGHT_REPO_URL", "https://github.com/StanLukuvka/raylight.git")
_default("RAYLIGHT_DIR", f"{comfy_dir}/custom_nodes/raylight")
_default("USE_SYSTEM_SITE_PACKAGES", True)
_default("DEPENDENCY_PROFILE", "compatible-v5-kernels-0.14.0")
_default("EXTRA_PIP_PACKAGES", [
    "transformers==5.0.0",
    "diffusers==0.37.1",
    "kernels==0.14.0",
    "xfuser==0.4.5",
    "yunchang==0.6.4",
])
_default("COMFY_INSTANCES", [{"name": "comfy0", "gpu": None, "port": 8188}])
_default("COMFY_EXTRA_ARGS", ["--listen", "0.0.0.0", "--cache-none", "--preview-method", "none"])
_default("STARTUP_TIMEOUT_SECONDS", 900)
_default("MODEL_ROOTS", ["/kaggle/input"])
_default("REQUIRED_MODELS", [
    {"name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "folder": "diffusion_models", "expected_bytes": 20_970_379_616, "min_bytes": 20_000_000_000, "sha256": "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"},
    {"name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "folder": "text_encoders", "expected_bytes": 15_687_142_551, "min_bytes": 15_000_000_000, "sha256": "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6"},
    {"name": "minimax_h3_video_vae_fp16.safetensors", "folder": "vae", "expected_bytes": 5_207_808_496, "min_bytes": 5_000_000_000, "sha256": "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522"},
    {"name": "minimax_h3_audio_vae_fp32.safetensors", "folder": "vae", "expected_bytes": 605_254_808, "min_bytes": 500_000_000, "sha256": "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48"},
])
workflow_templates_commit = "a832a091491ce5b6341f4e4ca548b7ab536b6acd"
_default("OFFICIAL_T2V_WORKFLOW_URL", f"https://raw.githubusercontent.com/Comfy-Org/workflow_templates/{workflow_templates_commit}/templates/video_minimax_h3_t2v.json")
_default("WORKFLOW_DIR", f"{comfy_dir}/user/default/workflows")
workflow_dir = globals()["WORKFLOW_DIR"]
_default("OFFICIAL_WORKFLOW_PATH", f"{workflow_dir}/minimax_h3_t2v_official.json")
_default("SMOKE_WORKFLOW_PATH", f"{workflow_dir}/minimax_h3_t2v_kaggle_smoke.json")
_default("RAYLIGHT_WORKFLOW_PATH", f"{workflow_dir}/LOAD_ONLY_THIS__minimax_h3_raylight_bounded_fsdp.json")
_default("INSTALL_COMFY_MANAGER", False)
_default("COMFY_MANAGER_REPO_URL", "https://github.com/ltdrdata/ComfyUI-Manager.git")
_default("MANAGER_SECURITY_LEVEL", "weak")
_default("ENABLE_FILEBROWSER", False)
_default("FILEBROWSER_VERSION", "2.27.0")
_default("FILEBROWSER_ARCHIVE_SHA256", "95d5add44820c6f6bf03a1aa063c618f8f242f53b892c1665dc066b4e037ba80")
_default("FILEBROWSER_BINARY_SHA256", "6fa66f50c8f58284755ccd319b3c1a700ba01bc815c2811bb414e9e07ca915a1")
_default("FILEBROWSER_INSTALL_DIR", "/kaggle")
_default("FILEBROWSER_ROOT", "/kaggle")
_default("FILEBROWSER_PORT", 8080)
_default("FILEBROWSER_NO_AUTH", True)
_default("CLOUDFLARE_SOURCE_DIR", "/kaggle/input/datasets/stanlukuvka/cloudflare-files")
_default("CLOUDFLARE_WORK_DIR", "/kaggle/working/cloudflare")
_default("CLOUDFLARE_CONFIG_FILE", "/kaggle/working/cloudflare/config.yml")
_default("CLOUDFLARED_BINARY", "/kaggle/working/cloudflared")
cloudflared_version = "2026.7.3"
_default("CLOUDFLARED_DOWNLOAD_URL", f"https://github.com/cloudflare/cloudflared/releases/download/{cloudflared_version}/cloudflared-linux-amd64")
_default("CLOUDFLARED_SHA256", "9d71c677db00134c1bd4144b7783486b654ad281b1ea62b4972098d19f770f17")
_default("CLOUDFLARE_EXTRA_ARGS", [])

provisioner_url = f"https://raw.githubusercontent.com/StanLukuvka/raylight/{raylight_commit}/tools/kaggle_h3_interactive.py"
provisioner_sha256 = "5d9545667301687f59f3d237a6b16348d0abe39ac662ffc6d7bebe13f828b670"
source = urllib.request.urlopen(provisioner_url, timeout=120).read()
actual = hashlib.sha256(source).hexdigest()
if actual != provisioner_sha256:
    raise RuntimeError(f"Provisioner checksum mismatch: {actual} != {provisioner_sha256}")
exec(compile(source, provisioner_url, "exec"), globals())  # noqa: S102
if H3_AUTO_QUEUE_DIAGNOSTIC:
    runner = os.path.join(globals()["RAYLIGHT_DIR"], "tools", "kaggle_h3_queue_bounded_diagnostic.py")
    globals()["H3_AUTO_QUEUE_DIAGNOSTIC"] = False
    try:
        subprocess.run(
            [
                sys.executable,
                runner,
            ],
            check=True,
        )
    finally:
        globals()["ACTION"] = "diagnostics"
        exec(compile(source, provisioner_url, "exec"), globals())  # noqa: S102
