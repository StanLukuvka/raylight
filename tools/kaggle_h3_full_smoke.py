from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import kaggle_h3_loader_acceptance as loader

WORK = Path("/kaggle/working")
MODEL_ROOT = Path(os.environ.get("RAYLIGHT_MODEL_ROOT", "/kaggle/temp/raylight-models"))
SERVER_LOG = WORK / "raylight_h3_smoke_server.log"
RESULT = WORK / "raylight_h3_smoke_result.json"
FAILURE = WORK / "raylight_h3_smoke_failure.txt"
PORT = 8188
BASE_URL = f"http://127.0.0.1:{PORT}"

MODELS = {
    "diffusion_models": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "text_encoders": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "vae/video": "minimax_h3_video_vae_fp16.safetensors",
    "vae/audio": "minimax_h3_audio_vae_fp32.safetensors",
}


def request_json(path: str, payload: dict | None = None, timeout: float = 30) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def link_models() -> None:
    destinations = {
        "diffusion_models": loader.COMFY / "models" / "diffusion_models",
        "text_encoders": loader.COMFY / "models" / "text_encoders",
        "vae/video": loader.COMFY / "models" / "vae",
        "vae/audio": loader.COMFY / "models" / "vae",
    }
    for key, name in MODELS.items():
        source = MODEL_ROOT / key / name
        if not source.is_file():
            raise RuntimeError(f"Required model is missing: {source}")
        destination = destinations[key] / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.unlink(missing_ok=True)
        destination.symlink_to(source)


def prompt_graph() -> dict:
    prompt = "A red paper kite glides steadily through a clear blue sky. Gentle wind sound, no speech, no music."
    return {
        "init": {
            "class_type": "RayInitializer",
            "inputs": {
                "ray_cluster_address": "local",
                "ray_cluster_namespace": "minimax-h3-full-smoke",
                "GPU": 2,
                "ulysses_degree": 2,
                "ring_degree": 1,
                "cfg_degree": 1,
                "dp_degree": 1,
                "sync_ulysses": False,
                "clear_vram_after_sampling": True,
                "FSDP": True,
                "FSDP_CPU_OFFLOAD": False,
                "XFuser_attention": "TORCH_EFFICIENT",
                "skip_comm_test": False,
                "use_mmap": True,
                "ray_object_store_gb": 0.5,
                "load_after": ["conditioning", 0],
            },
        },
        "clip": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": MODELS["text_encoders"],
                "type": "minimax",
                "device": "default",
            },
        },
        "video_vae": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": MODELS["vae/video"]},
        },
        "audio_vae": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": MODELS["vae/audio"]},
        },
        "conditioning": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["clip", 0],
                "vae": ["video_vae", 0],
                "prompt": prompt,
                "width": 608,
                "height": 352,
                "length": 124,
            },
        },
        "unet": {
            "class_type": "RayUNETLoader",
            "inputs": {
                "ray_actors_init": ["init", 0],
                "unet_name": MODELS["diffusion_models"],
                "weight_dtype": "default",
                "load_after": ["conditioning", 0],
            },
        },
        "scheduler": {
            "class_type": "RayBasicScheduler",
            "inputs": {
                "ray_actors": ["unet", 0],
                "scheduler": "simple",
                "steps": 20,
                "denoise": 1.0,
            },
        },
        "guider": {
            "class_type": "RayBasicGuider",
            "inputs": {
                "ray_actors": ["unet", 0],
                "conditioning": ["conditioning", 0],
            },
        },
        "sampler": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "res_multistep"},
        },
        "sample": {
            "class_type": "XFuserSamplerCustomAdvanced",
            "inputs": {
                "add_noise": True,
                "noise_seed": 1,
                "guider": ["guider", 0],
                "sampler": ["sampler", 0],
                "sigmas": ["scheduler", 0],
                "latent_image": ["conditioning", 1],
            },
        },
        "decode_video": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["sample", 0], "vae": ["video_vae", 0]},
        },
        "decode_audio": {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["sample", 0], "vae": ["audio_vae", 0]},
        },
        "create_video": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["decode_video", 0],
                "audio": ["decode_audio", 0],
                "fps": 24.0,
            },
        },
        "save_video": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["create_video", 0],
                "filename_prefix": "raylight_h3_smoke",
                "format": "auto",
                "codec": "auto",
            },
        },
    }


def wait_for_server(process: subprocess.Popen, timeout: float = 900) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"ComfyUI exited {process.returncode}; see {SERVER_LOG}")
        try:
            info = request_json("/object_info", timeout=10)
            required = set(prompt_node["class_type"] for prompt_node in prompt_graph().values())
            missing = sorted(required.difference(info))
            if missing:
                raise RuntimeError(f"ComfyUI is missing required node classes: {missing}")
            return info
        except urllib.error.URLError:
            time.sleep(2)
    raise TimeoutError("Timed out waiting for ComfyUI")


def find_saved_video(history: dict) -> Path:
    outputs = history.get("outputs", {}).get("save_video", {})
    candidates = outputs.get("videos", []) + outputs.get("gifs", [])
    for item in candidates:
        filename = item.get("filename")
        if not filename:
            continue
        subfolder = item.get("subfolder", "")
        candidate = loader.COMFY / "output" / subfolder / filename
        if candidate.is_file():
            return candidate
    matches = sorted((loader.COMFY / "output").rglob("raylight_h3_smoke*"))
    if not matches:
        raise RuntimeError(f"SaveVideo returned no persisted video: {outputs}")
    return matches[-1]


def run_smoke() -> None:
    RESULT.unlink(missing_ok=True)
    FAILURE.unlink(missing_ok=True)
    checkpoint = loader.accelerator_preflight()
    loader.install(checkpoint)
    link_models()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("RAYLIGHT_RAY_TMPDIR", "/kaggle/temp/raylight-ray")
    command = [
        str(loader.PYTHON),
        "-u",
        str(loader.COMFY / "main.py"),
        "--listen",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--cache-none",
        "--preview-method",
        "none",
    ]
    with SERVER_LOG.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=loader.COMFY,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_for_server(process)
            queued = request_json("/prompt", {"prompt": prompt_graph(), "client_id": "raylight-h3-smoke"})
            prompt_id = queued["prompt_id"]
            deadline = time.monotonic() + 60 * 60
            history = None
            while time.monotonic() < deadline:
                payload = request_json(f"/history/{prompt_id}", timeout=15)
                if prompt_id in payload:
                    history = payload[prompt_id]
                    status = history.get("status", {})
                    if status.get("completed"):
                        break
                    messages = status.get("messages", [])
                    if any(message[0] == "execution_error" for message in messages):
                        raise RuntimeError(f"ComfyUI execution failed: {messages[-1]}")
                time.sleep(5)
            if history is None or not history.get("status", {}).get("completed"):
                raise TimeoutError(f"Smoke prompt did not complete: {prompt_id}")
            source_video = find_saved_video(history)
            destination = WORK / ("raylight_h3_smoke" + source_video.suffix)
            shutil.copy2(source_video, destination)
            result = {
                "status": "PASS",
                "prompt_id": prompt_id,
                "output": str(destination),
                "output_bytes": destination.stat().st_size,
                "frames": 5,
                "steps": 20,
                "resolution": [608, 352],
                "video_vae": MODELS["vae/video"],
                "audio_vae": MODELS["vae/audio"],
            }
            RESULT.write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2), flush=True)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)


if __name__ == "__main__":
    try:
        run_smoke()
    except BaseException:
        import traceback

        FAILURE.write_text(traceback.format_exc())
        if SERVER_LOG.exists():
            print(SERVER_LOG.read_text(errors="replace")[-50_000:], flush=True)
        raise
