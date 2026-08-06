#!/usr/bin/env python3
"""Queue the exact bounded H3 graph against an already-running ComfyUI.

Success is the configured, intentional stop after one warmup and one measured
denoiser forward. This script never provisions ComfyUI and never starts a
full smoke run.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "http://127.0.0.1:8188"
EXPECTED_STOP = "MiniMax-H3 diagnostic intentionally stopped after the first denoiser forward"
RESULT = Path("/kaggle/working/minimax-h3-raylight/bounded-diagnostic-result.json")

MODELS = {
    "diffusion": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "text": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "video_vae": "minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
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


def prompt_graph(width: int, height: int, length: int) -> dict:
    prompt = "A red paper kite glides steadily through a clear blue sky. Gentle wind sound, no speech, no music."
    return {
        "init": {
            "class_type": "RayInitializer",
            "inputs": {
                "ray_cluster_address": "local",
                "ray_cluster_namespace": "minimax-h3-bounded-diagnostic",
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
                "ray_dashboard_address": "None",
                "torch_dist_address": "127.0.0.1:29500",
                "load_after": ["conditioning", 0],
            },
        },
        "clip": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": MODELS["text"],
                "type": "minimax",
                "device": "default",
            },
        },
        "video_vae": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": MODELS["video_vae"]},
        },
        "audio_vae": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": MODELS["audio_vae"]},
        },
        "conditioning": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["clip", 0],
                "vae": ["video_vae", 0],
                "prompt": prompt,
                "width": width,
                "height": height,
                "length": length,
            },
        },
        "unet": {
            "class_type": "RayUNETLoader",
            "inputs": {
                "ray_actors_init": ["init", 0],
                "unet_name": MODELS["diffusion"],
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
                "filename_prefix": "raylight_h3_bounded_diagnostic",
                "format": "auto",
                "codec": "auto",
            },
        },
    }


def wait_for_server(timeout: float = 900) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            info = request_json("/object_info", timeout=10)
            required = {node["class_type"] for node in prompt_graph(736, 416, 124).values()}
            missing = sorted(required.difference(info))
            if missing:
                raise RuntimeError(f"ComfyUI is missing required node classes: {missing}")
            return
        except urllib.error.URLError:
            time.sleep(2)
    raise TimeoutError("Timed out waiting for ComfyUI")


def run(width: int, height: int, length: int) -> dict:
    RESULT.unlink(missing_ok=True)
    wait_for_server()
    queued = request_json(
        "/prompt",
        {"prompt": prompt_graph(width, height, length), "client_id": "raylight-h3-bounded-diagnostic"},
    )
    prompt_id = queued["prompt_id"]
    deadline = time.monotonic() + 60 * 20
    while time.monotonic() < deadline:
        payload = request_json(f"/history/{prompt_id}", timeout=15)
        if prompt_id not in payload:
            time.sleep(5)
            continue
        history = payload[prompt_id]
        status = history.get("status", {})
        messages = status.get("messages", [])
        for message in messages:
            if message[0] != "execution_error":
                continue
            detail = json.dumps(message[1], sort_keys=True)
            if EXPECTED_STOP not in detail:
                raise RuntimeError(f"Unexpected ComfyUI execution error: {detail}")
            result = {
                "status": "intentional_stop_observed",
                "prompt_id": prompt_id,
                "width": width,
                "height": height,
                "length": length,
                "error": EXPECTED_STOP,
            }
            RESULT.parent.mkdir(parents=True, exist_ok=True)
            RESULT.write_text(json.dumps(result, indent=2, sort_keys=True))
            return result
        if status.get("completed"):
            result = {
                "status": "completed_without_intentional_stop",
                "prompt_id": prompt_id,
                "width": width,
                "height": height,
                "length": length,
            }
            RESULT.parent.mkdir(parents=True, exist_ok=True)
            RESULT.write_text(json.dumps(result, indent=2, sort_keys=True))
            raise RuntimeError(f"Diagnostic unexpectedly completed: {result}")
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for bounded diagnostic prompt {prompt_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=736)
    parser.add_argument("--height", type=int, default=416)
    parser.add_argument("--length", type=int, default=124)
    args = parser.parse_args()
    print(json.dumps(run(args.width, args.height, args.length), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
