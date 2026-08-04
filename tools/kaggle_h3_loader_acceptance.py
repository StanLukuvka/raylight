"""Kaggle dual-T4 acceptance test for bounded MiniMax-H3 FSDP loading.

This is intentionally an initialization-only test. It verifies the expensive
quantized checkpoint mapping and FSDP shard materialization without loading the
Qwen encoder or VAEs.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
COMFY = WORK / "ComfyUI"
VENV = WORK / "raylight-test-venv"
PYTHON = VENV / "bin" / "python"
RAYLIGHT = COMFY / "custom_nodes" / "raylight"
RESULT = WORK / "raylight_loader_result.json"
LOG = WORK / "raylight_loader_test.log"
COMFY_COMMIT = "9a9fdb10ed144ce760d9682cb247526ea23cc525"
RAYLIGHT_REF = os.environ.get("RAYLIGHT_TEST_REF", "fix/bounded-quant-fsdp-load")
CHECKPOINT = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"


def run(command: list[str | Path], *, cwd: Path | None = None) -> None:
    printable = " ".join(str(item) for item in command)
    print(f"+ {printable}", flush=True)
    subprocess.run([str(item) for item in command], cwd=cwd, check=True)


def checkout(url: str, destination: Path, ref: str) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--filter=blob:none", "--no-checkout", url, destination])
    run(["git", "-C", destination, "fetch", "--depth", "1", "origin", ref])
    run(["git", "-C", destination, "checkout", "--detach", "FETCH_HEAD"])


def find_checkpoint() -> Path:
    matches = sorted(INPUT.rglob(CHECKPOINT))
    if not matches:
        raise FileNotFoundError(f"{CHECKPOINT} was not found below {INPUT}")
    source = matches[0]
    expected = 20_970_379_616
    if source.stat().st_size != expected:
        raise RuntimeError(f"Unexpected checkpoint size: {source.stat().st_size} != {expected}")
    return source


def install() -> None:
    checkout("https://github.com/Comfy-Org/ComfyUI.git", COMFY, COMFY_COMMIT)
    checkout("https://github.com/StanLukuvka/raylight.git", RAYLIGHT, RAYLIGHT_REF)

    if VENV.exists():
        shutil.rmtree(VENV)
    run([sys.executable, "-m", "venv", "--system-site-packages", VENV])
    run([PYTHON, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([PYTHON, "-m", "pip", "install", "-r", COMFY / "requirements.txt"])
    run([PYTHON, "-m", "pip", "install", "-r", RAYLIGHT / "requirements.txt"])

    destination = COMFY / "models" / "diffusion_models" / CHECKPOINT
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    destination.symlink_to(find_checkpoint())


ACCEPTANCE_CODE = r'''
import json
import os
import sys
from pathlib import Path

COMFY = Path("/kaggle/working/ComfyUI")
os.chdir(COMFY)
sys.path.insert(0, str(COMFY))
sys.path.insert(0, str(COMFY / "custom_nodes" / "raylight" / "src"))
os.environ.setdefault("RAYLIGHT_RAY_TMPDIR", "/kaggle/working/raylight-ray")

import ray
from raylight.nodes import RayInitializerAdvanced, RayUNETLoader

checkpoint = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
init_payload = RayInitializerAdvanced().spawn_actor(
    ray_cluster_address="local",
    ray_cluster_namespace="minimax-h3-loader-test",
    GPU=2,
    GPU_SELECT="0,1",
    ulysses_degree=2,
    ring_degree=1,
    cfg_degree=1,
    dp_degree=1,
    sync_ulysses=False,
    clear_vram_after_sampling=False,
    FSDP=True,
    FSDP_CPU_OFFLOAD=False,
    XFuser_attention="TORCH_FLASH",
    skip_comm_test=False,
    use_mmap=True,
    ray_object_store_gb=0.5,
    ray_dashboard_address="None",
    torch_dist_address="127.0.0.1:29500",
)[0]
actors = RayUNETLoader().load_ray_unet(
    init_payload,
    checkpoint,
    "default",
    load_after=None,
)[0]
workers = actors["workers"]
after_mapping = ray.get([worker.get_memory_snapshot.remote() for worker in workers])
ray.get([worker._patch_fsdp_for_sampling.remote() for worker in workers])
after_materialization = ray.get([worker.get_memory_snapshot.remote() for worker in workers])
result = {
    "status": "PASS",
    "checkpoint": checkpoint,
    "after_mapping": after_mapping,
    "after_materialization": after_materialization,
}
Path("/kaggle/working/raylight_loader_result.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
ray.shutdown()
'''


def main() -> None:
    RESULT.unlink(missing_ok=True)
    LOG.unlink(missing_ok=True)
    install()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with LOG.open("w") as output:
        completed = subprocess.run(
            [str(PYTHON), "-u", "-c", ACCEPTANCE_CODE],
            cwd=COMFY,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=45 * 60,
        )
    print(LOG.read_text()[-30_000:])
    if completed.returncode != 0:
        raise RuntimeError(f"Acceptance process exited {completed.returncode}; see {LOG}")
    result = json.loads(RESULT.read_text())
    if result.get("status") != "PASS":
        raise RuntimeError(f"Unexpected result: {result}")
    print("[PASS] Two-rank quantized FSDP checkpoint mapping and shard materialization completed.")


if __name__ == "__main__":
    main()
