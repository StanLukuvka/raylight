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
import threading
import time
import traceback
import urllib.request
from pathlib import Path


WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
ROOT = Path("/kaggle/temp/raylight-loader-acceptance")
COMFY = ROOT / "ComfyUI"
VENV = ROOT / "venv"
PYTHON = VENV / "bin" / "python"
RAYLIGHT = COMFY / "custom_nodes" / "raylight"
RESULT = WORK / "raylight_loader_result.json"
FAILURE = WORK / "raylight_loader_failure.txt"
LOG = WORK / "raylight_loader_test.log"
COMFY_COMMIT = "9a9fdb10ed144ce760d9682cb247526ea23cc525"
RAYLIGHT_REF = os.environ.get(
    "RAYLIGHT_TEST_REF", "473715b4aec7ff3a554c11556fa1c81e92ba075e"
)
CHECKPOINT = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"


def run(command: list[str | Path], *, cwd: Path | None = None) -> None:
    printable = " ".join(str(item) for item in command)
    print(f"+ {printable}", flush=True)
    subprocess.run([str(item) for item in command], cwd=cwd, check=True)


def prepare_owned_root() -> None:
    if ROOT.is_symlink():
        raise RuntimeError(f"Refusing symlinked acceptance root: {ROOT}")
    ROOT.mkdir(parents=True, exist_ok=True)
    if not ROOT.is_dir():
        raise RuntimeError(f"Acceptance root is not a directory: {ROOT}")
    for child in (COMFY, VENV):
        if child.is_symlink():
            raise RuntimeError(f"Refusing symlinked acceptance path: {child}")


def checkout(url: str, destination: Path, ref: str) -> None:
    destination.resolve().relative_to(ROOT.resolve())
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--filter=blob:none", "--no-checkout", url, destination])
    run(["git", "-C", destination, "fetch", "--depth", "1", "origin", ref])
    run(["git", "-C", destination, "checkout", "--detach", "FETCH_HEAD"])
    resolved = subprocess.check_output(
        ["git", "-C", destination, "rev-parse", "HEAD"], text=True
    ).strip()
    if resolved != ref:
        raise RuntimeError(f"Checkout mismatch for {url}: {resolved} != {ref}")


def find_checkpoint() -> Path:
    roots = [INPUT, Path("/kaggle/temp")]
    matches = sorted(path for root in roots if root.exists() for path in root.rglob(CHECKPOINT))
    if not matches:
        download_url = os.environ.get("RAYLIGHT_CHECKPOINT_URL")
        if not download_url:
            raise FileNotFoundError(
                f"{CHECKPOINT} is not mounted below /kaggle/input or /kaggle/temp; attach the Dataset "
                "or set RAYLIGHT_CHECKPOINT_URL to a temporary authenticated file URL"
            )
        destination = Path("/kaggle/temp") / CHECKPOINT
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        print(f"Downloading {CHECKPOINT} from RAYLIGHT_CHECKPOINT_URL", flush=True)
        with urllib.request.urlopen(download_url, timeout=120) as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
        partial.replace(destination)
        matches = [destination]
    source = matches[0]
    expected = 20_970_379_616
    if source.stat().st_size != expected:
        raise RuntimeError(f"Unexpected checkpoint size: {source.stat().st_size} != {expected}")
    return source


def accelerator_preflight() -> Path:
    result = subprocess.run(["nvidia-smi", "-L"], check=True, text=True, capture_output=True)
    gpus = [line for line in result.stdout.splitlines() if line.strip()]
    checkpoint = find_checkpoint()
    preflight = {
        "gpu_count": len(gpus),
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
    }
    (WORK / "raylight_loader_preflight.json").write_text(json.dumps(preflight, indent=2))
    if len(gpus) != 2:
        raise RuntimeError(f"Acceptance test requires exactly two GPUs; found {len(gpus)}")
    return checkpoint


def install(checkpoint: Path) -> None:
    prepare_owned_root()
    checkout("https://github.com/Comfy-Org/ComfyUI.git", COMFY, COMFY_COMMIT)
    checkout("https://github.com/StanLukuvka/raylight.git", RAYLIGHT, RAYLIGHT_REF)

    if VENV.exists():
        shutil.rmtree(VENV)
    run([sys.executable, "-m", "pip", "install", "-q", "virtualenv", "wrapt>=1.16"])
    run(
        [
            sys.executable,
            "-m",
            "virtualenv",
            "--system-site-packages",
            "-p",
            sys.executable,
            VENV,
        ]
    )
    run([PYTHON, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([PYTHON, "-m", "pip", "install", "-r", COMFY / "requirements.txt"])
    run([PYTHON, "-m", "pip", "install", "-r", RAYLIGHT / "requirements.txt"])

    destination = COMFY / "models" / "diffusion_models" / CHECKPOINT
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    destination.symlink_to(checkpoint)


ACCEPTANCE_CODE = r'''
import json
import os
import sys
from pathlib import Path

COMFY = Path("/kaggle/temp/raylight-loader-acceptance/ComfyUI")
os.chdir(COMFY)
sys.path.insert(0, str(COMFY))
sys.path.insert(0, str(COMFY / "custom_nodes" / "raylight" / "src"))
os.environ.setdefault("RAYLIGHT_RAY_TMPDIR", "/kaggle/temp/raylight-ray")

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
remaining_meta = {
    snapshot["rank"]: snapshot.get("meta_parameter_count")
    for snapshot in after_materialization
}
if any(count != 0 for count in remaining_meta.values()):
    raise RuntimeError(f"FSDP materialization left meta parameters: {remaining_meta}")
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


def monitor_cgroup_memory(stop: threading.Event) -> None:
    candidates = [
        Path("/sys/fs/cgroup/memory.current"),
        Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ]
    usage_path = next((path for path in candidates if path.exists()), None)
    if usage_path is None:
        return
    output_path = WORK / "raylight_cgroup_memory.csv"
    with output_path.open("w") as output:
        output.write("monotonic_seconds,memory_bytes\n")
        output.flush()
        while not stop.wait(1):
            try:
                usage = int(usage_path.read_text().strip())
            except (OSError, ValueError):
                continue
            output.write(f"{time.monotonic():.3f},{usage}\n")
            output.flush()


def main() -> None:
    RESULT.unlink(missing_ok=True)
    LOG.unlink(missing_ok=True)
    checkpoint = accelerator_preflight()
    install(checkpoint)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    monitor_stop = threading.Event()
    monitor = threading.Thread(target=monitor_cgroup_memory, args=(monitor_stop,), daemon=True)
    monitor.start()
    try:
        with LOG.open("w") as output:
            completed = subprocess.run(
                [str(PYTHON), "-u", "-c", ACCEPTANCE_CODE],
                cwd=COMFY,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=45 * 60,
            )
    finally:
        monitor_stop.set()
        monitor.join(timeout=5)
    print(LOG.read_text()[-30_000:])
    if completed.returncode != 0:
        raise RuntimeError(f"Acceptance process exited {completed.returncode}; see {LOG}")
    result = json.loads(RESULT.read_text())
    if result.get("status") != "PASS":
        raise RuntimeError(f"Unexpected result: {result}")
    print("[PASS] Two-rank quantized FSDP checkpoint mapping and shard materialization completed.")


if __name__ == "__main__":
    FAILURE.unlink(missing_ok=True)
    try:
        main()
    except BaseException:
        FAILURE.write_text(traceback.format_exc())
        raise
