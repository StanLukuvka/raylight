"""Pinned Kaggle interactive provisioner for MiniMax-H3 Raylight.

Executed by the companion notebook after its configuration cell. The script
intentionally reads that cell's globals and dispatches ACTION at the bottom.
"""

# ============================================================================
# PROVISIONING AND IMMUTABLE-CHECKOUT HELPERS
# ============================================================================
# ============================================================
# PROVISIONING AND IMMUTABLE-CHECKOUT HELPERS
# Safe to rerun. Existing valid symlinks and dependency installs are reused.
# ============================================================

import hashlib
import json
import os
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from copy import deepcopy
from pathlib import Path


def _require_config():

    required = [
        "ACTION", "ACTIVE_PROFILE", "WORK_DIR", "VENV_DIR", "COMFY_DIR",
        "COMFY_REPO_URL", "COMFY_INSTANCES", "REQUIRED_MODELS",
        "MODEL_ROOTS", "ENABLE_CLOUDFLARE",
    ]
    missing = [name for name in required if name not in globals()]
    if missing:
        raise RuntimeError("Run Section 1 first. Missing: " + ", ".join(missing))



def _run(cmd, *, cwd=None, check=True, env=None, quiet=False, capture=False):
    cmd = [str(part) for part in cmd]
    if not quiet:
        print("$", shlex.join(cmd))
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=check,
        env=env,
        text=True,
        capture_output=capture,
    )


def _tail(path, lines=100):
    path = Path(path)
    if not path.exists():
        return "(log file does not exist)"
    try:
        content = path.read_text(errors="replace").splitlines()
        return "\n".join(content[-lines:]) or "(log file is empty)"
    except Exception as exc:
        return f"(could not read log: {exc})"


def _sha256(path, chunk_size=32 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(path, expected, label):
    actual = _sha256(path)
    if actual.lower() != str(expected).lower():
        raise RuntimeError(f"{label} SHA256 mismatch: {actual} != {expected}")


def _download_small(url, destination, *, min_bytes=1, executable=False):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    partial.unlink(missing_ok=True)
    print(f"Downloading {destination.name}...")
    try:
        _run([
            "wget", "--show-progress", "--progress=bar:force:noscroll",
            "--retry-connrefused", "--waitretry=3", "--read-timeout=30",
            "--timeout=30", "--tries=5", "-O", partial, url,
        ], quiet=True)
    except FileNotFoundError:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=120) as response, open(partial, "wb") as out:
            shutil.copyfileobj(response, out)
    if not partial.exists() or partial.stat().st_size < min_bytes:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Download is missing or too small: {destination}")
    partial.replace(destination)
    if executable:
        destination.chmod(destination.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"[✓] {destination} ({destination.stat().st_size / 1e6:.1f} MB)")
    return destination


def _git_sync(url, destination, update=True):
    destination = Path(destination)
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--depth", "1", url, destination])
        return True
    if not (destination / ".git").exists():
        raise RuntimeError(
            f"{destination} exists but is not a git checkout. Set RESET_INSTALL=True."
        )
    if update:
        result = subprocess.run(
            ["git", "-C", str(destination), "pull", "--ff-only"], text=True
        )
        if result.returncode:
            print(f"Warning: could not update {destination}; using current checkout.")
    return False


def _pid_dir():
    path = Path(WORK_DIR) / ".minimax_h3_pids"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_pid(name, process, marker):
    (_pid_dir() / f"{name}.json").write_text(json.dumps({
        "name": name, "pid": process.pid, "marker": marker,
    }, indent=2))


def _cmdline_for_pid(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
    except Exception:
        return ""


def _terminate_pid(pid, marker=""):
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return
    if marker and marker not in _cmdline_for_pid(pid):
        print(f"Skipping stale PID {pid}; command no longer matches {marker!r}.")
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + 8
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.25)
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def stop_services():
    seen = set()
    for process in globals().get("SERVICE_PROCESSES", {}).values():
        pid = getattr(process, "pid", None)
        if pid and pid not in seen:
            seen.add(pid)
            _terminate_pid(pid)
    for pid_file in _pid_dir().glob("*.json"):
        try:
            payload = json.loads(pid_file.read_text())
            pid = int(payload["pid"])
            if pid not in seen:
                seen.add(pid)
                _terminate_pid(pid, str(payload.get("marker", "")))
        except Exception as exc:
            print(f"Warning: could not process {pid_file.name}: {exc}")
        finally:
            pid_file.unlink(missing_ok=True)
    globals()["SERVICE_PROCESSES"] = {}
    if seen:
        print("Stopped services from the previous run.")


def _ensure_venv():
    venv_dir = Path(VENV_DIR)
    venv_python = venv_dir / "bin" / "python"
    if RESET_INSTALL and venv_dir.exists():
        shutil.rmtree(venv_dir)
    if not venv_python.exists():
        if venv_dir.exists():
            shutil.rmtree(venv_dir)
        print(f"Creating virtual environment with {PYTHON_EXECUTABLE}...")
        _run([sys.executable, "-m", "pip", "install", "-q", "wrapt>=1.16", "virtualenv"])
        command = [sys.executable, "-m", "virtualenv"]
        if USE_SYSTEM_SITE_PACKAGES:
            command.append("--system-site-packages")
        _run([*command, "-p", PYTHON_EXECUTABLE, venv_dir])
    _run([venv_python, "-m", "pip", "install", "-q", "wrapt>=1.16"])
    return venv_python


def _install_dependencies(venv_python, new_checkout):
    marker = Path(VENV_DIR) / f".dependencies_{COMFY_COMMIT}_{RAYLIGHT_COMMIT}_{DEPENDENCY_PROFILE}"
    pip = [str(venv_python), "-m", "pip"]
    bootstrap = FORCE_DEPENDENCY_INSTALL or new_checkout or not marker.exists()
    if bootstrap:
        _run(pip + ["install", "--upgrade", "pip", "setuptools", "wheel"])
        _run(pip + ["install", "-r", Path(COMFY_DIR) / "requirements.txt"])
    if bootstrap and EXTRA_PIP_PACKAGES:
        _run(pip + ["install", "--upgrade", *EXTRA_PIP_PACKAGES])
    check = subprocess.run(pip + ["check"], capture_output=True, text=True)
    if check.returncode:
        print("pip check notes:\n" + (check.stdout + check.stderr).strip())
    verify = (
        "import torch; "
        "print('PyTorch:', torch.__version__); "
        "print('CUDA runtime:', torch.version.cuda); "
        "print('CUDA available:', torch.cuda.is_available()); "
        "print('Visible GPUs:', torch.cuda.device_count()); "
        "assert torch.cuda.is_available(), 'PyTorch cannot see CUDA'; "
        "assert torch.cuda.device_count() == 2, 'Expected exactly two visible GPUs'"
    )
    _run([venv_python, "-c", verify])
    return marker


def _checkout_pinned_repo(url, destination, commit, clean_paths=()):
    destination = Path(destination)
    if destination.exists() and not (destination / ".git").exists():
        app_root = Path(APP_ROOT).resolve()
        resolved = destination.resolve()
        if destination.is_symlink() or not resolved.is_relative_to(app_root):
            raise RuntimeError(
                f"Refusing to replace non-checkout outside the notebook root: {destination}"
            )
        print(f"Removing incomplete checkout from an earlier failed run: {destination}")
        shutil.rmtree(destination)
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--filter=blob:none", "--no-checkout", url, destination])
    if not (destination / ".git").exists():
        raise RuntimeError(f"{destination} exists but is not a git checkout")
    current = _run(
        ["git", "-C", destination, "rev-parse", "HEAD"],
        capture=True,
        quiet=True,
        check=False,
    )
    if current.returncode or current.stdout.strip() != commit:
        _run(["git", "-C", destination, "fetch", "--depth", "1", "origin", commit])
    _run(["git", "-C", destination, "checkout", "--detach", "--force", commit])
    _run(["git", "-C", destination, "reset", "--hard", commit])
    for clean_path in clean_paths:
        _run(["git", "-C", destination, "clean", "-ffd", "--", clean_path])
    actual = _run(
        ["git", "-C", destination, "rev-parse", "HEAD"],
        capture=True,
        quiet=True,
    ).stdout.strip()
    if actual != commit:
        raise RuntimeError(f"Pinned checkout mismatch: expected {commit}, got {actual}")
    return destination


def _install_custom_nodes(venv_python):
    custom_nodes = Path(COMFY_DIR) / "custom_nodes"
    custom_nodes.mkdir(parents=True, exist_ok=True)

    if INSTALL_COMFY_MANAGER:
        manager_dir = custom_nodes / "ComfyUI-Manager"
        _git_sync(COMFY_MANAGER_REPO_URL, manager_dir, UPDATE_REPOSITORIES)
        (manager_dir / "config.ini").write_text(
            "[default]\n" f"security_level = {MANAGER_SECURITY_LEVEL}\n"
        )

    if not INSTALL_RAYLIGHT:
        return False

    raylight_dir = Path(RAYLIGHT_DIR)
    marker = Path(VENV_DIR) / f".raylight_{RAYLIGHT_COMMIT}"
    try:
        _checkout_pinned_repo(
            RAYLIGHT_REPO_URL,
            raylight_dir,
            RAYLIGHT_COMMIT,
            clean_paths=(".",),
        )
        if not marker.exists():
            _run([
                venv_python, "-m", "pip", "install", "-r",
                raylight_dir / "requirements.txt",
            ])

        # Verify on every run so a stale marker cannot hide a broken environment.
        # This does not initialize CUDA or Ray workers.
        expected_versions = {
            package.split("==", 1)[0]: package.split("==", 1)[1]
            for package in EXTRA_PIP_PACKAGES
            if "==" in package
        }
        verify_code = (
            "import importlib.metadata as m; "
            f"expected={expected_versions!r}; "
            "actual={name:m.version(name) for name in expected}; "
            "assert actual == expected, (actual, expected); "
            "import ray, xfuser, yunchang; "
            "print('Raylight dependencies:', 'ray=' + ray.__version__, "
            "*(name + '=' + actual[name] for name in sorted(actual)))"
        )
        _run([venv_python, "-c", verify_code])
        final_check = subprocess.run(
            [venv_python, "-m", "pip", "check"], capture_output=True, text=True
        )
        if final_check.returncode:
            print(
                "Final pip check notes from Kaggle's shared base environment:\n"
                + (final_check.stdout + final_check.stderr).strip()
            )
        marker.write_text(RAYLIGHT_COMMIT)
        print(f"[✓] Raylight pinned at {RAYLIGHT_COMMIT}")
        return True
    except Exception as exc:
        print(f"[!] Required Raylight installation failed: {exc}")
        raise

# ============================================================================
# MODEL DISCOVERY AND BOUNDED WORKFLOW GENERATION
# ============================================================================
def _index_required_files(roots, names):
    matches = {name: [] for name in names}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        print(f"Scanning model root: {root}")
        for dirpath, _, filenames in os.walk(root):
            for name in names.intersection(filenames):
                matches[name].append(Path(dirpath) / name)
    return matches


def _candidate_score(path, spec):
    text = str(path).lower()
    expected_tail = str(Path(spec["folder"]) / spec["name"]).lower()
    return (0 if text.endswith(expected_tail) else 1, 0 if "minimax" in text else 1, len(path.parts), text)


def _prepare_fake_model_stubs():
    """Create sparse development placeholders without consuming model-sized storage."""
    root = Path(APP_ROOT) / "FAKE_MODEL_STUBS_DO_NOT_USE_FOR_INFERENCE"
    for spec in REQUIRED_MODELS:
        path = root / spec["folder"] / spec["name"]
        path.parent.mkdir(parents=True, exist_ok=True)
        expected = int(spec["expected_bytes"])
        if not path.exists() or path.stat().st_size != expected:
            with path.open("wb") as handle:
                handle.truncate(expected)
    (root / "README.txt").write_text(
        "Sparse fake model files for provisioning/UI iteration only.\n"
        "They contain no model data and must never be queued for inference.\n"
    )
    print("\n" + "!" * 78)
    print("FAKE MODEL MODE: services and workflow UI only; DO NOT QUEUE THE WORKFLOW.")
    print("The placeholders use sparse logical sizes and contain no model weights.")
    print("!" * 78 + "\n")
    return root


def _discover_kaggle_models():
    fake_mode = bool(globals().get("USE_FAKE_MODEL_STUBS", False))
    roots = (
        [_prepare_fake_model_stubs()]
        if fake_mode
        else [Path(root) for root in MODEL_ROOTS if Path(root).exists()]
    )
    if not roots:
        raise RuntimeError(
            "No Kaggle Input mount exists. Attach stanlukuvka/minimax-h3-comfyui-weights and rerun."
        )
    names = {spec["name"] for spec in REQUIRED_MODELS}
    matches = _index_required_files(roots, names)
    missing = [name for name, found in matches.items() if not found]
    if missing:
        raise FileNotFoundError(
            "Missing required H3 files:\n  - " + "\n  - ".join(missing)
            + "\nAttach stanlukuvka/minimax-h3-comfyui-weights and rerun."
        )
    selected = {}
    for spec in REQUIRED_MODELS:
        name = spec["name"]
        candidates = sorted(matches[name], key=lambda item: _candidate_score(item, spec))
        source = candidates[0]
        size = source.stat().st_size
        if size != int(spec["expected_bytes"]):
            raise RuntimeError(f"Unexpected size for {source}: {size:,} != {spec['expected_bytes']:,}")
        if VERIFY_MODEL_SHA256 and not fake_mode:
            actual = _sha256(source)
            if actual.lower() != spec["sha256"].lower():
                raise RuntimeError(f"SHA256 mismatch for {source}: {actual} != {spec['sha256']}")
        selected[name] = {
            "spec": spec, "source": source, "size": size, "fake": fake_mode
        }
        print(f"[preflight ✓] {name}: {size / 1e9:.2f} GB at {source}")
    return selected


def _link_kaggle_models(selected):
    model_root = Path(COMFY_DIR) / "models"
    linked = {}
    for name, item in selected.items():
        spec, source, size = item["spec"], item["source"], item["size"]
        destination = model_root / spec["folder"] / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            if destination.resolve() != source.resolve():
                destination.unlink()
        elif destination.exists():
            if not destination.is_file():
                raise RuntimeError(f"Model destination is not a regular file: {destination}")
            destination.unlink()
        if not destination.exists() and not destination.is_symlink():
            destination.symlink_to(source)
        linked[name] = {"source": source, "link": destination, "size": size}
        print(f"[link ✓] {destination} -> {source}")
    return linked


def _make_raylight_workflow(stock_workflow):
    """Convert the official H3 subgraph without changing H3 packing or decoding."""
    data = deepcopy(stock_workflow)
    subgraphs = data.get("definitions", {}).get("subgraphs", [])
    h3 = next((item for item in subgraphs if "MiniMax H3" in item.get("name", "")), None)
    if h3 is None:
        raise RuntimeError("Official workflow no longer contains the MiniMax H3 subgraph")

    nodes = h3["nodes"]
    by_type = {node.get("type"): node for node in nodes}
    required = {
        "UNETLoader", "BasicScheduler", "BasicGuider", "SamplerCustomAdvanced",
        "RandomNoise",
    }
    missing = sorted(required.difference(by_type))
    if missing:
        raise RuntimeError(f"Official H3 graph changed; missing nodes: {missing}")

    loader = by_type["UNETLoader"]
    scheduler = by_type["BasicScheduler"]
    guider = by_type["BasicGuider"]
    sampler = by_type["SamplerCustomAdvanced"]
    random_noise = by_type["RandomNoise"]

    next_node_id = max(node["id"] for node in nodes) + 1
    next_link_id = max(link["id"] for link in h3["links"]) + 1
    init_link_id = next_link_id
    load_after_link_id = next_link_id + 1
    init_after_link_id = next_link_id + 2
    conditioning_input = next(
        item for item in guider.get("inputs", []) if item.get("name") == "conditioning"
    )
    conditioning_source = next(
        link for link in h3["links"] if link["id"] == conditioning_input["link"]
    )
    conditioning_node = next(
        node for node in nodes if node.get("id") == conditioning_source["origin_id"]
    )
    if conditioning_node.get("type") != "MiniMaxH3ImageToVideo":
        raise RuntimeError(
            "Official workflow conditioning source changed; refusing to guess the H3 length widget"
        )
    conditioning_values = conditioning_node.get("widgets_values", [])
    if len(conditioning_values) < 4:
        raise RuntimeError("Official MiniMaxH3ImageToVideo widgets are incomplete")
    # Prompt is overridden for the bounded diagnostic; geometry stays as-is in the stock workflow.
    conditioning_values[0] = (
        "A single red ball rolls from left to right across a plain studio floor. "
        "Locked camera, one continuous shot, no text or cuts. Simple natural rolling sound."
    )

    initializer = {
        "id": next_node_id,
        "type": "RayInitializer",
        "pos": [-2390, 4440],
        "size": [500, 380],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [{
            "name": "load_after", "shape": 7, "type": "CONDITIONING",
            "link": init_after_link_id,
        }],
        "outputs": [{
            "name": "ray_actors_init", "type": "RAY_ACTORS_INIT",
            "links": [init_link_id],
        }],
        "properties": {"Node name for S&R": "RayInitializer"},
        # Exact accepted simple initializer: local, namespace, workers, Ulysses,
        # ring, CFG, DP, sync, clear, FSDP, CPU offload, attention, comm test,
        # mmap, Ray object store GiB, dashboard, torch distributed address.
        "widgets_values": [
            "local", "minimax-h3-raylight", 2, 2, 1, 1, 1, False,
            True, True, False, "TORCH_EFFICIENT", True, True,
            0.5, "None", "127.0.0.1:29500",
        ],
    }
    nodes.append(initializer)

    unet_external = next(
        (item for item in loader.get("inputs", []) if item.get("name") == "unet_name"),
        None,
    )
    loader["type"] = "RayUNETLoader"
    loader["inputs"] = [
        *([unet_external] if unet_external else []),
        {"name": "ray_actors_init", "type": "RAY_ACTORS_INIT", "link": init_link_id},
        {"name": "lora", "shape": 7, "type": "RAY_LORA", "link": None},
        {
            "name": "load_after", "shape": 7, "type": "CONDITIONING",
            "link": load_after_link_id,
        },
    ]
    loader["outputs"] = [{
        "name": "ray_actors", "type": "RAY_ACTORS", "links": [5, 193],
    }]
    loader["properties"] = {"Node name for S&R": "RayUNETLoader"}
    # Preserve checkpoint selector and explicit dtype=default so INT8 ConvRot metadata wins.
    loader["widgets_values"] = [loader.get("widgets_values", [
        "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    ])[0], "default"]

    scheduler["type"] = "RayBasicScheduler"
    scheduler["inputs"] = [{"name": "ray_actors", "type": "RAY_ACTORS", "link": 5}]
    scheduler["properties"] = {"Node name for S&R": "RayBasicScheduler"}
    scheduler["widgets_values"] = ["simple", 20, 1.0]

    guider["type"] = "RayBasicGuider"
    guider["inputs"] = [
        {"name": "ray_actors", "type": "RAY_ACTORS", "link": 193},
        {"name": "conditioning", "type": "CONDITIONING", "link": 187},
    ]
    guider["outputs"] = [{"name": "guider", "type": "RAY_GUIDER", "links": [12]}]
    guider["properties"] = {"Node name for S&R": "RayBasicGuider"}
    guider["widgets_values"] = []

    # Match the sampler that completed the accepted dual-T4 smoke run. Move the
    # subgraph's exposed seed link directly onto XFuserSamplerCustomAdvanced.
    sampler["type"] = "XFuserSamplerCustomAdvanced"
    sampler["inputs"] = [
        {"name": "add_noise", "type": "BOOLEAN", "widget": {"name": "add_noise"}, "link": None},
        {"name": "noise_seed", "type": "INT", "widget": {"name": "noise_seed"}, "link": 207},
        {"name": "guider", "type": "RAY_GUIDER", "link": 12},
        {"name": "sampler", "type": "SAMPLER", "link": 16},
        {"name": "sigmas", "type": "SIGMAS", "link": 18},
        {"name": "latent_image", "type": "LATENT", "link": 188},
        {"name": "shutdown_after_sampling", "type": "BOOLEAN", "widget": {"name": "shutdown_after_sampling"}, "link": None},
    ]
    sampler["outputs"] = [
        {"name": "output", "type": "LATENT", "links": [225, 226]},
        {"name": "denoised_output", "type": "LATENT", "links": None},
        {"name": "ray_actors", "type": "RAY_ACTORS", "links": None},
    ]
    sampler["properties"] = {"Node name for S&R": "XFuserSamplerCustomAdvanced"}
    sampler["widgets_values"] = [True, 1, "randomize", True]
    nodes.remove(random_noise)

    new_links = []
    for link in h3["links"]:
        link_id = link["id"]
        if link_id == 40:
            continue
        if link_id in (5, 193):
            link["type"] = "RAY_ACTORS"
        elif link_id == 207:
            link["target_id"] = sampler["id"]
            link["target_slot"] = 1
        elif link_id == 12:
            link["type"] = "RAY_GUIDER"
            link["target_slot"] = 2
        elif link_id == 16:
            link["target_slot"] = 3
        elif link_id == 18:
            link["target_slot"] = 4
        elif link_id == 188:
            link["target_slot"] = 5
        new_links.append(link)
    new_links.append({
        "id": init_link_id,
        "origin_id": initializer["id"], "origin_slot": 0,
        "target_id": loader["id"],
        "target_slot": 1 if unet_external else 0,
        "type": "RAY_ACTORS_INIT",
    })
    load_after_slot = next(
        index for index, item in enumerate(loader["inputs"])
        if item.get("name") == "load_after"
    )
    new_links.append({
        "id": load_after_link_id,
        "origin_id": conditioning_source["origin_id"],
        "origin_slot": conditioning_source["origin_slot"],
        "target_id": loader["id"],
        "target_slot": load_after_slot,
        "type": "CONDITIONING",
    })
    new_links.append({
        "id": init_after_link_id,
        "origin_id": conditioning_source["origin_id"],
        "origin_slot": conditioning_source["origin_slot"],
        "target_id": initializer["id"],
        "target_slot": 0,
        "type": "CONDITIONING",
    })
    source_node = next(node for node in nodes if node["id"] == conditioning_source["origin_id"])
    source_links = source_node["outputs"][conditioning_source["origin_slot"]].setdefault("links", [])
    source_links.extend([load_after_link_id, init_after_link_id])
    h3["links"] = new_links
    h3["state"]["lastNodeId"] = max(h3["state"].get("lastNodeId", 0), next_node_id)
    h3["state"]["lastLinkId"] = max(h3["state"].get("lastLinkId", 0), init_after_link_id)

    data.setdefault("extra", {})["raylight"] = {
        "commit": RAYLIGHT_COMMIT,
        "mode": "FSDP+USP",
        "gpus": 2,
        "ulysses_degree": 2,
        "fsdp_cpu_offload": False,
        "attention": "TORCH_EFFICIENT",
        "fake_model_mode": bool(globals().get("USE_FAKE_MODEL_STUBS", False)),
        "int8_accumulator_mib": int(globals().get("INT8_ACCUMULATOR_MIB", 128)),
        "memory_trace": bool(globals().get("H3_MEMORY_TRACE", False)),
        "phase_profile": bool(globals().get("H3_PHASE_PROFILE", False)),
    }
    return data


def _install_workflows():
    workflow_dir = Path(WORKFLOW_DIR)
    workflow_dir.mkdir(parents=True, exist_ok=True)
    source = Path(WORK_DIR) / "minimax_h3_official_template_source.json"
    _download_small(OFFICIAL_T2V_WORKFLOW_URL, source, min_bytes=10_000)

    data = json.loads(source.read_text())
    for node in data.get("nodes", []):
        if node.get("type") == "ResolutionSelector":
            values = node.get("widgets_values", [])
            if len(values) >= 2:
                values[1] = 0.2  # roughly 608x352 at 16:9

    unsafe_or_stale = (
        Path(OFFICIAL_WORKFLOW_PATH),
        Path(SMOKE_WORKFLOW_PATH),
        workflow_dir / "minimax_h3_t2v_raylight_fsdp_usp_smoke.json",
    )
    for old in unsafe_or_stale:
        old.unlink(missing_ok=True)

    raylight = Path(RAYLIGHT_WORKFLOW_PATH)
    graph = _make_raylight_workflow(data)
    raylight.write_text(json.dumps(graph, indent=2))
    print(f"[✓] Only bounded H3 workflow installed: {raylight}")
    print("[safety] Stock and stale H3 workflows were removed because they can OOM this runtime.")

# ============================================================================
# OPTIONAL SERVICES, TELEMETRY, AND RUNTIME HELPERS
# ============================================================================
def _install_filebrowser():
    install_dir = Path(FILEBROWSER_INSTALL_DIR)
    install_dir.mkdir(parents=True, exist_ok=True)
    binary = install_dir / "filebrowser"
    if not binary.exists():
        archive = Path(WORK_DIR) / f"linux-amd64-filebrowser-v{FILEBROWSER_VERSION}.tar.gz"
        url = (
            "https://github.com/filebrowser/filebrowser/releases/download/"
            f"v{FILEBROWSER_VERSION}/linux-amd64-filebrowser.tar.gz"
        )
        _download_small(url, archive, min_bytes=1_000_000)
        _require_sha256(archive, FILEBROWSER_ARCHIVE_SHA256, "FileBrowser archive")
        with tarfile.open(archive, "r:gz") as tar:
            member = next((m for m in tar.getmembers() if Path(m.name).name == "filebrowser"), None)
            if member is None:
                raise RuntimeError("FileBrowser archive did not contain the binary")
            extracted = tar.extractfile(member)
            if extracted is None:
                raise RuntimeError("Could not read FileBrowser binary from archive")
            with open(binary, "wb") as out:
                shutil.copyfileobj(extracted, out)
        archive.unlink(missing_ok=True)
    _require_sha256(binary, FILEBROWSER_BINARY_SHA256, "FileBrowser binary")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    database = Path(WORK_DIR) / "filebrowser.db"
    if not database.exists():
        _run([binary, "config", "init", "-d", database])
        if FILEBROWSER_NO_AUTH:
            _run([binary, "config", "set", "-d", database, "--auth.method=noauth"])
    return binary, database


def _is_linux_elf(path, min_bytes=1_000_000):
    path = Path(path)
    if not path.is_file() or path.stat().st_size < min_bytes:
        return False
    try:
        with open(path, "rb") as handle:
            return handle.read(4) == b"\x7fELF"
    except OSError:
        return False


def _find_cloudflare_files():
    source_dir = Path(CLOUDFLARE_SOURCE_DIR)
    work_dir = Path(CLOUDFLARE_WORK_DIR)
    config = Path(CLOUDFLARE_CONFIG_FILE)
    binary = Path(CLOUDFLARED_BINARY)
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Cloudflare dataset not found: {source_dir}. Attach it or set "
            "ENABLE_CLOUDFLARE=False."
        )
    if work_dir.exists():
        shutil.rmtree(work_dir)
    shutil.copytree(source_dir, work_dir, symlinks=False)
    if not config.exists():
        candidates = sorted(work_dir.rglob("config.yml")) + sorted(work_dir.rglob("config.yaml"))
        if not candidates:
            raise FileNotFoundError(f"No Cloudflare config found under {work_dir}")
        config.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[0], config)

    if not _is_linux_elf(binary):
        binary.unlink(missing_ok=True)
        preferred = [
            work_dir / "cloudflared",
            work_dir / "cloudflared-linux-amd64",
            source_dir / "cloudflared",
            source_dir / "cloudflared-linux-amd64",
        ]
        discovered = []
        for root in (work_dir, source_dir):
            discovered.extend(sorted(root.rglob("cloudflared*")))
        for candidate in [*preferred, *discovered]:
            if _is_linux_elf(candidate):
                shutil.copy2(candidate, binary)
                break
    if not _is_linux_elf(binary):
        _download_small(
            CLOUDFLARED_DOWNLOAD_URL,
            binary,
            min_bytes=5_000_000,
            executable=True,
        )
    _require_sha256(binary, CLOUDFLARED_SHA256, "cloudflared binary")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    for sensitive in (config, config.parent / "tunnel.json"):
        if sensitive.exists():
            sensitive.chmod(0o600)
    version = _run([binary, "--version"], capture=True)
    print(f"[✓] {version.stdout.strip() or version.stderr.strip()}")
    return binary, config


def _accelerator_preflight():
    result = _run(["nvidia-smi", "-L"], capture=True, quiet=True)
    gpus = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    print("Accelerator preflight:")
    for line in gpus:
        print(f"  {line}")
    if len(gpus) != 2 or any("Tesla T4" not in line for line in gpus):
        raise RuntimeError(f"This profile requires exactly two NVIDIA Tesla T4 GPUs; found: {gpus}")
    return gpus


def _gpu_count():
    try:
        return len(_accelerator_preflight())
    except Exception:
        return None


def _start_service(name, cmd, log_path, *, marker, env=None, cwd=None):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Starting {name}: {shlex.join([str(x) for x in cmd])}")
    with open(log_path, "w", buffering=1) as log:
        process = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
    time.sleep(1.5)
    if process.poll() is not None:
        raise RuntimeError(
            f"{name} exited immediately with code {process.returncode}.\n"
            f"--- {log_path} ---\n{_tail(log_path)}"
        )
    SERVICE_PROCESSES[name] = process
    _write_pid(name, process, marker)
    print(f"[✓] {name} started (PID {process.pid})")
    return process


def _install_host_ram_cap(venv_python: "Path | str") -> None:
    # FLOW-PRODUCED: cap-host-ram-on-kaggle
    """Install a sitecustomize.py into the venv that patches psutil before
    ComfyUI imports model_management.

    ComfyUI calls ``psutil.virtual_memory()`` at import time to compute
    ``MAX_PINNED_MEMORY``.  On Kaggle, psutil reports the host's physical RAM
    (~32 GB) instead of the cgroup limit (~30 GB).  This causes ComfyUI to
    over-allocate pinned memory and the cgroup OOM-killer brings down the
    whole notebook.

    The fix: write a ``sitecustomize.py`` into the venv's site-packages
    directory.  Python automatically imports this module at interpreter
    startup, *before* any user code (including ComfyUI's main.py).
    """
    import site

    site_packages = site.getsitepackages()
    if not site_packages:
        print("[!] Could not find site-packages directory; skipping host RAM cap")
        return

    target_dir = Path(site_packages[0])
    target = target_dir / "sitecustomize.py"

    raylight_dir = globals().get("RAYLIGHT_DIR", "")
    cgroup_module = f"{raylight_dir}/src" if raylight_dir else ""

    source = '''\
"""Auto-generated by Raylight provisioner — patches psutil before ComfyUI."""
import os, sys

_raylight_src = os.environ.get("RAYLIGHT_HOST_CAP_SRC", "")
if not _raylight_src:
    # Try to find it relative to the custom_nodes dir
    for candidate in (
        os.path.join(os.path.dirname(__file__), "..", "..", "custom_nodes", "raylight", "src"),
        os.path.join(os.path.dirname(__file__), "raylight", "src"),
    ):
        if os.path.isdir(candidate):
            _raylight_src = os.path.abspath(candidate)
            break

if _raylight_src and _raylight_src not in sys.path:
    sys.path.insert(0, _raylight_src)

try:
    from raylight.cgroup_ram import apply_host_ram_cap
    _cap = apply_host_ram_cap()
    if _cap is not None:
        print(f"[raylight] Host RAM capped to {_cap / (1024**3):.1f} GB", flush=True)
    else:
        print("[raylight] No cgroup RAM cap needed", flush=True)
except Exception as exc:
    print(f"[raylight] cgroup_ram patch failed: {exc}", flush=True)
'''
    target.write_text(source)
    print(f"[✓] sitecustomize.py installed → {target}")

    # Also set env var so the subprocess can find the raylight src
    os.environ.setdefault("RAYLIGHT_HOST_CAP_SRC", str(Path(raylight_dir) / "src") if raylight_dir else "")


def _start_memory_monitor():
    script = Path(WORK_DIR) / "memory_monitor.py"
    snapshots = Path(WORK_DIR) / "memory_snapshots.jsonl"
    events_copy = Path(WORK_DIR) / "memory_events_last.txt"
    monitor_source = r'''import json, os, subprocess, time
from pathlib import Path

OUT = Path(os.environ["H3_MEMORY_SNAPSHOTS"])
EVENTS_COPY = Path(os.environ["H3_MEMORY_EVENTS_COPY"])
CGROUP = Path("/sys/fs/cgroup")

def read(path):
    try:
        return Path(path).read_text(errors="replace").strip()
    except Exception:
        return ""

def integer(path):
    value = read(path)
    try:
        return int(value)
    except Exception:
        return value

def process_rss():
    rows = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            status = (proc / "status").read_text(errors="replace").splitlines()
            rss_line = next((x for x in status if x.startswith("VmRSS:")), "VmRSS: 0 kB")
            rss_kib = int(rss_line.split()[1])
            cmd = (proc / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace").strip()
            rows.append({"pid": int(proc.name), "rss_kib": rss_kib, "cmd": cmd[:500]})
        except Exception:
            pass
    return sorted(rows, key=lambda x: x["rss_kib"], reverse=True)[:15]

def gpu_processes():
    try:
        out = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=1)
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception as exc:
        return [f"ERROR: {exc}"]

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("a", buffering=1) as stream:
    while True:
        events = read(CGROUP / "memory.events")
        EVENTS_COPY.write_text(events + "\n")
        record = {
            "unix": time.time(),
            "memory_current": integer(CGROUP / "memory.current"),
            "memory_peak": integer(CGROUP / "memory.peak"),
            "memory_max": integer(CGROUP / "memory.max"),
            "memory_events": events,
            "processes": process_rss(),
            "gpu_processes": gpu_processes(),
        }
        stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        time.sleep(1)
'''
    script.write_text(monitor_source)
    snapshots.unlink(missing_ok=True)
    events_copy.unlink(missing_ok=True)
    env = os.environ.copy()
    env["H3_MEMORY_SNAPSHOTS"] = str(snapshots)
    env["H3_MEMORY_EVENTS_COPY"] = str(events_copy)
    process = _start_service(
        "memory-monitor",
        [sys.executable, "-u", script],
        Path(WORK_DIR) / "memory_monitor.log",
        marker="memory_monitor.py",
        env=env,
        cwd=WORK_DIR,
    )
    print(f"[✓] Memory telemetry: {snapshots}")
    return process


def print_memory_diagnostics(lines=12):
    snapshots = Path(WORK_DIR) / "memory_snapshots.jsonl"
    events = Path(WORK_DIR) / "memory_events_last.txt"
    comfy_log = Path(WORK_DIR) / "comfy0.log"
    print(f"===== {events} =====")
    print(events.read_text(errors="replace") if events.exists() else "MISSING")
    print(f"===== last {lines} memory snapshots: {snapshots} =====")
    if snapshots.exists():
        print("\n".join(snapshots.read_text(errors="replace").splitlines()[-lines:]))
    else:
        print("MISSING")
    print(f"===== last 50000 characters: {comfy_log} =====")
    if comfy_log.exists():
        print(comfy_log.read_text(errors="replace")[-50000:])
    else:
        print("MISSING")


def _phase_timing_summary(comfy_log):
    prefix = b"[raylight-load-phase] "
    max_input_bytes = 64 * 1024 * 1024
    max_line_bytes = 64 * 1024
    max_events = 10_000
    max_phases = 5_000
    max_field_chars = 512
    phases = []
    event_count = 0
    invalid_event_count = 0
    oversized_line_count = 0
    field_truncated_count = 0
    input_bytes = 0
    first_event_unix = None
    last_event_unix = None
    event_limit_reached = False
    phase_limit_reached = False
    input_truncated = False
    summary_error_type = None

    try:
        if comfy_log.is_file():
            with comfy_log.open("rb") as stream:
                while input_bytes < max_input_bytes:
                    remaining = max_input_bytes - input_bytes
                    line = stream.readline(min(max_line_bytes + 1, remaining))
                    if not line:
                        break
                    input_bytes += len(line)
                    if len(line) > max_line_bytes or not line.endswith(b"\n"):
                        oversized_line_count += 1
                        while line and not line.endswith(b"\n") and input_bytes < max_input_bytes:
                            remaining = max_input_bytes - input_bytes
                            line = stream.readline(min(max_line_bytes + 1, remaining))
                            input_bytes += len(line)
                        continue
                    if prefix not in line:
                        continue
                    if event_count >= max_events:
                        event_limit_reached = True
                        input_truncated = True
                        break
                    try:
                        payload = json.loads(
                            line.split(prefix, 1)[1].strip().decode("utf-8", errors="replace")
                        )
                    except (TypeError, ValueError, UnicodeError):
                        invalid_event_count += 1
                        continue
                    if not isinstance(payload, dict):
                        invalid_event_count += 1
                        continue

                    event_count += 1
                    event_unix = payload.get("unix")
                    if isinstance(event_unix, (int, float)):
                        first_event_unix = (
                            event_unix
                            if first_event_unix is None
                            else min(first_event_unix, event_unix)
                        )
                        last_event_unix = (
                            event_unix
                            if last_event_unix is None
                            else max(last_event_unix, event_unix)
                        )
                    if "elapsed_seconds" not in payload:
                        continue
                    if len(phases) >= max_phases:
                        phase_limit_reached = True
                        continue

                    phase = {}
                    for key in (
                        "event",
                        "phase_id",
                        "rank",
                        "process_role",
                        "status",
                        "elapsed_seconds",
                        "error_type",
                        "unix",
                    ):
                        value = payload.get(key)
                        if isinstance(value, str):
                            if len(value) > max_field_chars:
                                value = value[:max_field_chars]
                                field_truncated_count += 1
                            phase[key] = value
                        elif value is None or isinstance(value, (bool, int, float)):
                            if key in payload:
                                phase[key] = value
                    phases.append(phase)

                if input_bytes >= max_input_bytes:
                    input_truncated = True
    except BaseException as exc:
        summary_error_type = type(exc).__name__

    return {
        "schema_version": 1,
        "source": comfy_log.name,
        "event_count": event_count,
        "invalid_event_count": invalid_event_count,
        "oversized_line_count": oversized_line_count,
        "field_truncated_count": field_truncated_count,
        "input_bytes": input_bytes,
        "input_byte_limit": max_input_bytes,
        "event_limit_reached": event_limit_reached,
        "phase_limit_reached": phase_limit_reached,
        "input_truncated": input_truncated,
        "summary_error_type": summary_error_type,
        "first_event_unix": first_event_unix,
        "last_event_unix": last_event_unix,
        "phases": phases,
    }


def _write_run_manifest(venv_python, gpu_identities):
    probe_source = """
import json
import platform
import torch
import ray
from importlib.metadata import PackageNotFoundError, version

try:
    comfy_kitchen_version = version("comfy-kitchen")
except PackageNotFoundError:
    comfy_kitchen_version = None
print("RAYLIGHT_RUNTIME_JSON=" + json.dumps({
    "python": platform.python_version(),
    "pytorch": torch.__version__,
    "cuda": torch.version.cuda,
    "ray": ray.__version__,
    "comfy_kitchen": comfy_kitchen_version,
    "cuda_available": torch.cuda.is_available(),
    "visible_gpu_count": torch.cuda.device_count(),
    "visible_gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}, sort_keys=True))
"""
    observed = subprocess.run(
        [str(venv_python), "-c", probe_source],
        check=True,
        capture_output=True,
        text=True,
    )
    marker = "RAYLIGHT_RUNTIME_JSON="
    runtime_line = next(
        line for line in observed.stdout.splitlines() if line.startswith(marker)
    )
    runtime = json.loads(runtime_line[len(marker):])
    if runtime.get("cuda_available") is not True or runtime.get("visible_gpu_count") != 2:
        raise RuntimeError(f"Runtime observation did not confirm two CUDA GPUs: {runtime}")

    manifest = {
        "schema_version": 1,
        "run_started_unix": time.time(),
        "video_included": False,
        "active_profile": str(globals()["ACTIVE_PROFILE"]),
        "raylight_commit": str(globals()["RAYLIGHT_COMMIT"]),
        "comfy_commit": str(globals()["COMFY_COMMIT"]),
        "hardware_target": "2x Tesla T4",
        "gpu_count": 2,
        "gpu_identities": list(gpu_identities),
        "runtime": runtime,
        "evaluations": 20,
        "int8_accumulator_mib": int(globals().get("INT8_ACCUMULATOR_MIB", 128)),
        "distributed": {
            "fsdp": True,
            "ulysses_degree": 2,
            "sequential_rank_materialization": True,
            "shutdown_after_sampling": True,
            "cooperative_shutdown_timeout_seconds": 30.0,
            "force_shutdown_timeout_seconds": 10.0,
        },
        "telemetry_bounds": {
            "input_mib": 64,
            "line_kib": 64,
            "events": 10_000,
            "phases": 5_000,
            "field_chars": 512,
        },
        "archive_policy": "diagnostics and timing only; generated media excluded",
    }
    manifest_path = Path(str(globals()["WORK_DIR"])) / "h3-run-manifest.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(manifest_path)
    return manifest_path


def export_memory_diagnostics():
    output = Path("/kaggle/working/h3-diagnostics.zip")
    work_dir = Path(str(globals()["WORK_DIR"]))
    comfy_log = work_dir / "comfy0.log"
    manifest_path = work_dir / "h3-run-manifest.json"
    phase_summary_path = work_dir / "raylight-phase-summary.json"
    try:
        phase_summary_path.unlink(missing_ok=True)
    except BaseException:
        pass
    try:
        phase_summary = _phase_timing_summary(comfy_log)
        phase_summary_path.write_text(
            json.dumps(phase_summary, indent=2, sort_keys=True) + "\n"
        )
    except BaseException:
        pass
    candidates = [
        manifest_path,
        phase_summary_path,
        Path(APP_ROOT) / "bounded-diagnostic-result.json",
        work_dir / "memory_snapshots.jsonl",
        work_dir / "memory_events_last.txt",
        work_dir / "memory_monitor.log",
        comfy_log,
        work_dir / "cloudflared.log",
    ]
    output.unlink(missing_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        added = 0
        archive_errors = []
        for item in candidates:
            try:
                if not item.is_file():
                    continue
                archive.write(item, arcname=item.name)
                added += 1
            except BaseException as exc:
                if len(archive_errors) < 100:
                    archive_errors.append({
                        "candidate": str(getattr(item, "name", "unknown"))[:512],
                        "error_type": type(exc).__name__[:512],
                    })
        if archive_errors:
            try:
                archive.writestr(
                    "archive-errors.json",
                    json.dumps(
                        {
                            "schema_version": 1,
                            "errors": archive_errors,
                            "truncated": len(archive_errors) >= 100,
                        },
                        indent=2,
                        sort_keys=True,
                    ) + "\n",
                )
            except BaseException:
                pass
    if not added:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            "No diagnostics were found. Run ACTION='start' and reproduce the failure "
            "before using ACTION='diagnostics'."
        )
    print(f"Diagnostics archive: {output}")
    return output


def _wait_for_http(name, port, process, log_path):
    url = f"http://127.0.0.1:{port}/"
    deadline = time.time() + STARTUP_TIMEOUT_SECONDS
    next_status = 0
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"{name} exited with code {process.returncode}.\n"
                f"--- {log_path} ---\n{_tail(log_path)}"
            )
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "health-check"})
            with urllib.request.urlopen(request, timeout=3) as response:
                if response.status < 500:
                    print(f"[✓] {name} is ready at {url}")
                    return
        except Exception:
            pass
        elapsed = int(STARTUP_TIMEOUT_SECONDS - max(0, deadline - time.time()))
        if elapsed >= next_status:
            print(f"Waiting for {name}... {elapsed}s")
            next_status = elapsed + 15
        time.sleep(2)
    raise TimeoutError(
        f"{name} did not become ready within {STARTUP_TIMEOUT_SECONDS}s.\n"
        f"--- {log_path} ---\n{_tail(log_path)}"
    )


def _print_storage():
    print("\nStorage mounts:")
    paths = [path for path in ("/kaggle/working", "/kaggle/input", "/kaggle/temp") if Path(path).exists()]
    subprocess.run(["df", "-h", *paths], text=True)

# ============================================================================
# ACTION DISPATCHER
# ============================================================================
def main():
    _require_config()
    print("=" * 78)
    print(f"ACTIVE PROFILE: {ACTIVE_PROFILE}")
    print("H3 weights: attached through /kaggle/input; no model downloads required.")
    print("=" * 78)
    Path(WORK_DIR).mkdir(parents=True, exist_ok=True)
    stop_services()
    action = str(ACTION).strip().lower()
    if action == "stop":
        print("All tracked services are stopped.")
        return
    if action == "diagnostics":
        print_memory_diagnostics(lines=20)
        export_memory_diagnostics()
        return
    if action not in {"start", "restart", "int8-probe"}:
        raise ValueError(
            'ACTION must be "start", "restart", "stop", "diagnostics", or "int8-probe"'
        )

    if action in {"start", "restart"}:
        (Path(str(globals()["WORK_DIR"])) / "h3-run-manifest.json").unlink(missing_ok=True)

    global SERVICE_PROCESSES
    SERVICE_PROCESSES = {}
    try:
        _print_storage()
        gpus = _accelerator_preflight()
        if action == "int8-probe":
            comfy_dir = Path(COMFY_DIR)
            if RESET_INSTALL and Path(APP_ROOT).exists():
                shutil.rmtree(APP_ROOT)
            existed_before = comfy_dir.exists()
            _checkout_pinned_repo(
                COMFY_REPO_URL,
                comfy_dir,
                COMFY_COMMIT,
                clean_paths=("custom_nodes",),
            )
            venv_python = _ensure_venv()
            dependency_marker = _install_dependencies(venv_python, not existed_before)
            _install_custom_nodes(venv_python)
            dependency_marker.write_text(str(time.time()))
            probe_command = [
                str(venv_python),
                str(Path(RAYLIGHT_DIR) / "tools" / "kaggle_h3_int8_backend_probe.py"),
                "--rows", str(int(globals().get("H3_INT8_PROBE_ROWS", 128))),
                "--warmups", str(int(globals().get("H3_INT8_PROBE_WARMUPS", 1))),
                "--iterations", str(int(globals().get("H3_INT8_PROBE_ITERATIONS", 3))),
                "--output-dir", str(Path(WORK_DIR) / "h3-int8-backend-probe"),
            ]
            if bool(globals().get("H3_INT8_PROBE_FULL_ROWS", False)):
                probe_command.append("--full-rows")
            if bool(globals().get("H3_INT8_PROBE_ALLOW_CUDA_UNDER_13", False)):
                probe_command.append("--allow-cuda-under-13")
            _run(probe_command, cwd=RAYLIGHT_DIR)
            print(f"INT8 probe report: {Path(WORK_DIR) / 'h3-int8-backend-probe' / 'comparison.json'}")
            return
        selected_models = _discover_kaggle_models()
        comfy_dir = Path(COMFY_DIR)
        if RESET_INSTALL and Path(APP_ROOT).exists():
            shutil.rmtree(APP_ROOT)
        existed_before = comfy_dir.exists()
        _checkout_pinned_repo(
            COMFY_REPO_URL,
            comfy_dir,
            COMFY_COMMIT,
            clean_paths=("custom_nodes",),
        )
        new_checkout = not existed_before
        linked = _link_kaggle_models(selected_models)
        venv_python = _ensure_venv()
        dependency_marker = _install_dependencies(venv_python, new_checkout)
        raylight_ready = _install_custom_nodes(venv_python)
        dependency_marker.write_text(str(time.time()))
        _install_workflows()
        _write_run_manifest(venv_python, gpus)

        filebrowser = database = None
        if ENABLE_FILEBROWSER:
            filebrowser, database = _install_filebrowser()

        cloudflared = cloudflare_config = None
        if ENABLE_CLOUDFLARE:
            cloudflared, cloudflare_config = _find_cloudflare_files()

        gpu_count = len(gpus)
        numeric_gpus = [
            instance.get("gpu") for instance in COMFY_INSTANCES
            if isinstance(instance.get("gpu"), int)
        ]
        if gpu_count is not None and numeric_gpus and max(numeric_gpus) >= gpu_count:
            raise RuntimeError(
                f"Configured GPU {max(numeric_gpus)}, but nvidia-smi reports {gpu_count} GPU(s)."
            )

        _start_memory_monitor()

        _install_host_ram_cap(venv_python)

        comfy_processes = []
        for index, instance in enumerate(COMFY_INSTANCES):
            name = str(instance.get("name") or f"comfy{index}")
            port = int(instance["port"])
            gpu = instance.get("gpu")
            command = [
                venv_python,
                Path(COMFY_DIR) / "main.py",
                "--port", str(port),
                *[str(arg) for arg in COMFY_EXTRA_ARGS],
            ]
            env = os.environ.copy()
            env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
            env["RAY_DEDUP_LOGS"] = "0"
            # Keep Ray's session files on expendable Kaggle storage, not the model mount.
            ray_tmp_root = Path("/kaggle/temp") if Path("/kaggle/temp").exists() else Path(WORK_DIR)
            env["RAYLIGHT_RAY_TMPDIR"] = str(ray_tmp_root / "raylight-ray")
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log_path = Path(WORK_DIR) / f"{name}.log"
            process = _start_service(
                name,
                command,
                log_path,
                marker="ComfyUI/main.py",
                env=env,
                cwd=COMFY_DIR,
            )
            comfy_processes.append((name, port, process, log_path))

        if ENABLE_FILEBROWSER:
            _start_service(
                "filebrowser",
                [
                    filebrowser,
                    "-d", database,
                    "-r", FILEBROWSER_ROOT,
                    "-p", str(FILEBROWSER_PORT),
                ],
                Path(WORK_DIR) / "filebrowser.log",
                marker="filebrowser",
                cwd=WORK_DIR,
            )

        for name, port, process, log_path in comfy_processes:
            _wait_for_http(name, port, process, log_path)

        if ENABLE_CLOUDFLARE:
            tunnel_log = Path(WORK_DIR) / "cloudflared.log"
            tunnel = _start_service(
                "cloudflared",
                [
                    cloudflared,
                    "tunnel",
                    "--config", cloudflare_config,
                    *[str(arg) for arg in CLOUDFLARE_EXTRA_ARGS],
                    "run",
                ],
                tunnel_log,
                marker="cloudflared",
                cwd=Path(cloudflare_config).parent,
            )
            time.sleep(2)
            if tunnel.poll() is not None:
                raise RuntimeError(
                    f"Cloudflare tunnel exited with code {tunnel.returncode}.\n"
                    f"--- {tunnel_log} ---\n{_tail(tunnel_log)}"
                )

        print("\n=== READY ===")
        print(f"Profile: {ACTIVE_PROFILE}")
        print("Linked H3 Dataset files:")
        for name, item in linked.items():
            print(f"  {name}: {item['source']}")
        for name, port, _, log_path in comfy_processes:
            print(f"{name}: http://127.0.0.1:{port}    log: {log_path}")
        if ENABLE_FILEBROWSER:
            print(f"FileBrowser: http://127.0.0.1:{FILEBROWSER_PORT}")
        print(f"SAFE Raylight workflow: {RAYLIGHT_WORKFLOW_PATH}")
        print(f"Memory snapshots:      {Path(WORK_DIR) / 'memory_snapshots.jsonl'}")
        print(f"OOM event counters:    {Path(WORK_DIR) / 'memory_events_last.txt'}")
        print("After a failure, rerun the helper cells and call print_memory_diagnostics().")
        print(f"Raylight available:   {raylight_ready}")
        if ENABLE_CLOUDFLARE:
            print(f"Cloudflare config: {cloudflare_config}")
            print(f"Cloudflare log: {Path(WORK_DIR) / 'cloudflared.log'}")
        print("\nLoad ONLY LOAD_ONLY_THIS__minimax_h3_raylight_bounded_fsdp.json.")
        print("Do not reuse a workflow restored by the browser from an earlier session.")
        print("The Raylight graph keeps 0.2 MP, five frames, 20 steps, simple schedule,")
        print("res_multistep, packed H3 audio/video conditioning, and ordinary VAE decoding.")
        print('After an OOM/kernel restart: set ACTION = "diagnostics" and rerun all cells.')
        print('To stop normally: set ACTION = "stop" and rerun all cells.')

    except Exception:
        print("\nSetup failed. Stopping services started by this run...")
        stop_services()
        raise


main()
