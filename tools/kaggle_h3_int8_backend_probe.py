#!/usr/bin/env python3
"""Isolated eager-versus-CUDA packed-INT8 capability probe for MiniMax-H3 on T4.

The CUDA leg fails closed below CUDA 13 unless the caller explicitly accepts the
ABI/extension risk. Run reduced rows first; ``--full-rows`` uses the observed
736x416 rank-local M=6103 shapes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile


H3_LINEAR_SHAPES = (
    ("qkv", 6103, 5376, 21504),
    ("attention_out", 6103, 7168, 5376),
    ("fc1", 6103, 5376, 28672),
    ("fc2", 6103, 14336, 5376),
)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _cuda_major(torch) -> int:
    cuda = torch.version.cuda
    if not cuda:
        return 0
    return int(str(cuda).split(".", 1)[0])


def _run_backend(args) -> dict:
    import numpy as np
    import torch

    backend = args.backend
    if backend == "cuda" and _cuda_major(torch) < 13 and not args.allow_cuda_under_13:
        raise RuntimeError(
            "CUDA backend probe below CUDA 13 requires --allow-cuda-under-13; "
            "use only for a disposable source-build/capability experiment"
        )

    # comfy_kitchen imports its CUDA extension eagerly. Keep this import after
    # the explicit ABI-risk guard so the default CUDA-12 path truly fails closed.
    import comfy_kitchen as ck

    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from raylight.comfy_dist.kitchen_patches.int8 import install_int8_patches

    if not torch.cuda.is_available():
        raise RuntimeError("The INT8 backend probe requires a CUDA GPU")

    install_int8_patches()
    if backend == "cuda":
        ck.enable_backend("cuda")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "backend": backend,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "backends": _jsonable(ck.list_backends()),
        "rows": 6103 if args.full_rows else args.rows,
        "warmups": args.warmups,
        "iterations": args.iterations,
        "shapes": [],
    }

    for shape_index, (name, full_m, k, n) in enumerate(H3_LINEAR_SHAPES):
        m = full_m if args.full_rows else min(args.rows, full_m)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed + shape_index)
        x = torch.randn((m, k), generator=generator, dtype=torch.float32).to(device=device, dtype=torch.bfloat16)
        weight = torch.randint(-127, 128, (n, k), generator=generator, dtype=torch.int8).to(device)
        weight_scale = (torch.rand((n,), generator=generator, dtype=torch.float32) * 0.01 + 0.001).to(device)
        bias = torch.randn((n,), generator=generator, dtype=torch.float32).to(device=device, dtype=torch.bfloat16)

        call = {
            "x": x,
            "weight": weight,
            "weight_scale": weight_scale,
            "bias": bias,
            "out_dtype": torch.bfloat16,
            "convrot": True,
            "convrot_groupsize": 256,
        }
        output = None
        with ck.use_backend(backend):
            for _ in range(args.warmups):
                output = ck.int8_linear(**call)
            torch.cuda.synchronize()
            if output is not None:
                del output
                output = None
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            free_before, total = torch.cuda.mem_get_info(device)
            event_pairs = []
            with torch.inference_mode():
                for _ in range(args.iterations):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    output = ck.int8_linear(**call)
                    end.record()
                    event_pairs.append((start, end))
            torch.cuda.synchronize()

        assert output is not None
        elapsed = [start.elapsed_time(end) for start, end in event_pairs]
        free_after, _ = torch.cuda.mem_get_info(device)
        finite = bool(torch.isfinite(output).all().item())
        output_path = output_dir / f"{backend}-{name}.npy"
        np.save(output_path, output.float().cpu().numpy())
        report["shapes"].append(
            {
                "name": name,
                "m": m,
                "k": k,
                "n": n,
                "output_shape": list(output.shape),
                "output_dtype": str(output.dtype),
                "finite": finite,
                "median_ms": statistics.median(elapsed),
                "samples_ms": elapsed,
                "peak_allocated": torch.cuda.max_memory_allocated(device),
                "peak_reserved": torch.cuda.max_memory_reserved(device),
                "free_before": free_before,
                "free_after": free_after,
                "total": total,
                "output_path": str(output_path),
            }
        )
        del output, x, weight, weight_scale, bias, call
        torch.cuda.empty_cache()

    report_path = Path(args.report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def _compare_arrays(reference_path: str, candidate_path: str, chunk_rows: int = 32) -> dict:
    import numpy as np

    reference = np.load(reference_path, mmap_mode="r")
    candidate = np.load(candidate_path, mmap_mode="r")
    if reference.shape != candidate.shape:
        raise RuntimeError(f"Output shape mismatch: {reference.shape} != {candidate.shape}")
    max_abs_error = 0.0
    abs_error_sum = 0.0
    value_count = 0
    mismatch_count = 0
    for start in range(0, reference.shape[0], chunk_rows):
        ref = np.asarray(reference[start:start + chunk_rows], dtype=np.float32)
        got = np.asarray(candidate[start:start + chunk_rows], dtype=np.float32)
        error = np.abs(ref - got)
        max_abs_error = max(max_abs_error, float(error.max(initial=0.0)))
        abs_error_sum += float(error.sum(dtype=np.float64))
        value_count += error.size
        mismatch_count += int(np.count_nonzero(error > (0.02 + 0.02 * np.abs(ref))))
    return {
        "max_abs_error": max_abs_error,
        "mean_abs_error": abs_error_sum / max(value_count, 1),
        "mismatch_fraction": mismatch_count / max(value_count, 1),
    }


def _run_comparison(args) -> dict:
    output_dir = Path(args.output_dir or tempfile.mkdtemp(prefix="h3-int8-probe-"))
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for backend in ("eager", "cuda"):
        report_path = output_dir / f"{backend}-report.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--backend", backend,
            "--rows", str(args.rows),
            "--warmups", str(args.warmups),
            "--iterations", str(args.iterations),
            "--seed", str(args.seed),
            "--output-dir", str(output_dir),
            "--report-path", str(report_path),
        ]
        if args.full_rows:
            command.append("--full-rows")
        if backend == "cuda" and args.allow_cuda_under_13:
            command.append("--allow-cuda-under-13")
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.returncode:
            raise RuntimeError(
                f"{backend} probe failed with exit {completed.returncode}:\n{completed.stderr}"
            )
        reports[backend] = json.loads(report_path.read_text())

    eager_shapes = {item["name"]: item for item in reports["eager"]["shapes"]}
    cuda_shapes = {item["name"]: item for item in reports["cuda"]["shapes"]}
    comparisons = []
    for name, _, _, _ in H3_LINEAR_SHAPES:
        numerical = _compare_arrays(eager_shapes[name]["output_path"], cuda_shapes[name]["output_path"])
        comparisons.append(
            {
                "name": name,
                **numerical,
                "eager_median_ms": eager_shapes[name]["median_ms"],
                "cuda_median_ms": cuda_shapes[name]["median_ms"],
                "speedup": eager_shapes[name]["median_ms"] / cuda_shapes[name]["median_ms"],
                "eager_peak_allocated": eager_shapes[name]["peak_allocated"],
                "cuda_peak_allocated": cuda_shapes[name]["peak_allocated"],
            }
        )
    combined = {"reports": reports, "comparisons": comparisons}
    combined_path = output_dir / "comparison.json"
    combined_path.write_text(json.dumps(combined, indent=2, sort_keys=True))
    print(json.dumps({"comparison_path": str(combined_path), "comparisons": comparisons}, indent=2))
    return combined


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--backend", choices=("eager", "cuda"), default="eager")
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--full-rows", action="store_true")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", default="/kaggle/working/h3-int8-backend-probe")
    parser.add_argument("--report-path", default="/kaggle/working/h3-int8-backend-probe/report.json")
    parser.add_argument("--allow-cuda-under-13", action="store_true")
    args = parser.parse_args()
    if args.rows < 1 or args.warmups < 0 or args.iterations < 1:
        parser.error("rows and iterations must be positive and warmups must be non-negative")
    return args


def main():
    args = _parse_args()
    if args.worker:
        report = _run_backend(args)
        print(json.dumps({"report_path": args.report_path, "backend": report["backend"]}))
    else:
        _run_comparison(args)


if __name__ == "__main__":
    main()
