from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]


def _load_easycache(monkeypatch):
    comfy = types.ModuleType("comfy")
    model_patcher = types.ModuleType("comfy.model_patcher")
    patcher_extension = types.ModuleType("comfy.patcher_extension")

    class WrappersMP:
        DIFFUSION_MODEL = "diffusion_model"
        CALC_COND_BATCH = "calc_cond_batch"
        OUTER_SAMPLE = "outer_sample"

    setattr(patcher_extension, "WrappersMP", WrappersMP)
    setattr(comfy, "model_patcher", model_patcher)
    setattr(comfy, "patcher_extension", patcher_extension)

    comfy_extras = types.ModuleType("comfy_extras")
    upstream = types.ModuleType("comfy_extras.nodes_easycache")

    class EasyCacheHolder:
        pass

    setattr(upstream, "EasyCacheHolder", EasyCacheHolder)

    decorator_module = types.ModuleType(
        "raylight.comfy_extra_dist.ray_patch_decorator"
    )
    setattr(decorator_module, "ray_patch", lambda fn: fn)

    for name, module in {
        "comfy": comfy,
        "comfy.model_patcher": model_patcher,
        "comfy.patcher_extension": patcher_extension,
        "comfy_extras": comfy_extras,
        "comfy_extras.nodes_easycache": upstream,
        "raylight.comfy_extra_dist.ray_patch_decorator": decorator_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    path = ROOT / "src/raylight/comfy_extra_dist/nodes_easycache.py"
    name = "raylight.comfy_extra_dist.nodes_easycache_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Holder:
    first_cond_uuid: str | None = None
    initial_step = True
    skip_current_step = False
    verbose = False
    is_log_rank = True
    x_prev_subsampled = None
    output_prev_subsampled = None
    output_prev_norm = None
    relative_transformation_rate = None
    cumulative_change_rate = 0.0
    uuid_cache_diffs = {}
    uuid_cache_diffs_audio = {}

    def __init__(self):
        self.updates = []

    def check_metadata(self, x):
        return None

    def is_past_end_timestep(self, sigmas):
        return False

    def should_do_easycache(self, sigmas):
        return True

    def has_first_cond_uuid(self, uuids):
        return self.first_cond_uuid in uuids

    def has_x_prev_subsampled(self):
        return False

    def has_output_prev_norm(self):
        return False

    def has_relative_transformation_rate(self):
        return False

    def subsample(self, x, uuids, clone=True):
        return x.clone() if clone else x

    def update_cache_diff(self, output, x, uuids, is_audio=False):
        self.updates.append((output, x, is_audio))

    def apply_cache_diff(self, x, uuids, is_audio=False):
        return x + (2 if is_audio else 1)

    def sync_scalar(self, value, op="max"):
        return float(value)


def test_distributed_easycache_preserves_h3_av_output_and_caches_both_streams(monkeypatch):
    module = _load_easycache(monkeypatch)
    holder = _Holder()
    video_in = torch.zeros((1, 2, 2, 2))
    audio_in = torch.zeros((1, 2, 4))
    video_out = torch.ones_like(video_in)
    audio_out = torch.ones_like(audio_in)
    full_output = [video_out, audio_out]
    options = {
        "easycache": holder,
        "sigmas": torch.tensor([1.0]),
        "uuids": ["cond"],
    }

    result = module.distributed_easycache_forward_wrapper(
        lambda *args, **kwargs: full_output,
        [video_in, audio_in],
        None,
        options,
    )

    assert result is full_output
    assert len(holder.updates) == 2
    assert holder.updates[0][2] is False
    assert holder.updates[1][2] is True
    assert holder.updates[0][0] is video_out
    assert holder.updates[1][0] is audio_out


def test_distributed_easycache_skip_returns_h3_av_list(monkeypatch):
    module = _load_easycache(monkeypatch)
    holder = _Holder()
    holder.first_cond_uuid = "cond"
    holder.initial_step = False
    holder.skip_current_step = True
    video = torch.zeros((1, 2, 2, 2))
    audio = torch.zeros((1, 2, 4))
    options = {
        "easycache": holder,
        "sigmas": torch.tensor([1.0]),
        "uuids": ["cond"],
    }

    result = module.distributed_easycache_forward_wrapper(
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must skip")),
        [video, audio],
        None,
        options,
    )

    assert isinstance(result, list)
    torch.testing.assert_close(result[0], video + 1)
    torch.testing.assert_close(result[1], audio + 2)
