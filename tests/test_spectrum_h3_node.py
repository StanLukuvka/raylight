from __future__ import annotations

import sys
import types

from raylight.comfy_extra_dist.nodes_spectrum_h3 import RaySpectrumApplyMiniMaxH3


class _Model:
    def __init__(self):
        self.cloned = None

    def clone(self):
        self.cloned = _Model()
        return self.cloned


def test_ray_node_uses_community_runtime_and_sampler_wrappers(monkeypatch):
    package = types.ModuleType("comfyui_spectrum_h3")
    config_module = types.ModuleType("comfyui_spectrum_h3.config")
    minimax_module = types.ModuleType("comfyui_spectrum_h3.minimax_h3")
    runtime_module = types.ModuleType("comfyui_spectrum_h3.runtime")
    sampling_module = types.ModuleType("comfyui_spectrum_h3.sampling")
    events = []

    class Config:
        def __init__(self, **kwargs):
            self.values = kwargs

        def validate(self):
            events.append(("validate", self.values))
            return self

    class Runtime:
        def __init__(self, config):
            self.config = config

    def require_native(model):
        events.append(("require", model))

    def install(model, runtime):
        events.append(("install", model, runtime))

    setattr(config_module, "SpectrumH3Config", Config)
    setattr(minimax_module, "require_native_minimax_h3", require_native)
    setattr(runtime_module, "SpectrumH3Runtime", Runtime)
    setattr(sampling_module, "install_sampler_wrappers", install)
    for name, module in {
        "comfyui_spectrum_h3": package,
        "comfyui_spectrum_h3.config": config_module,
        "comfyui_spectrum_h3.minimax_h3": minimax_module,
        "comfyui_spectrum_h3.runtime": runtime_module,
        "comfyui_spectrum_h3.sampling": sampling_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    source = _Model()
    apply_on_worker = getattr(RaySpectrumApplyMiniMaxH3.apply, "__wrapped__")
    result = apply_on_worker(
        object(),
        source,
        True,
        0.5,
        4,
        0.1,
        2.0,
        0.75,
        5,
        1,
        8,
        False,
        "system_ram",
    )

    assert result is source
    assert source.cloned is None
    assert [event[0] for event in events] == ["validate", "require", "install"]
    assert events[0][1]["history_storage"] == "system_ram"


def test_disabled_ray_node_does_not_import_or_clone():
    source = _Model()
    apply_on_worker = getattr(RaySpectrumApplyMiniMaxH3.apply, "__wrapped__")

    result = apply_on_worker(
        object(),
        source,
        False,
        0.5,
        4,
        0.1,
        2.0,
        0.75,
        5,
        1,
        8,
        False,
        "system_ram",
    )

    assert result is source
    assert source.cloned is None


def test_ray_node_accepts_fsdp_dynamic_subclass_of_native_minimax_h3(monkeypatch):
    package = types.ModuleType("comfyui_spectrum_h3")
    config_module = types.ModuleType("comfyui_spectrum_h3.config")
    minimax_module = types.ModuleType("comfyui_spectrum_h3.minimax_h3")
    runtime_module = types.ModuleType("comfyui_spectrum_h3.runtime")
    sampling_module = types.ModuleType("comfyui_spectrum_h3.sampling")

    native = type("MiniMaxH3Model", (), {"__module__": "comfy.ldm.minimax.model"})
    fsdp = type(
        "FSDPMiniMaxH3Model",
        (native,),
        {"__module__": "torch.distributed.fsdp._fully_shard._fully_shard"},
    )()
    for name in (
        "blocks",
        "final_layer",
        "hidden_size",
        "patch_size",
        "latents_dim",
        "audio_latents_dim",
        "sigma_shift_video",
        "sigma_shift_audio",
        "time_embedder",
    ):
        setattr(fsdp, name, object())
    fsdp.use_adaln_curves = False

    class Model:
        def __init__(self):
            self.model = types.SimpleNamespace(diffusion_model=fsdp)

        def clone(self):
            return self

    class Config:
        def __init__(self, **kwargs):
            self.values = kwargs

        def validate(self):
            return self

    class Runtime:
        def __init__(self, config):
            self.config = config

    installed = []
    setattr(config_module, "SpectrumH3Config", Config)
    setattr(minimax_module, "locate_minimax_h3_inner", lambda model: (fsdp, "model.diffusion_model"))
    setattr(
        minimax_module,
        "require_native_minimax_h3",
        lambda model: (_ for _ in ()).throw(TypeError("exact native type required")),
    )
    setattr(runtime_module, "SpectrumH3Runtime", Runtime)
    setattr(sampling_module, "install_sampler_wrappers", lambda model, runtime: installed.append(model))
    for name, module in {
        "comfyui_spectrum_h3": package,
        "comfyui_spectrum_h3.config": config_module,
        "comfyui_spectrum_h3.minimax_h3": minimax_module,
        "comfyui_spectrum_h3.runtime": runtime_module,
        "comfyui_spectrum_h3.sampling": sampling_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    model = Model()
    apply_on_worker = getattr(RaySpectrumApplyMiniMaxH3.apply, "__wrapped__")
    result = apply_on_worker(
        RaySpectrumApplyMiniMaxH3(), model, True, 0.5, 4, 0.1, 2.0, 0.75, 5, 1, 8, False, "system_ram"
    )

    assert result is model
    assert installed == [model]
