from __future__ import annotations

from .ray_patch_decorator import ray_patch


def _is_fsdp_wrapped_native_minimax_h3(model, locate_inner) -> bool:
    inner, path = locate_inner(model)
    if inner is None or path is None or type(inner).__name__ == "MiniMaxH3Model":
        return False
    native_base = any(
        cls.__name__ == "MiniMaxH3Model" and cls.__module__ == "comfy.ldm.minimax.model"
        for cls in type(inner).__mro__[1:]
    )
    required = (
        "blocks",
        "final_layer",
        "hidden_size",
        "patch_size",
        "latents_dim",
        "audio_latents_dim",
        "sigma_shift_video",
        "sigma_shift_audio",
        "use_adaln_curves",
    )
    if not native_base or not all(hasattr(inner, name) for name in required):
        return False
    if not isinstance(inner.use_adaln_curves, bool):
        return False
    timestep_attribute = "adaln_t_table" if inner.use_adaln_curves else "time_embedder"
    return hasattr(inner, timestep_attribute)


class RaySpectrumApplyMiniMaxH3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "enabled": ("BOOLEAN", {"default": True}),
                "blend_weight": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "degree": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1}),
                "ridge_lambda": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 10.0, "step": 0.01}),
                "window_size": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 16.0, "step": 0.05}),
                "flex_window": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 8.0, "step": 0.05}),
                "warmup_steps": ("INT", {"default": 5, "min": 0, "max": 64, "step": 1}),
                "tail_actual_steps": ("INT", {"default": 1, "min": 0, "max": 64, "step": 1}),
                "max_history": ("INT", {"default": 8, "min": 2, "max": 64, "step": 1}),
                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "history_storage": (["system_ram", "vram"], {"default": "system_ram"}),
            },
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "apply"
    CATEGORY = "Raylight/sampling/spectrum"

    @ray_patch
    def apply(
        self,
        model,
        enabled,
        blend_weight,
        degree,
        ridge_lambda,
        window_size,
        flex_window,
        warmup_steps,
        tail_actual_steps,
        max_history,
        debug,
        history_storage="system_ram",
    ):
        if not enabled:
            return model

        # The forecasting algorithm, sampler transaction, and configuration are
        # supplied by the existing community plugin. Raylight only adapts its
        # model-call boundary to rank-local Ulysses hidden state.
        from comfyui_spectrum_h3.config import SpectrumH3Config
        from comfyui_spectrum_h3 import minimax_h3
        from comfyui_spectrum_h3.runtime import SpectrumH3Runtime
        from comfyui_spectrum_h3.sampling import install_sampler_wrappers

        config = SpectrumH3Config(
            enabled=True,
            blend_weight=float(blend_weight),
            degree=int(degree),
            ridge_lambda=float(ridge_lambda),
            window_size=float(window_size),
            flex_window=float(flex_window),
            warmup_steps=int(warmup_steps),
            tail_actual_steps=int(tail_actual_steps),
            max_history=int(max_history),
            history_storage=str(history_storage),
            debug=bool(debug),
        ).validate()
        patched = model.clone()
        locate_inner = getattr(minimax_h3, "locate_minimax_h3_inner", None)
        fsdp_native = locate_inner is not None and _is_fsdp_wrapped_native_minimax_h3(
            patched, locate_inner
        )
        if not fsdp_native:
            minimax_h3.require_native_minimax_h3(patched)
        install_sampler_wrappers(patched, SpectrumH3Runtime(config))
        return patched


NODE_CLASS_MAPPINGS = {"RaySpectrumApplyMiniMaxH3": RaySpectrumApplyMiniMaxH3}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RaySpectrumApplyMiniMaxH3": "Spectrum Apply MiniMax H3 (Ray)",
}
