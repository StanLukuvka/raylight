from __future__ import annotations

from .ray_patch_decorator import ray_patch


class RayTESpeedMiniMaxH3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "processing_control_value": (
                    "FLOAT",
                    {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "processing_percent_1": (
                    "FLOAT",
                    {"default": 0.10, "min": 0.0, "max": 0.49, "step": 0.01},
                ),
                "processing_percent_2": (
                    "FLOAT",
                    {"default": 0.90, "min": 0.51, "max": 1.0, "step": 0.01},
                ),
                "mcs": ("INT", {"default": 2, "min": 0, "max": 10, "step": 1}),
                "device": (["cpu", "auto"], {"default": "cpu"}),
            },
            "optional": {
                "cache_depth": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.0, "max": 0.95, "step": 0.05},
                ),
            },
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "apply"
    CATEGORY = "Raylight/sampling/cache"

    @ray_patch
    def apply(
        self,
        model,
        processing_control_value,
        processing_percent_1,
        processing_percent_2,
        mcs,
        device,
        cache_depth=0.75,
    ):
        # Use the pinned community implementation unchanged. Raylight exposes
        # its block-loop contract on rank-local Ulysses state and checks that
        # every rank requests the same block range before entering collectives.
        from te_speed_minimax_h3_oss.nodes import TESpeedMiniMaxH3

        patched = TESpeedMiniMaxH3().patch(
            model,
            float(processing_control_value),
            float(processing_percent_1),
            float(processing_percent_2),
            int(mcs),
            str(device),
            float(cache_depth),
        )
        if not isinstance(patched, tuple) or len(patched) != 1:
            raise RuntimeError("TE-Speed MiniMax H3 returned an invalid model result")
        return patched[0]


NODE_CLASS_MAPPINGS = {"RayTESpeedMiniMaxH3": RayTESpeedMiniMaxH3}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RayTESpeedMiniMaxH3": "TE-Speed MiniMax H3 (Ray)",
}
