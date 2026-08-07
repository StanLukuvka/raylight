# Community MiniMax-H3 caching and Spectrum audit

Date: 2026-08-06

This audit is for Raylight's exact two-rank Ulysses MiniMax-H3 path on 2× Tesla T4. The v6 notebook and commit `908b1857e69721d83240bd6e7213eed6abe8f06d` remain the unchanged control.

## Result

Two existing Raylight/ComfyUI paths are suitable for bounded experiments:

1. [ComfyUI-Spectrum-MiniMax-H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3), pinned at `85ec1da66277e893079ecd46e32cc865c56cfe53`.
2. Raylight's existing `RayEasyCache` node, backed by ComfyUI's established EasyCache implementation and Raylight's distributed wrapper.

Spectrum remains an external pinned dependency. EasyCache is already part of Raylight, so the notebook adds no second cache plugin or block-loop adapter.

Neither optimization is accepted for production yet. Both change the denoising trajectory and require bounded dual-T4 testing and fixed-seed video/audio review.

## What the community caches

### EasyCache and TeaCache in ComfyUI/Raylight

The existing `RayEasyCache` node operates around the complete diffusion-model call. It compares consecutive model inputs and reuses a cached output residual when the accumulated change remains below a threshold.

Why it is selected here:

- it is the established ComfyUI/Raylight cache surface rather than a niche H3-only plugin;
- Raylight already owns its worker patching, cache lifecycle, and distributed decision synchronization;
- no external cache checkout, copied H3 `_forward`, or partial block-loop hook is required;
- `distributed_sync=True` is retained for the exact two-rank Ulysses notebook.

H3 video/audio quality and skip-rate behavior are still unproven. EasyCache is therefore supplied as a separate workflow from Spectrum and must pass the same bounded two-T4 test before a combined graph is attempted.

### MiniMaxH3-Cache / Trent MiniMax H3 Cache

Sources:

- [ComfyUI-MiniMaxH3-Cache](https://github.com/lihaoyun6/ComfyUI-MiniMaxH3-Cache), GPL-3.0, reviewed at `8a45e096a2a05c140dd4d909eb74e4279b673819`.
- [TrentNodes MiniMax H3 cache](https://github.com/TrentHunter82/TrentNodes/blob/a8dd22f43befa345ca67a68c34a6ff0609afadd9/nodes/minimax_h3_cache.py), repository MIT but the file states it is ported from the GPL-3.0 implementation; reviewed at `a8dd22f43befa345ca67a68c34a6ff0609afadd9`.

Mechanism:

- Patch the complete 50-block loop.
- On a real step, cache `final_hidden - initial_hidden`.
- On a skipped step, add that residual to the current hidden state and skip all transformer blocks.
- Decide using accumulated relative-L1 drift sampled from target audio/video rows.
- Force a real step after a bounded number of consecutive skips.

Cache identity and invalidation:

- sampling-scope reset at the start/end of each run;
- reset on hidden shape, dtype, device, or block-count change;
- step identity is derived from timestep changes so CFG branches do not advance the counter twice;
- target audio/video ranges contribute to the drift signature;
- prompt/reference identity is not an explicit key, but normal run teardown resets the state.

Why it was not selected:

- It monkey-patches or copies ComfyUI's complete H3 `_forward`, creating an upstream-version and audio-slope hazard.
- A skipped step executes zero blocks. Independent per-rank decisions therefore cannot be checked at the first shared block collective.
- The original is GPL-3.0. The Trent repository-level MIT label does not erase the source file's stated GPL provenance.
- Advertised quality and speed are not backed by a controlled H3 video/audio benchmark in the reviewed repository.

### Cache-DiT / SGLang

Sources:

- [cache-dit](https://github.com/vipshop/cache-dit), Apache-2.0.
- [SGLang cache-dit integration](https://github.com/sgl-project/sglang/blob/21225aba3d63000ba89b98fc9c85e4b6517c4b72/python/sglang/multimodal_gen/runtime/cache/cache_dit_integration.py), Apache-2.0, reviewed at `21225aba3d63000ba89b98fc9c85e4b6517c4b72`.

Mechanism:

- configurable front/back blocks are computed;
- intermediate residuals/features are reused when relative difference is below threshold;
- warmup and maximum consecutive cached steps bound reuse;
- optional static step-computation masks and TaylorSeer forecasting are available.

Distributed behavior is the strongest reviewed design: SGLang all-reduces similarity statistics over the sequence/tensor-parallel group so every rank makes one decision. SGLang also contains a custom `MiniMaxH3DiTModel` adapter.

It was not plugged into Raylight now because the adapter targets SGLang's H3 class and forward signature, not ComfyUI's `MiniMaxH3Model`, and Cache-DiT would add a second block-hook/runtime layer. Raylight's existing EasyCache is the smaller established Comfy-native experiment.

## Spectrum integration

Source: [ComfyUI-Spectrum-MiniMax-H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3), GPL-3.0, pinned at `85ec1da66277e893079ecd46e32cc865c56cfe53`.

The upstream implementation is not a conventional cache. It retains actual post-transformer features and forecasts future features using:

- a small Chebyshev ridge fit over solver coordinates;
- low-frequency spectral history weights blended with a two-point linear forecast;
- a sampler-aware actual/forecast schedule;
- bounded FP32 chunk accumulation;
- current-step final video/audio heads on every evaluation.

Upstream conservative RES multistep policy for 20 evaluations is 14 actual and 6 forecast steps, a 30% transformer-evaluation reduction. This is not a 30% end-to-end guarantee.

Upstream invalidation and safety contracts include:

- one sampling run ID and monotonically tracked step ID;
- exact feature shape and packed topology;
- stable, unique CFG branch labels;
- sampler allowlist;
- warmup and native tail;
- actual retry on prediction failure;
- no ancestry/noise-injecting sampler forecasting;
- complete history reset at run end or abort.

Upstream explicitly keeps generic multi-GPU sampling native because it has not validated distributed forecast transactions. Raylight supplies the missing narrow adapter:

- history is the rank-local packed hidden state after the final block;
- every rank uses the existing upstream runtime and forecaster;
- actual-vs-forecast decisions are synchronized before any block collective;
- prediction and archive success are synchronized;
- any prediction failure raises the upstream `ForecastRetryActual` on every rank;
- predicted rank-local hidden states are gathered through the existing Ulysses final-gather path;
- current H3 video/audio final heads and sigma/audio-slope behavior remain unchanged.

Raylight's rank-local history includes non-target packed rows because the integration enters after the existing distributed split. Those extra rows are never consumed by the final target heads, but they increase storage. At 5,750 local rows, hidden width 5,376, BF16, and eight histories:

- approximately 58.96 MiB per local snapshot;
- approximately 471.68 MiB host history per rank;
- approximately 943.36 MiB across two ranks.

That fits under v6's 2.353 GiB host headroom on paper, leaving roughly 1.4 GiB before transfer and allocator overhead. It must still pass the real cgroup gate. VRAM history is inappropriate for the T4 baseline.

Upstream reports real fidelity risks: changed motion trajectories and localized degradation of fast-moving eyes, fingers, nails, and other small articulated details. Fixed-seed video and audio review is mandatory.

## Experimental workflow policy

- v6 remains untouched.
- Community repositories are pinned and installed as separate dependencies; their algorithm sources are not vendored.
- `RaySpectrumApplyMiniMaxH3` is present in the generated experimental workflow with conservative settings and system-RAM history.
- `RayEasyCache` is available as the separate cache workflow; do not stack it with Spectrum until each path independently passes parity and memory tests.
- First run: `608×352`, one bounded forward.
- Second run: accepted `736×416×124`, one bounded forward.
- Only then run all 20 evaluations.
- Compare each accelerator separately against exact v6 using fixed prompt, seed, sampler, and checkpoint.
- Reject on rank disagreement, unbounded cleanup, OOM/cgroup pressure, missing audio, A/V desynchronization, or material quality regression.
