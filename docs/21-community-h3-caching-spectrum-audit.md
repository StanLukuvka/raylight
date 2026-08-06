# Community MiniMax-H3 caching and Spectrum audit

Date: 2026-08-06

This audit is for Raylight's exact two-rank Ulysses MiniMax-H3 path on 2× Tesla T4. The v6 notebook and commit `908b1857e69721d83240bd6e7213eed6abe8f06d` remain the unchanged control.

## Result

Two existing community implementations are suitable for bounded experiments:

1. [ComfyUI-Spectrum-MiniMax-H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3), pinned at `85ec1da66277e893079ecd46e32cc865c56cfe53`.
2. [TE-Speed-MiniMaxH3-OSS](https://github.com/HELPMEEADICE/TE-Speed-MiniMaxH3-OSS), pinned at `c1dacf47bc02cb9326f7b93c69280529b93d391b`.

Raylight does not copy either algorithm. It installs each repository at its pinned commit and exposes a thin `RAY_ACTORS` adapter. Spectrum's existing runtime and forecaster are used unchanged. TE-Speed's existing node and cache state are used unchanged.

Neither optimization is accepted for production yet. Both change the denoising trajectory and require bounded dual-T4 testing and fixed-seed video/audio review.

## What the community caches

### EasyCache and TeaCache in ComfyUI/Raylight

The existing `RayEasyCache` and `RayTeaCache` nodes operate around the complete diffusion-model call. They compare consecutive model input tensors, then reuse a cached output residual or output-derived estimate.

This is not a safe MiniMax-H3 integration:

- H3 receives joint video/audio input rather than one tensor, while the generic implementation assumes properties such as `x.shape` and tensor subtraction.
- The generic metadata hook is not an H3 topology contract.
- It does not prove correspondence for packed text, reference, target-video, and target-audio segments.
- It does not synchronize cache decisions across Ulysses ranks.
- A differing rank decision can deadlock on the next attention collective.

These nodes remain available for their existing supported models but are not wired into the H3 workflow.

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

### TE-Speed MiniMax-H3 OSS

Source: [TE-Speed-MiniMaxH3-OSS](https://github.com/HELPMEEADICE/TE-Speed-MiniMaxH3-OSS), LGPL-3.0, pinned at `c1dacf47bc02cb9326f7b93c69280529b93d391b`.

Mechanism:

- On a full step, run all blocks and retain the residual from the output of a warm prefix to the complete output.
- On a cache step, recompute the warm prefix and add the prior tail residual instead of running the remaining blocks.
- Default `cache_depth=0.75` recomputes approximately the first quarter of the 50 blocks.
- Eligibility depends on schedule position, adjacent sigma difference, available residual, and a maximum consecutive-cache count.
- CFG calls at the same sigma reuse the same full/cache decision.

Cache identity and invalidation:

- first sigma of a run is full;
- an increasing sigma indicates a new run and resets state;
- the cache stores one prior tail residual plus a warm-boundary snapshot;
- there is no explicit layout, conditioning, or model-weight hash;
- changing geometry inside one sampling run is therefore unsupported;
- normal separate Comfy sampling runs reset because their schedule restarts at high sigma.

Distributed compatibility:

- The residual and snapshot are rank-local after Raylight's Ulysses split.
- The decision inputs are schedule scalars and counters, so they should be identical on both ranks.
- Raylight additionally all-reduces the requested `[start, end)` block range before executing any block. A disagreement fails on both ranks before entering attention collectives.
- The plugin's algorithm is imported unchanged; Raylight only exposes the block-loop contract it expects.

Memory:

- At the accepted workload, one rank-local hidden state is approximately 58.96 MiB.
- With Raylight's default `device=cpu`, the residual occupies about 58.96 MiB of host memory per rank and the latest warm-boundary snapshot about 58.96 MiB of GPU memory per rank.
- A full-step residual calculation can transiently hold another rank-local tensor, so the bounded first-forward and full-run memory gates remain mandatory.

The repository claims approximately 45% speedup for its 30-step reference workflow. That is an upstream claim, not evidence for the 20-evaluation dual-T4 workload.

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

It was not plugged into Raylight now because the adapter targets SGLang's H3 class and forward signature, not ComfyUI's `MiniMaxH3Model`, and cache-dit would add a second block-hook/runtime layer. TE-Speed is the smaller existing Comfy-native experiment. Cache-DiT remains the preferred next candidate if TE-Speed demonstrates value but needs better policy or if an Apache-only path becomes necessary.

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
- `RayTESpeedMiniMaxH3` is available as a separate cache node; do not stack it with Spectrum until each path independently passes parity and memory tests.
- First run: `608×352`, one bounded forward.
- Second run: accepted `736×416×124`, one bounded forward.
- Only then run all 20 evaluations.
- Compare each accelerator separately against exact v6 using fixed prompt, seed, sampler, and checkpoint.
- Reject on rank disagreement, unbounded cleanup, OOM/cgroup pressure, missing audio, A/V desynchronization, or material quality regression.
