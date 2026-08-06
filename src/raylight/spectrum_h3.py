from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.distributed as dist


RUNTIME_KEY = "spectrum_h3_runtime"
RUN_ID_KEY = "spectrum_h3_run_id"
STEP_ID_KEY = "spectrum_h3_step_id"


@dataclass(frozen=True, slots=True)
class SpectrumCall:
    runtime: Any
    run_id: int
    step_id: int
    call_id: int
    actual: bool


def _distributed_bool(value: bool, reduce_op: dist.ReduceOp) -> bool:
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return bool(value)
    device = torch.device("cuda", torch.cuda.current_device())
    flag = torch.tensor(int(bool(value)), device=device, dtype=torch.int32)
    from xfuser.core.distributed import get_sp_group

    get_sp_group().all_reduce(flag, op=reduce_op)
    return bool(flag.item())


def _any_rank(value: bool) -> bool:
    return _distributed_bool(value, dist.ReduceOp.MAX)


def _all_ranks(value: bool) -> bool:
    return _distributed_bool(value, dist.ReduceOp.MIN)


def begin_spectrum_call(
    transformer_options: dict[str, Any],
    expected_shape: tuple[int, ...],
    topology: tuple[Any, ...],
    labels: tuple[Any, ...] | None,
    *,
    sync_any: Callable[[bool], bool] = _any_rank,
    sync_all: Callable[[bool], bool] = _all_ranks,
) -> SpectrumCall | None:
    """Join an existing Spectrum runtime decision across every sequence rank."""
    runtime = transformer_options.get(RUNTIME_KEY)
    run_id = transformer_options.get(RUN_ID_KEY)
    step_id = transformer_options.get(STEP_ID_KEY)
    if runtime is None or run_id is None or step_id is None:
        return None

    begin_error: Exception | None = None
    call_id = -1
    local_actual = True
    try:
        call_id, local_actual = runtime.begin_model_call(
            int(run_id),
            int(step_id),
            topology=tuple(topology),
            labels=None if labels is None else tuple(labels),
            expected_shape=tuple(int(value) for value in expected_shape),
        )
    except Exception as exc:
        begin_error = exc
    if not sync_all(begin_error is None):
        reason = "Spectrum begin failed on at least one distributed rank"
        if begin_error is not None:
            reason = f"{reason}: {type(begin_error).__name__}: {begin_error}"
        raise RuntimeError(reason) from begin_error

    actual = sync_any(bool(local_actual))
    fallback_error: Exception | None = None
    if actual and not local_actual:
        try:
            runtime.fallback_current_step(
                int(run_id),
                int(step_id),
                "another distributed rank requires an actual Spectrum step",
            )
        except Exception as exc:
            fallback_error = exc
    if not sync_all(fallback_error is None):
        reason = "Spectrum distributed fallback failed on at least one rank"
        if fallback_error is not None:
            reason = f"{reason}: {type(fallback_error).__name__}: {fallback_error}"
        raise RuntimeError(reason) from fallback_error
    return SpectrumCall(runtime, int(run_id), int(step_id), int(call_id), actual)


def predict_spectrum_feature(
    call: SpectrumCall,
    *,
    device: torch.device,
    dtype: torch.dtype,
    sync_all: Callable[[bool], bool] = _all_ranks,
) -> torch.Tensor:
    """Forecast rank-local hidden state or transactionally retry on every rank."""
    predicted = None
    error: Exception | None = None
    try:
        predicted = call.runtime.predict(
            call.run_id,
            call.step_id,
            call.call_id,
            device=device,
            dtype=dtype,
        )
        if predicted is not None:
            # Use the pinned community implementation's own finite/range policy;
            # the distributed adapter must not create a weaker forecast path.
            from comfyui_spectrum_h3.minimax_h3 import _sanitize_prediction

            predicted, _ = _sanitize_prediction(predicted, dtype)
    except Exception as exc:  # synchronize failure before any later collective
        error = exc

    if not sync_all(error is None and predicted is not None):
        from comfyui_spectrum_h3.runtime import ForecastRetryActual

        reason = "Spectrum forecast failed on at least one distributed rank"
        if error is not None:
            reason = f"{reason}: {type(error).__name__}: {error}"
        raise ForecastRetryActual(reason)
    assert predicted is not None
    return predicted


def observe_spectrum_feature(
    call: SpectrumCall,
    feature: torch.Tensor,
    *,
    sync_all: Callable[[bool], bool] = _all_ranks,
) -> None:
    """Archive an actual rank-local hidden state with collective failure agreement."""
    error: Exception | None = None
    try:
        call.runtime.observe_actual(
            call.run_id,
            call.step_id,
            call.call_id,
            feature,
        )
    except Exception as exc:  # do not let a peer enter all-gather alone
        error = exc
    if not sync_all(error is None):
        reason = "Spectrum history archive failed on at least one distributed rank"
        if error is not None:
            reason = f"{reason}: {type(error).__name__}: {error}"
        raise RuntimeError(reason) from error
