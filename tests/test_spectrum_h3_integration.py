from __future__ import annotations

import sys
import types

import pytest
import torch

from raylight.spectrum_h3 import (
    begin_spectrum_call,
    observe_spectrum_feature,
    predict_spectrum_feature,
)


class _FakeRuntime:
    def __init__(self, *, actual: bool, prediction: torch.Tensor | None = None):
        self.actual = actual
        self.prediction = prediction
        self.fallbacks = []
        self.observed = []
        self.calls = []

    def begin_model_call(self, run_id, step_id, *, topology, labels, expected_shape):
        self.calls.append((run_id, step_id, topology, labels, expected_shape))
        return 0, self.actual

    def fallback_current_step(self, run_id, step_id, reason):
        self.fallbacks.append((run_id, step_id, reason))
        self.actual = True

    def predict(self, run_id, step_id, call_id, *, device, dtype):
        if self.prediction is None:
            return None
        return self.prediction.to(device=device, dtype=dtype)

    def observe_actual(self, run_id, step_id, call_id, feature):
        self.observed.append((run_id, step_id, call_id, feature.clone()))


def _options(runtime):
    return {
        "spectrum_h3_runtime": runtime,
        "spectrum_h3_run_id": 7,
        "spectrum_h3_step_id": 3,
    }


def _install_community_forecast_contract(monkeypatch):
    class RetryActual(RuntimeError):
        pass

    def sanitize(feature, dtype):
        fp32 = feature.to(torch.float32)
        finite = torch.isfinite(fp32)
        if not bool(finite.any().item()):
            return None, {"reason": "forecast contains no finite values"}
        finfo = torch.finfo(dtype)
        sanitized = torch.nan_to_num(fp32, nan=0.0, posinf=finfo.max, neginf=finfo.min)
        return sanitized.clamp(min=finfo.min, max=finfo.max).to(dtype), None

    runtime_module = types.ModuleType("comfyui_spectrum_h3.runtime")
    setattr(runtime_module, "ForecastRetryActual", RetryActual)
    minimax_module = types.ModuleType("comfyui_spectrum_h3.minimax_h3")
    setattr(minimax_module, "_sanitize_prediction", sanitize)
    monkeypatch.setitem(sys.modules, "comfyui_spectrum_h3.runtime", runtime_module)
    monkeypatch.setitem(sys.modules, "comfyui_spectrum_h3.minimax_h3", minimax_module)
    return RetryActual


def test_absent_community_runtime_keeps_native_path():
    assert begin_spectrum_call({}, (1, 4, 8), ("topology",), ((0, "uuid"),)) is None


def test_distributed_actual_decision_forces_local_forecast_rank_to_compute():
    runtime = _FakeRuntime(actual=False)

    call = begin_spectrum_call(
        _options(runtime),
        (1, 4, 8),
        ("topology",),
        ((0, "uuid"),),
        sync_any=lambda _: True,
    )

    assert call is not None and call.actual
    assert runtime.fallbacks == [(7, 3, "another distributed rank requires an actual Spectrum step")]


def test_begin_failure_is_raised_on_every_rank_before_blocks():
    runtime = _FakeRuntime(actual=False)
    options = {
        "spectrum_h3_runtime": runtime,
        "spectrum_h3_run_id": 3,
        "spectrum_h3_step_id": 4,
    }

    with pytest.raises(RuntimeError, match="begin failed on at least one distributed rank"):
        begin_spectrum_call(
            options,
            (1, 4, 8),
            ("rank-local", 0),
            ("cond",),
            sync_any=lambda value: value,
            sync_all=lambda value: False,
        )


def test_forecast_uses_existing_community_runtime_prediction(monkeypatch):
    _install_community_forecast_contract(monkeypatch)
    expected = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
    runtime = _FakeRuntime(actual=False, prediction=expected)
    call = begin_spectrum_call(
        _options(runtime),
        tuple(expected.shape),
        ("topology",),
        ((0, "uuid"),),
        sync_any=lambda flag: flag,
    )

    predicted = predict_spectrum_feature(
        call,
        device=torch.device("cpu"),
        dtype=torch.float32,
        sync_all=lambda flag: flag,
    )

    torch.testing.assert_close(predicted, expected)


def test_nonfinite_forecast_uses_community_sanitizer_and_retries(monkeypatch):
    RetryActual = _install_community_forecast_contract(monkeypatch)
    runtime = _FakeRuntime(
        actual=False,
        prediction=torch.full((1, 4, 8), float("nan")),
    )
    call = begin_spectrum_call(
        _options(runtime),
        (1, 4, 8),
        ("topology",),
        ((0, "uuid"),),
        sync_any=lambda flag: flag,
    )

    assert call is not None
    with pytest.raises(RetryActual, match="failed on at least one distributed rank"):
        predict_spectrum_feature(
            call,
            device=torch.device("cpu"),
            dtype=torch.float32,
            sync_all=lambda flag: flag,
        )


def test_failed_prediction_raises_community_retry_on_every_rank(monkeypatch):
    RetryActual = _install_community_forecast_contract(monkeypatch)

    runtime = _FakeRuntime(actual=False, prediction=torch.ones((1, 4, 8)))
    call = begin_spectrum_call(
        _options(runtime),
        (1, 4, 8),
        ("topology",),
        ((0, "uuid"),),
        sync_any=lambda flag: flag,
    )

    assert call is not None
    with pytest.raises(RetryActual, match="failed on at least one distributed rank"):
        predict_spectrum_feature(
            call,
            device=torch.device("cpu"),
            dtype=torch.float32,
            sync_all=lambda _: False,
        )


def test_actual_history_archive_is_synchronized():
    runtime = _FakeRuntime(actual=True)
    feature = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
    call = begin_spectrum_call(
        _options(runtime),
        tuple(feature.shape),
        ("topology",),
        ((0, "uuid"),),
        sync_any=lambda flag: flag,
    )

    observe_spectrum_feature(call, feature, sync_all=lambda flag: flag)

    assert len(runtime.observed) == 1
    torch.testing.assert_close(runtime.observed[0][3], feature)
