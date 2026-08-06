from __future__ import annotations

import sys
import types

from raylight.comfy_extra_dist.nodes_te_speed_h3 import RayTESpeedMiniMaxH3


class _Model:
    pass


def test_ray_te_speed_node_delegates_to_pinned_community_node(monkeypatch):
    calls = {}
    result_model = _Model()

    class CommunityNode:
        def patch(self, model, control, start, end, mcs, device, depth):
            calls["args"] = (model, control, start, end, mcs, device, depth)
            return (result_model,)

    package = types.ModuleType("te_speed_minimax_h3_oss")
    nodes = types.ModuleType("te_speed_minimax_h3_oss.nodes")
    setattr(nodes, "TESpeedMiniMaxH3", CommunityNode)
    monkeypatch.setitem(sys.modules, "te_speed_minimax_h3_oss", package)
    monkeypatch.setitem(sys.modules, "te_speed_minimax_h3_oss.nodes", nodes)

    model = _Model()
    apply_on_worker = getattr(RayTESpeedMiniMaxH3.apply, "__wrapped__")
    result = apply_on_worker(
        RayTESpeedMiniMaxH3(),
        model,
        0.12,
        0.1,
        0.9,
        2,
        "cpu",
        0.75,
    )

    assert result is result_model
    assert calls["args"] == (model, 0.12, 0.1, 0.9, 2, "cpu", 0.75)


def test_ray_te_speed_node_rejects_invalid_community_result(monkeypatch):
    class CommunityNode:
        def patch(self, *args):
            return "not-a-model-tuple"

    package = types.ModuleType("te_speed_minimax_h3_oss")
    nodes = types.ModuleType("te_speed_minimax_h3_oss.nodes")
    setattr(nodes, "TESpeedMiniMaxH3", CommunityNode)
    monkeypatch.setitem(sys.modules, "te_speed_minimax_h3_oss", package)
    monkeypatch.setitem(sys.modules, "te_speed_minimax_h3_oss.nodes", nodes)

    apply_on_worker = getattr(RayTESpeedMiniMaxH3.apply, "__wrapped__")
    try:
        apply_on_worker(
            RayTESpeedMiniMaxH3(),
            _Model(),
            0.12,
            0.1,
            0.9,
            2,
            "cpu",
            0.75,
        )
    except RuntimeError as exc:
        assert "invalid model result" in str(exc)
    else:
        raise AssertionError("invalid community result was accepted")
