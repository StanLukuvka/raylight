import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
POLICY_PATH = ROOT / "src/raylight/kitchen_backend_policy.py"


def _load_policy():
    spec = importlib.util.spec_from_file_location("raylight_test_kitchen_backend_policy", POLICY_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_int8_backend_policy_defaults_to_eager_and_rejects_unknown_values(monkeypatch):
    policy = _load_policy()
    monkeypatch.delenv("RAYLIGHT_INT8_BACKEND", raising=False)
    assert policy.requested_int8_backend() == "eager"

    monkeypatch.setenv("RAYLIGHT_INT8_BACKEND", "triton")
    with pytest.raises(ValueError, match="RAYLIGHT_INT8_BACKEND must be eager or cuda"):
        policy.requested_int8_backend()


def test_cuda_under_13_guard_runs_before_comfy_kitchen_import(monkeypatch):
    policy = _load_policy()
    monkeypatch.setenv("RAYLIGHT_INT8_BACKEND", "cuda")
    monkeypatch.delenv("RAYLIGHT_INT8_CUDA_ALLOW_UNDER_13", raising=False)

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_capability():
            return (7, 5)

    class FakeTorch:
        cuda = FakeCuda()
        version = type("Version", (), {"cuda": "12.8"})()

    imported = []

    def import_kitchen(name):
        imported.append(name)
        raise AssertionError("Comfy Kitchen must not import before the CUDA guard")

    with pytest.raises(RuntimeError, match="requires RAYLIGHT_INT8_CUDA_ALLOW_UNDER_13=1"):
        policy.initialize_worker_int8_backend(FakeTorch(), import_kitchen=import_kitchen)
    assert imported == []


def test_cuda_under_13_explicit_override_imports_and_validates_backend(monkeypatch):
    policy = _load_policy()
    monkeypatch.setenv("RAYLIGHT_INT8_BACKEND", "cuda")
    monkeypatch.setenv("RAYLIGHT_INT8_CUDA_ALLOW_UNDER_13", "1")
    monkeypatch.setenv("RAYLIGHT_H3_STOP_AFTER_FIRST_FORWARD", "1")
    monkeypatch.setenv("RAYLIGHT_H3_PHASE_PROFILE", "1")

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_capability():
            return (7, 5)

    class FakeTorch:
        cuda = FakeCuda()
        version = type("Version", (), {"cuda": "12.8"})()

    class FakeKitchen:
        @staticmethod
        def list_backends():
            return {"cuda": {"available": True, "disabled": True}}

    imported = []

    def import_kitchen(name):
        imported.append(name)
        return FakeKitchen()

    backend, kitchen = policy.initialize_worker_int8_backend(
        FakeTorch(), import_kitchen=import_kitchen
    )
    assert backend == "cuda"
    assert isinstance(kitchen, FakeKitchen)
    assert imported == ["comfy_kitchen"]


def test_ray_worker_applies_policy_before_any_comfy_kitchen_import_path():
    source = (ROOT / "src/raylight/distributed_worker/ray_worker.py").read_text()
    policy_call = source.index("initialize_worker_int8_backend(torch)")
    quant_ops_import = source.index("from raylight.comfy_dist.quant_ops import patch_temp_fix_ck_ops")
    assert policy_call < quant_ops_import


def test_sampler_scopes_and_restores_requested_backend():
    source = (ROOT / "src/raylight/comfy_dist/quant_ops.py").read_text()
    assert 'with ck.use_backend("eager"):' in source
    assert "with ck.use_backend(backend):" not in source


def test_runtime_env_propagates_int8_backend_policy_to_each_ray_worker():
    source = (ROOT / "src/raylight/nodes.py").read_text()
    assert '"RAYLIGHT_INT8_BACKEND"' in source
    assert '"RAYLIGHT_INT8_CUDA_ALLOW_UNDER_13"' in source


def test_cuda_int8_is_timed_and_eager_fallback_fails_closed():
    source = (
        ROOT / "src/raylight/comfy_dist/kitchen_patches/int8.py"
    ).read_text()
    assert "_ORIG_CUDA_INT8_LINEAR" in source
    assert "def _profiled_cuda_int8_linear(" in source
    assert 'group="h3_first_block_cuda_int8"' in source
    assert "return _profiled_cuda_int8_linear(" in source
    assert "cuda_backend.int8_linear = _profiled_cuda_int8_linear" not in source
    assert 'backend="cuda"' in source


def test_cuda_worker_bootstrap_requires_bounded_profile_before_import(monkeypatch):
    policy = _load_policy()
    monkeypatch.setenv("RAYLIGHT_INT8_BACKEND", "cuda")
    monkeypatch.setenv("RAYLIGHT_INT8_CUDA_ALLOW_UNDER_13", "1")
    monkeypatch.delenv("RAYLIGHT_H3_STOP_AFTER_FIRST_FORWARD", raising=False)
    monkeypatch.setenv("RAYLIGHT_H3_PHASE_PROFILE", "1")

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_capability():
            return (7, 5)

    class FakeTorch:
        cuda = FakeCuda()
        version = type("Version", (), {"cuda": "12.8"})()

    imported = []

    with pytest.raises(RuntimeError, match="bounded stop and phase profile"):
        policy.initialize_worker_int8_backend(
            FakeTorch(), import_kitchen=lambda name: imported.append(name)
        )
    assert imported == []


def test_worker_emits_rank_specific_backend_evidence():
    source = (ROOT / "src/raylight/distributed_worker/ray_worker.py").read_text()
    assert 'os.environ["RAYLIGHT_RANK"] = str(self.local_rank)' in source
    assert '"marker": "raylight_int8_backend_policy"' in source
    assert '"rank": self.local_rank' in source
