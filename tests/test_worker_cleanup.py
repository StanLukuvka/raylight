from __future__ import annotations

import ast
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from raylight.worker_cleanup import (
    get_results_or_shutdown,
    raise_for_shutdown_result,
    run_lifecycle_or_shutdown,
    run_worker_shutdown,
    shutdown_workers,
    submit_results_or_shutdown,
)


ROOT = Path(__file__).parents[1]


class FakeRayActorError(Exception):
    pass


class FakeRemoteMethod:
    def __init__(self, ref: str, submissions: list[str]):
        self.ref = ref
        self.submissions = submissions

    def remote(self):
        self.submissions.append(self.ref)
        return self.ref


class FakeActor:
    def __init__(self, name: str, submissions: list[str]):
        self.name = name
        self.kill = FakeRemoteMethod(f"cleanup:{name}", submissions)
        self.__ray_ready__ = FakeRemoteMethod(f"confirm:{name}", submissions)


def _actor_exit(ref):
    raise FakeRayActorError(f"{ref} actor exited")


def test_shutdown_submits_all_workers_and_proves_every_exit() -> None:
    submissions: list[str] = []
    waits: list[tuple[list[str], int, float]] = []
    forced: list[tuple[str, bool]] = []
    actors = [FakeActor("rank-0", submissions), FakeActor("rank-1", submissions)]

    def wait(refs, num_returns, timeout):
        waits.append((refs, num_returns, timeout))
        return refs, []

    ray_module = SimpleNamespace(
        wait=wait,
        get=_actor_exit,
        kill=lambda actor, no_restart: forced.append((actor.name, no_restart)),
        exceptions=SimpleNamespace(RayActorError=FakeRayActorError),
    )

    result = shutdown_workers(
        ray_module,
        actors,
        timeout_seconds=30.0,
        force_timeout_seconds=5.0,
    )

    assert submissions == ["cleanup:rank-0", "cleanup:rank-1"]
    assert waits == [
        (["cleanup:rank-0", "cleanup:rank-1"], 2, 30.0),
    ]
    assert forced == []
    assert result == {"status": "clean", "worker_count": 2}


def test_cleanup_submission_failure_does_not_skip_later_workers() -> None:
    submissions: list[str] = []

    class FailingCleanup:
        @staticmethod
        def remote():
            submissions.append("cleanup-attempt:rank-0")
            raise RuntimeError("rank 0 cleanup submission failed")

    rank_0 = SimpleNamespace(
        name="rank-0",
        kill=FailingCleanup(),
        __ray_ready__=FakeRemoteMethod("confirm:rank-0", submissions),
    )
    rank_1 = FakeActor("rank-1", submissions)
    actors = [rank_0, rank_1]

    ray_module = SimpleNamespace(
        wait=lambda refs, num_returns, timeout: (refs, []),
        get=_actor_exit,
        kill=lambda actor, no_restart: submissions.append(f"force:{actor.name}"),
        exceptions=SimpleNamespace(RayActorError=FakeRayActorError),
    )

    result = shutdown_workers(
        ray_module,
        actors,
        timeout_seconds=0.01,
        force_timeout_seconds=0.01,
    )

    assert submissions[:2] == ["cleanup-attempt:rank-0", "cleanup:rank-1"]
    assert result["status"] == "forced"


def test_pending_cleanup_force_kills_and_confirms_every_actor_dead() -> None:
    submissions: list[str] = []
    forced: list[tuple[str, bool]] = []
    actors = [FakeActor("rank-0", submissions), FakeActor("rank-1", submissions)]
    wait_calls = 0

    def wait(refs, num_returns, timeout):
        nonlocal wait_calls
        wait_calls += 1
        if wait_calls == 1:
            return [refs[0]], [refs[1]]
        return refs, []

    ray_module = SimpleNamespace(
        wait=wait,
        get=_actor_exit,
        kill=lambda actor, no_restart: forced.append((actor.name, no_restart)),
        exceptions=SimpleNamespace(RayActorError=FakeRayActorError),
    )

    result = shutdown_workers(
        ray_module,
        actors,
        timeout_seconds=5.0,
        force_timeout_seconds=2.0,
    )

    assert forced == [("rank-0", True), ("rank-1", True)]
    assert submissions == [
        "cleanup:rank-0",
        "cleanup:rank-1",
        "confirm:rank-0",
        "confirm:rank-1",
    ]
    assert result == {
        "status": "forced",
        "worker_count": 2,
        "error_type": "TimeoutError",
        "unconfirmed_worker_count": 0,
    }


def test_cleanup_task_failure_force_kills_every_actor() -> None:
    submissions: list[str] = []
    forced: list[tuple[str, bool]] = []
    actors = [FakeActor("rank-0", submissions), FakeActor("rank-1", submissions)]

    def get(ref):
        if ref == "cleanup:rank-0":
            raise RuntimeError("resource release failed")
        return _actor_exit(ref)

    ray_module = SimpleNamespace(
        wait=lambda refs, num_returns, timeout: (refs, []),
        get=get,
        kill=lambda actor, no_restart: forced.append((actor.name, no_restart)),
        exceptions=SimpleNamespace(RayActorError=FakeRayActorError),
    )

    result = shutdown_workers(
        ray_module,
        actors,
        timeout_seconds=5.0,
        force_timeout_seconds=2.0,
    )

    assert result == {
        "status": "forced",
        "worker_count": 2,
        "error_type": "RuntimeError",
        "unconfirmed_worker_count": 0,
    }
    assert forced == [("rank-0", True), ("rank-1", True)]


def test_explicit_shutdown_re_raises_cleanup_failure_after_confirmed_force() -> None:
    failure = RuntimeError("resource release failed")
    result = {
        "status": "forced",
        "worker_count": 2,
        "error_type": "RuntimeError",
        "unconfirmed_worker_count": 0,
        "_failure": failure,
    }

    with pytest.raises(RuntimeError, match="resource release failed") as caught:
        raise_for_shutdown_result(result)

    assert caught.value is failure


def test_explicit_shutdown_fails_when_actor_death_is_unconfirmed() -> None:
    failure = TimeoutError("cleanup deadline expired")
    result = {
        "status": "force_unconfirmed",
        "worker_count": 2,
        "error_type": "TimeoutError",
        "unconfirmed_worker_count": 1,
        "_failure": failure,
    }

    with pytest.raises(RuntimeError, match="could not be confirmed") as caught:
        raise_for_shutdown_result(result)

    assert caught.value.__cause__ is failure


def test_force_kill_itself_is_bounded_and_reports_unconfirmed_workers() -> None:
    submissions: list[str] = []
    actors = [FakeActor("rank-0", submissions), FakeActor("rank-1", submissions)]
    never = threading.Event()

    ray_module = SimpleNamespace(
        wait=lambda refs, num_returns, timeout: ([], refs),
        get=_actor_exit,
        kill=lambda actor, no_restart: never.wait(1.0),
        exceptions=SimpleNamespace(RayActorError=FakeRayActorError),
    )

    result = shutdown_workers(
        ray_module,
        actors,
        timeout_seconds=0.01,
        force_timeout_seconds=0.01,
    )

    assert result == {
        "status": "force_unconfirmed",
        "worker_count": 2,
        "error_type": "TimeoutError",
        "unconfirmed_worker_count": 2,
    }


def test_sampling_failure_runs_bounded_shutdown_and_preserves_original_error() -> None:
    submissions: list[str] = []
    actors = [FakeActor("rank-0", submissions), FakeActor("rank-1", submissions)]

    def get(ref):
        if isinstance(ref, list):
            raise LookupError("rank 1 failed during sampling")
        return _actor_exit(ref)

    ray_module = SimpleNamespace(
        get=get,
        wait=lambda refs, num_returns, timeout: (refs, []),
        kill=lambda actor, no_restart: None,
        exceptions=SimpleNamespace(RayActorError=FakeRayActorError),
    )

    with pytest.raises(LookupError, match="rank 1 failed during sampling"):
        get_results_or_shutdown(
            ray_module,
            ["sample:rank-0", "sample:rank-1"],
            actors,
            timeout_seconds=12.0,
            force_timeout_seconds=2.0,
        )

    assert submissions == ["cleanup:rank-0", "cleanup:rank-1"]


def test_thread_start_failure_cannot_replace_sampling_error(monkeypatch) -> None:
    submissions: list[str] = []
    actors = [FakeActor("rank-0", submissions), FakeActor("rank-1", submissions)]

    def get(ref):
        if isinstance(ref, list):
            raise LookupError("sampling failed first")
        return _actor_exit(ref)

    ray_module = SimpleNamespace(
        get=get,
        wait=lambda refs, num_returns, timeout: ([], refs),
        kill=lambda actor, no_restart: None,
        exceptions=SimpleNamespace(RayActorError=FakeRayActorError),
    )
    monkeypatch.setattr(
        threading.Thread,
        "start",
        lambda self: (_ for _ in ()).throw(RuntimeError("cannot create thread")),
    )

    with pytest.raises(LookupError, match="sampling failed first"):
        get_results_or_shutdown(
            ray_module,
            ["sample:rank-0", "sample:rank-1"],
            actors,
            timeout_seconds=0.01,
            force_timeout_seconds=0.01,
        )


def test_partial_sampling_submission_failure_cleans_every_worker() -> None:
    submissions: list[str] = []
    actors = [FakeActor("rank-0", submissions), FakeActor("rank-1", submissions)]

    def submit(index, actor):
        submissions.append(f"sample:{actor.name}")
        if index == 1:
            raise ValueError("rank 1 submission failed")
        return f"future:{actor.name}"

    ray_module = SimpleNamespace(
        wait=lambda refs, num_returns, timeout: (refs, []),
        get=_actor_exit,
        kill=lambda actor, no_restart: None,
        exceptions=SimpleNamespace(RayActorError=FakeRayActorError),
    )

    with pytest.raises(ValueError, match="rank 1 submission failed"):
        submit_results_or_shutdown(
            ray_module,
            actors,
            submit,
            timeout_seconds=0.01,
            force_timeout_seconds=0.01,
        )

    assert submissions[:4] == [
        "sample:rank-0",
        "sample:rank-1",
        "cleanup:rank-0",
        "cleanup:rank-1",
    ]


@pytest.mark.parametrize(
    ("fault_name", "lifecycle", "error_type", "message"),
    [
        (
            "empty_results",
            lambda: [][0],
            IndexError,
            "list index out of range",
        ),
        (
            "malformed_results",
            lambda: (_ for _ in ()).throw(
                ValueError("not enough values to unpack from malformed result")
            ),
            ValueError,
            "not enough values to unpack",
        ),
        (
            "grouped_result_validation",
            lambda: (_ for _ in ()).throw(RuntimeError("missing outputs for dp_rank [1]")),
            RuntimeError,
            "missing outputs for dp_rank",
        ),
        (
            "pre_dispatch_metadata_rpc",
            lambda: (_ for _ in ()).throw(LookupError("metadata RPC failed")),
            LookupError,
            "metadata RPC failed",
        ),
    ],
)
def test_complete_sampling_lifecycle_faults_cleanup_every_worker_and_preserve_error(
    fault_name,
    lifecycle,
    error_type,
    message,
) -> None:
    submissions: list[str] = []
    actors = [FakeActor("rank-0", submissions), FakeActor("rank-1", submissions)]
    ray_module = SimpleNamespace(
        wait=lambda refs, num_returns, timeout: (refs, []),
        get=_actor_exit,
        kill=lambda actor, no_restart: None,
        exceptions=SimpleNamespace(RayActorError=FakeRayActorError),
    )

    with pytest.raises(error_type, match=message):
        run_lifecycle_or_shutdown(
            ray_module,
            actors,
            lifecycle,
            timeout_seconds=0.01,
            force_timeout_seconds=0.01,
        )

    assert submissions == ["cleanup:rank-0", "cleanup:rank-1"], fault_name


def test_ray_shutdown_failure_is_secondary_to_confirmed_force_cleanup_error() -> None:
    cleanup_failure = RuntimeError("worker resource release failed")
    ray_shutdown_failure = OSError("ray.shutdown failed")
    result = {
        "status": "forced",
        "worker_count": 2,
        "error_type": "RuntimeError",
        "unconfirmed_worker_count": 0,
        "_failure": cleanup_failure,
    }

    with pytest.raises(RuntimeError, match="worker resource release failed") as caught:
        raise_for_shutdown_result(result, secondary_error=ray_shutdown_failure)

    assert caught.value is cleanup_failure
    assert caught.value._raylight_secondary_error is ray_shutdown_failure


def test_ray_shutdown_failure_is_secondary_to_unconfirmed_worker_death() -> None:
    cleanup_failure = TimeoutError("cleanup deadline expired")
    ray_shutdown_failure = OSError("ray.shutdown failed")
    result = {
        "status": "force_unconfirmed",
        "worker_count": 2,
        "error_type": "TimeoutError",
        "unconfirmed_worker_count": 1,
        "_failure": cleanup_failure,
    }

    with pytest.raises(RuntimeError, match="could not be confirmed") as caught:
        raise_for_shutdown_result(result, secondary_error=ray_shutdown_failure)

    assert caught.value.__cause__ is cleanup_failure
    assert caught.value._raylight_secondary_error is ray_shutdown_failure


def test_clean_worker_shutdown_still_reports_ray_shutdown_failure() -> None:
    ray_shutdown_failure = OSError("ray.shutdown failed")

    with pytest.raises(OSError, match="ray.shutdown failed") as caught:
        raise_for_shutdown_result(
            {"status": "clean", "worker_count": 2},
            secondary_error=ray_shutdown_failure,
        )

    assert caught.value is ray_shutdown_failure


def test_worker_shutdown_destroys_group_but_does_not_exit_after_release_failure() -> None:
    calls: list[str] = []

    def fail_release():
        calls.append("release")
        raise RuntimeError("resource release failed")

    with pytest.raises(RuntimeError, match="resource release failed"):
        run_worker_shutdown(
            release_resources=fail_release,
            destroy_process_group=lambda: calls.append("destroy"),
            exit_actor=lambda: calls.append("exit"),
        )

    assert calls == ["release", "destroy"]


def test_worker_shutdown_does_not_exit_after_group_destroy_failure() -> None:
    calls: list[str] = []

    def fail_destroy():
        calls.append("destroy")
        raise RuntimeError("process group destroy failed")

    with pytest.raises(RuntimeError, match="process group destroy failed"):
        run_worker_shutdown(
            release_resources=lambda: calls.append("release"),
            destroy_process_group=fail_destroy,
            exit_actor=lambda: calls.append("exit"),
        )

    assert calls == ["release", "destroy"]


def test_worker_shutdown_exits_only_after_successful_cleanup() -> None:
    calls: list[str] = []

    def exit_actor():
        calls.append("exit")
        raise SystemExit("intentional actor exit")

    with pytest.raises(SystemExit, match="intentional actor exit"):
        run_worker_shutdown(
            release_resources=lambda: calls.append("release"),
            destroy_process_group=lambda: calls.append("destroy"),
            exit_actor=exit_actor,
        )

    assert calls == ["release", "destroy", "exit"]


def test_production_worker_and_sampler_paths_use_bounded_cleanup() -> None:
    nodes = (ROOT / "src/raylight/nodes.py").read_text()
    custom_sampler = (
        ROOT / "src/raylight/comfy_extra_dist/nodes_custom_sampler.py"
    ).read_text()
    worker = (
        ROOT / "src/raylight/distributed_worker/ray_worker.py"
    ).read_text()

    direct_driver_waits = []
    for class_node in (
        node for node in ast.parse(nodes).body if isinstance(node, ast.ClassDef)
    ):
        for function_node in (
            node for node in class_node.body if isinstance(node, ast.FunctionDef)
        ):
            for call in (
                node for node in ast.walk(function_node) if isinstance(node, ast.Call)
            ):
                if (
                    isinstance(call.func, ast.Attribute)
                    and call.func.attr == "get"
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "ray"
                    and call.args
                    and isinstance(call.args[0], ast.Name)
                    and call.args[0].id == "futures"
                ):
                    direct_driver_waits.append((class_node.name, function_node.name))

    assert direct_driver_waits == [
        ("RayControlNetLoader", "load_controlnet"),
        ("RayVAEDecodeDistributed", "ray_decode"),
    ]
    assert "ray.get(futures)" not in custom_sampler
    assert "get_results_or_shutdown" not in nodes
    assert "get_results_or_shutdown" not in custom_sampler
    assert "submit_results_or_shutdown" not in nodes
    assert "submit_results_or_shutdown" not in custom_sampler
    assert nodes.count("run_lifecycle_or_shutdown(") == 3
    assert custom_sampler.count("run_lifecycle_or_shutdown(") == 6
    assert nodes.count("submit_results(") == 3
    assert custom_sampler.count("submit_results(") == 6
    assert '"shutdown_after_sampling": ("BOOLEAN", {"default": False})' in custom_sampler
    assert "if shutdown_after_sampling:" in custom_sampler
    assert "shutdown_workers(ray, gpu_actors, timeout_seconds=30.0)" in custom_sampler
    assert "retain_failure=True" in nodes
    assert "raise_for_shutdown_result(result, secondary_error=cluster_shutdown_error)" in nodes
    assert 'shutdown_result["status"] == "force_unconfirmed"' in custom_sampler
    assert "run_worker_shutdown(" in worker
