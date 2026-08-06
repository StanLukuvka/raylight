from __future__ import annotations

import threading
import time


def run_worker_shutdown(
    *,
    release_resources,
    destroy_process_group,
    exit_actor,
) -> None:
    release_error = None
    destroy_error = None

    try:
        release_resources()
    except BaseException as exc:
        release_error = exc

    try:
        destroy_process_group()
    except BaseException as exc:
        destroy_error = exc

    if release_error is not None:
        if destroy_error is not None:
            raise release_error from destroy_error
        raise release_error
    if destroy_error is not None:
        raise destroy_error

    exit_actor()


def _force_kill_and_confirm(
    ray_module,
    actors,
    *,
    actor_error,
    timeout_seconds: float,
) -> int:
    deadline = time.monotonic() + timeout_seconds
    kill_errors: list[BaseException | None] = [None] * len(actors)

    def force_kill(index, actor):
        try:
            ray_module.kill(actor, no_restart=True)
        except BaseException as exc:
            kill_errors[index] = exc

    threads = []
    for index, actor in enumerate(actors):
        try:
            thread = threading.Thread(
                target=force_kill,
                args=(index, actor),
                daemon=True,
                name=f"raylight-force-kill-{index}",
            )
            thread.start()
        except BaseException as exc:
            kill_errors[index] = exc
        else:
            threads.append(thread)

    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))

    unconfirmed = sum(thread.is_alive() for thread in threads)
    unconfirmed += sum(error is not None for error in kill_errors)
    if unconfirmed:
        return unconfirmed

    confirmation_refs = []
    for actor in actors:
        try:
            confirmation_refs.append(actor.__ray_ready__.remote())
        except BaseException:
            unconfirmed += 1

    if not confirmation_refs:
        return unconfirmed

    remaining_seconds = max(0.0, deadline - time.monotonic())
    try:
        ready_refs, pending_refs = ray_module.wait(
            confirmation_refs,
            num_returns=len(confirmation_refs),
            timeout=remaining_seconds,
        )
    except BaseException:
        return unconfirmed + len(confirmation_refs)

    unconfirmed += len(pending_refs)
    for ref in ready_refs:
        try:
            ray_module.get(ref)
        except actor_error:
            continue
        except BaseException:
            unconfirmed += 1
        else:
            unconfirmed += 1
    return unconfirmed


def shutdown_workers(
    ray_module,
    actors,
    *,
    timeout_seconds: float,
    force_timeout_seconds: float = 10.0,
    retain_failure: bool = False,
) -> dict[str, object]:
    actors = list(actors)
    if not actors:
        return {"status": "clean", "worker_count": 0}

    actor_error = getattr(
        getattr(ray_module, "exceptions", None),
        "RayActorError",
        (),
    )
    cleanup_refs = []
    failure = None
    for actor in actors:
        try:
            cleanup_refs.append(actor.kill.remote())
        except BaseException as exc:
            if failure is None:
                failure = exc

    if cleanup_refs:
        try:
            ready_refs, pending_refs = ray_module.wait(
                cleanup_refs,
                num_returns=len(cleanup_refs),
                timeout=timeout_seconds,
            )
        except BaseException as exc:
            if failure is None:
                failure = exc
        else:
            if pending_refs and failure is None:
                failure = TimeoutError(
                    f"{len(pending_refs)} worker cleanup RPCs exceeded the shutdown deadline"
                )
            for ref in ready_refs:
                try:
                    ray_module.get(ref)
                except actor_error:
                    continue
                except BaseException as exc:
                    if failure is None:
                        failure = exc
                else:
                    if failure is None:
                        failure = RuntimeError(
                            "worker cleanup RPC returned without terminating its actor"
                        )

    if failure is None:
        return {"status": "clean", "worker_count": len(actors)}

    force_error_type = None
    try:
        unconfirmed = _force_kill_and_confirm(
            ray_module,
            actors,
            actor_error=actor_error,
            timeout_seconds=force_timeout_seconds,
        )
    except BaseException as exc:
        unconfirmed = len(actors)
        force_error_type = type(exc).__name__
    result = {
        "status": "forced" if unconfirmed == 0 else "force_unconfirmed",
        "worker_count": len(actors),
        "error_type": type(failure).__name__,
        "unconfirmed_worker_count": unconfirmed,
    }
    if force_error_type is not None:
        result["force_error_type"] = force_error_type
    if retain_failure:
        result["_failure"] = failure
    return result


def _record_secondary_error(authoritative, secondary_error) -> None:
    if not isinstance(secondary_error, BaseException):
        return
    try:
        authoritative._raylight_secondary_error = secondary_error
        add_note = getattr(authoritative, "add_note", None)
        if callable(add_note):
            add_note(
                f"Secondary cluster shutdown failure: "
                f"{type(secondary_error).__name__}: {secondary_error}"
            )
    except BaseException:
        pass


def raise_for_shutdown_result(result, *, secondary_error=None) -> None:
    status = result.get("status")
    failure = result.get("_failure")
    if status == "force_unconfirmed":
        error = RuntimeError(
            f"Worker termination could not be confirmed for "
            f"{result.get('unconfirmed_worker_count', 'unknown')} actor(s)"
        )
        _record_secondary_error(error, secondary_error)
        if isinstance(failure, BaseException):
            raise error from failure
        raise error
    if status == "forced":
        if isinstance(failure, BaseException):
            _record_secondary_error(failure, secondary_error)
            raise failure
        error = RuntimeError("Worker cleanup failed and required force termination")
        _record_secondary_error(error, secondary_error)
        raise error
    if isinstance(secondary_error, BaseException):
        raise secondary_error


def run_lifecycle_or_shutdown(
    ray_module,
    actors,
    lifecycle,
    *,
    timeout_seconds: float,
    force_timeout_seconds: float = 10.0,
):
    """Run a complete worker lifecycle under one workload-error authority."""
    actors = list(actors)
    try:
        return lifecycle()
    except BaseException:
        try:
            shutdown_workers(
                ray_module,
                actors,
                timeout_seconds=timeout_seconds,
                force_timeout_seconds=force_timeout_seconds,
            )
        except BaseException:
            pass
        raise


def submit_results(ray_module, actors, submit):
    """Submit every worker RPC before waiting for the result collection."""
    futures = []
    for index, actor in enumerate(actors):
        futures.append(submit(index, actor))
    return ray_module.get(futures)


def get_results_or_shutdown(
    ray_module,
    futures,
    actors,
    *,
    timeout_seconds: float,
    force_timeout_seconds: float = 10.0,
):
    return run_lifecycle_or_shutdown(
        ray_module,
        actors,
        lambda: ray_module.get(futures),
        timeout_seconds=timeout_seconds,
        force_timeout_seconds=force_timeout_seconds,
    )


def submit_results_or_shutdown(
    ray_module,
    actors,
    submit,
    *,
    timeout_seconds: float,
    force_timeout_seconds: float = 10.0,
):
    actors = list(actors)
    return run_lifecycle_or_shutdown(
        ray_module,
        actors,
        lambda: submit_results(ray_module, actors, submit),
        timeout_seconds=timeout_seconds,
        force_timeout_seconds=force_timeout_seconds,
    )
