"""Inactive sessions retain queryable metadata, not per-step RGB payloads."""

import asyncio
import gc
import weakref
from typing import Any

import pytest

from libero_remote.protocol.models import (
    CreateSessionRequest,
    ErrorCode,
    ResetRequest,
    StepRequest,
)
from libero_remote.server.errors import PlatformError
from libero_remote.server.session_manager import SessionManager
from libero_remote.worker.process import WorkerError


@pytest.mark.parametrize(
    "route",
    ["close", "close_error", "expire", "expire_error", "step_error", "reset_error", "reset"],
)
def test_lifecycle_releases_previous_rgb_payloads(route: str, monkeypatch: Any) -> None:
    async def scenario() -> None:
        now = [100.0]
        manager = SessionManager(
            backend="fake",
            max_sessions=1,
            worker_timeout_seconds=5.0,
            request_cache_size=16,
            session_idle_timeout_seconds=10.0,
            clock=lambda: now[0],
        )
        created = await manager.create(
            CreateSessionRequest(camera_height=256, camera_width=256, max_steps=32)
        )
        session_id = created["session_id"]
        record = manager._sessions[session_id]
        try:
            for step in range(16):
                await manager.step(
                    session_id,
                    StepRequest(
                        episode_id=created["episode_id"],
                        expected_step_index=step,
                        request_id=f"step-{step}",
                        action=[0.0, 0.0],
                    ),
                )
            images = [
                weakref.ref(image)
                for entry in record.cache.values()
                for key, image in entry.response["observation"].items()
                if key.endswith("_image")
            ]
            assert len(images) == 32
            assert all(image() is not None for image in images)

            if route in {"close_error", "expire_error"}:

                def fail_close() -> None:
                    raise WorkerError("forced close error")

                monkeypatch.setattr(record.worker, "close", fail_close)

            if route == "close":
                await manager.close(session_id)
                await manager.close(session_id)
            elif route == "close_error":
                with pytest.raises(WorkerError, match="forced close error"):
                    await manager.close(session_id)
                await manager.close(session_id)
            elif route.startswith("expire"):
                now[0] = 111.0
                assert await manager.expire_idle_sessions() == 1
                assert await manager.expire_idle_sessions() == 0
            elif route in {"step_error", "reset_error"}:
                record.worker.terminate()
                command = StepRequest(
                    episode_id=created["episode_id"],
                    expected_step_index=16,
                    request_id="failed-request",
                    action=[0.0, 0.0],
                )
                with pytest.raises(PlatformError) as failed:
                    if route == "step_error":
                        await manager.step(session_id, command)
                    else:
                        await manager.reset(session_id, ResetRequest(init_state_id=0, seed=7))
                assert failed.value.code == ErrorCode.WORKER_FAILED
                if route == "step_error":
                    with pytest.raises(PlatformError) as retried:
                        await manager.step(session_id, command)
                    assert retried.value.code == failed.value.code
                    assert retried.value.message == failed.value.message
                    assert len(record.cache) == 1
                    assert record.cache[command.request_id].response is None
            else:
                reset = await manager.reset(session_id, ResetRequest(init_state_id=0, seed=7))
                assert reset["episode_id"] != created["episode_id"]
                assert record.last_committed is None

            await asyncio.sleep(0)
            gc.collect()
            assert all(image() is None for image in images)
            assert all(entry.response is None for entry in record.cache.values())
            if route != "reset":
                assert not record.worker.is_alive
                assert "observation" not in (record.last_committed or {})
            snapshot = await manager.get(session_id)
            assert snapshot["session_id"] == session_id
            assert snapshot["last_request_id"] == (None if route == "reset" else "step-15")
        finally:
            await manager.close_all()

    asyncio.run(scenario())


@pytest.mark.parametrize("finished_by", ["success", "cap"])
@pytest.mark.parametrize("release_by", ["close", "expire"])
def test_result_queries_survive_payload_release(finished_by: str, release_by: str) -> None:
    async def scenario() -> None:
        now = [100.0]
        manager = SessionManager(
            backend="fake",
            worker_timeout_seconds=5.0,
            session_idle_timeout_seconds=10.0,
            clock=lambda: now[0],
        )
        created = await manager.create(
            CreateSessionRequest(camera_height=16, camera_width=16, max_steps=1)
        )
        session_id = created["session_id"]
        command = StepRequest(
            episode_id=created["episode_id"],
            expected_step_index=0,
            request_id="final-step",
            action=[1.0 if finished_by == "success" else 0.0, 0.0],
        )
        try:
            first = await manager.step(session_id, command)
            replay = await manager.step(session_id, command)
            assert replay is first  # Final-action retries remain available until close/expiry.
            expected = await manager.episode_outcome(session_id)
            if release_by == "close":
                await manager.close(session_id)
                await manager.close(session_id)
            else:
                now[0] = 111.0
                assert await manager.expire_idle_sessions() == 1
            assert await manager.episode_outcome(session_id) == expected
            assert (await manager.get(session_id))["last_request_id"] == "final-step"
            with pytest.raises(PlatformError) as inactive:
                await manager.step(session_id, command)
            assert inactive.value.code == ErrorCode.SESSION_NOT_ACTIVE
            record = manager._sessions[session_id]
            assert not record.cache
            assert set(record.last_committed) == {
                "request_id",
                "reward",
                "success",
                "terminated",
                "truncated",
            }
        finally:
            await manager.close_all()

    asyncio.run(scenario())
