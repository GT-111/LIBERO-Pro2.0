from pathlib import Path

import numpy as np
import pytest

from libero_remote.worker.process import EnvWorker


def test_spawned_worker_preserves_environment_state_across_steps() -> None:
    worker = EnvWorker.start(backend="fake", timeout_seconds=5.0)
    try:
        created = worker.create(
            {
                "suite": "libero_goal",
                "task_id": 0,
                "init_state_id": 0,
                "seed": 42,
                "camera_names": ["agentview", "robot0_eye_in_hand"],
                "camera_height": 16,
                "camera_width": 16,
                "max_steps": 10,
            }
        )
        assert created["observation"]["robot0_joint_pos"][0] == 0.0

        first = worker.step(np.array([0.25, 0.0], dtype=np.float32))
        second = worker.step(np.array([0.25, 0.0], dtype=np.float32))

        assert first["observation"]["robot0_joint_pos"][0] == 0.25
        assert second["observation"]["robot0_joint_pos"][0] == 0.5
        assert second["worker_step_index"] == 2
    finally:
        worker.close()

    assert not worker.is_alive


@pytest.mark.parametrize("operation", ["close", "terminate"])
def test_retained_closed_workers_release_process_file_descriptors(operation: str) -> None:
    fd_directory = Path("/proc/self/fd")
    if not fd_directory.is_dir():
        pytest.skip("Linux process descriptor accounting is required")
    warmup = EnvWorker.start(backend="fake", timeout_seconds=5.0)
    warmup.close()  # Start multiprocessing's shared resource tracker before the baseline.
    baseline = len(list(fd_directory.iterdir()))
    retained = []
    for _ in range(6):
        worker = EnvWorker.start(backend="fake", timeout_seconds=5.0)
        pid = worker.pid
        getattr(worker, operation)()
        assert worker.pid == pid
        assert not worker.is_alive
        retained.append(worker)  # Session records intentionally retain these handles.
    assert len(list(fd_directory.iterdir())) <= baseline
