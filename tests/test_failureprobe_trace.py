import json
from types import SimpleNamespace

import numpy as np
import pytest

from libero_remote.environment.failureprobe_trace import OracleTrace, runtime_sources
from libero_remote.environment.libero import LiberoEnvironmentAdapter


def _observation():
    return {
        "agentview_image": np.zeros((16, 16, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((16, 16, 3), dtype=np.uint8),
        "robot0_joint_pos": np.zeros(7),
        "robot0_joint_vel": np.zeros(7),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0, 0, 0, 1]),
        "robot0_gripper_qpos": np.zeros(2),
    }


class FakeEnvironment:
    language_instruction = "pick up the object"
    sim = SimpleNamespace(
        data=SimpleNamespace(
            ncon=1,
            contact=[
                SimpleNamespace(
                    geom1=1, geom2=2, dist=-0.001, pos=np.zeros(3), frame=np.eye(3).flatten()
                )
            ],
        )
    )

    def get_sim_state(self):
        return np.array([1.0, 2.0])

    def check_success(self):
        return False

    def step(self, action):
        return (
            _observation(),
            0.0,
            False,
            {
                "perturbation": {"case_id": "D08", "events": [{"step": 0, "event": "injected"}]},
                "failureprobe_commands": {
                    "requested_action": action.copy(),
                    "transformed_action": -action.copy(),
                },
            },
        )


def test_nested_oracle_is_saved_before_scalar_filter(tmp_path, monkeypatch):
    monkeypatch.setenv("FAILUREPROBE_ORACLE_DIR", str(tmp_path))
    trace = OracleTrace({"suite": "libero_object", "_failureprobe_session_id": "ses_test"})
    env = FakeEnvironment()
    trace.reset(3, 314, env)
    adapter = LiberoEnvironmentAdapter()
    adapter._env = env
    adapter._camera_names = ["agentview", "robot0_eye_in_hand"]
    adapter._max_steps = 280
    adapter._failureprobe_trace = trace
    action = np.arange(7, dtype=np.float32) / 10
    result = adapter.step(action)
    rows = [json.loads(line) for line in trace.path.read_text().splitlines()]
    assert rows[0]["runtime"]["sources"]["service_adapter"]["sha256"]
    assert rows[1]["remote_session_id"] == "ses_test"
    assert rows[1]["step"] == 0 and rows[1]["env_step"] == 1
    assert rows[1]["info"]["perturbation"]["events"][0]["event"] == "injected"
    assert rows[1]["info"]["failureprobe_commands"]["transformed_action"] == (-action).tolist()
    assert rows[1]["contacts"][0]["geom2"] == 2
    assert "perturbation" not in result["info"]
    assert "failureprobe_commands" not in result["info"]
    assert "failureprobe_oracle_trace" in result["info"]
    assert all("oracle" not in key and "perturbation" not in key for key in result["observation"])


def test_runtime_rejects_a_wrong_worker_source_hash(tmp_path, monkeypatch):
    expected = tmp_path / "runtime.json"
    expected.write_text(
        json.dumps(
            {
                "fork_commit": "fork",
                "upstream_baseline": {},
                "sources": {"trace_helper": {"sha256": "wrong"}},
            }
        )
    )
    monkeypatch.setenv("FAILUREPROBE_EXPECTED_RUNTIME", str(expected))
    with pytest.raises(RuntimeError, match="wrong trace_helper source"):
        runtime_sources()
