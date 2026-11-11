import hashlib
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


def test_reset_records_native_source_count_and_hash_separately_from_settled_state(tmp_path, monkeypatch):
    monkeypatch.setenv("FAILUREPROBE_ORACLE_DIR", str(tmp_path))

    class ResetEnvironment(FakeEnvironment):
        robots = [SimpleNamespace(controller=SimpleNamespace(use_delta=True))]

        def seed(self, seed):
            pass

        def set_perturbation_seed(self, seed):
            pass

        def reset(self):
            return _observation()

        def set_init_state(self, state):
            return _observation()

        def begin_perturbation_rollout(self):
            return _observation()

        def get_perturbation_plan(self):
            return {}

    source = tmp_path / "actual.pruned_init"
    source.write_bytes(b"source file bytes differ from the selected numerical state")
    bddl = tmp_path / "actual.bddl"
    bddl.write_bytes(b"(define (problem actual))")
    states = np.arange(12, dtype=np.float64).reshape(3, 4)
    adapter = LiberoEnvironmentAdapter()
    adapter._env = ResetEnvironment()
    adapter._env.env = SimpleNamespace(bddl_file_name=str(bddl))
    adapter._init_states = states
    adapter._initial_state_source = {
        "path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    adapter._camera_names = ["agentview", "robot0_eye_in_hand"]
    adapter._failureprobe_trace = OracleTrace({"_failureprobe_session_id": "ses_native_reset"})
    adapter.reset(2, 42)
    event = json.loads(adapter._failureprobe_trace.path.read_text())
    assert event["source_initial_state_count"] == 3
    assert event["source_initial_state_source"] == adapter._initial_state_source
    assert event["source_initial_state_sha256"] == hashlib.sha256(states[2].tobytes()).hexdigest()
    assert event["source_initial_state_dtype"] == "<f8"
    assert event["source_initial_state_shape"] == [4]
    assert event["source_initial_state_sha256"] != event["source_initial_state_source"]["sha256"]
    assert event["bddl"] == {"path": str(bddl), "sha256": hashlib.sha256(bddl.read_bytes()).hexdigest()}
    assert event["sim_state"] == [1.0, 2.0]
