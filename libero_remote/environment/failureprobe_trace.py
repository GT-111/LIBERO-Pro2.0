"""Opt-in worker-local oracle logging; never part of observations or policy input."""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported oracle value: {type(value).__name__}")


def runtime_sources() -> Dict[str, Any]:
    paths = {
        "service_adapter": Path(__file__).with_name("libero.py"),
        "trace_helper": Path(__file__),
    }
    for key, module_name in (
        ("bddl", "libero.libero.envs.bddl_base_domain"),
        ("perturbations", "libero.libero.envs.nonphysical_perturbations"),
        ("worker", "libero_remote.worker.process"),
    ):
        module = sys.modules.get(module_name)
        location = getattr(module, "__file__", None)
        if location:
            paths[key] = Path(location)
    sources = {
        key: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for key, path in paths.items()
    }
    expected_path = os.environ.get("FAILUREPROBE_EXPECTED_RUNTIME")
    if expected_path:
        expected = json.loads(Path(expected_path).read_text(encoding="utf-8"))
        for key, reference in expected["sources"].items():
            if key not in sources or sources[key]["sha256"] != reference["sha256"]:
                raise RuntimeError(f"FailureProbe worker imported the wrong {key} source")
        return {
            "sources": sources,
            "fork_commit": expected["fork_commit"],
            "upstream_baseline": expected["upstream_baseline"],
        }
    return {"sources": sources, "fork_commit": None}


class OracleTrace:
    def __init__(self, spec: Dict[str, Any]) -> None:
        root = os.environ.get("FAILUREPROBE_ORACLE_DIR")
        self.path: Optional[Path] = None
        self.session_id = spec.get("_failureprobe_session_id")
        self.reset_index = -1
        self.spec = {key: value for key, value in spec.items() if not key.startswith("_")}
        if root:
            if not self.session_id or not str(self.session_id).startswith("ses_"):
                raise RuntimeError("Oracle capture needs the server-assigned session identity")
            self.path = Path(root) / (str(self.session_id) + ".jsonl")
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: str, **fields: Any) -> None:
        if self.path is None:
            return
        row = {
            "schema_version": "failureprobe.pro2.oracle.v1",
            "event": event,
            "remote_session_id": self.session_id,
            "reset_index": self.reset_index,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    row, default=_json_value, ensure_ascii=False, allow_nan=False, sort_keys=True
                )
                + "\n"
            )
            stream.flush()

    def reset(self, init_state_id: int, seed: int, environment: Any, **fields: Any) -> None:
        self.reset_index += 1
        if self.path is not None:
            self.record(
                "reset",
                init_state_id=init_state_id,
                seed=seed,
                spec=self.spec,
                runtime=runtime_sources(),
                sim_state=environment.get_sim_state(),
                **fields,
            )
