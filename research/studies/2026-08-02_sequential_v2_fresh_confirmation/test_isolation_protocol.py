"""Simulation-free contract tests for the fresh-arm isolation protocol."""

from __future__ import annotations

import ast
import copy
import sys
import unittest
from pathlib import Path


STUDY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY_DIR))
import isolation_protocol as protocol  # noqa: E402


class IsolationProtocolTests(unittest.TestCase):
    def sample_job(self):
        return protocol.seal_job({
            "arm": "primary",
            "candidate_count": 24,
            "state": {"base_seed": 17, "pick_slot": 6},
            "configuration": {},
            "source_fingerprints": {"run_study.py": "abc"},
            "runtime_fingerprint": {"native_binaries": {"manifest_sha256": "def"}},
        })

    def sample_result(self, job):
        return protocol.seal_result({
            "job_sha256": job["job_sha256"],
            "arm": job["arm"],
            "candidate_count": job["candidate_count"],
            "state": job["state"],
            "source_fingerprints": job["source_fingerprints"],
            "runtime_fingerprint": job["runtime_fingerprint"],
            "observed_worker_environment": dict(protocol.WORKER_ENVIRONMENT),
        })

    def test_job_and_result_hashes_fail_closed(self):
        job = self.sample_job()
        protocol.validate_job(job)
        result = self.sample_result(job)
        protocol.validate_result_envelope(result, job)
        corrupted = copy.deepcopy(result)
        corrupted["candidate_count"] = 32
        with self.assertRaises(ValueError):
            protocol.validate_result_envelope(corrupted, job)

    def test_worker_environment_overrides_parent_values(self):
        environment = protocol.worker_environment({"OPENBLAS_NUM_THREADS": "8"})
        self.assertEqual(
            {key: environment[key] for key in protocol.WORKER_ENVIRONMENT},
            protocol.WORKER_ENVIRONMENT,
        )

    def test_controller_has_no_pool_or_retry_construct(self):
        source = (STUDY_DIR / "run_study.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertNotIn("ProcessPoolExecutor", names)
        self.assertNotIn("ThreadPoolExecutor", names)
        runner = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "run_arm_subprocess"
        )
        self.assertFalse(
            any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(runner))
        )
        controller_main = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        called_names = {
            node.func.id
            for node in ast.walk(controller_main)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("run_policy", called_names)
        self.assertIn("run_arm_subprocess", called_names)


if __name__ == "__main__":
    unittest.main()
