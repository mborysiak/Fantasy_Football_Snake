"""Synthetic, simulation-free tests for cross-process result aggregation."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


STUDY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY_DIR))
import run_study  # noqa: E402


def tensor(player):
    return {
        "components": {
            "rooms": {"sha256": f"rooms-{player}"},
            "values": {"sha256": f"values-{player}"},
            "stack_utilities": {"sha256": f"stack-{player}"},
        },
        "payload_sha256": f"tensor-{player}",
    }


def arm_payload(arm, players, action, reference_matrix):
    tensors = {
        bank: {player: tensor(player) for player in players}
        for bank in ("candidate", "decision", "audit")
    }
    scores = {player: float(index) for index, player in enumerate(players, start=1)}
    reference_scores = {
        player: float(index * 2) for index, player in enumerate(players, start=1)
    }
    alignment = {
        "rooms": [0, 1],
        "banks": {
            "candidate": {
                "candidate_count": len(players),
                "values_shape": [2, 3],
                "stack_shape": [2],
            },
            "decision": {
                "candidate_count": len(players),
                "values_shape": [2, 4],
                "stack_shape": [2],
            },
            "audit": {
                "candidate_count": len(players),
                "values_shape": [2, 4],
                "stack_shape": [2],
            },
        },
        "policy_path_room_indices": [0, 1],
    }
    alignment["receipt_sha256"] = run_study.protocol.stable_json_hash(alignment)
    return {
        "arm": arm,
        "candidate_count": len(players),
        "players": list(players),
        "all_rooms_complete": True,
        "production_d128_prefix_pass": True,
        "bank_disjointness_pass": True,
        "canonical_room_alignment_pass": True,
        "canonical_room_alignment_receipt": alignment,
        "scenario_banks": {
            "construction_ppg_columns": [0],
            "evaluation_ppg_columns": [1],
            "decision_ppg_columns": [2, 3],
            "audit_ppg_columns": [4, 5, 6, 7],
        },
        "decision_superbank": [2, 3],
        "reference_bank": [4, 5, 6, 7],
        "draft_room_adp_columns": [[1, 2], [3, 4]],
        "path_hashes": {player: f"path-{player}" for player in players},
        "tensor_fingerprints": tensors,
        "d128_action": action,
        "d256_action": action,
        "d128_scores": scores,
        "d256_scores": scores,
        "reference_scores": reference_scores,
        "reference_half_1_scores": reference_scores,
        "reference_half_2_scores": reference_scores,
        "current_pick_ev": scores,
        "selected_reference_adjusted_matrices": {action: reference_matrix},
        "job_sha256": f"job-{arm}",
        "result_sha256": f"result-{arm}",
        "runtime_fingerprint": {
            "native_binaries": {"manifest_sha256": "native"}
        },
        "policy_seconds": 4.0,
        "reference_seconds": 2.0,
        "worker_wall_seconds": 7.0,
        "_controller_parent_wall_seconds": 8.0,
        "worker_pid": 100 if arm == "primary" else 101,
    }


class ControllerContractTests(unittest.TestCase):
    def test_confirmation_design_is_pinned_to_canonical_results_directory(self):
        args = SimpleNamespace(
            base_seed_values=list(run_study.FRESH_BASE_SEEDS),
            slot_values=list(run_study.FROZEN_SLOTS),
            completed_pick_values=list(run_study.FROZEN_COMPLETED_PICKS),
            year=2026,
            teams=12,
            rounds=20,
            rooms=24,
            primary_candidates=24,
            wide_candidates=32,
            construction_samples=16,
            evaluation_samples=64,
            control_decision_samples=128,
            expanded_decision_samples=256,
            reference_samples=512,
            fail_fast=True,
            prelaunch_stress_dir=run_study.DEFAULT_PRELAUNCH_STRESS_DIR,
            output_dir=run_study.CANONICAL_RESULTS_DIR,
            db=run_study.DEFAULT_DB,
        )
        self.assertTrue(run_study.frozen_design_exact(args))
        with tempfile.TemporaryDirectory() as temp_dir:
            args.output_dir = Path(temp_dir)
            self.assertFalse(run_study.frozen_design_exact(args))

    def setUp(self):
        self.args = SimpleNamespace(
            primary_candidates=2,
            wide_candidates=3,
            rooms=2,
            evaluation_samples=3,
            expanded_decision_samples=4,
            reference_samples=4,
        )
        self.primary = arm_payload(
            "primary", ["A", "B"], "B", [[10.0, 12.0], [14.0, 16.0]]
        )
        self.wide = arm_payload(
            "wide", ["A", "B", "C"], "C", [[11.0, 13.0], [15.0, 17.0]]
        )

    def test_overlap_tensor_hash_mismatch_fails_closed(self):
        corrupted = copy.deepcopy(self.wide)
        corrupted["tensor_fingerprints"]["audit"]["A"][
            "payload_sha256"
        ] = "different"
        with self.assertRaises(AssertionError):
            run_study.assert_contracts(self.primary, corrupted, self.args)

    def test_path_hash_preserves_container_types(self):
        self.assertNotEqual(
            run_study.exact_structure_hash({"path": ["A", "B"]}),
            run_study.exact_structure_hash({"path": ("A", "B")}),
        )

    def test_canonical_room_alignment_rejects_reordered_rooms(self):
        args = SimpleNamespace(
            rooms=2,
            evaluation_samples=3,
            expanded_decision_samples=4,
            reference_samples=5,
        )
        result = run_study.pd.DataFrame({
            "player": ["A"],
            "PolicyCompletedRooms": [2],
        })

        def payload(samples):
            return {
                "rooms": run_study.np.array([0, 1], dtype=run_study.np.int64),
                "values": run_study.np.ones((2, samples), dtype=run_study.np.float32),
                "stack_utilities": run_study.np.zeros(2, dtype=run_study.np.float32),
            }

        result.attrs["candidate_value_matrices"] = {"A": payload(3)}
        result.attrs["decision_value_matrices"] = {"A": payload(4)}
        result.attrs["audit_value_matrices"] = {"A": payload(5)}
        result.attrs["policy_paths"] = {
            "A": [{"room_idx": 0}, {"room_idx": 1}]
        }
        receipt = run_study.validate_canonical_room_alignment(result, args)
        self.assertEqual(receipt["rooms"], [0, 1])
        result.attrs["audit_value_matrices"]["A"]["rooms"] = run_study.np.array(
            [1, 0], dtype=run_study.np.int64
        )
        with self.assertRaises(AssertionError):
            run_study.validate_canonical_room_alignment(result, args)

    def test_paired_selected_matrices_survive_aggregation(self):
        record = run_study.compare_state(
            self.primary,
            self.wide,
            self.args,
            base_seed=17,
            stream_seed=23,
            pick_slot=6,
            completed_picks=0,
            to_add=[],
            to_drop=[],
            id_to_name={},
        )
        self.assertEqual(record["d256_action"], "B")
        self.assertEqual(record["wide_d256_action"], "C")
        self.assertAlmostEqual(record["wide_minus_primary_reference_value"], 1.0)
        self.assertTrue(record["arm_isolation_contract_pass"])

    def test_stale_journal_blocks_resume_and_clears_only_after_commit(self):
        state = {
            "base_seed": 17,
            "stream_seed": 23,
            "pick_slot": 6,
            "completed_picks": 0,
            "to_add": [],
            "to_drop": [],
            "state_sha256": "state",
        }

        def job(arm, candidates):
            return run_study.protocol.seal_job({
                "arm": arm,
                "candidate_count": candidates,
                "state": state,
                "configuration": {},
                "source_fingerprints": {},
                "runtime_fingerprint": {},
            })

        primary_job = job("primary", 24)
        wide_job = job("wide", 32)
        receipt = {"resume_guard": "guard"}
        primary = {"result_sha256": "primary-result", "worker_pid": 100}
        wide = {"result_sha256": "wide-result", "worker_pid": 101}
        with tempfile.TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(output_dir=Path(temp_dir), resume=True)
            run_study.begin_in_progress_journal(
                args, receipt, primary_job, wide_job
            )
            with self.assertRaises(ValueError):
                run_study.load_resume_records(args, receipt)
            run_study.advance_in_progress_journal(
                args, receipt, primary_job, wide_job, primary
            )
            with self.assertRaises(ValueError):
                run_study.clear_committed_in_progress_journal(
                    args, receipt, primary_job, wide_job, primary, wide
                )

            (args.output_dir / "source_receipt.json").write_text(
                json.dumps(receipt), encoding="utf-8"
            )
            (args.output_dir / "summary.json").write_text("{}", encoding="utf-8")
            run_study.pd.DataFrame([{
                "base_seed": 17,
                "pick_slot": 6,
                "completed_picks": 0,
                "status": "complete",
                "primary_arm_job_sha256": primary_job["job_sha256"],
                "wide_arm_job_sha256": wide_job["job_sha256"],
                "primary_arm_result_sha256": primary["result_sha256"],
                "wide_arm_result_sha256": wide["result_sha256"],
            }]).to_csv(args.output_dir / "state_metrics.csv", index=False)
            run_study.clear_committed_in_progress_journal(
                args, receipt, primary_job, wide_job, primary, wide
            )
            self.assertFalse(run_study.in_progress_journal_path(args).exists())


if __name__ == "__main__":
    unittest.main()
