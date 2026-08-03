"""Fresh-seed confirmation of nested D128/D256 and 24-vs-32 coverage."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import isolation_protocol as protocol


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
DEFAULT_DB = (REPO_ROOT / "app" / "Simulation.sqlite3").resolve()
FROZEN_DB_SHA256 = "47658fab0a2a98a1714890e8c57d45dbfbce63dd62c5455fad4ccc15374065a2"
PRIOR_PATH = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-08-02_sequential_v2_bank_stability"
    / "run_study.py"
)
PRIOR_SPEC = importlib.util.spec_from_file_location(
    "sequential_v2_bank_stability_harness",
    PRIOR_PATH,
)
if PRIOR_SPEC is None or PRIOR_SPEC.loader is None:
    raise RuntimeError(f"Could not import prior harness: {PRIOR_PATH}")
prior = importlib.util.module_from_spec(PRIOR_SPEC)
PRIOR_SPEC.loader.exec_module(prior)


FRESH_BASE_SEEDS = [
    20017,
    21017,
    22017,
    23017,
    24017,
    25017,
    26017,
    27017,
    28017,
]
PRELAUNCH_BASE_SEEDS = [12017, 13017]
FROZEN_SLOTS = [1, 6, 12]
FROZEN_COMPLETED_PICKS = [0, 7, 14]
PRELAUNCH_COMPLETED_PICKS = [0]
DEFAULT_PRELAUNCH_STRESS_DIR = (
    STUDY_DIR / "artifacts" / "local" / "prelaunch_arm_isolation_v2"
)
CANONICAL_RESULTS_DIR = (STUDY_DIR / "results").resolve()
STREAM_NAMESPACE = "sequential-v2-d256-fresh-confirmation-v2"
BOOTSTRAP_SEED = 20260803
BOOTSTRAP_DRAWS = 10_000
REGRET_THRESHOLD = 10.0
COVERAGE_THRESHOLD = 10.0
VALUE_NONINFERIORITY_PCT = -0.25
INTERNAL_POLICY_P90_SECONDS = 15.0
END_TO_END_P90_SECONDS = 30.0
ARM_TIMEOUT_SECONDS = 180.0
PREDICTION_COLUMNS = 1000
ARM_WORKER_PATH = STUDY_DIR / "arm_worker.py"
IN_PROGRESS_JOURNAL_NAME = "in_progress.json"
IN_PROGRESS_JOURNAL_SCHEMA_VERSION = "sequential-v2-in-progress-journal-v1"


class TimedNestedBankFootballSimulation(prior.NestedBankFootballSimulation):
    """Prior validated bank allocator with audit-selection timing exposed."""

    def __init__(self, *args, **kwargs):
        self.study_audit_selector_seconds = 0.0
        super().__init__(*args, **kwargs)

    def select_additional_policy_ppg_columns(
        self, num_columns, excluded_columns, samples, seed, bank_name
    ):
        started = time.perf_counter()
        try:
            return super().select_additional_policy_ppg_columns(
                num_columns,
                excluded_columns,
                samples,
                seed,
                bank_name,
            )
        finally:
            if bank_name == "Audit":
                self.study_audit_selector_seconds += time.perf_counter() - started


def make_policy_sim(conn, args, pick_slot):
    return TimedNestedBankFootballSimulation(
        conn=conn,
        set_year=args.year,
        pos_require_start=prior.POS_REQUIRE,
        num_teams=args.teams,
        num_rounds=args.rounds,
        my_pick_position=pick_slot,
        pred_vers="final_ensemble",
        league="dk",
        position_ranges=prior.POSITION_RANGES,
        template_resid_blend=1.0,
        use_stack_bonus=True,
        stack_bonus_pct=prior.SEQUENTIAL_STACK_BONUS_PCT,
        stack_pair_cap=prior.SEQUENTIAL_STACK_PAIR_CAP,
        stack_team_cap=prior.SEQUENTIAL_STACK_TEAM_CAP,
        study_decision_superbank_samples=args.expanded_decision_samples,
        study_control_decision_samples=args.control_decision_samples,
        study_reference_samples=args.reference_samples,
    )


def parse_csv_values(raw, value_type=str):
    return [value_type(value.strip()) for value in raw.split(",") if value.strip()]


def slot_stream_seed(base_seed, pick_slot):
    payload = f"{STREAM_NAMESPACE}|{int(base_seed)}|slot={int(pick_slot)}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def seed_manifest(base_seeds, slots):
    rows = [
        {
            "base_seed": int(base_seed),
            "pick_slot": int(slot),
            "stream_seed": slot_stream_seed(base_seed, slot),
        }
        for base_seed in base_seeds
        for slot in slots
    ]
    streams = [row["stream_seed"] for row in rows]
    if len(streams) != len(set(streams)):
        raise ValueError("Slot-specific stream seeds are not unique.")
    return rows


def stable_hash(value):
    return prior.stable_json_hash(value)


def run_policy(sim, args, to_add, to_drop, stream_seed, candidate_count, audit_samples):
    return sim.run_sim_best_ball_policy(
        to_add,
        to_drop,
        num_iters=args.rooms,
        construction_samples=args.construction_samples,
        evaluation_samples=args.evaluation_samples,
        decision_samples=args.expanded_decision_samples,
        decision_candidate_count=candidate_count,
        audit_samples=audit_samples,
        candidate_pool_size=candidate_count,
        seed=stream_seed,
        evaluation_seed=stream_seed + 202,
        decision_seed=stream_seed + 404,
        audit_seed=stream_seed + 505,
    )


def frozen_design_exact(args):
    return bool(
        args.base_seed_values == FRESH_BASE_SEEDS
        and args.slot_values == FROZEN_SLOTS
        and args.completed_pick_values == FROZEN_COMPLETED_PICKS
        and args.year == 2026
        and args.teams == 12
        and args.rounds == 20
        and args.rooms == 24
        and args.primary_candidates == 24
        and args.wide_candidates == 32
        and args.construction_samples == 16
        and args.evaluation_samples == 64
        and args.control_decision_samples == 128
        and args.expanded_decision_samples == 256
        and args.reference_samples == 512
        and args.fail_fast
        and args.prelaunch_stress_dir.resolve()
        == DEFAULT_PRELAUNCH_STRESS_DIR.resolve()
        and args.output_dir.resolve() == CANONICAL_RESULTS_DIR
        and args.db.resolve() == DEFAULT_DB
    )


def configured_state_keys(args):
    return {
        (int(base_seed), int(slot), int(depth))
        for base_seed in args.base_seed_values
        for slot in args.slot_values
        for depth in args.completed_pick_values
    }


def timed_policy_seconds(result, sim):
    policy_seconds, reference_seconds = prior.policy_seconds(result)
    selector_seconds = float(getattr(sim, "study_audit_selector_seconds", 0.0))
    return max(policy_seconds - selector_seconds, 0.0), reference_seconds + selector_seconds


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def array_fingerprint(values):
    array = np.ascontiguousarray(np.asarray(values))
    header = {
        "dtype": array.dtype.str,
        "shape": [int(size) for size in array.shape],
        "order": "C",
    }
    digest = hashlib.sha256()
    digest.update(protocol.canonical_json_bytes(header))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return {
        **header,
        "nbytes": int(array.nbytes),
        "sha256": digest.hexdigest(),
    }


def exact_structure_payload(value):
    """Type-tag a path so list/tuple/scalar distinctions remain auditable."""

    if isinstance(value, dict):
        items = [
            [exact_structure_payload(key), exact_structure_payload(item)]
            for key, item in value.items()
        ]
        items.sort(key=lambda pair: protocol.canonical_json_bytes(pair[0]))
        return {"type": "dict", "items": items}
    if isinstance(value, list):
        return {
            "type": "list",
            "items": [exact_structure_payload(item) for item in value],
        }
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [exact_structure_payload(item) for item in value],
        }
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "fingerprint": array_fingerprint(value)}
    if isinstance(value, np.generic):
        return {
            "type": "numpy_scalar",
            "fingerprint": array_fingerprint(np.asarray(value)),
        }
    if value is None:
        return {"type": "none"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": value}
    if isinstance(value, float):
        return {
            "type": "float",
            "fingerprint": array_fingerprint(np.asarray(value, dtype=np.float64)),
        }
    if isinstance(value, str):
        return {"type": "str", "value": value}
    raise TypeError(f"Unsupported policy-path value type: {type(value).__name__}")


def exact_structure_hash(value):
    return protocol.stable_json_hash(exact_structure_payload(value))


def matrix_payload_fingerprints(matrices):
    output = {}
    for player, payload in sorted(matrices.items()):
        components = {
            key: array_fingerprint(payload[key])
            for key in ("rooms", "values", "stack_utilities")
        }
        output[str(player)] = {
            "components": components,
            "payload_sha256": protocol.stable_json_hash(components),
        }
    return output


def adjusted_matrix_payloads(result, bank_name):
    adjusted, _, _ = prior.adjusted_matrices(result, bank_name)
    return {
        str(player): np.asarray(values, dtype=np.float64)
        for player, values in adjusted.items()
    }


def scores_for_slice(matrices, scenario_slice=slice(None)):
    return {
        player: float(values[:, scenario_slice].mean())
        for player, values in matrices.items()
    }


def validate_canonical_room_alignment(result, args):
    players = set(result.player.astype(str))
    expected_rooms = np.arange(args.rooms, dtype=np.int64)
    bank_specs = {
        "candidate": (
            result.attrs["candidate_value_matrices"],
            args.evaluation_samples,
        ),
        "decision": (
            result.attrs["decision_value_matrices"],
            args.expanded_decision_samples,
        ),
        "audit": (
            result.attrs["audit_value_matrices"],
            args.reference_samples,
        ),
    }
    receipt = {"rooms": expected_rooms.tolist(), "banks": {}}
    for bank_name, (matrices, expected_samples) in bank_specs.items():
        if set(map(str, matrices)) != players:
            raise AssertionError(
                f"{bank_name} matrices do not cover the exact candidate set."
            )
        for player, payload in matrices.items():
            rooms = np.asarray(payload["rooms"], dtype=np.int64)
            values = np.asarray(payload["values"])
            stack = np.asarray(payload["stack_utilities"])
            if not np.array_equal(rooms, expected_rooms):
                raise AssertionError(
                    f"{bank_name} rooms are not canonical 0..{args.rooms - 1} "
                    f"for {player}."
                )
            if values.shape != (args.rooms, expected_samples):
                raise AssertionError(
                    f"{bank_name} values have shape {values.shape} for {player}; "
                    f"expected {(args.rooms, expected_samples)}."
                )
            if stack.shape != (args.rooms,):
                raise AssertionError(
                    f"{bank_name} stack utilities have shape {stack.shape} for "
                    f"{player}; expected {(args.rooms,)}."
                )
            if not np.isfinite(values).all() or not np.isfinite(stack).all():
                raise AssertionError(
                    f"{bank_name} contains a non-finite value for {player}."
                )
        receipt["banks"][bank_name] = {
            "candidate_count": len(matrices),
            "values_shape": [args.rooms, expected_samples],
            "stack_shape": [args.rooms],
        }

    paths = result.attrs["policy_paths"]
    if set(map(str, paths)) != players:
        raise AssertionError("Policy paths do not cover the exact candidate set.")
    for player, player_paths in paths.items():
        if len(player_paths) != args.rooms:
            raise AssertionError(f"Policy path room count differs for {player}.")
        room_indices = [int(path["room_idx"]) for path in player_paths]
        if room_indices != expected_rooms.tolist():
            raise AssertionError(
                f"Policy path room_idx is not canonical for {player}."
            )
    receipt["policy_path_room_indices"] = expected_rooms.tolist()
    receipt["receipt_sha256"] = protocol.stable_json_hash(receipt)
    return receipt


def validate_worker_local_contracts(result, sim, args, stream_seed):
    banks = result.attrs["scenario_banks"]
    construction = np.asarray(banks["construction_ppg_columns"], dtype=np.int64)
    evaluation = np.asarray(banks["evaluation_ppg_columns"], dtype=np.int64)
    decision = np.asarray(banks["decision_ppg_columns"], dtype=np.int64)
    reference = np.asarray(banks["audit_ppg_columns"], dtype=np.int64)
    if len(decision) != args.expanded_decision_samples:
        raise AssertionError("D256 does not contain exactly 256 columns.")
    if len(reference) != args.reference_samples:
        raise AssertionError("R512 does not contain exactly 512 columns.")
    allocated = [construction, evaluation, decision, reference]
    for left_idx in range(len(allocated)):
        for right_idx in range(left_idx + 1, len(allocated)):
            if np.intersect1d(allocated[left_idx], allocated[right_idx]).size:
                raise AssertionError("C/E/D256/R512 banks overlap.")
    production_d128 = prior.FootballSimulation.select_additional_policy_ppg_columns(
        PREDICTION_COLUMNS,
        np.concatenate([construction, evaluation]),
        args.control_decision_samples,
        stream_seed + 404,
        "Decision",
    )
    if not np.array_equal(decision[: args.control_decision_samples], production_d128):
        raise AssertionError("D128 is not the exact production allocator prefix.")
    if not np.array_equal(sim.study_decision_superbank, decision):
        raise AssertionError("D256 superbank receipt differs from the result bank.")
    if not np.array_equal(sim.study_reference_bank, reference):
        raise AssertionError("R512 receipt differs from the result bank.")
    return banks


def serialize_arm_output(
    result,
    sim,
    args,
    job,
    hydration_seconds,
    worker_wall_seconds,
    source_fingerprint,
    runtime,
):
    players = result.player.astype(str).tolist()
    if len(players) != int(job["candidate_count"]):
        raise AssertionError("Policy arm did not retain its configured root screen.")
    if not (result.PolicyCompletedRooms == args.rooms).all():
        raise AssertionError("Policy arm has an incomplete candidate room.")
    canonical_alignment = validate_canonical_room_alignment(result, args)
    banks = validate_worker_local_contracts(
        result,
        sim,
        args,
        int(job["state"]["stream_seed"]),
    )
    decision = adjusted_matrix_payloads(result, "decision")
    reference = adjusted_matrix_payloads(result, "audit")
    d128_scores = scores_for_slice(
        decision, slice(0, args.control_decision_samples)
    )
    d256_scores = scores_for_slice(decision)
    reference_scores = scores_for_slice(reference)
    reference_half_1_scores = scores_for_slice(reference, slice(0, 256))
    reference_half_2_scores = scores_for_slice(reference, slice(256, 512))
    d128_action = max(d128_scores, key=d128_scores.get)
    d256_action = max(d256_scores, key=d256_scores.get)
    if d256_action != str(result.attrs["decision_top_player"]):
        raise AssertionError("Derived D256 action differs from app output.")

    raw_policy_paths = result.attrs["policy_paths"]
    policy_paths = {
        str(player): json_ready(path)
        for player, path in raw_policy_paths.items()
    }
    path_hashes = {
        str(player): exact_structure_hash(path)
        for player, path in sorted(raw_policy_paths.items())
    }
    selected_reference = {
        action: json_ready(reference[action])
        for action in sorted({d128_action, d256_action})
    }
    policy_seconds, reference_seconds = timed_policy_seconds(result, sim)
    tensor_fingerprints = {
        "candidate": matrix_payload_fingerprints(
            result.attrs["candidate_value_matrices"]
        ),
        "decision": matrix_payload_fingerprints(
            result.attrs["decision_value_matrices"]
        ),
        "audit": matrix_payload_fingerprints(
            result.attrs["audit_value_matrices"]
        ),
    }
    return {
        "job_sha256": job["job_sha256"],
        "arm": job["arm"],
        "candidate_count": int(job["candidate_count"]),
        "state": job["state"],
        "source_fingerprints": source_fingerprint,
        "runtime_fingerprint": runtime,
        "observed_worker_environment": protocol.observed_worker_environment(),
        "worker_pid": int(os.getpid()),
        "players": players,
        "all_rooms_complete": True,
        "production_d128_prefix_pass": True,
        "bank_disjointness_pass": True,
        "canonical_room_alignment_pass": True,
        "canonical_room_alignment_receipt": canonical_alignment,
        "scenario_banks": json_ready(banks),
        "decision_superbank": json_ready(sim.study_decision_superbank),
        "reference_bank": json_ready(sim.study_reference_bank),
        "draft_room_adp_columns": json_ready(
            result.attrs["draft_room_adp_columns"]
        ),
        "path_hashes": path_hashes,
        "tensor_fingerprints": tensor_fingerprints,
        "overlap_attestation_payload_sha256": protocol.stable_json_hash({
            "path_hashes": path_hashes,
            "tensor_fingerprints": tensor_fingerprints,
        }),
        "d128_action": d128_action,
        "d256_action": d256_action,
        "d128_scores": d128_scores,
        "d256_scores": d256_scores,
        "reference_scores": reference_scores,
        "reference_half_1_scores": reference_half_1_scores,
        "reference_half_2_scores": reference_half_2_scores,
        "current_pick_ev": {
            str(row.player): float(row.CurrentPickEV)
            for row in result[["player", "CurrentPickEV"]].itertuples(index=False)
        },
        "selected_reference_adjusted_matrices": selected_reference,
        "selected_room_zero_path": policy_paths[d256_action][0],
        "selected_room_zero_path_sha256": protocol.stable_json_hash(
            exact_structure_payload(raw_policy_paths[d256_action][0])
        ),
        "policy_seconds": float(policy_seconds),
        "reference_seconds": float(reference_seconds),
        "hydration_seconds": float(hydration_seconds),
        "worker_wall_seconds": float(worker_wall_seconds),
        "timings": json_ready(result.attrs["timings"]),
    }


def execute_arm_job(job):
    """Execute exactly one arm; called only by the one-shot worker process."""

    protocol.validate_job(job)
    protocol.assert_worker_environment()
    configuration = dict(job["configuration"])
    configuration["db"] = Path(configuration["db"]).resolve()
    args = SimpleNamespace(**configuration)
    actual_sources = source_fingerprints(args)
    if actual_sources != job["source_fingerprints"]:
        raise RuntimeError("Worker source fingerprint differs from its sealed job.")
    actual_runtime = runtime_fingerprint()
    if actual_runtime != job["runtime_fingerprint"]:
        raise RuntimeError("Worker runtime/native fingerprint differs from its sealed job.")

    worker_start = time.perf_counter()
    db_uri = f"file:{args.db.as_posix()}?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True)
    try:
        sim = make_policy_sim(conn, args, int(job["state"]["pick_slot"]))
        hydration = prior.prehydrate_weekly_templates(sim)
        result = run_policy(
            sim,
            args,
            list(job["state"]["to_add"]),
            list(job["state"]["to_drop"]),
            int(job["state"]["stream_seed"]),
            int(job["candidate_count"]),
            args.reference_samples,
        )
        return serialize_arm_output(
            result,
            sim,
            args,
            job,
            hydration,
            time.perf_counter() - worker_start,
            actual_sources,
            actual_runtime,
        )
    finally:
        conn.close()


def arm_configuration(args):
    return {
        "db": str(args.db.resolve()),
        "year": int(args.year),
        "teams": int(args.teams),
        "rounds": int(args.rounds),
        "rooms": int(args.rooms),
        "construction_samples": int(args.construction_samples),
        "evaluation_samples": int(args.evaluation_samples),
        "control_decision_samples": int(args.control_decision_samples),
        "expanded_decision_samples": int(args.expanded_decision_samples),
        "reference_samples": int(args.reference_samples),
    }


def make_arm_job(
    arm,
    candidate_count,
    args,
    receipt,
    base_seed,
    stream_seed,
    pick_slot,
    completed_picks,
    to_add,
    to_drop,
):
    state = {
        "base_seed": int(base_seed),
        "stream_seed": int(stream_seed),
        "pick_slot": int(pick_slot),
        "completed_picks": int(completed_picks),
        "to_add": list(to_add),
        "to_drop": list(to_drop),
        "state_sha256": stable_hash({
            "to_add": list(to_add),
            "to_drop": list(to_drop),
        }),
    }
    return protocol.seal_job({
        "arm": arm,
        "candidate_count": int(candidate_count),
        "state": state,
        "configuration": arm_configuration(args),
        "source_fingerprints": receipt["fingerprints"],
        "runtime_fingerprint": receipt["runtime_fingerprint"],
    })


def run_arm_subprocess(job, timeout_seconds=ARM_TIMEOUT_SECONDS):
    """Run one sealed job in one fresh process, exactly once."""

    parent_started = time.perf_counter()
    protocol.validate_job(job)
    with tempfile.TemporaryDirectory(prefix="sequential-v2-policy-arm-") as temp_dir:
        temp_root = Path(temp_dir)
        job_path = temp_root / "job.json"
        result_path = temp_root / "result.json"
        protocol.write_json_atomic(job_path, job)
        command = [
            sys.executable,
            str(ARM_WORKER_PATH),
            "--job",
            str(job_path),
            "--result",
            str(result_path),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env=protocol.worker_environment(),
                timeout=float(timeout_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"{job['arm']} policy worker exceeded {timeout_seconds:.0f}s; "
                "the arm is invalid and will not be retried."
            ) from exc
        if completed.returncode != 0:
            diagnostic = (completed.stderr or completed.stdout or "").strip()[-4000:]
            raise RuntimeError(
                f"{job['arm']} policy worker exited {completed.returncode}; "
                "the arm is invalid and will not be retried.\n"
                f"{diagnostic}"
            )
        if not result_path.is_file():
            raise RuntimeError(
                f"{job['arm']} policy worker exited cleanly without a result."
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        protocol.validate_result_envelope(result, job)
    result["_controller_parent_wall_seconds"] = float(
        time.perf_counter() - parent_started
    )
    return result


def assert_contracts(primary, wide, args):
    for arm, payload in (("primary", primary), ("wide", wide)):
        if not payload["all_rooms_complete"]:
            raise AssertionError(f"{arm} has an incomplete candidate room.")
        if not payload["production_d128_prefix_pass"]:
            raise AssertionError(f"{arm} failed the production D128 prefix contract.")
        if not payload["bank_disjointness_pass"]:
            raise AssertionError(f"{arm} failed bank disjointness.")
        if not payload["canonical_room_alignment_pass"]:
            raise AssertionError(f"{arm} failed canonical room alignment.")
        alignment = dict(payload["canonical_room_alignment_receipt"])
        claimed_alignment_hash = alignment.pop("receipt_sha256", None)
        if claimed_alignment_hash != protocol.stable_json_hash(alignment):
            raise AssertionError(f"{arm} room-alignment receipt hash differs.")
        expected_count = (
            args.primary_candidates if arm == "primary" else args.wide_candidates
        )
        expected_alignment = {
            "rooms": list(range(args.rooms)),
            "banks": {
                "candidate": {
                    "candidate_count": expected_count,
                    "values_shape": [args.rooms, args.evaluation_samples],
                    "stack_shape": [args.rooms],
                },
                "decision": {
                    "candidate_count": expected_count,
                    "values_shape": [args.rooms, args.expanded_decision_samples],
                    "stack_shape": [args.rooms],
                },
                "audit": {
                    "candidate_count": expected_count,
                    "values_shape": [args.rooms, args.reference_samples],
                    "stack_shape": [args.rooms],
                },
            },
            "policy_path_room_indices": list(range(args.rooms)),
        }
        if alignment != expected_alignment:
            raise AssertionError(f"{arm} room-alignment receipt differs from design.")

    primary_banks = primary["scenario_banks"]
    wide_banks = wide["scenario_banks"]
    bank_keys = (
        "construction_ppg_columns",
        "evaluation_ppg_columns",
        "decision_ppg_columns",
        "audit_ppg_columns",
    )
    for key in bank_keys:
        if primary_banks[key] != wide_banks[key]:
            raise AssertionError(f"Primary/wide {key} differ.")
    if primary["decision_superbank"] != primary_banks["decision_ppg_columns"]:
        raise AssertionError("Primary D256 superbank receipt differs.")
    if wide["decision_superbank"] != wide_banks["decision_ppg_columns"]:
        raise AssertionError("Wide D256 superbank receipt differs.")
    if primary["reference_bank"] != primary_banks["audit_ppg_columns"]:
        raise AssertionError("Primary R512 receipt differs.")
    if wide["reference_bank"] != wide_banks["audit_ppg_columns"]:
        raise AssertionError("Wide R512 receipt differs.")
    if primary["draft_room_adp_columns"] != wide["draft_room_adp_columns"]:
        raise AssertionError("Primary/wide ADP rooms differ.")

    primary_players = set(primary["players"])
    wide_players = set(wide["players"])
    if len(primary_players) != args.primary_candidates:
        raise AssertionError("Primary root screen has duplicate or missing candidates.")
    if len(wide_players) != args.wide_candidates:
        raise AssertionError("Wide root screen has duplicate or missing candidates.")
    overlap = sorted(primary_players & wide_players)
    if not overlap:
        raise AssertionError("Primary and wide candidate screens have no overlap.")
    overlap_receipt = {}
    for player in overlap:
        if primary["path_hashes"][player] != wide["path_hashes"][player]:
            raise AssertionError(f"Overlapping full policy path differs for {player}.")
        overlap_receipt[player] = {
            "path_sha256": primary["path_hashes"][player],
            "tensors": {},
        }
        for bank_name in ("candidate", "decision", "audit"):
            primary_tensor = primary["tensor_fingerprints"][bank_name][player]
            wide_tensor = wide["tensor_fingerprints"][bank_name][player]
            if primary_tensor != wide_tensor:
                raise AssertionError(
                    f"Overlapping {bank_name} room/value/stack tensor differs "
                    f"for {player}."
                )
            overlap_receipt[player]["tensors"][bank_name] = primary_tensor[
                "payload_sha256"
            ]
    return {
        "primary_candidates": primary_players,
        "wide_candidates": wide_players,
        "overlap": overlap,
        "primary_only": sorted(primary_players - wide_players),
        "wide_only": sorted(wide_players - primary_players),
        "overlap_contract_sha256": protocol.stable_json_hash(overlap_receipt),
    }


def compare_state(
    primary,
    wide,
    args,
    base_seed,
    stream_seed,
    pick_slot,
    completed_picks,
    to_add,
    to_drop,
    id_to_name,
):
    sets = assert_contracts(primary, wide, args)
    d128_action = primary["d128_action"]
    d256_action = primary["d256_action"]
    wide_action = wide["d256_action"]
    d128_scores = primary["d128_scores"]
    d256_scores = primary["d256_scores"]
    wide_d256_scores = wide["d256_scores"]
    reference_scores = primary["reference_scores"]
    reference_top = max(reference_scores, key=reference_scores.get)
    ref_half_1_scores = primary["reference_half_1_scores"]
    ref_half_2_scores = primary["reference_half_2_scores"]
    ref_half_1 = max(ref_half_1_scores, key=ref_half_1_scores.get)
    ref_half_2 = max(ref_half_2_scores, key=ref_half_2_scores.get)

    def cross_fitted_regret(action):
        return 0.5 * (
            ref_half_2_scores[ref_half_1]
            - ref_half_2_scores[action]
            + ref_half_1_scores[ref_half_2]
            - ref_half_1_scores[action]
        )

    d128_cf = cross_fitted_regret(d128_action)
    d256_cf = cross_fitted_regret(d256_action)
    d128_full = reference_scores[reference_top] - reference_scores[d128_action]
    d256_full = reference_scores[reference_top] - reference_scores[d256_action]
    primary_reference = primary["selected_reference_adjusted_matrices"]
    wide_reference = wide["selected_reference_adjusted_matrices"]
    d128_reference_matrix = np.asarray(primary_reference[d128_action], dtype=np.float64)
    d256_reference_matrix = np.asarray(primary_reference[d256_action], dtype=np.float64)
    wide_reference_matrix = np.asarray(wide_reference[wide_action], dtype=np.float64)
    if not (
        d128_reference_matrix.shape
        == d256_reference_matrix.shape
        == wide_reference_matrix.shape
    ):
        raise AssertionError("Selected R512 matrices are not physically paired.")
    value_delta_matrix = d256_reference_matrix - d128_reference_matrix
    coverage_delta_matrix = wide_reference_matrix - d256_reference_matrix
    value_delta = float(value_delta_matrix.mean())
    coverage_delta = float(coverage_delta_matrix.mean())
    control_value = float(d128_reference_matrix.mean())
    challenger_value = float(d256_reference_matrix.mean())
    d128_corr, d128_top5 = prior.rank_diagnostics(d128_scores, reference_scores)
    d256_corr, d256_top5 = prior.rank_diagnostics(d256_scores, reference_scores)
    wide_pilot = wide["current_pick_ev"]
    covered_wide_pilot = {
        player: score
        for player, score in wide_pilot.items()
        if player in sets["primary_candidates"]
    }
    pilot_omission = float(max(wide_pilot.values()) - max(covered_wide_pilot.values()))
    state_payload = {"to_add": list(to_add), "to_drop": list(to_drop)}
    return {
        "league": "dk",
        "base_seed": int(base_seed),
        "stream_seed": int(stream_seed),
        "pick_slot": int(pick_slot),
        "trajectory_id": f"{base_seed}:{pick_slot}",
        "completed_picks": int(completed_picks),
        "current_round": int(completed_picks + 1),
        "state_hash": stable_hash(state_payload),
        "to_add_keys": json.dumps(list(to_add)),
        "to_drop_keys": json.dumps(list(to_drop)),
        "to_add_names": json.dumps([id_to_name[key] for key in to_add]),
        "to_drop_names": json.dumps([id_to_name[key] for key in to_drop]),
        "primary_candidate_count": int(len(primary["players"])),
        "wide_candidate_count": int(len(wide["players"])),
        "candidate_overlap_count": int(len(sets["overlap"])),
        "primary_only_count": int(len(sets["primary_only"])),
        "wide_only_count": int(len(sets["wide_only"])),
        "candidate_overlap": json.dumps(sets["overlap"]),
        "primary_only": json.dumps(sets["primary_only"]),
        "wide_only": json.dumps(sets["wide_only"]),
        "all_rooms_complete": True,
        "bank_contracts_pass": True,
        "overlap_path_tensor_invariance_pass": True,
        "canonical_room_alignment_pass": True,
        "arm_isolation_contract_pass": True,
        "primary_worker_exit_ok": True,
        "wide_worker_exit_ok": True,
        "primary_worker_pid": int(primary["worker_pid"]),
        "wide_worker_pid": int(wide["worker_pid"]),
        "overlap_contract_sha256": sets["overlap_contract_sha256"],
        "primary_room_alignment_receipt_sha256": primary[
            "canonical_room_alignment_receipt"
        ]["receipt_sha256"],
        "wide_room_alignment_receipt_sha256": wide[
            "canonical_room_alignment_receipt"
        ]["receipt_sha256"],
        "primary_arm_job_sha256": primary["job_sha256"],
        "wide_arm_job_sha256": wide["job_sha256"],
        "primary_arm_result_sha256": primary["result_sha256"],
        "wide_arm_result_sha256": wide["result_sha256"],
        "primary_native_manifest_sha256": primary["runtime_fingerprint"][
            "native_binaries"
        ]["manifest_sha256"],
        "wide_native_manifest_sha256": wide["runtime_fingerprint"]["native_binaries"][
            "manifest_sha256"
        ],
        "d128_action": d128_action,
        "d256_action": d256_action,
        "wide_d256_action": wide_action,
        "d128_d256_action_agreement": bool(d128_action == d256_action),
        "wide_action_in_primary": bool(wide_action in sets["primary_candidates"]),
        "reference_top": reference_top,
        "reference_half_1_top": ref_half_1,
        "reference_half_2_top": ref_half_2,
        "d128_reference_exact": bool(d128_action == reference_top),
        "d256_reference_exact": bool(d256_action == reference_top),
        "d128_crossfit_regret": float(d128_cf),
        "d256_crossfit_regret": float(d256_cf),
        "d128_full_reference_regret": float(d128_full),
        "d256_full_reference_regret": float(d256_full),
        "d128_reference_value": control_value,
        "d256_reference_value": challenger_value,
        "d256_minus_d128_value": value_delta,
        "wide_minus_primary_reference_value": coverage_delta,
        "wide_positive_coverage_harm": max(coverage_delta, 0.0),
        "historical_pilot_omission_regret": pilot_omission,
        "d128_rank_correlation_vs_reference": d128_corr,
        "d256_rank_correlation_vs_reference": d256_corr,
        "d128_top5_overlap_reference": d128_top5,
        "d256_top5_overlap_reference": d256_top5,
        "primary_policy_seconds": float(primary["policy_seconds"]),
        "primary_reference_seconds": float(primary["reference_seconds"]),
        "wide_policy_seconds": float(wide["policy_seconds"]),
        "wide_reference_seconds": float(wide["reference_seconds"]),
        "primary_worker_wall_seconds": float(primary["worker_wall_seconds"]),
        "wide_worker_wall_seconds": float(wide["worker_wall_seconds"]),
        "primary_parent_wall_seconds": float(
            primary["_controller_parent_wall_seconds"]
        ),
        "wide_parent_wall_seconds": float(wide["_controller_parent_wall_seconds"]),
        "wide_d256_score_advantage": float(
            wide_d256_scores[wide_action]
            - max(
                score
                for player, score in wide_d256_scores.items()
                if player in sets["primary_candidates"]
            )
        ),
    }


def bootstrap_effect(frame):
    complete = frame[frame.status == "complete"].copy()
    trajectory_frames = {
        (int(slot), int(seed)): group
        for (slot, seed), group in complete.groupby(["pick_slot", "base_seed"])
    }
    trajectories = {
        key: {
            "delta": group.d256_minus_d128_value.to_numpy(dtype=np.float64),
            "control": group.d128_reference_value.to_numpy(dtype=np.float64),
        }
        for key, group in trajectory_frames.items()
    }
    slots = sorted({slot for slot, _ in trajectories})
    clusters_by_slot = {
        slot: sorted(seed for cluster_slot, seed in trajectories if cluster_slot == slot)
        for slot in slots
    }
    if not trajectories:
        return {
            "draws": BOOTSTRAP_DRAWS,
            "clusters": 0,
            "mean_delta": None,
            "ci95": [None, None],
            "mean_delta_pct": None,
            "ci95_pct": [None, None],
        }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    raw_draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    pct_draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for draw_idx in range(BOOTSTRAP_DRAWS):
        sampled_delta = []
        sampled_control = []
        for slot in slots:
            seeds = clusters_by_slot[slot]
            sampled = rng.choice(seeds, size=len(seeds), replace=True)
            for seed in sampled:
                trajectory = trajectories[(slot, int(seed))]
                sampled_delta.append(trajectory["delta"])
                sampled_control.append(trajectory["control"])
        delta = np.concatenate(sampled_delta).mean()
        control = np.concatenate(sampled_control).mean()
        raw_draws[draw_idx] = delta
        pct_draws[draw_idx] = 100.0 * delta / control
    mean_delta = float(complete.d256_minus_d128_value.mean())
    mean_delta_pct = float(100.0 * mean_delta / complete.d128_reference_value.mean())
    return {
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "cluster_unit": "base_seed_x_slot_trajectory",
        "stratified_by": "pick_slot",
        "clusters": len(trajectories),
        "clusters_by_slot": {
            str(slot): len(seeds) for slot, seeds in clusters_by_slot.items()
        },
        "mean_delta": mean_delta,
        "ci95": np.quantile(raw_draws, [0.025, 0.975]).tolist(),
        "mean_delta_pct": mean_delta_pct,
        "ci95_pct": np.quantile(pct_draws, [0.025, 0.975]).tolist(),
    }


def runtime_summary(frame, column):
    return prior.runtime_summary(frame, column)


def summarize(records, args, receipt, source_unchanged=None):
    frame = pd.DataFrame(records)
    complete = frame[frame.status == "complete"].copy()
    effect = bootstrap_effect(frame)
    configured_keys = configured_state_keys(args)
    complete_keys = [
        (int(row.base_seed), int(row.pick_slot), int(row.completed_picks))
        for row in complete.itertuples()
    ]
    error_count = int((frame.status == "error").sum())
    execution_ok = bool(
        len(complete_keys) == args.expected_states
        and len(complete_keys) == len(set(complete_keys))
        and set(complete_keys) == configured_keys
        and error_count == 0
    )
    contracts_ok = bool(
        execution_ok
        and complete.all_rooms_complete.all()
        and complete.bank_contracts_pass.all()
        and complete.overlap_path_tensor_invariance_pass.all()
        and complete.canonical_room_alignment_pass.all()
        and complete.arm_isolation_contract_pass.all()
        and complete.primary_worker_exit_ok.all()
        and complete.wide_worker_exit_ok.all()
        and complete.physical_state_valid.all()
    )
    d256_threshold = bool(
        execution_ok
        and (complete.d256_crossfit_regret <= REGRET_THRESHOLD).all()
    )
    d256_nonworse = bool(
        execution_ok
        and complete.d256_crossfit_regret.mean()
        <= complete.d128_crossfit_regret.mean()
        and complete.d256_crossfit_regret.max()
        <= complete.d128_crossfit_regret.max()
    )
    positive_effect = bool(
        execution_ok and effect["mean_delta"] is not None and effect["mean_delta"] > 0
    )
    value_noninferior = bool(
        execution_ok
        and effect["ci95_pct"][0] is not None
        and effect["ci95_pct"][0] >= VALUE_NONINFERIORITY_PCT
    )
    coverage_ok = bool(
        execution_ok
        and (complete.wide_positive_coverage_harm <= COVERAGE_THRESHOLD).all()
    )
    primary_runtime = runtime_summary(frame, "primary_policy_seconds")
    primary_parent_runtime = runtime_summary(frame, "primary_parent_wall_seconds")
    internal_usability_ok = bool(
        execution_ok
        and primary_runtime
        and primary_runtime["p90"] < INTERNAL_POLICY_P90_SECONDS
    )
    end_to_end_usability_ok = bool(
        execution_ok
        and primary_parent_runtime
        and primary_parent_runtime["p90"] < END_TO_END_P90_SECONDS
    )
    exact_design = bool(
        frozen_design_exact(args)
        and receipt["fingerprints"]["database"] == FROZEN_DB_SHA256
    )
    gates = {
        "frozen_design_exact": exact_design,
        "source_unchanged": bool(source_unchanged),
        "all_states_complete": execution_ok,
        "physical_room_bank_path_and_overlap_contracts": contracts_ok,
        "d256_crossfit_regret_at_most_10": d256_threshold,
        "d256_regret_mean_and_max_nonworse_than_d128": d256_nonworse,
        "fresh_mean_d256_value_positive": positive_effect,
        "fresh_reference_value_noninferior_at_minus_0_25_pct": value_noninferior,
        "wide_positive_reference_advantage_at_most_10": coverage_ok,
        "primary24_parent_end_to_end_p90_below_30_seconds": end_to_end_usability_ok,
    }
    quality = {}
    if not complete.empty:
        for label in ("d128", "d256"):
            quality[label] = {
                "crossfit_regret_mean": float(
                    complete[f"{label}_crossfit_regret"].mean()
                ),
                "crossfit_regret_max": float(
                    complete[f"{label}_crossfit_regret"].max()
                ),
                "crossfit_regret_above_10": int(
                    (complete[f"{label}_crossfit_regret"] > REGRET_THRESHOLD).sum()
                ),
                "full_reference_regret_mean": float(
                    complete[f"{label}_full_reference_regret"].mean()
                ),
                "full_reference_regret_max": float(
                    complete[f"{label}_full_reference_regret"].max()
                ),
                "reference_exact_rate": float(
                    complete[f"{label}_reference_exact"].mean()
                ),
            }
    coverage = {}
    if not complete.empty:
        coverage = {
            "mean_candidate_overlap": float(complete.candidate_overlap_count.mean()),
            "minimum_candidate_overlap": int(complete.candidate_overlap_count.min()),
            "states_with_nonnested_sets": int(
                ((complete.primary_only_count > 0) | (complete.wide_only_count > 0)).sum()
            ),
            "wide_action_in_primary_rate": float(complete.wide_action_in_primary.mean()),
            "wide_minus_primary_reference_mean": float(
                complete.wide_minus_primary_reference_value.mean()
            ),
            "wide_positive_reference_advantage_max": float(
                complete.wide_positive_coverage_harm.max()
            ),
            "historical_pilot_omission_regret_max": float(
                complete.historical_pilot_omission_regret.max()
            ),
        }
    return {
        "completed_at_utc": prior.utc_now(),
        "configuration": active_configuration(args),
        "source_receipt": receipt,
        "state_counts": {
            "configured": args.expected_states,
            "complete": int(len(complete)),
            "errors": error_count,
        },
        "quality": quality,
        "paired_reference_effect": effect,
        "coverage": coverage,
        "runtime": {
            "primary24_internal_policy": primary_runtime,
            "primary24_parent_end_to_end": primary_parent_runtime,
            "wide32_internal_policy": runtime_summary(frame, "wide_policy_seconds"),
            "wide32_parent_end_to_end": runtime_summary(
                frame, "wide_parent_wall_seconds"
            ),
            "usability_sla": {
                "internal_policy_p90_diagnostic": "15_seconds",
                "parent_end_to_end_p90": "strictly_below_30_seconds",
            },
            "diagnostics": {
                "primary24_internal_policy_p90_below_15_seconds": (
                    internal_usability_ok
                ),
            },
        },
        "gates": gates,
        "fresh_confirmation_pass": bool(all(gates.values())),
        "preview_promotion_ready": False,
        "remaining_promotion_blockers": [
            "historical_forced_pick_and_opponent_sensitivity_not_run"
        ],
        "errors": frame.loc[frame.status == "error", [
            "base_seed", "stream_seed", "pick_slot", "completed_picks", "message"
        ]].to_dict("records"),
    }


def active_configuration(args):
    return {
        "database_path": str(args.db.resolve()),
        "output_directory": str(args.output_dir.resolve()),
        "year": args.year,
        "league": "dk",
        "teams": args.teams,
        "rounds": args.rounds,
        "base_seeds": args.base_seed_values,
        "slots": args.slot_values,
        "slot_stream_manifest": seed_manifest(args.base_seed_values, args.slot_values),
        "completed_picks": args.completed_pick_values,
        "rooms": args.rooms,
        "primary_candidates": args.primary_candidates,
        "wide_candidates": args.wide_candidates,
        "construction_samples": args.construction_samples,
        "evaluation_samples": args.evaluation_samples,
        "control_decision_samples": args.control_decision_samples,
        "expanded_decision_samples": args.expanded_decision_samples,
        "reference_samples": args.reference_samples,
        "reference_halves": [256, 256],
        "legacy_in_confirmation_process": False,
        "execution_isolation": protocol.EXECUTION_ISOLATION,
        "one_fresh_subprocess_per_arm": True,
        "worker_pool": False,
        "worker_retry": False,
        "worker_job_schema_version": protocol.JOB_SCHEMA_VERSION,
        "worker_result_schema_version": protocol.RESULT_SCHEMA_VERSION,
        "worker_environment": dict(protocol.WORKER_ENVIRONMENT),
        "arm_timeout_seconds": ARM_TIMEOUT_SECONDS,
        "in_progress_journal": IN_PROGRESS_JOURNAL_NAME,
        "in_progress_journal_schema_version": (
            IN_PROGRESS_JOURNAL_SCHEMA_VERSION
        ),
        "stale_journal_blocks_resume_and_retry": True,
        "prelaunch_stress_dir": str(args.prelaunch_stress_dir.resolve()),
        "fail_fast": bool(args.fail_fast),
        "regret_threshold": REGRET_THRESHOLD,
        "coverage_threshold": COVERAGE_THRESHOLD,
        "value_noninferiority_pct": VALUE_NONINFERIORITY_PCT,
        "internal_policy_p90_seconds_diagnostic_threshold": (
            INTERNAL_POLICY_P90_SECONDS
        ),
        "parent_end_to_end_p90_seconds_strict_upper_bound": (
            END_TO_END_P90_SECONDS
        ),
    }


def runtime_fingerprint():
    return protocol.runtime_fingerprint()


def source_fingerprints(args):
    paths = {
        "database": args.db.resolve(),
        "zSim_Helper.py": REPO_ROOT / "app" / "zSim_Helper.py",
        "snake_draft_app.py": REPO_ROOT / "app" / "snake_draft_app.py",
        "simulation_worker.py": REPO_ROOT / "app" / "simulation_worker.py",
        "prior_run_study.py": PRIOR_PATH,
        "run_study.py": Path(__file__).resolve(),
        "arm_worker.py": ARM_WORKER_PATH,
        "isolation_protocol.py": STUDY_DIR / "isolation_protocol.py",
        "README.md": STUDY_DIR / "README.md",
        "ABORTED.md": STUDY_DIR / "ABORTED.md",
    }
    return {name: prior.sha256_file(path) for name, path in paths.items()}


def expected_prelaunch_configuration(args):
    expected = active_configuration(args)
    expected["output_directory"] = str(args.prelaunch_stress_dir.resolve())
    expected["base_seeds"] = PRELAUNCH_BASE_SEEDS
    expected["slots"] = FROZEN_SLOTS
    expected["slot_stream_manifest"] = seed_manifest(
        PRELAUNCH_BASE_SEEDS, FROZEN_SLOTS
    )
    expected["completed_picks"] = PRELAUNCH_COMPLETED_PICKS
    return expected


def verify_prelaunch_stress(args, fingerprints, runtime):
    stress_dir = args.prelaunch_stress_dir.resolve()
    stale_journal = stress_dir / IN_PROGRESS_JOURNAL_NAME
    if stale_journal.exists():
        raise ValueError(
            f"Prelaunch stress has a stale in-progress journal: {stale_journal}"
        )
    receipt_path = stress_dir / "source_receipt.json"
    summary_path = stress_dir / "summary.json"
    metrics_path = stress_dir / "state_metrics.csv"
    for path in (receipt_path, summary_path, metrics_path):
        if not path.is_file():
            raise ValueError(f"Required prelaunch-stress artifact is missing: {path}")

    stress_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    stress_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if stress_receipt.get("fingerprints") != fingerprints:
        raise ValueError("Prelaunch stress source fingerprints differ from this launch.")
    if stress_receipt.get("runtime_fingerprint") != runtime:
        raise ValueError("Prelaunch stress runtime fingerprint differs from this launch.")
    if stress_receipt.get("launch", {}).get("resume_requested") is not False:
        raise ValueError("Prelaunch stress must be a non-resumed controller launch.")
    if stress_receipt.get("active_configuration") != expected_prelaunch_configuration(args):
        raise ValueError("Prelaunch stress did not use the frozen stress configuration.")

    expected_keys = {
        (seed, slot, depth)
        for seed in PRELAUNCH_BASE_SEEDS
        for slot in FROZEN_SLOTS
        for depth in PRELAUNCH_COMPLETED_PICKS
    }
    frame = pd.read_csv(metrics_path)
    if len(frame) != len(expected_keys) or set(frame.status.astype(str)) != {"complete"}:
        raise ValueError("Prelaunch stress is not a clean complete checkpoint.")
    actual_keys = {
        (int(row.base_seed), int(row.pick_slot), int(row.completed_picks))
        for row in frame.itertuples()
    }
    if actual_keys != expected_keys or len(actual_keys) != len(frame):
        raise ValueError("Prelaunch stress state keys are incomplete or duplicated.")
    for column in (
        "all_rooms_complete",
        "bank_contracts_pass",
        "overlap_path_tensor_invariance_pass",
        "canonical_room_alignment_pass",
        "arm_isolation_contract_pass",
        "primary_worker_exit_ok",
        "wide_worker_exit_ok",
        "physical_state_valid",
    ):
        if (
            column not in frame
            or set(frame[column].astype(str).str.lower()) != {"true"}
        ):
            raise ValueError(f"Prelaunch stress failed required contract: {column}")
    state_counts = stress_summary.get("state_counts", {})
    gates = stress_summary.get("gates", {})
    if state_counts != {"configured": 6, "complete": 6, "errors": 0}:
        raise ValueError("Prelaunch stress summary counts are not exact.")
    for gate in (
        "source_unchanged",
        "all_states_complete",
        "physical_room_bank_path_and_overlap_contracts",
        "primary24_parent_end_to_end_p90_below_30_seconds",
    ):
        if gates.get(gate) is not True:
            raise ValueError(f"Prelaunch stress summary failed required gate: {gate}")
    return {
        "directory": str(stress_dir),
        "configuration": stress_receipt["active_configuration"],
        "runtime_fingerprint": stress_receipt["runtime_fingerprint"],
        "source_receipt_sha256": prior.sha256_file(receipt_path),
        "state_metrics_sha256": prior.sha256_file(metrics_path),
        "summary_sha256": prior.sha256_file(summary_path),
        "state_counts": state_counts,
    }


def build_source_receipt(args, conn):
    fingerprints = source_fingerprints(args)
    runtime = runtime_fingerprint()
    if frozen_design_exact(args) and fingerprints["database"] != FROZEN_DB_SHA256:
        raise ValueError("Fresh confirmation database differs from the frozen hash.")
    prelaunch_attestation = (
        verify_prelaunch_stress(args, fingerprints, runtime)
        if frozen_design_exact(args)
        else None
    )
    manifest = {
        "default_database_path": str(DEFAULT_DB),
        "frozen_database_sha256": FROZEN_DB_SHA256,
        "frozen_base_seeds": FRESH_BASE_SEEDS,
        "frozen_slots": FROZEN_SLOTS,
        "frozen_completed_picks": FROZEN_COMPLETED_PICKS,
        "slot_stream_manifest": seed_manifest(FRESH_BASE_SEEDS, FROZEN_SLOTS),
        "fail_fast_on_state_error": True,
        "prelaunch_stress": {
            "base_seeds": PRELAUNCH_BASE_SEEDS,
            "slots": FROZEN_SLOTS,
            "completed_picks": PRELAUNCH_COMPLETED_PICKS,
            "rooms": 24,
            "required_before_confirmation": True,
        },
        "gates": {
            "regret_threshold": REGRET_THRESHOLD,
            "coverage_threshold": COVERAGE_THRESHOLD,
            "value_noninferiority_pct": VALUE_NONINFERIORITY_PCT,
            "internal_policy_p90_seconds_diagnostic_threshold": (
                INTERNAL_POLICY_P90_SECONDS
            ),
            "parent_end_to_end_p90_seconds_strict_upper_bound": (
                END_TO_END_P90_SECONDS
            ),
        },
        "arm_execution": {
            "isolation": protocol.EXECUTION_ISOLATION,
            "fresh_subprocesses_per_state": 2,
            "order": ["primary", "wide"],
            "pool": False,
            "retry": False,
            "timeout_seconds": ARM_TIMEOUT_SECONDS,
            "job_transport": "os_temp_json",
            "result_transport": "versioned_os_temp_json",
            "worker_environment": dict(protocol.WORKER_ENVIRONMENT),
            "failed_final_arm_invalidates_launch": True,
            "in_progress_journal": IN_PROGRESS_JOURNAL_NAME,
            "stale_journal_blocks_resume_and_retry": True,
            "journal_clears_only_after_committed_pair_validation": True,
        },
    }
    probe = prior.make_sim(conn, args, args.slot_values[0])
    if not probe.uses_v2_joint_template:
        raise ValueError("Fresh confirmation requires current V2 joint templates.")
    if probe.template_resid_method_version != "joint_centered_template_v2_v1":
        raise ValueError("Unexpected V2 template residual method.")
    receipt = {
        "created_at_utc": prior.utc_now(),
        "git_head": prior.git_head(),
        "launch": {"resume_requested": bool(args.resume)},
        "fingerprints": fingerprints,
        "runtime_fingerprint": runtime,
        "prelaunch_stress_attestation": prelaunch_attestation,
        "database": {
            "path": str(args.db.resolve()),
            "size_bytes": args.db.resolve().stat().st_size,
            "quick_check": conn.execute("PRAGMA quick_check").fetchone()[0],
            "integrity_check": conn.execute("PRAGMA integrity_check").fetchone()[0],
            "freelist_count": conn.execute("PRAGMA freelist_count").fetchone()[0],
        },
        "model_contract": {
            "league": probe.league,
            "player_count": int(len(probe.player_data)),
            "identity_column": probe.identity_column(probe.player_data),
            "uses_v2_joint_template": bool(probe.uses_v2_joint_template),
            "template_resid_method_version": probe.template_resid_method_version,
            "weekly_horizon": 16,
        },
        "frozen_design_manifest": manifest,
        "frozen_design_manifest_hash": stable_hash(manifest),
        "active_configuration": active_configuration(args),
    }
    receipt["resume_guard"] = stable_hash({
        "fingerprints": fingerprints,
        "runtime_fingerprint": runtime,
        "prelaunch_stress_attestation": prelaunch_attestation,
        "active_configuration": receipt["active_configuration"],
        "frozen_design_manifest_hash": receipt["frozen_design_manifest_hash"],
    })
    return receipt


def verify_source_unchanged(args, receipt, conn):
    return bool(
        source_fingerprints(args) == receipt["fingerprints"]
        and runtime_fingerprint() == receipt["runtime_fingerprint"]
        and conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        and conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    )


def in_progress_journal_path(args):
    return args.output_dir.resolve() / IN_PROGRESS_JOURNAL_NAME


def seal_in_progress_journal(payload):
    journal = dict(payload)
    journal["schema_version"] = IN_PROGRESS_JOURNAL_SCHEMA_VERSION
    journal.pop("journal_sha256", None)
    journal["journal_sha256"] = protocol.stable_json_hash(journal)
    return journal


def validate_in_progress_journal(journal):
    if journal.get("schema_version") != IN_PROGRESS_JOURNAL_SCHEMA_VERSION:
        raise ValueError("In-progress journal schema is unsupported.")
    claimed = journal.get("journal_sha256")
    unsigned = dict(journal)
    unsigned.pop("journal_sha256", None)
    if not isinstance(claimed, str) or claimed != protocol.stable_json_hash(unsigned):
        raise ValueError("In-progress journal hash is missing or invalid.")
    if journal.get("phase") not in {"primary_running", "wide_running"}:
        raise ValueError("In-progress journal phase is invalid.")
    return journal


def begin_in_progress_journal(args, receipt, primary_job, wide_job):
    path = in_progress_journal_path(args)
    if path.exists():
        raise ValueError(
            f"Stale in-progress journal blocks launch and retry: {path}"
        )
    if primary_job["state"] != wide_job["state"]:
        raise ValueError("Primary/wide jobs do not describe the same state.")
    journal = seal_in_progress_journal({
        "created_at_utc": prior.utc_now(),
        "updated_at_utc": prior.utc_now(),
        "resume_guard": receipt["resume_guard"],
        "phase": "primary_running",
        "state": primary_job["state"],
        "primary_job_sha256": primary_job["job_sha256"],
        "wide_job_sha256": wide_job["job_sha256"],
        "primary_result_sha256": None,
    })
    protocol.write_json_atomic(path, journal)
    return journal


def advance_in_progress_journal(args, receipt, primary_job, wide_job, primary):
    path = in_progress_journal_path(args)
    if not path.is_file():
        raise ValueError("Primary arm returned without its in-progress journal.")
    journal = validate_in_progress_journal(
        json.loads(path.read_text(encoding="utf-8"))
    )
    expected = {
        "resume_guard": receipt["resume_guard"],
        "phase": "primary_running",
        "state": primary_job["state"],
        "primary_job_sha256": primary_job["job_sha256"],
        "wide_job_sha256": wide_job["job_sha256"],
    }
    for key, value in expected.items():
        if journal.get(key) != value:
            raise ValueError(f"In-progress journal differs before wide arm: {key}.")
    journal.update({
        "updated_at_utc": prior.utc_now(),
        "phase": "wide_running",
        "primary_result_sha256": primary["result_sha256"],
        "primary_worker_pid": int(primary["worker_pid"]),
    })
    protocol.write_json_atomic(path, seal_in_progress_journal(journal))


def clear_committed_in_progress_journal(
    args,
    receipt,
    primary_job,
    wide_job,
    primary,
    wide,
):
    """Remove the marker only after re-reading the matching committed row."""

    path = in_progress_journal_path(args)
    if not path.is_file():
        raise ValueError("Committed state is missing its in-progress journal.")
    journal = validate_in_progress_journal(
        json.loads(path.read_text(encoding="utf-8"))
    )
    expected = {
        "resume_guard": receipt["resume_guard"],
        "phase": "wide_running",
        "state": primary_job["state"],
        "primary_job_sha256": primary_job["job_sha256"],
        "wide_job_sha256": wide_job["job_sha256"],
        "primary_result_sha256": primary["result_sha256"],
    }
    for key, value in expected.items():
        if journal.get(key) != value:
            raise ValueError(f"In-progress journal differs at commit: {key}.")

    metrics_path = args.output_dir / "state_metrics.csv"
    receipt_path = args.output_dir / "source_receipt.json"
    summary_path = args.output_dir / "summary.json"
    if not all(item.is_file() for item in (metrics_path, receipt_path, summary_path)):
        raise ValueError("Checkpoint files are incomplete; journal will not be cleared.")
    committed_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if committed_receipt.get("resume_guard") != receipt["resume_guard"]:
        raise ValueError("Committed receipt differs; journal will not be cleared.")
    frame = pd.read_csv(metrics_path)
    state = primary_job["state"]
    matching = frame[
        (frame.base_seed.astype(int) == int(state["base_seed"]))
        & (frame.pick_slot.astype(int) == int(state["pick_slot"]))
        & (frame.completed_picks.astype(int) == int(state["completed_picks"]))
    ]
    if len(matching) != 1:
        raise ValueError("Committed state row is missing or duplicated.")
    row = matching.iloc[0]
    committed = {
        "status": str(row.status),
        "primary_arm_job_sha256": str(row.primary_arm_job_sha256),
        "wide_arm_job_sha256": str(row.wide_arm_job_sha256),
        "primary_arm_result_sha256": str(row.primary_arm_result_sha256),
        "wide_arm_result_sha256": str(row.wide_arm_result_sha256),
    }
    required = {
        "status": "complete",
        "primary_arm_job_sha256": primary_job["job_sha256"],
        "wide_arm_job_sha256": wide_job["job_sha256"],
        "primary_arm_result_sha256": primary["result_sha256"],
        "wide_arm_result_sha256": wide["result_sha256"],
    }
    if committed != required:
        raise ValueError("Committed arm hashes differ; journal will not be cleared.")
    path.unlink()


def write_checkpoint(records, args, receipt, source_unchanged=None):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output_dir / "state_metrics.csv", index=False)
    (args.output_dir / "source_receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    if records:
        (args.output_dir / "summary.json").write_text(
            json.dumps(
                summarize(records, args, receipt, source_unchanged=source_unchanged),
                indent=2,
            ),
            encoding="utf-8",
        )


def load_resume_records(args, receipt):
    metrics_path = args.output_dir / "state_metrics.csv"
    receipt_path = args.output_dir / "source_receipt.json"
    journal_path = in_progress_journal_path(args)
    if journal_path.exists():
        raise ValueError(
            "A stale in-progress journal proves that an arm/state launch did not "
            "commit cleanly. This output is invalid and cannot be resumed or retried: "
            f"{journal_path}"
        )
    if not args.resume:
        if metrics_path.exists() or receipt_path.exists():
            raise ValueError("Output exists; use --resume only with an unchanged guard.")
        return []
    if not metrics_path.exists() or not receipt_path.exists():
        raise ValueError("Resume requested without both checkpoint files.")
    old_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if old_receipt.get("resume_guard") != receipt.get("resume_guard"):
        raise ValueError("Resume guard differs from the frozen source/design receipt.")
    rows = pd.read_csv(metrics_path).to_dict("records")
    allowed = configured_state_keys(args)
    complete_by_key = {}
    for row in rows:
        key = (
            int(row["base_seed"]),
            int(row["pick_slot"]),
            int(row["completed_picks"]),
        )
        if key not in allowed:
            raise ValueError(f"Resume checkpoint contains unconfigured state {key}.")
        expected_stream = slot_stream_seed(key[0], key[1])
        if int(row["stream_seed"]) != expected_stream:
            raise ValueError(f"Resume checkpoint stream seed differs for {key}.")
        status = str(row.get("status", ""))
        if status == "error":
            raise ValueError(
                f"Resume checkpoint contains a failed state {key}; "
                "failed launches cannot be resumed as evidence."
            )
        if status != "complete":
            raise ValueError(f"Resume checkpoint has invalid status for {key}: {status}")
        if key in complete_by_key:
            raise ValueError(f"Resume checkpoint duplicates completed state {key}.")
        complete_by_key[key] = row
    return [
        complete_by_key[key]
        for key in sorted(complete_by_key)
    ]


def opening_path_from_record(record):
    raw = record.get("opening_room_path_json")
    if not isinstance(raw, str) or not raw:
        raise ValueError("Completed opening checkpoint lacks its D256 room-zero path.")
    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--teams", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--base-seeds", default=",".join(map(str, FRESH_BASE_SEEDS)))
    parser.add_argument("--slots", default="1,6,12")
    parser.add_argument("--completed-picks", default="0,7,14")
    parser.add_argument("--rooms", type=int, default=24)
    parser.add_argument("--primary-candidates", type=int, default=24)
    parser.add_argument("--wide-candidates", type=int, default=32)
    parser.add_argument("--construction-samples", type=int, default=16)
    parser.add_argument("--evaluation-samples", type=int, default=64)
    parser.add_argument("--control-decision-samples", type=int, default=128)
    parser.add_argument("--expanded-decision-samples", type=int, default=256)
    parser.add_argument("--reference-samples", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=CANONICAL_RESULTS_DIR)
    parser.add_argument(
        "--prelaunch-stress-dir",
        type=Path,
        default=DEFAULT_PRELAUNCH_STRESS_DIR,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--receipt-only", action="store_true")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort immediately after checkpointing the first state error.",
    )
    args = parser.parse_args()
    args.base_seed_values = parse_csv_values(args.base_seeds, int)
    args.slot_values = parse_csv_values(args.slots, int)
    args.completed_pick_values = parse_csv_values(args.completed_picks, int)
    args.expected_states = (
        len(args.base_seed_values)
        * len(args.slot_values)
        * len(args.completed_pick_values)
    )
    if any(seed in FRESH_BASE_SEEDS for seed in args.base_seed_values) and not frozen_design_exact(args):
        parser.error("Confirmation seeds may be used only by the exact frozen design.")
    if args.reference_samples != 512:
        parser.error("Cross-fitting requires R512.")
    total_samples = (
        args.construction_samples
        + args.evaluation_samples
        + args.expanded_decision_samples
        + args.reference_samples
    )
    if total_samples > PREDICTION_COLUMNS:
        parser.error("Disjoint banks exceed 1,000 prediction columns.")
    seed_manifest(args.base_seed_values, args.slot_values)

    db_uri = f"file:{args.db.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True)
    records = []
    try:
        receipt = build_source_receipt(args, conn)
        if args.receipt_only:
            print(json.dumps({"source_receipt": receipt}, indent=2), flush=True)
            return
        records = load_resume_records(args, receipt)
        completed_keys = {
            (int(row["base_seed"]), int(row["pick_slot"]), int(row["completed_picks"]))
            for row in records
            if row["status"] == "complete"
        }
        print(json.dumps({"source_receipt": receipt}, indent=2), flush=True)
        state_index = 0
        for base_seed in args.base_seed_values:
            for pick_slot in args.slot_values:
                stream_seed = slot_stream_seed(base_seed, pick_slot)
                fixture_sim = prior.make_sim(conn, args, pick_slot)
                name_to_id, id_to_name = prior.identity_name_maps(fixture_sim)
                initial_opponents = prior.derive_initial_opponent_picks(
                    fixture_sim, stream_seed
                )
                primary_room_path = None
                opening_checkpoint = next(
                    (
                        row
                        for row in records
                        if int(row["base_seed"]) == base_seed
                        and int(row["pick_slot"]) == pick_slot
                        and int(row["completed_picks"]) == 0
                        and row["status"] == "complete"
                    ),
                    None,
                )
                if opening_checkpoint is not None:
                    primary_room_path = opening_path_from_record(opening_checkpoint)
                for completed_picks in args.completed_pick_values:
                    state_index += 1
                    key = (base_seed, pick_slot, completed_picks)
                    if key in completed_keys:
                        print(
                            f"resume-skip {state_index}/{args.expected_states} "
                            f"base={base_seed} slot={pick_slot} round={completed_picks + 1}",
                            flush=True,
                        )
                        continue
                    state_start = time.perf_counter()
                    state_complete = False
                    primary_job = None
                    wide_job = None
                    primary = None
                    wide = None
                    try:
                        if completed_picks == 0:
                            to_add, to_drop = [], list(initial_opponents)
                        elif primary_room_path is None:
                            raise RuntimeError("Primary D256 opening trajectory unavailable.")
                        else:
                            to_add, to_drop = prior.derive_state_from_control_path(
                                primary_room_path,
                                initial_opponents,
                                completed_picks,
                                name_to_id,
                            )
                        if not prior.physical_state_is_valid(
                            fixture_sim, to_add, to_drop, completed_picks
                        ):
                            raise ValueError("Derived draft state is not physical.")
                        coverage = prior.player_pool_coverage(fixture_sim, to_add, to_drop)
                        if not coverage[0]:
                            raise ValueError(
                                f"State has {coverage[1]} modeled players for "
                                f"{coverage[2]} required picks."
                            )

                        primary_job = make_arm_job(
                            "primary",
                            args.primary_candidates,
                            args,
                            receipt,
                            base_seed,
                            stream_seed,
                            pick_slot,
                            completed_picks,
                            to_add,
                            to_drop,
                        )
                        wide_job = make_arm_job(
                            "wide",
                            args.wide_candidates,
                            args,
                            receipt,
                            base_seed,
                            stream_seed,
                            pick_slot,
                            completed_picks,
                            to_add,
                            to_drop,
                        )
                        begin_in_progress_journal(
                            args, receipt, primary_job, wide_job
                        )
                        primary = run_arm_subprocess(primary_job)
                        advance_in_progress_journal(
                            args,
                            receipt,
                            primary_job,
                            wide_job,
                            primary,
                        )
                        wide = run_arm_subprocess(wide_job)
                        record = compare_state(
                            primary,
                            wide,
                            args,
                            base_seed,
                            stream_seed,
                            pick_slot,
                            completed_picks,
                            to_add,
                            to_drop,
                            id_to_name,
                        )
                        record.update({
                            "status": "complete",
                            "message": "",
                            "traceback": "",
                            "physical_state_valid": True,
                            "pool_available": coverage[1],
                            "pool_required": coverage[2],
                            "primary_hydration_seconds": primary[
                                "hydration_seconds"
                            ],
                            "wide_hydration_seconds": wide["hydration_seconds"],
                            "state_wall_seconds": float(time.perf_counter() - state_start),
                            "opening_room_path_json": "",
                        })
                        if completed_picks == 0:
                            primary_room_path = primary["selected_room_zero_path"]
                            record["opening_room_path_json"] = json.dumps(primary_room_path)
                        records.append(record)
                        print(
                            f"complete {state_index}/{args.expected_states} "
                            f"base={base_seed} stream={stream_seed} slot={pick_slot} "
                            f"round={completed_picks + 1} d128={record['d128_action']} "
                            f"d256={record['d256_action']} wide={record['wide_d256_action']} "
                            f"delta={record['d256_minus_d128_value']:.3f} "
                            f"coverage={record['wide_minus_primary_reference_value']:.3f}",
                            flush=True,
                        )
                        state_complete = True
                    except Exception as exc:
                        state_complete = False
                        traceback_text = traceback.format_exc()
                        records.append({
                            "league": "dk",
                            "base_seed": base_seed,
                            "stream_seed": stream_seed,
                            "pick_slot": pick_slot,
                            "trajectory_id": f"{base_seed}:{pick_slot}",
                            "completed_picks": completed_picks,
                            "current_round": completed_picks + 1,
                            "status": "error",
                            "message": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback_text,
                            "state_wall_seconds": float(time.perf_counter() - state_start),
                            "opening_room_path_json": "",
                        })
                        print(
                            f"error {state_index}/{args.expected_states} base={base_seed} "
                            f"slot={pick_slot} round={completed_picks + 1}: "
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        print(traceback_text, flush=True)
                        if args.fail_fast:
                            raise
                        if completed_picks == 0:
                            break
                    finally:
                        write_checkpoint(records, args, receipt)
                        if state_complete:
                            clear_committed_in_progress_journal(
                                args,
                                receipt,
                                primary_job,
                                wide_job,
                                primary,
                                wide,
                            )
        unchanged = verify_source_unchanged(args, receipt, conn)
        write_checkpoint(records, args, receipt, source_unchanged=unchanged)
        summary = summarize(records, args, receipt, source_unchanged=unchanged)
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
