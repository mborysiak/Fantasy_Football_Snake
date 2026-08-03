"""Leakage-safe rolling-origin forced-pick replay for DK snake best ball.

The study deliberately has two phases.  ``freeze`` creates every synthetic
draft state, recommendation, and downstream roster without reading the target
season outcomes.  ``score`` refuses to run until that complete artifact is
sealed, then evaluates the frozen rosters on target-season weekly outcomes.

This file is research-only.  It imports the production simulation engine but
does not modify production app code or databases.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

if __name__ == "__main__":
    _bootstrap_model_repo = Path(__file__).resolve().parents[4] / "Fantasy_Football"
    _bootstrap_interpreter = (
        _bootstrap_model_repo / ".venv_ff_312" / "Scripts" / "python.exe"
    ).resolve()
    _bootstrap_actual = Path(sys.executable).resolve()
    if (
        not _bootstrap_interpreter.is_file()
        or str(_bootstrap_actual).casefold() != str(_bootstrap_interpreter).casefold()
        or sys.version_info[:2] != (3, 12)
    ):
        raise RuntimeError(
            "Use the maintained Python 3.12 model interpreter before loading "
            f"historical replay dependencies: {_bootstrap_interpreter}; current "
            f"interpreter is {_bootstrap_actual} ({platform.python_version()})."
        )

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
MODEL_REPO = REPO_ROOT.parent / "Fantasy_Football"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.zSim_Helper import (  # noqa: E402
    SEQUENTIAL_DECISION_EXTENSION_SEED_OFFSET,
    SEQUENTIAL_DECISION_PREFIX_SAMPLES,
    SEQUENTIAL_STACK_BONUS_PCT,
    SEQUENTIAL_STACK_PAIR_CAP,
    SEQUENTIAL_STACK_TEAM_CAP,
    FootballSimulation,
)


POSITION_RANGES = {
    "QB": (2, 3),
    "RB": (5, 7),
    "WR": (7, 9),
    "TE": (2, 3),
}
POS_REQUIRE = {position: maximum for position, (_, maximum) in POSITION_RANGES.items()}

# These constants are the preregistered/default design.  Changing a CLI value
# makes ``frozen_design_exact`` false and therefore cannot pass the study gate.
FROZEN_ORIGINS = list(range(2017, 2026))
FROZEN_SLOTS = [1, 6, 12]
FROZEN_DEPTHS = [0, 7, 14]
FROZEN_TEAMS = 12
FROZEN_ROUNDS = 20
FROZEN_ROOMS = 24
FROZEN_CANDIDATES = 24
FROZEN_CONSTRUCTION_SAMPLES = 16
FROZEN_EVALUATION_SAMPLES = 64
FROZEN_D128 = 128
FROZEN_D256 = 256
FROZEN_SEED_BASE = 20260802
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260803
NONINFERIORITY_MARGIN_PCT = -0.25
DECISION_EXTENSION_SEED_OFFSET = 1_000_003
WEEKLY_HORIZON = 16
ARMS = ("d128", "d256")
FREEZE_CHILD_PROTOCOL = 1
FREEZE_CHILD_STAGES = ("nested", "forced")
RUNTIME_DISTRIBUTIONS = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "cvxopt",
    "PuLP",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_csv_values(raw: str, value_type=int) -> list:
    return [value_type(value.strip()) for value in raw.split(",") if value.strip()]


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def maintained_interpreter(model_repo: Path) -> Path:
    return (
        Path(model_repo).expanduser().resolve()
        / ".venv_ff_312"
        / "Scripts"
        / "python.exe"
    ).resolve()


def assert_maintained_interpreter(model_repo: Path) -> Path:
    expected = maintained_interpreter(model_repo)
    actual = Path(sys.executable).resolve()
    if not expected.is_file():
        raise FileNotFoundError(f"Maintained model interpreter is missing: {expected}")
    if str(actual).casefold() != str(expected).casefold() or sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "Historical replay must run under the maintained Python 3.12 model "
            f"interpreter: {expected}; current interpreter is {actual} "
            f"({platform.python_version()})."
        )
    return expected


def runtime_contract() -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    distributions: dict[str, str | None] = {}
    for distribution in RUNTIME_DISTRIBUTIONS:
        try:
            distributions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            distributions[distribution] = None
    return {
        "python": sys.version,
        "python_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": str(executable),
        "executable_sha256": sha256_file(executable),
        "distributions": distributions,
    }


def code_contract(args) -> dict[str, Any]:
    weekly_builder = args.model_repo / "Scripts" / "Modeling" / "s4_Best_Ball_Weekly.py"
    paths = {
        "run_study.py": Path(__file__),
        "historical_data.py": STUDY_DIR / "historical_data.py",
        "app/zSim_Helper.py": REPO_ROOT / "app" / "zSim_Helper.py",
        "app/simulation_worker.py": REPO_ROOT / "app" / "simulation_worker.py",
        "Scripts/Modeling/s4_Best_Ball_Weekly.py": weekly_builder,
        "Scripts/V2/build_player_outcomes.py": args.model_repo
        / "Scripts"
        / "V2"
        / "build_player_outcomes.py",
        "Scripts/V2/contracts.py": args.model_repo / "Scripts" / "V2" / "contracts.py",
        "Scripts/V2/config.py": args.model_repo / "Scripts" / "V2" / "config.py",
        "Scripts/config.py": args.model_repo / "Scripts" / "config.py",
    }
    missing = [label for label, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Study code contract is missing files: {missing}")
    return {
        label: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for label, path in paths.items()
    }


def domain_seed(seed_base: int, *parts: Any) -> int:
    """Stable, domain-separated seed below NumPy's signed 32-bit ceiling."""
    digest = hashlib.blake2b(
        "|".join([str(seed_base), *map(str, parts)]).encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little") % (2**31 - 1)


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(json_safe(value), indent=2, sort_keys=True))


def atomic_write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


@dataclass(frozen=True)
class SyntheticState:
    depth: int
    to_add: tuple[str, ...]
    to_drop: tuple[str, ...]
    adp_column: int
    board_seed: int


class NestedDecisionSimulation(FootballSimulation):
    """Allocate D128 exactly as production, then append an independent D128."""

    def __init__(self, *args, control_samples: int, expanded_samples: int, **kwargs):
        self.study_control_samples = int(control_samples)
        self.study_expanded_samples = int(expanded_samples)
        self.study_decision_superbank = np.zeros(0, dtype=np.int64)
        super().__init__(*args, **kwargs)

    def select_additional_policy_ppg_columns(
        self,
        num_columns,
        excluded_columns,
        samples,
        seed,
        bank_name,
    ):
        samples = int(samples)
        if bank_name != "Decision" or samples <= 0:
            return super().select_additional_policy_ppg_columns(
                num_columns,
                excluded_columns,
                samples,
                seed,
                bank_name,
            )

        if self.study_expanded_samples < self.study_control_samples:
            raise ValueError("The expanded decision bank is smaller than D128.")
        if self.study_control_samples == SEQUENTIAL_DECISION_PREFIX_SAMPLES:
            self.study_decision_superbank = (
                super().select_additional_policy_ppg_columns(
                    num_columns,
                    excluded_columns,
                    self.study_expanded_samples,
                    seed,
                    bank_name,
                )
            )
        else:
            # Mechanical smokes use D8/D16. Production nesting is defined for
            # D128/D256, so reproduce the same prefix-plus-independent-extension
            # construction at the smoke sizes instead of assuming that one
            # NumPy choice call is prefix-stable across arbitrary sample sizes.
            prefix = super().select_additional_policy_ppg_columns(
                num_columns,
                excluded_columns,
                self.study_control_samples,
                seed,
                bank_name,
            )
            extension = super().select_additional_policy_ppg_columns(
                num_columns,
                np.concatenate([excluded_columns, prefix]),
                self.study_expanded_samples - self.study_control_samples,
                int(seed) + SEQUENTIAL_DECISION_EXTENSION_SEED_OFFSET,
                "DecisionExtension",
            )
            self.study_decision_superbank = np.concatenate([prefix, extension])
        if samples > len(self.study_decision_superbank):
            raise ValueError("Requested decision samples exceed the nested superbank.")
        return self.study_decision_superbank[:samples].copy()


class ForcedRootSimulation(FootballSimulation):
    """Use the production downstream policy for an explicitly frozen root set."""

    def __init__(self, *args, forced_root_keys: Sequence[str], **kwargs):
        self.forced_root_keys = tuple(dict.fromkeys(map(str, forced_root_keys)))
        self._active_player_ids: np.ndarray | None = None
        super().__init__(*args, **kwargs)

    def run_sim_best_ball_policy(self, to_add, to_drop, *args, **kwargs):
        active = self.drop_players(self.player_data, set(to_drop))
        self._active_player_ids = self.identity_values(active, validate_unique=True)
        return super().run_sim_best_ball_policy(to_add, to_drop, *args, **kwargs)

    def select_sequential_root_candidates(
        self,
        candidate_indices,
        immediate_values,
        policy_scores,
        survival,
        player_positions,
        candidate_pool_size,
        position_quotas=None,
    ):
        del immediate_values, policy_scores, survival, player_positions
        del candidate_pool_size, position_quotas
        if self._active_player_ids is None:
            raise RuntimeError("Forced-root identities were not initialized.")
        legal = set(np.asarray(candidate_indices, dtype=np.int64).tolist())
        index_by_key = {
            str(player_key): idx for idx, player_key in enumerate(self._active_player_ids)
        }
        missing = [key for key in self.forced_root_keys if key not in index_by_key]
        illegal = [key for key in self.forced_root_keys if index_by_key.get(key) not in legal]
        if missing or illegal:
            raise ValueError(f"Forced actions are absent or illegal: missing={missing}, illegal={illegal}")
        selected = np.asarray([index_by_key[key] for key in self.forced_root_keys], dtype=np.int64)
        return selected, {int(idx): ("frozen_forced_action",) for idx in selected}


def simulation_kwargs(origin, args, pick_slot: int) -> dict[str, Any]:
    return {
        "conn": origin.conn,
        "set_year": int(origin.set_year),
        "pos_require_start": POS_REQUIRE,
        "num_teams": args.teams,
        "num_rounds": args.rounds,
        "my_pick_position": pick_slot,
        "pred_vers": str(origin.pred_vers),
        "league": "dk",
        "position_ranges": POSITION_RANGES,
        "template_resid_blend": 1.0,
    }


def make_nested_sim(origin, args, pick_slot: int) -> NestedDecisionSimulation:
    return NestedDecisionSimulation(
        **simulation_kwargs(origin, args, pick_slot),
        use_stack_bonus=True,
        stack_bonus_pct=SEQUENTIAL_STACK_BONUS_PCT,
        stack_pair_cap=SEQUENTIAL_STACK_PAIR_CAP,
        stack_team_cap=SEQUENTIAL_STACK_TEAM_CAP,
        control_samples=args.control_decision_samples,
        expanded_samples=args.expanded_decision_samples,
    )


def make_forced_sim(origin, args, pick_slot: int, action_keys: Sequence[str]):
    return ForcedRootSimulation(
        **simulation_kwargs(origin, args, pick_slot),
        use_stack_bonus=True,
        stack_bonus_pct=SEQUENTIAL_STACK_BONUS_PCT,
        stack_pair_cap=SEQUENTIAL_STACK_PAIR_CAP,
        stack_team_cap=SEQUENTIAL_STACK_TEAM_CAP,
        forced_root_keys=action_keys,
    )


def make_legacy_sim(origin, args, pick_slot: int) -> FootballSimulation:
    return FootballSimulation(
        **simulation_kwargs(origin, args, pick_slot),
        use_stack_bonus=True,
        stack_bonus_pct=0.25,
        stack_pair_cap=12.0,
        stack_team_cap=18.0,
    )


def identity_maps(sim: FootballSimulation) -> tuple[dict[str, str], dict[str, str]]:
    identity_column = sim.identity_column(sim.player_data)
    frame = sim.player_data[[identity_column, "player"]].copy()
    if frame[identity_column].duplicated().any() or frame.player.duplicated().any():
        raise ValueError("Historical runtime identities and display names must both be unique.")
    name_to_key = dict(zip(frame.player.astype(str), frame[identity_column].astype(str)))
    key_to_name = dict(zip(frame[identity_column].astype(str), frame.player.astype(str)))
    return name_to_key, key_to_name


def build_synthetic_adp_states(
    sim: FootballSimulation,
    depths: Sequence[int],
    board_seed: int,
) -> dict[int, SyntheticState]:
    """Create one nested, neutral board prefix using ADP and legality only."""
    depths = sorted(set(map(int, depths)))
    if not depths or depths[0] < 0 or depths[-1] >= sim.num_rounds:
        raise ValueError("Synthetic depths must be between 0 and rounds - 1.")
    with sim.temp_seed(board_seed):
        adp_samples = sim.get_adp_samples(num_options=1000)
    player_ids = sim.identity_values(adp_samples, validate_unique=True)
    positions = adp_samples.pos.to_numpy()
    adp_matrix = adp_samples[sim.sample_value_columns(adp_samples)].to_numpy(dtype=np.float32)
    orders, adp_columns = sim.build_sequential_draft_orders(
        adp_matrix,
        1,
        seed=board_seed + 303,
    )
    order = orders[0]
    remaining = np.ones(len(player_ids), dtype=bool)
    selected: list[int] = []
    opponents: list[int] = []
    pointer = 0

    def advance(count: int) -> None:
        nonlocal pointer
        pointer, drafted = sim.advance_sequential_opponents(
            remaining,
            order,
            pointer,
            count,
        )
        if len(drafted) != count:
            raise ValueError("Synthetic ADP board exhausted before the requested prefix.")
        opponents.extend(map(int, drafted))

    advance(max(int(sim.my_picks[0]) - 1, 0))
    states: dict[int, SyntheticState] = {}
    if 0 in depths:
        states[0] = SyntheticState(
            0,
            (),
            tuple(map(str, player_ids[opponents])),
            int(adp_columns[0]),
            int(board_seed),
        )

    for completed_depth in range(1, max(depths) + 1):
        picks_left = sim.num_rounds - (completed_depth - 1)
        legal = sim.sequential_legal_candidate_indices(
            remaining,
            positions,
            np.asarray(selected, dtype=np.int64),
            picks_left,
            pos_ranges=POSITION_RANGES,
        )
        legal_set = set(map(int, legal))
        choice = next((int(idx) for idx in order if remaining[idx] and int(idx) in legal_set), None)
        if choice is None:
            raise ValueError("Neutral ADP drafter has no legal user selection.")
        selected.append(choice)
        remaining[choice] = False
        next_user_pick = int(sim.my_picks[completed_depth])
        current_user_pick = int(sim.my_picks[completed_depth - 1])
        advance(next_user_pick - current_user_pick - 1)
        if completed_depth in depths:
            states[completed_depth] = SyntheticState(
                completed_depth,
                tuple(map(str, player_ids[selected])),
                tuple(map(str, player_ids[opponents])),
                int(adp_columns[0]),
                int(board_seed),
            )
    if set(states) != set(depths):
        raise AssertionError("Not every configured synthetic depth was generated.")
    return states


def run_nested_policy(sim, args, state: SyntheticState, policy_seed: int):
    return sim.run_sim_best_ball_policy(
        state.to_add,
        state.to_drop,
        num_iters=args.rooms,
        construction_samples=args.construction_samples,
        evaluation_samples=args.evaluation_samples,
        decision_samples=args.expanded_decision_samples,
        decision_candidate_count=args.candidates,
        audit_samples=0,
        candidate_pool_size=args.candidates,
        seed=policy_seed,
        evaluation_seed=policy_seed + 202,
        decision_seed=policy_seed + 404,
        audit_seed=policy_seed + 505,
    )


def run_legacy(sim, args, state: SyntheticState, policy_seed: int):
    with sim.temp_seed(policy_seed):
        return sim.run_sim_best_ball_ilp(
            state.to_add,
            state.to_drop,
            args.rooms,
            num_weeks=WEEKLY_HORIZON,
            weekly_score_mode="template",
            current_pick_ev=False,
            parallel_workers=1,
        )


def run_forced_rollout(
    sim,
    args,
    state: SyntheticState,
    policy_seed: int,
    action_count: int,
):
    return sim.run_sim_best_ball_policy(
        state.to_add,
        state.to_drop,
        num_iters=args.rooms,
        construction_samples=args.construction_samples,
        evaluation_samples=args.evaluation_samples,
        decision_samples=0,
        decision_candidate_count=max(1, action_count),
        audit_samples=0,
        candidate_pool_size=max(1, action_count),
        seed=policy_seed,
        evaluation_seed=policy_seed + 202,
        decision_seed=policy_seed + 404,
        audit_seed=policy_seed + 505,
    )


def nested_actions(result, control_samples: int) -> tuple[str, str, dict[str, float]]:
    matrices = result.attrs["decision_value_matrices"]
    if not matrices:
        raise ValueError("Sequential policy produced no decision matrices.")
    ranking_rows = []
    for player, matrix in matrices.items():
        # Preserve production float32 addition/mean semantics exactly; a
        # float64 research recast could change a knife-edge tie.
        values = np.asarray(matrix["values"])
        stack = np.asarray(matrix["stack_utilities"]).reshape(-1, 1)
        adjusted = values + stack
        if adjusted.shape[1] < control_samples:
            raise ValueError("Nested decision matrix is shorter than D128.")
        result_row = result.loc[result.player.eq(player)].iloc[0]
        ranking_rows.append({
            "player": str(player),
            "d128_adjusted": float(adjusted[:, :control_samples].mean()),
            "d128_raw": float(values[:, :control_samples].mean()),
            "d256_adjusted": float(adjusted.mean()),
            "d256_raw": float(values.mean()),
            "pilot_rank": int(result_row.PilotRank),
        })
    ranking = pd.DataFrame(ranking_rows)
    d128 = str(
        ranking.sort_values(
            ["d128_adjusted", "pilot_rank"],
            ascending=[False, True],
            kind="mergesort",
        ).iloc[0].player
    )
    d256 = str(
        ranking.sort_values(
            ["d256_adjusted", "pilot_rank"],
            ascending=[False, True],
            kind="mergesort",
        ).iloc[0].player
    )
    if d256 != str(result.attrs["decision_top_player"]):
        raise AssertionError("D256 action does not match the production ranking output.")
    return d128, d256, {
        "d128_score": float(
            ranking.loc[ranking.player.eq(d128), "d128_adjusted"].iloc[0]
        ),
        "d256_score": float(
            ranking.loc[ranking.player.eq(d256), "d256_adjusted"].iloc[0]
        ),
    }


def legacy_current_action(result, depth: int) -> str:
    round_number = int(depth) + 1
    count_column = f"Round{round_number}Count"
    available_column = f"Round{round_number}Available"
    required = {"player", count_column, available_column, "TotalSelectionCounts"}
    if not required.issubset(result.columns):
        raise ValueError(f"Legacy output lacks current-round columns: {sorted(required - set(result.columns))}")
    choices = result[(result[available_column] > 0) & (result[count_column] > 0)].copy()
    if choices.empty:
        raise ValueError("Legacy produced no current-pick action.")
    # Match the app's ``round_results.nlargest(..., count_col)`` exactly.  On a
    # count tie, nlargest keeps the inherited helper-result order (already
    # ordered by TotalSelectionCounts); an added alphabetical key would change
    # the user-visible Legacy action on a complete tie.
    return str(choices.nlargest(1, count_column).iloc[0].player)


def assert_physical_state(
    sim: FootballSimulation,
    state: SyntheticState,
) -> None:
    if len(state.to_add) != state.depth:
        raise ValueError("Synthetic state user-pick count does not match its depth.")
    if len(state.to_add) != len(set(state.to_add)):
        raise ValueError("Synthetic state contains duplicate user players.")
    if len(state.to_drop) != len(set(state.to_drop)):
        raise ValueError("Synthetic state contains duplicate opponent players.")
    if set(state.to_add) & set(state.to_drop):
        raise ValueError("Synthetic user and opponent player sets overlap.")
    sim.validate_selection_coverage(state.to_add, state.to_drop)
    adjusted = sim.calculate_adjusted_picks(state.depth)
    if not adjusted:
        raise ValueError("Configured state has no remaining user pick.")
    expected_opponents = int(adjusted[0] - 1 - state.depth)
    if len(state.to_drop) != expected_opponents:
        raise ValueError(
            f"Synthetic state has {len(state.to_drop)} opponent picks; "
            f"physical schedule requires {expected_opponents}."
        )


def freeze_forced_rosters(
    sim: FootballSimulation,
    result,
    action_keys: Sequence[str],
    state: SyntheticState,
    name_to_key: dict[str, str],
    args,
) -> dict[str, list[dict[str, Any]]]:
    roster_validator = importlib.import_module(
        "historical_data"
    ).validate_final_roster_state_contract
    paths = result.attrs["policy_paths"]
    frozen: dict[str, list[dict[str, Any]]] = {}
    key_to_name = {key: name for name, key in name_to_key.items()}
    identity_column = sim.identity_column(sim.player_data)
    position_by_key = dict(
        zip(
            sim.player_data[identity_column].astype(str),
            sim.player_data.pos.astype(str),
        )
    )
    for action_key in action_keys:
        action_name = key_to_name[action_key]
        action_paths = sorted(paths.get(action_name, []), key=lambda row: int(row["room_idx"]))
        if len(action_paths) != args.rooms:
            raise ValueError(f"Forced action {action_name} completed {len(action_paths)} of {args.rooms} rooms.")
        if [int(path["room_idx"]) for path in action_paths] != list(range(args.rooms)):
            raise ValueError(f"Forced action {action_name} does not cover exact room IDs 0..{args.rooms - 1}.")
        rows = []
        for path in action_paths:
            path_keys = [name_to_key[str(name)] for name in path["path"]]
            if not path_keys or path_keys[0] != action_key:
                raise ValueError(f"Forced path for {action_name} does not begin with its frozen action.")
            roster = [*state.to_add, *path_keys]
            roster = roster_validator(
                roster,
                to_add=state.to_add,
                to_drop=state.to_drop,
                depth=state.depth,
                action_key=action_key,
                rounds=args.rounds,
                label=f"Forced action {action_name} room {path['room_idx']}",
            )
            missing_positions = [key for key in roster if key not in position_by_key]
            if missing_positions:
                raise ValueError(f"Final roster contains unknown identities: {missing_positions}")
            positions = pd.Series([position_by_key[key] for key in roster]).value_counts()
            unsupported = set(positions.index) - set(POSITION_RANGES)
            if unsupported:
                raise ValueError(f"Final roster contains unsupported positions: {sorted(unsupported)}")
            for position, (minimum, maximum) in POSITION_RANGES.items():
                count = int(positions.get(position, 0))
                if not minimum <= count <= maximum:
                    raise ValueError(
                        f"Final roster violates {position} range {minimum}-{maximum}: {count}"
                    )
            rows.append({"room_idx": int(path["room_idx"]), "player_keys": roster})
        frozen[action_key] = rows
    return frozen


def validate_origin_contract(origin, origin_year: int) -> dict[str, Any]:
    origin.assert_decision_inputs_clean()
    if str(origin.league).lower() != "dk":
        raise ValueError("Historical replay is frozen to DK.")
    if int(origin.set_year) != int(origin_year):
        raise ValueError("Historical origin set_year mismatch.")
    donors = sorted(set(map(int, origin.donor_years)))
    if not donors or max(donors) >= origin_year:
        raise ValueError(f"Origin {origin_year} does not have a strict-prior donor set: {donors[-5:]}")
    quick_check = origin.conn.execute("PRAGMA quick_check").fetchone()[0]
    if quick_check != "ok":
        raise ValueError(f"Origin database quick_check failed: {quick_check}")
    database_hash = sha256_file(Path(origin.db_path))
    fingerprint = str(getattr(origin, "source_fingerprint", "") or database_hash)
    return {
        "origin_year": int(origin_year),
        "donor_years": donors,
        "strict_prior": True,
        "database_sha256": database_hash,
        "source_fingerprint": fingerprint,
        "receipt": json_safe(origin.receipt),
    }


def validate_simulation_helper_path(contract: dict[str, Any]) -> None:
    imported = sys.modules.get("app.zSim_Helper")
    if imported is None or not getattr(imported, "__file__", None):
        raise ImportError("Production simulation helper module is not loaded.")
    observed = Path(imported.__file__).resolve()
    expected = Path(contract["app/zSim_Helper.py"]["path"]).resolve()
    if observed != expected:
        raise ImportError(
            f"Production simulation helper imported from {observed}, expected {expected}."
        )


def validate_adapter_module_paths(module, args) -> None:
    contract = getattr(args, "sealed_code_contract", None)
    if not isinstance(contract, dict):
        raise ValueError("Historical adapter import lacks a sealed code contract.")
    observed_modules = {
        "app/zSim_Helper.py": sys.modules.get("app.zSim_Helper"),
        "historical_data.py": module,
        "Scripts/Modeling/s4_Best_Ball_Weekly.py": getattr(
            module, "weekly_builder", None
        ),
        "Scripts/V2/build_player_outcomes.py": getattr(
            module, "governed_outcomes", None
        ),
        "Scripts/V2/contracts.py": getattr(module, "v2_contracts", None),
        "Scripts/V2/config.py": getattr(module, "v2_config", None),
        "Scripts/config.py": getattr(module, "scripts_config", None),
    }
    for label, imported in observed_modules.items():
        if imported is None or not getattr(imported, "__file__", None):
            raise ImportError(f"Historical adapter did not import contracted module {label}.")
        observed = Path(imported.__file__).resolve()
        expected = Path(contract[label]["path"]).resolve()
        if observed != expected:
            raise ImportError(
                f"Historical adapter imported {label} from {observed}, expected {expected}."
            )
    weekly_config = getattr(module, "weekly_config", None)
    if (
        weekly_config is None
        or not getattr(weekly_config, "__file__", None)
        or Path(weekly_config.__file__).resolve()
        != Path(contract["Scripts/config.py"]["path"]).resolve()
    ):
        raise ImportError("Historical adapter weekly config import is outside sealed code.")


@contextlib.contextmanager
def open_origin(args, origin_year: int):
    """Late-bound adapter supplied by the adjacent ``historical_data.py``.

    Required helper interface is documented in README.md.  Keeping this import
    late lets ``--dry-run`` and syntax checks work while origin assembly is
    developed independently.
    """
    module = importlib.import_module("historical_data")
    validate_adapter_module_paths(module, args)
    if not hasattr(module, "open_origin"):
        raise RuntimeError(
            "historical_data.py must expose open_origin(source_db=..., "
            "origin_year=..., work_dir=..., league='dk', strict_prior=True, smoke=...)."
        )
    manager = module.open_origin(
        source_db=args.model_repo,
        origin_year=int(origin_year),
        work_dir=args.work_dir,
        league="dk",
        strict_prior=True,
        smoke=bool(args.smoke),
    )
    with manager as origin:
        required = (
            "db_path",
            "set_year",
            "pred_vers",
            "league",
            "predictions",
            "donor_years",
            "receipt",
            "source_fingerprint",
            "target_outcome_fingerprint",
            "assert_decision_inputs_clean",
            "assert_target_outcomes_unread",
            "score_rosters",
        )
        missing = [name for name in required if not hasattr(origin, name)]
        if missing:
            raise RuntimeError(f"Historical origin adapter is missing: {missing}")
        # HistoricalOriginData intentionally owns only a disposable path.  The
        # runner owns and closes the read/write SQLite handle consumed by the
        # production simulation class.
        with contextlib.closing(sqlite3.connect(origin.db_path)) as connection:
            connection.execute("PRAGMA query_only=ON")
            if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("Disposable historical database failed quick_check.")
            origin.conn = connection
            try:
                yield origin
            finally:
                origin.conn = None


def state_id(origin_year: int, slot: int, depth: int) -> str:
    return f"dk:{origin_year}:slot{slot}:depth{depth}"


def configured_state_ids(args) -> list[str]:
    return [
        state_id(origin_year, slot, depth)
        for origin_year in args.origin_values
        for slot in args.slot_values
        for depth in args.depth_values
    ]


def frozen_design_exact(args) -> bool:
    return bool(
        args.origin_values == FROZEN_ORIGINS
        and args.slot_values == FROZEN_SLOTS
        and args.depth_values == FROZEN_DEPTHS
        and args.teams == FROZEN_TEAMS
        and args.rounds == FROZEN_ROUNDS
        and args.rooms == FROZEN_ROOMS
        and args.candidates == FROZEN_CANDIDATES
        and args.construction_samples == FROZEN_CONSTRUCTION_SAMPLES
        and args.evaluation_samples == FROZEN_EVALUATION_SAMPLES
        and args.control_decision_samples == FROZEN_D128
        and args.expanded_decision_samples == FROZEN_D256
        and args.seed_base == FROZEN_SEED_BASE
        and args.bootstrap_draws == BOOTSTRAP_DRAWS
        and args.bootstrap_seed == BOOTSTRAP_SEED
    )


def design_payload(args) -> dict[str, Any]:
    design = {
        "study": "sequential_historical_forced_pick_replay",
        "league": "dk",
        "origins": args.origin_values,
        "slots": args.slot_values,
        "completed_user_picks": args.depth_values,
        "synthetic_board": "single_noisy_adp_priority_with_adp_only_legal_user_picks",
        "teams": args.teams,
        "rounds": args.rounds,
        "rooms": args.rooms,
        "candidates": args.candidates,
        "construction_samples": args.construction_samples,
        "evaluation_samples": args.evaluation_samples,
        "decision_samples": {"d128": args.control_decision_samples, "d256": args.expanded_decision_samples},
        "comparison_estimand": "d256_vs_exact_d128_prefix_with_common_sequential_downstream_policy",
        "legacy_in_approval_process": False,
        "strict_prior_donors": True,
        "target_outcomes": (
            "held_out_origin_configured_dk_all_played_weekly_outcomes_"
            "only_after_global_freeze"
        ),
        "target_scoring_scope": (
            "configured-DK all-played; not official contest-realized DK; "
            "repo scoring dictionaries omit two-point and individual return/ST TD"
        ),
        "bootstrap": {
            "cluster": "origin_season",
            "draws": args.bootstrap_draws,
            "seed": args.bootstrap_seed,
        },
        "primary_gate": {
            "contrast": "d256_minus_d128_target_best_ball_points",
            "noninferiority_margin_pct": NONINFERIORITY_MARGIN_PCT,
            "interval": "two_sided_95pct_season_cluster_bootstrap",
            "margin_points_formula": "-0.0025 * observed_paired_d128_mean_points",
        },
        "seed_base": args.seed_base,
        "code_contract": code_contract(args),
        "frozen_design_exact": frozen_design_exact(args),
        "expected_states": len(configured_state_ids(args)),
        "created_before_outcome_scoring": True,
    }
    design["design_hash"] = stable_hash(design)
    return design


def prepare_design(args) -> dict[str, Any]:
    design = design_payload(args)
    path = args.output_dir / "design.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing_payload = dict(existing)
        existing_hash = existing_payload.pop("design_hash", None)
        if existing_hash != stable_hash(existing_payload):
            raise ValueError("Existing design artifact has an invalid payload hash.")
        if existing != json_safe(design):
            raise ValueError("Existing output directory contains a different frozen design.")
        return existing
    atomic_write_json(path, design)
    return design


def prepare_runner_receipt(args, design: dict[str, Any]) -> dict[str, Any]:
    path = args.output_dir / "runner_receipt.json"
    immutable = {
        "git_head": git_head(),
        "design_hash": design["design_hash"],
        "code_contract": design["code_contract"],
        "runtime": runtime_contract(),
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if (
            "immutable" not in existing
            or existing.get("receipt_hash") != stable_hash(existing["immutable"])
        ):
            raise ValueError("Existing runner receipt has an invalid immutable-payload hash.")
        if existing.get("immutable") != json_safe(immutable):
            raise ValueError(
                "Existing runner receipt was produced by different code/runtime; "
                "use a new output directory or restart the freeze cleanly."
            )
        return existing
    receipt = {"created_at_utc": utc_now(), "immutable": immutable}
    receipt["receipt_hash"] = stable_hash(receipt["immutable"])
    atomic_write_json(path, receipt)
    return receipt


def load_frozen_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        key = record["state_id"]
        if key in records:
            raise ValueError(f"Duplicate frozen state {key} on line {line_number}.")
        if record["freeze_hash"] != stable_hash(record["freeze_payload"]):
            raise ValueError(f"Frozen state hash mismatch for {key}.")
        records[key] = record
    return records


def write_frozen_records(path: Path, records: dict[str, dict[str, Any]], args) -> None:
    order = {key: idx for idx, key in enumerate(configured_state_ids(args))}
    lines = [
        json.dumps(records[key], sort_keys=True, separators=(",", ":"))
        for key in sorted(records, key=lambda key: order.get(key, 10**9))
    ]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))
    summary = []
    for record in records.values():
        payload = record["freeze_payload"]
        summary.append({
            "state_id": record["state_id"],
            "origin_year": payload["origin_year"],
            "pick_slot": payload["pick_slot"],
            "completed_picks": payload["completed_picks"],
            "d128_action": payload["actions"]["d128"]["player"],
            "d256_action": payload["actions"]["d256"]["player"],
            "unique_actions": len(payload["rosters_by_action"]),
            "rooms": args.rooms,
        })
    atomic_write_frame(args.output_dir / "frozen_action_summary.csv", pd.DataFrame(summary))


def synthetic_state_receipt(state: SyntheticState) -> dict[str, Any]:
    return {
        "depth": int(state.depth),
        "to_add": list(state.to_add),
        "to_drop": list(state.to_drop),
        "adp_column": int(state.adp_column),
        "board_seed": int(state.board_seed),
    }


def build_stage_state(sim, args, origin_year: int, slot: int, depth: int) -> SyntheticState:
    board_seed = domain_seed(args.seed_base, "board", origin_year, slot)
    state = build_synthetic_adp_states(sim, [depth], board_seed)[int(depth)]
    assert_physical_state(sim, state)
    return state


def assert_v2_template_contract(sim: FootballSimulation, label: str) -> None:
    if (
        not sim.uses_v2_joint_template
        or sim.template_resid_method_version != "joint_centered_template_v2_v1"
    ):
        raise ValueError(f"{label} requires the V2 joint-template handoff.")
    if sim.load_weekly_template_profiles() != WEEKLY_HORIZON:
        raise ValueError(f"{label} does not have exactly 16 weekly template columns.")


def freeze_nested_stage(origin, args, slot: int, depth: int, design_hash: str) -> dict[str, Any]:
    start = time.perf_counter()
    policy_seed = domain_seed(args.seed_base, "policy", origin.set_year, slot, depth)
    sim = make_nested_sim(origin, args, slot)
    state = build_stage_state(sim, args, int(origin.set_year), slot, depth)
    assert_v2_template_contract(sim, "Nested historical replay")
    name_to_key, _ = identity_maps(sim)
    result = run_nested_policy(sim, args, state, policy_seed)
    if len(result) != args.candidates or not (result.PolicyCompletedRooms == args.rooms).all():
        raise ValueError("Sequential candidate screen or rollout completion contract failed.")
    d128_name, d256_name, decision_scores = nested_actions(
        result,
        args.control_decision_samples,
    )
    actions = {
        "d128": {"player": d128_name, "player_key": name_to_key[d128_name]},
        "d256": {"player": d256_name, "player_key": name_to_key[d256_name]},
    }
    selected_paths: dict[str, Any] = {}
    for action in actions.values():
        action_name = action["player"]
        action_key = action["player_key"]
        if action_name not in result.attrs["policy_paths"]:
            raise ValueError(f"Nested rollout lacks the selected path for {action_name}.")
        paths = sorted(
            result.attrs["policy_paths"][action_name],
            key=lambda row: int(row["room_idx"]),
        )
        if [int(path["room_idx"]) for path in paths] != list(range(args.rooms)):
            raise ValueError(f"Nested selected path lacks exact rooms for {action_name}.")
        selected_paths[action_key] = json_safe(paths)

    banks = result.attrs["scenario_banks"]
    decision_columns = list(map(int, banks["decision_ppg_columns"]))
    if len(decision_columns) != args.expanded_decision_samples:
        raise AssertionError("Expanded decision-bank size mismatch.")
    if decision_columns != sim.study_decision_superbank.tolist():
        raise AssertionError("Nested decision-bank receipt mismatch.")
    construction_columns = np.asarray(
        banks["construction_ppg_columns"], dtype=np.int64
    )
    evaluation_columns = np.asarray(
        banks["evaluation_ppg_columns"], dtype=np.int64
    )
    production_control = FootballSimulation.select_additional_policy_ppg_columns(
        1000,
        np.concatenate([construction_columns, evaluation_columns]),
        args.control_decision_samples,
        policy_seed + 404,
        "Decision",
    )
    if not np.array_equal(
        production_control,
        np.asarray(decision_columns[: args.control_decision_samples]),
    ):
        raise AssertionError("Nested control prefix is not the exact production allocation.")
    return {
        "stage": "nested",
        "design_hash": design_hash,
        "state_id": state_id(int(origin.set_year), slot, depth),
        "origin_year": int(origin.set_year),
        "pick_slot": int(slot),
        "completed_picks": int(depth),
        "synthetic_state": synthetic_state_receipt(state),
        "policy_seed": int(policy_seed),
        "actions": actions,
        "decision_scores": decision_scores,
        "selected_policy_paths": selected_paths,
        "bank_contract": {
            "construction_columns": list(map(int, banks["construction_ppg_columns"])),
            "evaluation_columns": list(map(int, banks["evaluation_ppg_columns"])),
            "d128_prefix_columns": decision_columns[: args.control_decision_samples],
            "d256_columns_hash": stable_hash(decision_columns),
            "decision_columns": decision_columns,
            "nested_prefix": True,
            "production_control_prefix_exact": True,
            "production_d128_prefix_exact": bool(
                args.control_decision_samples == FROZEN_D128
            ),
            "disjoint": bool(banks["disjoint"]),
            "draft_room_adp_columns": list(
                map(int, result.attrs["draft_room_adp_columns"])
            ),
        },
        "target_outcomes_read": False,
        "runtime_seconds": float(time.perf_counter() - start),
    }


def freeze_legacy_stage(origin, args, slot: int, depth: int, design_hash: str) -> dict[str, Any]:
    start = time.perf_counter()
    policy_seed = domain_seed(args.seed_base, "policy", origin.set_year, slot, depth)
    sim = make_legacy_sim(origin, args, slot)
    state = build_stage_state(sim, args, int(origin.set_year), slot, depth)
    assert_v2_template_contract(sim, "Legacy historical replay")
    name_to_key, _ = identity_maps(sim)
    result = run_legacy(sim, args, state, policy_seed)
    timings = result.attrs.get("timings", {})
    if (
        int(timings.get("success_trials", -1)) != args.rooms
        or int(timings.get("failed_exception_count", -1)) != 0
    ):
        raise ValueError("Legacy did not complete every configured current-action room.")
    action_name = legacy_current_action(result, state.depth)
    if action_name not in name_to_key:
        raise ValueError("Legacy selected an action absent from the canonical population.")
    return {
        "stage": "legacy",
        "design_hash": design_hash,
        "state_id": state_id(int(origin.set_year), slot, depth),
        "origin_year": int(origin.set_year),
        "pick_slot": int(slot),
        "completed_picks": int(depth),
        "synthetic_state": synthetic_state_receipt(state),
        "policy_seed": int(policy_seed),
        "action": {"player": action_name, "player_key": name_to_key[action_name]},
        "all_rooms_complete": True,
        "target_outcomes_read": False,
        "runtime_seconds": float(time.perf_counter() - start),
    }


def freeze_forced_stage(
    origin,
    args,
    slot: int,
    depth: int,
    design_hash: str,
    origin_contract: dict[str, Any],
    nested_envelope: dict[str, Any],
    request_hash: str,
) -> dict[str, Any]:
    start = time.perf_counter()
    nested = nested_envelope["stage_payload"]
    actions = {
        "d128": nested["actions"]["d128"],
        "d256": nested["actions"]["d256"],
    }
    action_keys = tuple(
        dict.fromkeys(str(action["player_key"]) for action in actions.values())
    )
    policy_seed = domain_seed(args.seed_base, "policy", origin.set_year, slot, depth)
    sim = make_forced_sim(origin, args, slot, action_keys)
    state = build_stage_state(sim, args, int(origin.set_year), slot, depth)
    state_receipt = synthetic_state_receipt(state)
    if state_receipt != nested["synthetic_state"]:
        raise AssertionError("Forced and Nested children rebuilt different ADP states.")
    if int(policy_seed) != int(nested["policy_seed"]):
        raise AssertionError("Freeze stages disagree on the shared policy seed.")
    assert_v2_template_contract(sim, "Forced historical replay")
    name_to_key, _ = identity_maps(sim)
    for arm, action in actions.items():
        if name_to_key.get(str(action["player"])) != str(action["player_key"]):
            raise ValueError(f"{arm} action identity changed in the forced child.")
    result = run_forced_rollout(sim, args, state, policy_seed, len(action_keys))
    if len(result) != len(action_keys) or not (result.PolicyCompletedRooms == args.rooms).all():
        raise ValueError("A forced action did not complete all common rooms.")
    for arm in ("d128", "d256"):
        action = actions[arm]
        action_name = str(action["player"])
        action_key = str(action["player_key"])
        if action_name not in result.attrs["policy_paths"]:
            raise AssertionError(f"Forced rollout lacks the selected path for {action_name}.")
        forced_paths = sorted(
            result.attrs["policy_paths"][action_name],
            key=lambda row: int(row["room_idx"]),
        )
        if json_safe(forced_paths) != nested["selected_policy_paths"][action_key]:
            raise AssertionError(f"Forced rollout changed the Nested path for {action_name}.")
    nested_banks = nested["bank_contract"]
    forced_banks = result.attrs["scenario_banks"]
    for receipt_key, bank_name in (
        ("construction_columns", "construction_ppg_columns"),
        ("evaluation_columns", "evaluation_ppg_columns"),
    ):
        if list(map(int, forced_banks[bank_name])) != nested_banks[receipt_key]:
            raise AssertionError(f"Forced rollout changed the shared {bank_name}.")
    forced_adp_columns = list(map(int, result.attrs["draft_room_adp_columns"]))
    if forced_adp_columns != nested_banks["draft_room_adp_columns"]:
        raise AssertionError("Forced rollout changed the shared latent ADP rooms.")
    rosters = freeze_forced_rosters(
        sim,
        result,
        action_keys,
        state,
        name_to_key,
        args,
    )
    payload = {
        "design_hash": design_hash,
        "origin_year": int(origin.set_year),
        "pick_slot": int(slot),
        "completed_picks": int(depth),
        "board_kind": "synthetic_adp_only_not_observed_draft_room",
        "board_seed": int(state.board_seed),
        "board_adp_column": int(state.adp_column),
        "policy_seed": int(policy_seed),
        "synthetic_state": {
            "to_add": list(state.to_add),
            "to_drop": list(state.to_drop),
        },
        "actions": actions,
        "rosters_by_action": rosters,
        "decision_scores": nested["decision_scores"],
        "bank_contract": {
            "construction_columns": nested_banks["construction_columns"],
            "evaluation_columns": nested_banks["evaluation_columns"],
            "d128_prefix_columns": nested_banks["d128_prefix_columns"],
            "d256_columns_hash": nested_banks["d256_columns_hash"],
            "nested_prefix": True,
            "production_control_prefix_exact": True,
            "v2_joint_template_16_week_contract": True,
            "production_d128_prefix_exact": nested_banks[
                "production_d128_prefix_exact"
            ],
            "disjoint": bool(nested_banks["disjoint"]),
            "forced_construction_evaluation_match": True,
            "common_downstream_adp_columns": forced_adp_columns,
        },
        "origin_source_fingerprint": origin_contract["source_fingerprint"],
        "origin_database_sha256": origin_contract["database_sha256"],
        "origin_receipt_hash": stable_hash(origin_contract["receipt"]),
        "strict_prior_donor_years": origin_contract["donor_years"],
        "frozen_contracts": {
            "physical_state_valid": True,
            "all_final_rosters_legal": True,
            "final_rosters_preserve_user_prefix_and_exclude_opponents": True,
            "exact_common_room_domain": True,
            "forced_bank_and_adp_match": True,
            "production_control_prefix_exact": True,
            "two_stage_process_isolation": True,
        },
        "isolation_contract": {
            "protocol": FREEZE_CHILD_PROTOCOL,
            "stages": list(FREEZE_CHILD_STAGES),
            "nested_envelope_hash": nested_envelope["envelope_hash"],
            "forced_request_hash": request_hash,
            "child_runtime": runtime_contract(),
        },
        "target_outcomes_read": False,
        "runtime_seconds_by_stage": {
            "nested": float(nested["runtime_seconds"]),
            "forced": float(time.perf_counter() - start),
        },
    }
    return {
        "state_id": state_id(int(origin.set_year), slot, depth),
        "freeze_payload": payload,
        "freeze_hash": stable_hash(payload),
        "frozen_at_utc": utc_now(),
    }


def child_execution_payload(args) -> dict[str, Any]:
    return {
        "model_repo": str(Path(args.model_repo).resolve()),
        "work_dir": str(Path(args.work_dir).resolve()),
        "teams": int(args.teams),
        "rounds": int(args.rounds),
        "rooms": int(args.rooms),
        "candidates": int(args.candidates),
        "construction_samples": int(args.construction_samples),
        "evaluation_samples": int(args.evaluation_samples),
        "control_decision_samples": int(args.control_decision_samples),
        "expanded_decision_samples": int(args.expanded_decision_samples),
        "seed_base": int(args.seed_base),
        "smoke": bool(args.smoke),
    }


def child_args_from_request(request: dict[str, Any]):
    execution = request.get("execution")
    required = {
        "model_repo",
        "work_dir",
        "teams",
        "rounds",
        "rooms",
        "candidates",
        "construction_samples",
        "evaluation_samples",
        "control_decision_samples",
        "expanded_decision_samples",
        "seed_base",
        "smoke",
    }
    if not isinstance(execution, dict) or set(execution) != required:
        raise ValueError("Freeze-child execution contract is malformed.")
    child_args = argparse.Namespace(**execution)
    child_args.model_repo = Path(child_args.model_repo).resolve()
    child_args.work_dir = Path(child_args.work_dir).resolve()
    child_args.sealed_code_contract = request["code_contract"]
    assert_maintained_interpreter(child_args.model_repo)
    if child_args.model_repo != MODEL_REPO.resolve():
        raise ValueError("Freeze child model_repo is not the canonical sibling repository.")
    return child_args


def path_is_within(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def validate_hashed_envelope(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or "envelope_hash" not in value:
        raise ValueError("Freeze-child stage envelope is malformed.")
    envelope_hash = str(value["envelope_hash"])
    envelope = {key: item for key, item in value.items() if key != "envelope_hash"}
    if envelope_hash != stable_hash(envelope):
        raise ValueError("Freeze-child stage envelope hash is invalid.")
    if envelope.get("stage_payload_hash") != stable_hash(envelope.get("stage_payload")):
        raise ValueError("Freeze-child stage payload hash is invalid.")
    if envelope.get("origin_contract_hash") != stable_hash(envelope.get("origin_contract")):
        raise ValueError("Freeze-child origin contract hash is invalid.")
    return value


def validate_stage_envelope(
    envelope: dict[str, Any],
    *,
    stage: str,
    key: str,
    design: dict[str, Any],
    request_hash: str | None = None,
    origin_contract: dict[str, Any] | None = None,
) -> None:
    validate_hashed_envelope(envelope)
    expected = {
        "protocol": FREEZE_CHILD_PROTOCOL,
        "stage": stage,
        "state_id": key,
        "design_hash": design["design_hash"],
        "code_contract": design["code_contract"],
        "runtime": runtime_contract(),
        "target_outcomes_unread": True,
    }
    for field, value in expected.items():
        if envelope.get(field) != value:
            raise ValueError(f"Freeze-child {stage} envelope has a stale {field}.")
    if request_hash is not None and envelope.get("request_hash") != request_hash:
        raise ValueError(f"Freeze-child {stage} did not echo the exact request hash.")
    payload = envelope["stage_payload"]
    if (
        payload.get("stage") != stage
        or payload.get("state_id") != key
        or payload.get("design_hash") != design["design_hash"]
        or payload.get("target_outcomes_read") is not False
    ):
        raise ValueError(f"Freeze-child {stage} payload domain is invalid.")
    payload_key = state_id(
        int(payload.get("origin_year", -1)),
        int(payload.get("pick_slot", -1)),
        int(payload.get("completed_picks", -1)),
    )
    synthetic_state = payload.get("synthetic_state")
    if (
        payload_key != key
        or not isinstance(synthetic_state, dict)
        or int(synthetic_state.get("depth", -1))
        != int(payload.get("completed_picks", -1))
    ):
        raise ValueError(f"Freeze-child {stage} physical-state domain is invalid.")
    contract = envelope["origin_contract"]
    if envelope.get("origin_database_sha256") != contract.get("database_sha256"):
        raise ValueError("Freeze-child materialized database hash is inconsistent.")
    if envelope.get("origin_source_fingerprint") != contract.get("source_fingerprint"):
        raise ValueError("Freeze-child decision-source fingerprint is inconsistent.")
    if envelope.get("origin_receipt_hash") != stable_hash(contract.get("receipt")):
        raise ValueError("Freeze-child decision receipt hash is inconsistent.")
    if envelope.get("strict_prior_donor_years") != contract.get("donor_years"):
        raise ValueError("Freeze-child strict-prior donor receipt is inconsistent.")
    if origin_contract is not None and contract != json_safe(origin_contract):
        raise ValueError(f"Freeze-child {stage} rebuilt a different origin contract.")


def load_child_request(path: Path) -> tuple[dict[str, Any], str]:
    wrapper = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(wrapper, dict) or set(wrapper) != {"request", "request_hash"}:
        raise ValueError("Freeze-child request wrapper is malformed.")
    request = wrapper["request"]
    request_hash = str(wrapper["request_hash"])
    if not isinstance(request, dict) or request_hash != stable_hash(request):
        raise ValueError("Freeze-child request hash is invalid.")
    return request, request_hash


def write_child_result(
    result_path: Path,
    *,
    request: dict[str, Any],
    request_hash: str,
    origin_contract: dict[str, Any],
    stage_payload: dict[str, Any],
) -> None:
    envelope = {
        "protocol": FREEZE_CHILD_PROTOCOL,
        "stage": request["stage"],
        "state_id": request["state_id"],
        "design_hash": request["design_hash"],
        "request_hash": request_hash,
        "code_contract": request["code_contract"],
        "runtime": runtime_contract(),
        "origin_contract": json_safe(origin_contract),
        "origin_contract_hash": stable_hash(origin_contract),
        "origin_database_sha256": origin_contract["database_sha256"],
        "origin_source_fingerprint": origin_contract["source_fingerprint"],
        "origin_receipt_hash": stable_hash(origin_contract["receipt"]),
        "strict_prior_donor_years": origin_contract["donor_years"],
        "stage_payload": json_safe(stage_payload),
        "stage_payload_hash": stable_hash(stage_payload),
        "target_outcomes_unread": True,
        "completed_at_utc": utc_now(),
    }
    wrapper = {**envelope, "envelope_hash": stable_hash(envelope)}
    atomic_write_json(result_path, wrapper)


def run_freeze_child(cli_args) -> None:
    system_temp = Path(tempfile.gettempdir()).resolve()
    request_path = Path(cli_args.freeze_child_request).resolve()
    result_path = Path(cli_args.freeze_child_result).resolve()
    if not path_is_within(request_path, system_temp) or not path_is_within(result_path, system_temp):
        raise ValueError("Freeze-child request/result files must live under OS temporary storage.")
    request, request_hash = load_child_request(request_path)
    if request.get("protocol") != FREEZE_CHILD_PROTOCOL:
        raise ValueError("Unsupported freeze-child protocol.")
    stage = request.get("stage")
    if stage not in FREEZE_CHILD_STAGES:
        raise ValueError(f"Unknown freeze-child stage: {stage}")
    child_args = child_args_from_request(request)
    current_contract = code_contract(child_args)
    if current_contract != request.get("code_contract"):
        raise ValueError("Freeze-child code differs from the parent-sealed contract.")
    validate_simulation_helper_path(current_contract)
    origin_year = int(request["origin_year"])
    slot = int(request["pick_slot"])
    depth = int(request["completed_picks"])
    key = state_id(origin_year, slot, depth)
    if request.get("state_id") != key:
        raise ValueError("Freeze-child request state ID is inconsistent.")
    design_stub = {
        "design_hash": request["design_hash"],
        "code_contract": request["code_contract"],
    }
    dependencies = request.get("dependencies", {})
    if not isinstance(dependencies, dict):
        raise ValueError("Freeze-child dependencies must be a dictionary.")
    nested_envelope = None
    if stage == "legacy":
        if set(dependencies) != {"expected_synthetic_state"}:
            raise ValueError("Legacy child requires only the Nested synthetic-state receipt.")
    elif stage == "forced":
        if set(dependencies) != {"nested"}:
            raise ValueError("Forced child requires the exact Nested envelope.")
        nested_envelope = dependencies["nested"]
        validate_stage_envelope(
            nested_envelope,
            stage="nested",
            key=key,
            design=design_stub,
        )
    elif dependencies:
        raise ValueError("Nested child does not accept upstream dependencies.")

    with open_origin(child_args, origin_year) as origin:
        origin_contract = validate_origin_contract(origin, origin_year)
        if stage == "nested":
            stage_payload = freeze_nested_stage(
                origin, child_args, slot, depth, request["design_hash"]
            )
        elif stage == "legacy":
            stage_payload = freeze_legacy_stage(
                origin, child_args, slot, depth, request["design_hash"]
            )
            if stage_payload["synthetic_state"] != dependencies["expected_synthetic_state"]:
                raise AssertionError("Legacy child rebuilt a different Nested synthetic state.")
        else:
            if nested_envelope["origin_contract"] != json_safe(origin_contract):
                raise ValueError("Freeze stages rebuilt different origin contracts.")
            stage_payload = {
                "stage": "forced",
                "design_hash": request["design_hash"],
                "state_id": key,
                "origin_year": origin_year,
                "pick_slot": slot,
                "completed_picks": depth,
                "synthetic_state": nested_envelope["stage_payload"]["synthetic_state"],
                "record": freeze_forced_stage(
                    origin,
                    child_args,
                    slot,
                    depth,
                    request["design_hash"],
                    origin_contract,
                    nested_envelope,
                    request_hash,
                ),
                "target_outcomes_read": False,
            }
        origin.assert_target_outcomes_unread()
    if code_contract(child_args) != request["code_contract"]:
        raise ValueError("Study code changed while a freeze child was running.")
    write_child_result(
        result_path,
        request=request,
        request_hash=request_hash,
        origin_contract=origin_contract,
        stage_payload=stage_payload,
    )


def load_child_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    validate_hashed_envelope(result)
    return result


def run_isolated_freeze_stage(
    args,
    design: dict[str, Any],
    *,
    stage: str,
    origin_year: int,
    slot: int,
    depth: int,
    dependencies: dict[str, Any],
    origin_contract: dict[str, Any],
) -> dict[str, Any]:
    key = state_id(origin_year, slot, depth)
    request = {
        "protocol": FREEZE_CHILD_PROTOCOL,
        "stage": stage,
        "state_id": key,
        "origin_year": int(origin_year),
        "pick_slot": int(slot),
        "completed_picks": int(depth),
        "design_hash": design["design_hash"],
        "code_contract": design["code_contract"],
        "execution": child_execution_payload(args),
        "dependencies": json_safe(dependencies),
    }
    request_wrapper = {"request": request, "request_hash": stable_hash(request)}
    child_root = Path(tempfile.gettempdir()).resolve() / "fantasy_football_replay_children"
    child_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"{origin_year}_s{slot}_d{depth}_{stage}_",
        dir=child_root,
    ) as temporary_directory:
        temporary_path = Path(temporary_directory)
        request_path = temporary_path / "request.json"
        result_path = temporary_path / "result.json"
        atomic_write_json(request_path, request_wrapper)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--freeze-child-request",
            str(request_path),
            "--freeze-child-result",
            str(result_path),
        ]
        print(f"launching isolated {stage} child for {key}", flush=True)
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
        if completed.returncode != 0:
            unsigned_code = completed.returncode & 0xFFFFFFFF
            diagnostics = (
                f"stdout:\n{completed.stdout[-8000:]}\n"
                f"stderr:\n{completed.stderr[-8000:]}"
            )
            raise RuntimeError(
                f"Isolated {stage} child failed for {key} with exit "
                f"{completed.returncode} (0x{unsigned_code:08X}).\n{diagnostics}"
            )
        if not result_path.is_file():
            raise RuntimeError(f"Isolated {stage} child for {key} exited without a result.")
        result = load_child_result(result_path)
        validate_stage_envelope(
            result,
            stage=stage,
            key=key,
            design=design,
            request_hash=request_wrapper["request_hash"],
            origin_contract=origin_contract,
        )
        if completed.stdout.strip():
            print(completed.stdout[-4000:].rstrip(), flush=True)
        if completed.stderr.strip():
            print(completed.stderr[-4000:].rstrip(), file=sys.stderr, flush=True)
        return result


def validate_final_child_record(
    record: dict[str, Any],
    *,
    key: str,
    origin_year: int,
    slot: int,
    depth: int,
    args,
    design: dict[str, Any],
    origin_contract: dict[str, Any],
    nested_envelope: dict[str, Any],
    forced_envelope: dict[str, Any],
) -> None:
    roster_validator = importlib.import_module(
        "historical_data"
    ).validate_final_roster_state_contract
    if record.get("state_id") != key:
        raise ValueError("Forced child returned the wrong frozen state ID.")
    payload = record.get("freeze_payload")
    if not isinstance(payload, dict) or record.get("freeze_hash") != stable_hash(payload):
        raise ValueError("Forced child returned an invalid final freeze hash.")
    if (
        payload.get("design_hash") != design["design_hash"]
        or int(payload.get("origin_year", -1)) != origin_year
        or int(payload.get("pick_slot", -1)) != slot
        or int(payload.get("completed_picks", -1)) != depth
        or payload.get("target_outcomes_read") is not False
    ):
        raise ValueError("Forced child returned a stale final freeze domain.")
    if payload.get("origin_source_fingerprint") != origin_contract["source_fingerprint"]:
        raise ValueError("Final freeze record has a stale decision-source fingerprint.")
    if payload.get("origin_database_sha256") != origin_contract["database_sha256"]:
        raise ValueError("Final freeze record has a stale materialized database hash.")
    if payload.get("origin_receipt_hash") != stable_hash(origin_contract["receipt"]):
        raise ValueError("Final freeze record has a stale decision receipt.")
    if payload.get("strict_prior_donor_years") != origin_contract["donor_years"]:
        raise ValueError("Final freeze record has a stale strict-prior donor set.")
    expected_state = nested_envelope["stage_payload"]["synthetic_state"]
    if payload.get("board_seed") != expected_state["board_seed"] or payload.get(
        "board_adp_column"
    ) != expected_state["adp_column"]:
        raise ValueError("Final freeze board receipt differs from the isolated stages.")
    if payload.get("synthetic_state") != {
        "to_add": expected_state["to_add"],
        "to_drop": expected_state["to_drop"],
    }:
        raise ValueError("Final freeze physical state differs from the isolated stages.")
    if set(payload.get("actions", {})) != set(ARMS):
        raise ValueError("Final freeze record lacks the exact D128/D256 action domain.")
    expected_actions = nested_envelope["stage_payload"]["actions"]
    if payload["actions"] != expected_actions:
        raise ValueError("Final freeze actions differ from the isolated action stages.")
    action_keys = {action["player_key"] for action in expected_actions.values()}
    if set(payload.get("rosters_by_action", {})) != action_keys:
        raise ValueError("Final freeze roster union differs from the isolated actions.")
    final_state = payload["synthetic_state"]
    for action_key, roster_rows in payload["rosters_by_action"].items():
        if [int(row["room_idx"]) for row in roster_rows] != list(range(args.rooms)):
            raise ValueError("Final freeze action lacks the exact common room domain.")
        for row in roster_rows:
            roster_validator(
                row["player_keys"],
                to_add=final_state["to_add"],
                to_drop=final_state["to_drop"],
                depth=depth,
                action_key=action_key,
                rounds=args.rounds,
                label=f"Final child {key} action {action_key} room {row['room_idx']}",
            )
    if not all(payload.get("frozen_contracts", {}).values()):
        raise ValueError("Final freeze child did not seal every physical/isolation contract.")
    isolation = payload.get("isolation_contract", {})
    if (
        isolation.get("protocol") != FREEZE_CHILD_PROTOCOL
        or isolation.get("stages") != list(FREEZE_CHILD_STAGES)
        or isolation.get("nested_envelope_hash") != nested_envelope["envelope_hash"]
        or isolation.get("forced_request_hash") != forced_envelope["request_hash"]
        or isolation.get("child_runtime") != runtime_contract()
    ):
        raise ValueError("Final freeze process-isolation receipt is invalid.")


def validate_nested_synthetic_boards(
    records: dict[str, dict[str, Any]], args
) -> None:
    for origin_year in args.origin_values:
        for slot in args.slot_values:
            states = [
                records[state_id(origin_year, slot, depth)]["freeze_payload"]
                for depth in sorted(args.depth_values)
            ]
            seeds = {payload["board_seed"] for payload in states}
            adp_columns = {payload["board_adp_column"] for payload in states}
            if len(seeds) != 1 or len(adp_columns) != 1:
                raise ValueError("Independently isolated depths do not share one ADP board.")
            previous_add: list[str] = []
            previous_drop: list[str] = []
            for payload in states:
                current_add = payload["synthetic_state"]["to_add"]
                current_drop = payload["synthetic_state"]["to_drop"]
                if (
                    current_add[: len(previous_add)] != previous_add
                    or current_drop[: len(previous_drop)] != previous_drop
                ):
                    raise ValueError("Independently isolated states are not nested board prefixes.")
                previous_add = current_add
                previous_drop = current_drop


def validate_durable_isolation_receipt(payload: dict[str, Any]) -> None:
    isolation = payload.get("isolation_contract")
    if (
        not isinstance(isolation, dict)
        or isolation.get("protocol") != FREEZE_CHILD_PROTOCOL
        or isolation.get("stages") != list(FREEZE_CHILD_STAGES)
        or isolation.get("child_runtime") != runtime_contract()
        or not payload.get("frozen_contracts", {}).get(
            "two_stage_process_isolation", False
        )
    ):
        raise ValueError("Frozen record lacks its durable two-stage isolation receipt.")
    for key in (
        "nested_envelope_hash",
        "forced_request_hash",
    ):
        digest = str(isolation.get(key, "")).strip().lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"Frozen isolation receipt has invalid {key}.")


def validate_all_origin_snapshots_before_seal(
    args,
    design: dict[str, Any],
    records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Reopen every origin immediately before sealing the global freeze."""
    if code_contract(args) != design["code_contract"]:
        raise ValueError("Study code changed during the freeze phase.")
    receipts: dict[str, Any] = {}
    for origin_year in args.origin_values:
        with open_origin(args, origin_year) as origin:
            contract = validate_origin_contract(origin, origin_year)
            origin_records = [
                record
                for record in records.values()
                if record["freeze_payload"]["origin_year"] == origin_year
            ]
            if not origin_records:
                raise ValueError(f"Origin {origin_year} has no frozen states.")
            for record in origin_records:
                payload = record["freeze_payload"]
                if payload.get("design_hash") != design["design_hash"]:
                    raise ValueError("A pre-seal origin record has a stale design hash.")
                if payload["origin_source_fingerprint"] != contract["source_fingerprint"]:
                    raise ValueError(f"Origin {origin_year} source changed before global seal.")
                if payload["origin_database_sha256"] != contract["database_sha256"]:
                    raise ValueError(f"Origin {origin_year} materialized database changed before global seal.")
                if payload["origin_receipt_hash"] != stable_hash(contract["receipt"]):
                    raise ValueError(f"Origin {origin_year} receipt changed before global seal.")
                if payload["strict_prior_donor_years"] != contract["donor_years"]:
                    raise ValueError(f"Origin {origin_year} donor set changed before global seal.")
            origin.assert_target_outcomes_unread()
            receipts[str(origin_year)] = contract
    return receipts


def run_freeze(args, design: dict) -> None:
    freeze_path = args.output_dir / "frozen_states.jsonl"
    records = load_frozen_records(freeze_path) if args.resume else {}
    expected = configured_state_ids(args)
    freeze_in_progress = {
        "started_at_utc": utc_now(),
        "design_hash": design["design_hash"],
        "freeze_complete": False,
        "outcomes_scored": False,
    }
    atomic_write_json(args.output_dir / "freeze_manifest.json", freeze_in_progress)
    atomic_write_json(
        args.output_dir / "summary.json",
        {"status": "freeze_in_progress", **freeze_in_progress},
    )
    completed_this_run = 0
    receipts_path = args.output_dir / "origin_receipts.json"
    receipts = json.loads(receipts_path.read_text(encoding="utf-8")) if receipts_path.exists() else {}

    for origin_year in args.origin_values:
        origin_ids = [key for key in expected if key.startswith(f"dk:{origin_year}:")]
        with open_origin(args, origin_year) as origin:
            contract = validate_origin_contract(origin, origin_year)
            current_receipt_hash = stable_hash(contract["receipt"])
            for key in origin_ids:
                if key not in records:
                    continue
                payload = records[key]["freeze_payload"]
                validate_durable_isolation_receipt(payload)
                if payload.get("design_hash") != design["design_hash"]:
                    raise ValueError(f"Checkpoint {key} belongs to a different design.")
                if payload.get("origin_source_fingerprint") != contract["source_fingerprint"]:
                    raise ValueError(f"Checkpoint {key} has a stale source fingerprint.")
                if payload.get("origin_database_sha256") != contract["database_sha256"]:
                    raise ValueError(f"Checkpoint {key} has a different materialized database.")
                if payload.get("origin_receipt_hash") != current_receipt_hash:
                    raise ValueError(f"Checkpoint {key} has a stale origin receipt.")
                if payload.get("strict_prior_donor_years") != contract["donor_years"]:
                    raise ValueError(f"Checkpoint {key} has a different strict-prior donor set.")
            receipts[str(origin_year)] = contract
            atomic_write_json(receipts_path, receipts)
            origin.assert_target_outcomes_unread()
        if all(key in records for key in origin_ids):
            continue
        for slot in args.slot_values:
            for depth in args.depth_values:
                key = state_id(origin_year, slot, depth)
                if key in records:
                    continue
                nested_envelope = run_isolated_freeze_stage(
                    args,
                    design,
                    stage="nested",
                    origin_year=origin_year,
                    slot=slot,
                    depth=depth,
                    dependencies={},
                    origin_contract=contract,
                )
                forced_envelope = run_isolated_freeze_stage(
                    args,
                    design,
                    stage="forced",
                    origin_year=origin_year,
                    slot=slot,
                    depth=depth,
                    dependencies={"nested": nested_envelope},
                    origin_contract=contract,
                )
                record = forced_envelope["stage_payload"]["record"]
                validate_final_child_record(
                    record,
                    key=key,
                    origin_year=origin_year,
                    slot=slot,
                    depth=depth,
                    args=args,
                    design=design,
                    origin_contract=contract,
                    nested_envelope=nested_envelope,
                    forced_envelope=forced_envelope,
                )
                records[key] = record
                write_frozen_records(freeze_path, records, args)
                completed_this_run += 1
                print(f"frozen {len(records)}/{len(expected)} {key}", flush=True)
                gc.collect()
                if (
                    args.max_states
                    and completed_this_run >= args.max_states
                    and set(records) != set(expected)
                ):
                    return

    if set(records) != set(expected):
        raise ValueError("Freeze phase ended without every configured state.")
    validate_nested_synthetic_boards(records, args)
    for record in records.values():
        payload = record["freeze_payload"]
        validate_durable_isolation_receipt(payload)
        if payload.get("design_hash") != design["design_hash"]:
            raise AssertionError("Frozen-state design hash mismatch.")
        if payload["target_outcomes_read"] or len(payload["rosters_by_action"]) > len(ARMS):
            raise AssertionError("Frozen-state leakage or action-union contract failed.")
        if not all(payload.get("frozen_contracts", {}).values()):
            raise AssertionError("At least one frozen physical/bank/roster contract failed.")
        for rosters in payload["rosters_by_action"].values():
            if len(rosters) != args.rooms:
                raise AssertionError("Frozen state does not contain every downstream room.")
    receipts = validate_all_origin_snapshots_before_seal(args, design, records)
    atomic_write_json(receipts_path, receipts)
    runner_receipt = json.loads(
        (args.output_dir / "runner_receipt.json").read_text(encoding="utf-8")
    )
    manifest = {
        "sealed_at_utc": utc_now(),
        "design_hash": design["design_hash"],
        "code_contract": design["code_contract"],
        "runner_receipt_hash": runner_receipt["receipt_hash"],
        "frozen_states_sha256": sha256_file(freeze_path),
        "origin_receipts_sha256": sha256_file(receipts_path),
        "state_count": len(records),
        "expected_state_count": len(expected),
        "outcomes_scored": False,
        "freeze_complete": True,
        "strict_prior_all_origins": bool(all(
            max(record["freeze_payload"]["strict_prior_donor_years"])
            < record["freeze_payload"]["origin_year"]
            for record in records.values()
        )),
        "physical_bank_and_roster_contracts": bool(all(
            all(record["freeze_payload"]["frozen_contracts"].values())
            for record in records.values()
        )),
        "two_stage_process_isolation": bool(all(
            record["freeze_payload"]["frozen_contracts"].get(
                "two_stage_process_isolation", False
            )
            for record in records.values()
        )),
        "production_d128_prefix_exact": bool(all(
            record["freeze_payload"]["bank_contract"].get(
                "production_d128_prefix_exact", False
            )
            for record in records.values()
        )),
    }
    atomic_write_json(args.output_dir / "freeze_manifest.json", manifest)
    atomic_write_json(
        args.output_dir / "summary.json",
        {
            "status": "freeze_complete_waiting_for_score",
            "completed_at_utc": utc_now(),
            "design_hash": design["design_hash"],
            "frozen_design_exact": bool(design["frozen_design_exact"]),
            "freeze_manifest": manifest,
            "freeze_manifest_hash": stable_hash(manifest),
            "counts": {
                "states": len(records),
                "origin_seasons": len(args.origin_values),
                "slots": len(args.slot_values),
                "depths": len(args.depth_values),
            },
            "target_outcomes_scored": False,
            "promotion_ready": False,
        },
    )


def validate_freeze_for_scoring(args, design: dict):
    roster_validator = importlib.import_module(
        "historical_data"
    ).validate_final_roster_state_contract
    freeze_path = args.output_dir / "frozen_states.jsonl"
    manifest_path = args.output_dir / "freeze_manifest.json"
    receipts_path = args.output_dir / "origin_receipts.json"
    if not freeze_path.exists() or not manifest_path.exists() or not receipts_path.exists():
        raise ValueError("Score phase requires a complete sealed freeze phase.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("design_hash") != design["design_hash"]:
        raise ValueError("Freeze manifest design hash mismatch.")
    if not manifest.get("freeze_complete"):
        raise ValueError("Freeze manifest is not sealed complete.")
    if manifest.get("code_contract") != design["code_contract"]:
        raise ValueError("Freeze manifest code contract mismatch.")
    runner_receipt = json.loads(
        (args.output_dir / "runner_receipt.json").read_text(encoding="utf-8")
    )
    if manifest.get("runner_receipt_hash") != runner_receipt.get("receipt_hash"):
        raise ValueError("Freeze manifest runner-receipt hash mismatch.")
    if manifest.get("frozen_states_sha256") != sha256_file(freeze_path):
        raise ValueError("Frozen roster artifact changed after sealing.")
    if manifest.get("origin_receipts_sha256") != sha256_file(receipts_path):
        raise ValueError("Frozen origin-receipt artifact changed after sealing.")
    records = load_frozen_records(freeze_path)
    if set(records) != set(configured_state_ids(args)):
        raise ValueError("Frozen roster artifact does not contain the exact configured state set.")
    if (
        int(manifest.get("state_count", -1)) != len(records)
        or int(manifest.get("expected_state_count", -1)) != len(records)
        or not manifest.get("two_stage_process_isolation")
    ):
        raise ValueError("Freeze manifest count/isolation contract is invalid.")
    validate_nested_synthetic_boards(records, args)
    if any(
        record["freeze_payload"].get("design_hash") != design["design_hash"]
        for record in records.values()
    ):
        raise ValueError("At least one frozen record belongs to a different design.")
    for key, record in records.items():
        payload = record["freeze_payload"]
        validate_durable_isolation_receipt(payload)
        expected_key = state_id(
            payload["origin_year"],
            payload["pick_slot"],
            payload["completed_picks"],
        )
        if key != expected_key or set(payload["actions"]) != set(ARMS):
            raise ValueError("Frozen record state/action domain is malformed.")
        action_keys = {
            action["player_key"] for action in payload["actions"].values()
        }
        if set(payload["rosters_by_action"]) != action_keys:
            raise ValueError("Frozen action union does not match arm actions.")
        if payload.get("target_outcomes_read") or not all(
            payload.get("frozen_contracts", {}).values()
        ):
            raise ValueError("Frozen leakage/physical/roster contract is not sealed true.")
        donors = payload["strict_prior_donor_years"]
        if not donors or max(donors) >= payload["origin_year"]:
            raise ValueError("Frozen record has a non-prior donor season.")
        depth = int(payload["completed_picks"])
        state = payload["synthetic_state"]
        if (
            len(state["to_add"]) != depth
            or len(state["to_add"]) != len(set(state["to_add"]))
            or len(state["to_drop"]) != len(set(state["to_drop"]))
            or set(state["to_add"]) & set(state["to_drop"])
        ):
            raise ValueError("Frozen synthetic state structure is invalid.")
        for action_key, roster_rows in payload["rosters_by_action"].items():
            if [int(row["room_idx"]) for row in roster_rows] != list(range(args.rooms)):
                raise ValueError("Frozen action lacks the exact common room domain.")
            for row in roster_rows:
                roster_validator(
                    row["player_keys"],
                    to_add=state["to_add"],
                    to_drop=state["to_drop"],
                    depth=depth,
                    action_key=action_key,
                    rounds=args.rounds,
                    label=f"Scoring freeze {key} action {action_key} room {row['room_idx']}",
                )
    return manifest, records


def normalize_scores(scores: Any, expected: int) -> np.ndarray:
    if isinstance(scores, pd.DataFrame):
        if "roster_id" in scores:
            observed_ids = scores.roster_id.astype(str).tolist()
            expected_ids = [str(index) for index in range(expected)]
            if observed_ids != expected_ids:
                raise ValueError("score_rosters changed the frozen roster order/domain.")
        score_column = next(
            (column for column in ("target_points", "score", "best_ball_points") if column in scores),
            None,
        )
        if score_column is None:
            raise ValueError("score_rosters DataFrame lacks a supported score column.")
        scores = scores[score_column].to_numpy()
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(values) != expected or not np.isfinite(values).all():
        raise ValueError("score_rosters returned missing, non-finite, or mis-sized results.")
    return values


def build_target_source_receipt(origin, design: dict[str, Any]) -> dict[str, Any]:
    """Capture and validate the canonical lazy governed target provenance."""
    published = origin.receipt.get("target_source_receipt")
    if not isinstance(published, dict):
        raise ValueError("Lazy scoring did not publish target_source_receipt.")
    required = {
        "source_name",
        "source_uri",
        "source_sha256",
        "row_count",
        "origin_year",
        "foundation_run_id",
        "scoring_hash",
        "scoring_code_path",
        "scoring_code_sha256",
        "mask_code_path",
        "mask_code_sha256",
        "scoring_dependency_sha256",
        "governed_last_week",
        "scoring_week_count",
        "roster_scoring_population",
        "roster_position_attribution",
        "reconciliation_population",
        "configured_scoring_limitation",
        "evaluation_audit",
    }
    missing = required - set(published)
    if missing:
        raise ValueError(f"target_source_receipt lacks canonical fields: {sorted(missing)}")
    published = json_safe(published)
    if int(published["origin_year"]) != int(origin.set_year):
        raise ValueError("Target-source receipt origin mismatch.")
    if int(published["row_count"]) <= 0 or int(published["scoring_week_count"]) != WEEKLY_HORIZON:
        raise ValueError("Target-source receipt has an invalid row count or scoring horizon.")
    for key in (
        "source_name",
        "source_uri",
        "source_sha256",
        "foundation_run_id",
        "scoring_hash",
        "scoring_code_path",
        "scoring_code_sha256",
        "mask_code_path",
        "mask_code_sha256",
        "roster_scoring_population",
        "roster_position_attribution",
        "reconciliation_population",
    ):
        if not str(published[key]).strip():
            raise ValueError(f"Target-source receipt has blank {key}.")
    for key in ("source_sha256", "scoring_code_sha256", "mask_code_sha256"):
        value = str(published[key]).strip().lower()
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"Target-source receipt has invalid {key}.")
    expected_last_week = 16 if int(origin.set_year) < 2021 else 17
    if int(published["governed_last_week"]) != expected_last_week:
        raise ValueError("Target-source governed full-horizon week boundary is wrong.")
    if (
        published["roster_scoring_population"]
        != "configured_dk_all_regular_season_played_rows"
        or published["roster_position_attribution"]
        != "candidate_preseason_position"
        or published["reconciliation_population"]
        != "build_player_outcomes.aggregate_player_outcomes"
        or not published["configured_scoring_limitation"]
    ):
        raise ValueError("Target-source scoring-population semantics are invalid.")
    evaluation_audit = published["evaluation_audit"]
    if not isinstance(evaluation_audit, dict):
        raise ValueError("Target-source evaluation_audit must be a dictionary.")
    audit_count_fields = {
        "outcome_rows",
        "raw_payload_rows",
        "origin_regular_rows",
        "raw_all_player_full_governed_horizon_rows",
        "mapped_full_horizon_rows",
        "unmapped_full_horizon_rows",
        "mapped_candidate_count",
        "all_played_full_horizon_raw_rows",
        "all_played_full_horizon_player_week_rows",
        "all_played_week1_16_raw_rows",
        "all_played_week1_16_player_week_rows",
        "governed_reconciliation_rows",
        "opportunity_excluded_rows",
        "week17_excluded_rows",
        "governed_last_week",
        "governed_seasonal_rows",
        "exact_outcome_rows",
        "required_weekly_mapping_rows",
        "missing_required_weekly_mapping_rows",
        "outcome_reconciliation_compared_rows",
        "outcome_reconciliation_mismatch_rows",
        "known_mapping_regression_rows_checked",
        "raw_candidate_position_mismatch_rows",
        "raw_candidate_position_mismatch_players",
    }
    audit_point_fields = {
        "origin_regular_points",
        "raw_all_player_full_governed_horizon_points",
        "mapped_full_horizon_points",
        "unmapped_full_horizon_points",
        "all_played_full_horizon_points",
        "all_played_week1_16_points",
        "governed_reconciliation_points",
        "opportunity_excluded_points",
        "week17_excluded_points",
        "outcome_reconciliation_atol",
        "outcome_reconciliation_max_abs_delta",
    }
    audit_hash_fields = {
        "raw_schema_sha256",
        "configured_dk_week1_16_outcome_frame_sha256",
        "governed_seasonal_sha256",
        "candidate_identity_sha256",
        "full_player_identity_sha256",
        "target_alias_position_sha256",
    }
    audit_required = audit_count_fields | audit_point_fields | audit_hash_fields | {
        "raw_schema_columns",
        "decision_frame_sha256_revalidated",
    }
    missing_audit = audit_required - set(evaluation_audit)
    if missing_audit:
        raise ValueError(
            f"target_source_receipt evaluation_audit lacks fields: {sorted(missing_audit)}"
        )
    for key in audit_count_fields:
        value = evaluation_audit[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"evaluation_audit {key} must be a nonnegative integer.")
    for key in audit_point_fields:
        value = evaluation_audit[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"evaluation_audit {key} must be a finite number.")
    for key in audit_hash_fields:
        value = str(evaluation_audit[key]).strip().lower()
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"evaluation_audit {key} must be a SHA256 value.")
    raw_schema_columns = evaluation_audit["raw_schema_columns"]
    if (
        not isinstance(raw_schema_columns, list)
        or not raw_schema_columns
        or any(
            not isinstance(row, dict)
            or not str(row.get("column", "")).strip()
            or not str(row.get("dtype", "")).strip()
            for row in raw_schema_columns
        )
    ):
        raise ValueError("evaluation_audit raw_schema_columns is malformed.")
    if evaluation_audit["raw_payload_rows"] != int(published["row_count"]):
        raise ValueError("Target-source payload and evaluation-audit row counts differ.")
    if evaluation_audit["governed_last_week"] != expected_last_week:
        raise ValueError("evaluation_audit governed_last_week is inconsistent.")
    revalidated_decision_hashes = evaluation_audit["decision_frame_sha256_revalidated"]
    frozen_decision_hashes = json_safe(origin.receipt.get("table_sha256"))
    if (
        not isinstance(revalidated_decision_hashes, dict)
        or not revalidated_decision_hashes
        or revalidated_decision_hashes != frozen_decision_hashes
    ):
        raise ValueError(
            "Target evaluation did not exactly revalidate the frozen decision frames."
        )
    for label, value in revalidated_decision_hashes.items():
        digest = str(value).strip().lower()
        if not str(label).strip() or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("evaluation_audit decision-frame hashes are malformed.")
    if (
        evaluation_audit["missing_required_weekly_mapping_rows"] != 0
        or evaluation_audit["outcome_reconciliation_mismatch_rows"] != 0
    ):
        raise ValueError("Target evaluation audit contains unresolved mapping/reconciliation rows.")
    target_fingerprint = str(origin.target_outcome_fingerprint or "").strip()
    if not target_fingerprint:
        raise ValueError("Governed target source did not publish an outcome fingerprint.")
    expected_scoring_code_sha256 = design["code_contract"][
        "Scripts/V2/build_player_outcomes.py"
    ]["sha256"]
    if str(published["scoring_code_sha256"]).lower() != expected_scoring_code_sha256:
        raise ValueError("Target scorer code hash differs from the sealed V2 outcome builder.")
    expected_scoring_code_path = Path(
        design["code_contract"]["Scripts/V2/build_player_outcomes.py"]["path"]
    ).resolve()
    if Path(str(published["scoring_code_path"])).expanduser().resolve() != expected_scoring_code_path:
        raise ValueError("Target scorer path differs from the sealed V2 outcome builder.")
    if (
        Path(str(published["mask_code_path"])).expanduser().resolve()
        != expected_scoring_code_path
        or str(published["mask_code_sha256"]).lower()
        != expected_scoring_code_sha256
    ):
        raise ValueError("Target mask code differs from the sealed V2 outcome builder.")
    dependency_contract = {
        "build_player_outcomes": design["code_contract"][
            "Scripts/V2/build_player_outcomes.py"
        ],
        "contracts": design["code_contract"]["Scripts/V2/contracts.py"],
        "v2_config": design["code_contract"]["Scripts/V2/config.py"],
        "scripts_config": design["code_contract"]["Scripts/config.py"],
        "weekly_config": design["code_contract"]["Scripts/config.py"],
        "weekly_builder": design["code_contract"][
            "Scripts/Modeling/s4_Best_Ball_Weekly.py"
        ],
    }
    if published["scoring_dependency_sha256"] != dependency_contract:
        raise ValueError("Target scoring dependency receipt differs from sealed code.")
    origin_scoring_hashes = set(
        origin.predictions["v2_scoring_hash"].dropna().astype(str)
    )
    if len(origin_scoring_hashes) != 1 or origin_scoring_hashes != {
        str(published["scoring_hash"])
    }:
        raise ValueError("Target-source scoring hash differs from the exact origin V2 hash.")
    fingerprint_lower = target_fingerprint.lower()
    if len(fingerprint_lower) != 64 or any(
        character not in "0123456789abcdef" for character in fingerprint_lower
    ):
        raise ValueError("Target-outcome fingerprint is not a SHA256 value.")
    # The source receipt owns the scoring-code hash.  The study code contract
    # independently seals the historical adapter that invokes it.
    return {
        "target_source_receipt": published,
        "target_source_receipt_hash": stable_hash(published),
        "target_outcome_fingerprint": target_fingerprint,
        "historical_data_sha256": design["code_contract"]["historical_data.py"]["sha256"],
    }


def target_source_receipt_hash(receipt: dict[str, Any]) -> str:
    source_receipt = receipt.get("target_source_receipt")
    if not isinstance(source_receipt, dict):
        raise ValueError("Per-origin score receipt lacks its canonical target-source dict.")
    stored = str(receipt.get("target_source_receipt_hash", "")).strip()
    calculated = stable_hash(source_receipt)
    if stored and stored != calculated:
        raise ValueError("Per-origin governed target-source receipt hash is invalid.")
    return calculated


def score_origin_records(origin, origin_records: Sequence[dict], args) -> list[dict[str, Any]]:
    rows = []
    for record in origin_records:
        payload = record["freeze_payload"]
        score_cache: dict[str, np.ndarray] = {}
        for action_key, roster_rows in payload["rosters_by_action"].items():
            rosters = [row["player_keys"] for row in roster_rows]
            score_cache[action_key] = normalize_scores(origin.score_rosters(rosters), len(rosters))
        for arm in ARMS:
            action = payload["actions"][arm]
            action_key = action["player_key"]
            roster_rows = payload["rosters_by_action"][action_key]
            for roster_row, target_points in zip(roster_rows, score_cache[action_key]):
                rows.append({
                    "state_id": record["state_id"],
                    "origin_year": payload["origin_year"],
                    "pick_slot": payload["pick_slot"],
                    "completed_picks": payload["completed_picks"],
                    "arm": arm,
                    "action_player": action["player"],
                    "action_player_key": action_key,
                    "room_idx": int(roster_row["room_idx"]),
                    "target_best_ball_points": float(target_points),
                    "freeze_hash": record["freeze_hash"],
                })
    return rows


def assert_origin_roster_contracts(origin, origin_records: Sequence[dict], args) -> None:
    roster_validator = importlib.import_module(
        "historical_data"
    ).validate_final_roster_state_contract
    player_frame = origin.predictions[["player_key", "pos"]].copy()
    if player_frame.player_key.duplicated().any():
        raise ValueError("Origin prediction surface contains duplicate player keys.")
    position_by_key = dict(
        zip(player_frame.player_key.astype(str), player_frame.pos.astype(str))
    )
    for record in origin_records:
        payload = record["freeze_payload"]
        depth = int(payload["completed_picks"])
        state = payload["synthetic_state"]
        state_keys = [
            *payload["synthetic_state"]["to_add"],
            *payload["synthetic_state"]["to_drop"],
        ]
        if any(key not in position_by_key for key in state_keys):
            raise ValueError("Frozen physical state is absent from the reopened origin population.")
        for action_key, roster_rows in payload["rosters_by_action"].items():
            for row in roster_rows:
                roster = roster_validator(
                    row["player_keys"],
                    to_add=state["to_add"],
                    to_drop=state["to_drop"],
                    depth=depth,
                    action_key=action_key,
                    rounds=args.rounds,
                    label=(
                        f"Reopened origin {payload['origin_year']} action {action_key} "
                        f"room {row['room_idx']}"
                    ),
                )
                if any(key not in position_by_key for key in roster):
                    raise ValueError("Frozen roster is absent from the reopened origin population.")
                counts = pd.Series([position_by_key[key] for key in roster]).value_counts()
                if set(counts.index) - set(POSITION_RANGES):
                    raise ValueError("Frozen roster contains an unsupported position.")
                for position, (minimum, maximum) in POSITION_RANGES.items():
                    if not minimum <= int(counts.get(position, 0)) <= maximum:
                        raise ValueError("Frozen roster violates a position-range contract.")


def validate_score_checkpoint(
    frame: pd.DataFrame,
    records: dict[str, dict[str, Any]],
    args,
    score_receipts: dict[str, Any] | None = None,
) -> None:
    if frame.empty:
        return
    required = {
        "state_id",
        "origin_year",
        "pick_slot",
        "completed_picks",
        "arm",
        "action_player",
        "action_player_key",
        "room_idx",
        "target_best_ball_points",
        "target_outcome_fingerprint",
        "target_source_receipt_hash",
        "freeze_hash",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Room-score checkpoint lacks integrity columns: {sorted(missing)}")
    if frame.duplicated(["state_id", "arm", "room_idx"]).any():
        raise ValueError("Room-score checkpoint contains duplicate state/arm/room rows.")
    for row in frame.itertuples(index=False):
        if row.state_id not in records or row.arm not in ARMS:
            raise ValueError("Room-score checkpoint contains an unknown state or arm.")
        if not 0 <= int(row.room_idx) < args.rooms:
            raise ValueError("Room-score checkpoint contains an out-of-domain room ID.")
        record = records[row.state_id]
        payload = record["freeze_payload"]
        action = payload["actions"][row.arm]
        expected = (
            int(payload["origin_year"]),
            int(payload["pick_slot"]),
            int(payload["completed_picks"]),
            str(action["player"]),
            str(action["player_key"]),
            str(record["freeze_hash"]),
        )
        observed = (
            int(row.origin_year),
            int(row.pick_slot),
            int(row.completed_picks),
            str(row.action_player),
            str(row.action_player_key),
            str(row.freeze_hash),
        )
        if observed != expected:
            raise ValueError(f"Room-score checkpoint is stale or mismatched for {row.state_id}.")
        if pd.isna(row.target_outcome_fingerprint) or not str(row.target_outcome_fingerprint).strip():
            raise ValueError("Room-score checkpoint lacks a target-outcome fingerprint.")
        if pd.isna(row.target_source_receipt_hash) or not str(row.target_source_receipt_hash).strip():
            raise ValueError("Room-score checkpoint lacks a governed target-source receipt hash.")
        if score_receipts is not None:
            receipt = score_receipts.get(str(int(row.origin_year)))
            if receipt is None:
                raise ValueError("Room-score checkpoint lacks its per-origin target receipt.")
            if str(row.target_source_receipt_hash) != target_source_receipt_hash(receipt):
                raise ValueError("Room-score checkpoint target receipt hash is stale.")
        if not np.isfinite(float(row.target_best_ball_points)):
            raise ValueError("Room-score checkpoint contains a non-finite target score.")


def recover_score_checkpoint(
    frame: pd.DataFrame,
    score_receipts: dict[str, Any],
    records: dict[str, dict[str, Any]],
    args,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep only origin shards whose rows and exact receipt agree atomically."""
    if frame.empty:
        return pd.DataFrame(), {}
    if "origin_year" not in frame.columns:
        print("discarding malformed score checkpoint without origin_year", flush=True)
        return pd.DataFrame(), {}
    valid_frames = []
    valid_receipts: dict[str, Any] = {}
    for origin_year in args.origin_values:
        origin_frame = frame[frame.origin_year == origin_year].copy()
        if origin_frame.empty:
            continue
        origin_key = str(origin_year)
        receipt = score_receipts.get(origin_key)
        try:
            validate_score_checkpoint(
                origin_frame,
                records,
                args,
                {origin_key: receipt} if receipt is not None else {},
            )
        except (KeyError, TypeError, ValueError):
            print(
                f"discarding incomplete/mismatched score checkpoint for origin {origin_year}",
                flush=True,
            )
            continue
        valid_frames.append(origin_frame)
        valid_receipts[origin_key] = receipt
    recovered = (
        pd.concat(valid_frames, ignore_index=True)
        if valid_frames
        else pd.DataFrame()
    )
    return recovered, valid_receipts


def state_metrics_from_room_scores(room_scores: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        room_scores.groupby(
            ["state_id", "origin_year", "pick_slot", "completed_picks", "arm"],
            as_index=False,
        )
        .target_best_ball_points.mean()
    )
    pivot = grouped.pivot(
        index=["state_id", "origin_year", "pick_slot", "completed_picks"],
        columns="arm",
        values="target_best_ball_points",
    ).reset_index()
    pivot.columns.name = None
    for arm in ARMS:
        if arm not in pivot:
            raise ValueError(f"State metrics are missing arm {arm}.")
        pivot[f"{arm}_target_points"] = pivot.pop(arm)
    pivot["d256_minus_d128"] = pivot.d256_target_points - pivot.d128_target_points
    return pivot


def season_cluster_bootstrap(
    frame: pd.DataFrame,
    args,
    *,
    challenger: str,
    baseline: str,
) -> dict[str, Any]:
    seasons = sorted(frame.origin_year.unique())
    if not seasons:
        raise ValueError("No seasons are available for the primary bootstrap.")
    challenger_column = f"{challenger}_target_points"
    baseline_column = f"{baseline}_target_points"
    clusters = {season: frame[frame.origin_year == season] for season in seasons}
    rng = np.random.default_rng(args.bootstrap_seed)
    raw = np.empty(args.bootstrap_draws, dtype=np.float64)
    pct = np.empty(args.bootstrap_draws, dtype=np.float64)
    for draw in range(args.bootstrap_draws):
        sampled = rng.choice(seasons, size=len(seasons), replace=True)
        challenger_values = np.concatenate([
            clusters[season][challenger_column].to_numpy() for season in sampled
        ])
        control = np.concatenate([
            clusters[season][baseline_column].to_numpy() for season in sampled
        ])
        delta = challenger_values - control
        raw[draw] = delta.mean()
        pct[draw] = 100.0 * delta.mean() / control.mean()
    observed_delta = frame[challenger_column] - frame[baseline_column]
    mean_delta = float(observed_delta.mean())
    baseline_mean = float(frame[baseline_column].mean())
    if not np.isfinite(baseline_mean) or baseline_mean <= 0:
        raise ValueError("Noninferiority requires a finite positive configured-DK baseline mean.")
    mean_pct = float(100.0 * mean_delta / baseline_mean)
    margin_points = float((NONINFERIORITY_MARGIN_PCT / 100.0) * baseline_mean)
    ci95_points = np.quantile(raw, [0.025, 0.975]).tolist()
    return {
        "contrast": f"{challenger}_minus_{baseline}",
        "cluster_unit": "origin_season",
        "clusters": len(seasons),
        "draws": args.bootstrap_draws,
        "seed": args.bootstrap_seed,
        "baseline_mean_points": baseline_mean,
        "mean_delta_points": mean_delta,
        "ci95_points": ci95_points,
        "mean_delta_pct": mean_pct,
        "ci95_pct": np.quantile(pct, [0.025, 0.975]).tolist(),
        "noninferiority_margin_pct": NONINFERIORITY_MARGIN_PCT,
        "noninferiority_margin_points": margin_points,
        "noninferiority_pass": bool(ci95_points[0] >= margin_points),
    }


def summarize(
    args,
    design,
    manifest,
    score_manifest,
    room_scores,
    state_metrics,
) -> dict[str, Any]:
    d128_bootstrap = season_cluster_bootstrap(
        state_metrics,
        args,
        challenger="d256",
        baseline="d128",
    )
    expected_room_rows = len(configured_state_ids(args)) * len(ARMS) * args.rooms
    all_complete = bool(
        len(room_scores) == expected_room_rows
        and room_scores.groupby(["state_id", "arm"]).size().eq(args.rooms).all()
    )
    gates = {
        "frozen_design_exact": frozen_design_exact(args),
        "global_freeze_complete_before_outcomes": bool(manifest.get("freeze_complete")),
        "score_artifacts_sealed": bool(
            score_manifest.get("score_complete")
            and score_manifest.get("exact_state_arm_room_domain")
        ),
        "all_states_and_24_room_arm_scores_complete": all_complete,
        "strict_prior_donor_contract": bool(manifest.get("strict_prior_all_origins")),
        "physical_bank_and_roster_contracts": bool(
            manifest.get("physical_bank_and_roster_contracts")
        ),
        "two_stage_process_isolation": bool(
            manifest.get("two_stage_process_isolation")
        ),
        "production_d128_prefix_exact": bool(
            manifest.get("production_d128_prefix_exact")
        ),
        "d256_noninferior_to_d128_at_minus_0_25_pct": bool(
            d128_bootstrap["noninferiority_pass"]
        ),
    }
    return {
        "completed_at_utc": utc_now(),
        "design": design,
        "freeze_manifest": manifest,
        "score_manifest": score_manifest,
        "counts": {
            "states": int(len(state_metrics)),
            "origin_seasons": int(state_metrics.origin_year.nunique()),
            "room_arm_scores": int(len(room_scores)),
        },
        "target_outcome_means": {
            arm: float(state_metrics[f"{arm}_target_points"].mean()) for arm in ARMS
        },
        "primary_d256_vs_d128": d128_bootstrap,
        "gates": gates,
        "all_frozen_gates_pass": bool(all(gates.values())),
        "promotion_ready": False,
        "target_scoring_scope": design["target_scoring_scope"],
        "interpretation_boundary": (
            "Synthetic ADP boards and a common Sequential downstream policy isolate "
            "the current-pick action; this is not a replay of observed draft rooms or "
            "a full-season Legacy policy trajectory. Outcomes use configured-DK "
            "all-played scoring, not official contest-realized DK scoring."
        ),
    }


def run_score(args, design: dict) -> None:
    manifest, records = validate_freeze_for_scoring(args, design)
    score_path = args.output_dir / "room_scores.csv"
    score_receipts_path = args.output_dir / "score_receipts.json"
    existing = pd.read_csv(score_path) if args.resume and score_path.exists() else pd.DataFrame()
    previous_score_receipts = (
        json.loads(score_receipts_path.read_text(encoding="utf-8"))
        if args.resume and score_receipts_path.exists()
        else {}
    )
    if not isinstance(previous_score_receipts, dict):
        previous_score_receipts = {}
    existing, previous_score_receipts = recover_score_checkpoint(
        existing,
        previous_score_receipts,
        records,
        args,
    )
    in_progress = {
        "started_at_utc": utc_now(),
        "design_hash": design["design_hash"],
        "freeze_manifest_hash": stable_hash(manifest),
        "score_complete": False,
    }
    atomic_write_json(args.output_dir / "score_manifest.json", in_progress)
    atomic_write_json(
        args.output_dir / "summary.json",
        {"status": "score_in_progress", **in_progress},
    )
    combined = existing.copy()
    configured_origin_keys = set(map(str, args.origin_values))
    score_receipts: dict[str, Any] = {
        key: value
        for key, value in previous_score_receipts.items()
        if key in configured_origin_keys
    }

    for origin_year in args.origin_values:
        with open_origin(args, origin_year) as origin:
            contract = validate_origin_contract(origin, origin_year)
            origin_records = [
                record for record in records.values()
                if record["freeze_payload"]["origin_year"] == origin_year
            ]
            fingerprints = {record["freeze_payload"]["origin_source_fingerprint"] for record in origin_records}
            if fingerprints != {contract["source_fingerprint"]}:
                raise ValueError(f"Origin {origin_year} source changed between freeze and score.")
            database_hashes = {
                record["freeze_payload"]["origin_database_sha256"]
                for record in origin_records
            }
            if database_hashes != {contract["database_sha256"]}:
                raise ValueError(f"Origin {origin_year} database changed between freeze and score.")
            receipt_hashes = {
                record["freeze_payload"]["origin_receipt_hash"]
                for record in origin_records
            }
            if receipt_hashes != {stable_hash(contract["receipt"])}:
                raise ValueError(f"Origin {origin_year} decision receipt changed between freeze and score.")
            assert_origin_roster_contracts(origin, origin_records, args)
            origin_rows = pd.DataFrame(score_origin_records(origin, origin_records, args))
            current_score_receipt = build_target_source_receipt(origin, design)
            target_fingerprint = current_score_receipt["target_outcome_fingerprint"]
            target_receipt_hash = target_source_receipt_hash(current_score_receipt)
            origin_rows["target_outcome_fingerprint"] = target_fingerprint
            origin_rows["target_source_receipt_hash"] = target_receipt_hash
            prior_origin = (
                combined[combined.origin_year == origin_year]
                if not combined.empty
                else pd.DataFrame()
            )
            if not prior_origin.empty:
                prior_fingerprints = set(
                    prior_origin.target_outcome_fingerprint.astype(str)
                )
                if prior_fingerprints != {target_fingerprint}:
                    raise ValueError(
                        f"Origin {origin_year} target outcomes changed since its score checkpoint."
                    )
                prior_receipt_hashes = set(
                    prior_origin.target_source_receipt_hash.astype(str)
                )
                if prior_receipt_hashes != {target_receipt_hash}:
                    raise ValueError(
                        f"Origin {origin_year} target source changed since its score checkpoint."
                    )
                previous_receipt = previous_score_receipts.get(str(origin_year))
                if previous_receipt != current_score_receipt:
                    raise ValueError(
                        f"Origin {origin_year} canonical target receipt changed since checkpoint."
                    )
                combined = combined[combined.origin_year != origin_year]
            combined = pd.concat([combined, origin_rows], ignore_index=True)
            score_receipts[str(origin_year)] = current_score_receipt
            atomic_write_json(score_receipts_path, score_receipts)
            atomic_write_frame(score_path, combined)
            print(f"scored origin {origin_year}: {len(origin_rows)} room-arm rows", flush=True)

    room_scores = combined
    validate_score_checkpoint(room_scores, records, args, score_receipts)
    expected_domain = {
        (key, arm, room_idx)
        for key in configured_state_ids(args)
        for arm in ARMS
        for room_idx in range(args.rooms)
    }
    observed_domain = set(
        room_scores[["state_id", "arm", "room_idx"]]
        .itertuples(index=False, name=None)
    )
    if observed_domain != expected_domain:
        raise ValueError("Room-score checkpoint does not cover the exact frozen state/arm/room domain.")
    room_scores = room_scores.sort_values(
        ["origin_year", "pick_slot", "completed_picks", "arm", "room_idx"],
        kind="mergesort",
    ).reset_index(drop=True)
    atomic_write_frame(score_path, room_scores)
    if set(score_receipts) != set(map(str, args.origin_values)):
        raise ValueError("Per-origin score receipts are incomplete.")
    target_receipt_hashes = {
        origin: target_source_receipt_hash(receipt)
        for origin, receipt in score_receipts.items()
    }
    atomic_write_json(score_receipts_path, score_receipts)
    state_metrics = state_metrics_from_room_scores(room_scores)
    atomic_write_frame(args.output_dir / "state_metrics.csv", state_metrics)
    if code_contract(args) != design["code_contract"]:
        raise ValueError("Study code changed during target scoring.")
    score_manifest = {
        "sealed_at_utc": utc_now(),
        "design_hash": design["design_hash"],
        "freeze_manifest_hash": stable_hash(manifest),
        "frozen_states_sha256": manifest["frozen_states_sha256"],
        "code_contract": design["code_contract"],
        "room_scores_sha256": sha256_file(score_path),
        "state_metrics_sha256": sha256_file(args.output_dir / "state_metrics.csv"),
        "score_receipts_sha256": sha256_file(score_receipts_path),
        "target_source_receipt_hashes": target_receipt_hashes,
        "target_source_sha256": {
            origin: receipt["target_source_receipt"]["source_sha256"]
            for origin, receipt in score_receipts.items()
        },
        "target_outcome_fingerprints": {
            origin: receipt["target_outcome_fingerprint"]
            for origin, receipt in score_receipts.items()
        },
        "exact_state_arm_room_domain": True,
        "score_complete": True,
    }
    atomic_write_json(args.output_dir / "score_manifest.json", score_manifest)
    summary = summarize(
        args,
        design,
        manifest,
        score_manifest,
        room_scores,
        state_metrics,
    )
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-child-request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--freeze-child-result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--model-repo", type=Path, default=MODEL_REPO)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "fantasy_football_historical_replay",
    )
    parser.add_argument("--output-dir", type=Path, default=STUDY_DIR / "results_v2")
    parser.add_argument("--phase", choices=("all", "freeze", "score"), default="freeze")
    parser.add_argument("--origins", default="2017,2018,2019,2020,2021,2022,2023,2024,2025")
    parser.add_argument("--slots", default="1,6,12")
    parser.add_argument("--depths", default="0,7,14")
    parser.add_argument("--teams", type=int, default=FROZEN_TEAMS)
    parser.add_argument("--rounds", type=int, default=FROZEN_ROUNDS)
    parser.add_argument("--rooms", type=int, default=FROZEN_ROOMS)
    parser.add_argument("--candidates", type=int, default=FROZEN_CANDIDATES)
    parser.add_argument("--construction-samples", type=int, default=FROZEN_CONSTRUCTION_SAMPLES)
    parser.add_argument("--evaluation-samples", type=int, default=FROZEN_EVALUATION_SAMPLES)
    parser.add_argument("--control-decision-samples", type=int, default=FROZEN_D128)
    parser.add_argument("--expanded-decision-samples", type=int, default=FROZEN_D256)
    parser.add_argument("--seed-base", type=int, default=FROZEN_SEED_BASE)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-states", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def finalize_args(parser, args):
    args.origin_values = parse_csv_values(args.origins, int)
    args.slot_values = parse_csv_values(args.slots, int)
    args.depth_values = parse_csv_values(args.depths, int)
    if args.smoke:
        args.origin_values = [max(args.origin_values)]
        args.slot_values = [6]
        args.depth_values = [0]
        args.rooms = min(args.rooms, 2)
        args.candidates = min(args.candidates, 4)
        args.construction_samples = min(args.construction_samples, 4)
        args.evaluation_samples = min(args.evaluation_samples, 4)
        args.control_decision_samples = min(args.control_decision_samples, 8)
        args.expanded_decision_samples = min(args.expanded_decision_samples, 16)
        if args.output_dir == STUDY_DIR / "results_v2":
            args.output_dir = STUDY_DIR / "results_smoke"
        args.max_states = args.max_states or 1
    if args.control_decision_samples > args.expanded_decision_samples:
        parser.error("D128 must be a prefix no larger than D256.")
    total_samples = (
        args.construction_samples
        + args.evaluation_samples
        + args.expanded_decision_samples
    )
    if total_samples > 1000:
        parser.error("Disjoint construction/evaluation/D256 banks exceed 1,000 columns.")
    if any(depth < 0 or depth >= args.rounds for depth in args.depth_values):
        parser.error("Every depth must be between 0 and rounds - 1.")
    if args.rooms <= 0 or args.candidates <= 0 or not args.origin_values:
        parser.error("Origins, rooms, and candidates must be non-empty/positive.")
    args.output_dir = args.output_dir.resolve()
    args.work_dir = args.work_dir.resolve()
    args.model_repo = args.model_repo.resolve()
    if args.model_repo != MODEL_REPO.resolve():
        parser.error(
            f"--model-repo must be the canonical sibling repository: {MODEL_REPO.resolve()}"
        )
    if not path_is_within(args.work_dir, Path(tempfile.gettempdir()).resolve()):
        parser.error("--work-dir must resolve under the operating system temporary directory.")
    return args


def main() -> None:
    parser = build_parser()
    parsed = parser.parse_args()
    child_flags = (parsed.freeze_child_request, parsed.freeze_child_result)
    if any(child_flags):
        if not all(child_flags):
            parser.error("Internal freeze-child request and result paths must be supplied together.")
        run_freeze_child(parsed)
        return
    args = finalize_args(parser, parsed)
    assert_maintained_interpreter(args.model_repo)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    design = prepare_design(args)
    args.sealed_code_contract = design["code_contract"]
    validate_simulation_helper_path(design["code_contract"])
    prepare_runner_receipt(args, design)
    if args.dry_run:
        print(json.dumps(design, indent=2))
        return
    if args.phase in ("all", "freeze"):
        run_freeze(args, design)
    if args.phase in ("all", "score"):
        run_score(args, design)


if __name__ == "__main__":
    main()
