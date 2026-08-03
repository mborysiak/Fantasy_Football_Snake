"""Leakage-safe historical inputs for the Sequential forced-pick replay.

This module deliberately keeps the decision and evaluation surfaces separate.
The disposable ``Simulation.sqlite3`` contains only preseason projections,
preseason keyed ADP, and weekly templates from seasons strictly before the
origin.  Realized origin-season weekly points are loaded lazily, after a caller
has frozen its decisions, and are never written to that SQLite database.

The source databases are always opened read-only.  The only database written by
this module is a newly-created study artifact under the caller's work directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import sys
import tempfile
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
SNAKE_ROOT = STUDY_DIR.parents[2]
GITHUB_ROOT = SNAKE_ROOT.parent
MODEL_ROOT = GITHUB_ROOT / "Fantasy_Football"

if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))
if str(MODEL_ROOT / "Scripts") not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT / "Scripts"))

from Modeling import s4_Best_Ball_Weekly as weekly_builder  # noqa: E402
import config as weekly_config  # noqa: E402
from Scripts import config as scripts_config  # noqa: E402
from Scripts.V2 import build_player_outcomes as governed_outcomes  # noqa: E402
from Scripts.V2 import config as v2_config  # noqa: E402
from Scripts.V2 import contracts as v2_contracts  # noqa: E402


SUPPORTED_ORIGINS = tuple(range(2017, 2026))
SUPPORTED_POSITIONS = ("QB", "RB", "WR", "TE")
PRED_VERSION = "final_ensemble"
LEAGUE = "dk"
WEEK_COUNT = 16
BOARD_SIZE = 240
DEFAULT_CANDIDATE_LIMIT = 360

# Only sources whose market is explicitly DK or best-ball compatible may enter
# the historical room.  Sources later in the tuple are fallbacks, not extra
# votes.  Generic FantasyPros, MFL, NFFC, FFA, and FantasyPoints ranks are not
# silently treated as historical DK best-ball ADP.
ADP_SOURCE_PRIORITY = (
    "fantasypros_best_ball_adp",
    "draftkings_adp",
    "adp_average_dk",
)
ADP_POLICY_VERSION = "dk_compatible_single_source_priority_keyed_v2"
ADP_COVERAGE_POLICY_VERSION = (
    "all_positive_allowed_market_rows_resolve_joined_origin_universe_v1"
)
ADP_NOISE_POLICY_VERSION = "deterministic_truncated_normal_bounds_v1"
ADP_STD_FRACTION = 0.18
ADP_STD_MIN = 6.0
ADP_STD_MAX = 30.0
ADP_BOUND_SIGMAS = 2.0

# Some locked handoff rows explicitly have no valid point center.  They cannot
# simply disappear from a historical board.  The narrow fallback below uses
# only other preseason centers from the same origin and position, never the
# target outcome.  Every use is a receipt row and top-240 uses are a named gate.
CENTER_POLICY_VERSION = "locked_else_same_origin_position_adp8_v1"
CENTER_NEIGHBORS = 8
MAX_TOP240_CENTER_IMPUTATIONS = 8
POOL_POLICY_VERSION = "production_matcher_strict_prior_v1"
OUTCOME_POLICY_VERSION = (
    "manifest_pinned_nflverse_gsis_full_horizon_gate_score_week1_16_v3"
)
OUTCOME_RECONCILIATION_ATOL = 1e-6
RUNTIME_NAME_POLICY_VERSION = "disambiguate_duplicate_display_name_pos_key8_v1"

# Material historical failures that motivated the full-horizon gate.  These
# values name either governed opportunity totals or configured-DK all-played
# totals, never a scoring fallback.  The check runs only when the identity is
# in that origin's candidate population and refuses to turn a known nonzero
# season into a zero outcome.
KNOWN_WEEKLY_MAPPING_REGRESSIONS = (
    {
        "origin_year": 2018,
        "display_name": "Tyler Ervin",
        "position": "RB",
        "points": 6.5,
        "population": "all_played",
    },
    {
        "origin_year": 2019,
        "display_name": "Robbie Chosen",
        "alternate_display_names": ("Robby Anderson", "Robbie Anderson"),
        "position": "WR",
        "points": 164.5,
        "population": "governed",
    },
    {
        "origin_year": 2019,
        "display_name": "Deonte Harty",
        "position": "WR",
        "points": 10.2,
        "population": "governed",
    },
    {
        "origin_year": 2020,
        "display_name": "Lamar Miller",
        "position": "RB",
        "points": 2.6,
        "population": "all_played",
    },
    {
        "origin_year": 2020,
        "display_name": "Chris Herndon",
        "position": "TE",
        "points": 56.4,
        "population": "governed",
    },
    {
        "origin_year": 2020,
        "display_name": "Deonte Harty",
        "position": "WR",
        "points": 48.7,
        "population": "governed",
    },
)

# Corrupt/truncated generic-provider identities found near the old sampled
# board boundary in 2024/2025.  They are fixtures for the allow-list and
# fail-any-allowed-ADP rules; none may become a candidate via a generic source.
KNOWN_PROVISIONAL_ADP_REGRESSION_KEYS = (
    "9d3afb0f-00dc-5526-a2d3-b96f0eca2fd8",
    "3f8ad288-fa98-5f65-b71b-5e912df3507d",
    "b5143644-0e44-5f4a-9699-e6f42098832b",
    "be41c6e3-3d3d-50c1-a66b-cae4e7b9c39d",
    "506e5d11-b0f1-5e6a-a730-99598732694a",
    "1e10160f-b1a1-5140-9478-b001a3073d3a",
    "05fee98e-61c6-526c-abdb-283cb3bebfcd",
    "60f7fc07-2a7f-5767-b215-6344198382fb",
    "71f2c858-4c6d-5c96-9326-7778f74702fc",
    "1ebdd5f0-f9e3-5e47-9afc-be5a2ca72c3",
    "b94b851f-73b5-5695-9087-4bdda5e78696",
)

# Reviewed duplicate identities present in the historical feature spine.  The
# source aliases on the left are not independently draftable players; their
# market observations belong to the confirmed key on the right.  No fuzzy
# redirect is allowed at study runtime.
GOVERNED_PROVISIONAL_KEY_REDIRECTS = {
    # Robby/Robbie Anderson -> current canonical Robbie Chosen identity.
    "a62767ca-6e79-5c00-92d9-d8dd19260b89": "b3a4e510-01f3-5651-91e2-7b4c468294da",
    "5241e828-a13a-5ce1-82c8-95bb66b0ac32": "b3a4e510-01f3-5651-91e2-7b4c468294da",
    # Ben Watson -> Benjamin Watson.
    "c90ee060-7829-56e9-9113-0a6b3af5d96a": "e57e342f-82ad-5f06-9b13-bbc2cc8ea9f4",
    # Hollywood Brown -> Marquise Brown.
    "efc32ec0-af66-593d-8a38-0440c0687a90": "4bce7997-6962-558d-a207-44e60c5fc456",
    # Truncated 2024 ADP provider labels with a unique reviewed incumbent.
    "41d294a1-1aea-5d92-9ea2-18be6bf8b6c0": "3f0b675d-ef58-5606-8f9e-73bc2a9b4118",
    "934933ee-30ba-5d02-b13e-58403636a609": "447b185a-9ecd-51f6-8921-30a0206a573c",
}
PROVISIONAL_RESOLUTION_POLICY_VERSION = (
    "reviewed_key_redirects_fail_any_allowed_adp_v2"
)

STARTERS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
FLEX_POSITIONS = frozenset({"RB", "WR", "TE"})


class HistoricalDataError(RuntimeError):
    """Raised when a replay input would be incomplete or potentially leaky."""


def validate_final_roster_state_contract(
    roster: Sequence[str],
    *,
    to_add: Sequence[str],
    to_drop: Sequence[str],
    depth: int,
    action_key: str,
    rounds: int,
    label: str = "Final roster",
) -> list[str]:
    """Fail closed if a frozen roster no longer represents its draft state."""

    normalized_roster = [str(key) for key in roster]
    normalized_add = [str(key) for key in to_add]
    normalized_drop = [str(key) for key in to_drop]
    depth = int(depth)
    rounds = int(rounds)
    action_key = str(action_key)

    if len(normalized_add) != depth:
        raise HistoricalDataError(
            f"{label} user prefix has {len(normalized_add)} players for depth {depth}."
        )
    if len(normalized_roster) != rounds:
        raise HistoricalDataError(
            f"{label} has {len(normalized_roster)} players for {rounds} rounds."
        )
    if len(normalized_roster) != len(set(normalized_roster)):
        raise HistoricalDataError(f"{label} contains duplicate player identities.")
    if normalized_roster[:depth] != normalized_add:
        raise HistoricalDataError(
            f"{label} does not preserve the frozen user-pick prefix."
        )
    opponent_overlap = sorted(set(normalized_roster) & set(normalized_drop))
    if opponent_overlap:
        raise HistoricalDataError(
            f"{label} contains opponent-drafted identities: {opponent_overlap}."
        )
    if not 0 <= depth < len(normalized_roster):
        raise HistoricalDataError(f"{label} has no forced-pick position at depth {depth}.")
    if normalized_roster[depth] != action_key:
        raise HistoricalDataError(
            f"{label} forced root {normalized_roster[depth]!r} does not match "
            f"action {action_key!r}."
        )
    return normalized_roster


@dataclass(frozen=True)
class HistoricalSourcePaths:
    """Read-only sources used by one historical origin."""

    projection_v2_db: Path = MODEL_ROOT / "Data" / "Databases" / "Projection_V2.sqlite3"
    simulation_db: Path = MODEL_ROOT / "Data" / "Databases" / "Simulation.sqlite3"

    def resolved(self) -> "HistoricalSourcePaths":
        return HistoricalSourcePaths(
            projection_v2_db=self.projection_v2_db.expanduser().resolve(),
            simulation_db=self.simulation_db.expanduser().resolve(),
        )

    def validate(self) -> None:
        for label, path in (
            ("Projection V2", self.projection_v2_db),
            ("Simulation", self.simulation_db),
        ):
            if not path.is_file():
                raise HistoricalDataError(f"{label} database does not exist: {path}")


@contextmanager
def _connect_read_only(path: Path) -> Iterator[sqlite3.Connection]:
    path = path.resolve()
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    try:
        yield connection
    finally:
        connection.close()


@lru_cache(maxsize=16)
def _cached_file_sha256(path_text: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    stat = path.stat()
    return _cached_file_sha256(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _frame_sha256(frame: pd.DataFrame, sort_columns: Sequence[str]) -> str:
    ordered = frame.copy()
    usable = [column for column in sort_columns if column in ordered]
    if usable:
        ordered = ordered.sort_values(usable, kind="mergesort")
    ordered = ordered.reindex(sorted(ordered.columns), axis=1).reset_index(drop=True)
    hashed = pd.util.hash_pandas_object(ordered, index=True, categorize=True)
    return hashlib.sha256(hashed.to_numpy(dtype=np.uint64).tobytes()).hexdigest()


def _decision_source_context(
    paths: HistoricalSourcePaths,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return path/policy context without hashing a mixed target database."""

    source_paths = {
        "projection_v2": {
            "path": str(paths.projection_v2_db),
            "authentication": (
                "selected_decision_frames_and_allowed_adp_boundary_audit"
            ),
        },
        "simulation": {
            "path": str(paths.simulation_db),
            "authentication": "selected_strict_prior_template_frames_only",
        },
    }
    policies = {
        "historical_data_code_sha256": file_sha256(Path(__file__).resolve()),
        "adp": ADP_POLICY_VERSION,
        "adp_priority": ADP_SOURCE_PRIORITY,
        "adp_coverage": ADP_COVERAGE_POLICY_VERSION,
        "provisional_resolution": PROVISIONAL_RESOLUTION_POLICY_VERSION,
        "provisional_redirects": GOVERNED_PROVISIONAL_KEY_REDIRECTS,
        "adp_noise": ADP_NOISE_POLICY_VERSION,
        "center": CENTER_POLICY_VERSION,
        "pool": POOL_POLICY_VERSION,
        "outcome": OUTCOME_POLICY_VERSION,
        "outcome_reconciliation_atol": OUTCOME_RECONCILIATION_ATOL,
        "runtime_name": RUNTIME_NAME_POLICY_VERSION,
    }
    return source_paths, policies


def _decision_frame_hashes(
    predictions: pd.DataFrame,
    avg_adps: pd.DataFrame,
    player_map: pd.DataFrame,
    pools: pd.DataFrame,
    templates: pd.DataFrame,
) -> dict[str, str]:
    return {
        "Final_Predictions_Resid": _frame_sha256(predictions, ["player_key"]),
        "Avg_ADPs": _frame_sha256(avg_adps, ["player_key"]),
        "Best_Ball_Weekly_Player_Map": _frame_sha256(
            player_map, ["player_key"]
        ),
        "Best_Ball_Weekly_Template_Pools": _frame_sha256(
            pools, ["template_pool_key", "match_rank"]
        ),
        "Best_Ball_Weekly_Templates": _frame_sha256(
            templates, ["template_id"]
        ),
    }


def _verify_model_import_contract(paths: HistoricalSourcePaths) -> None:
    """Reject a data repo that disagrees with the imported model code root."""

    expected_python = (
        MODEL_ROOT / ".venv_ff_312" / "Scripts" / "python.exe"
    ).resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise HistoricalDataError(
            "Historical replay must run under the canonical model Python: "
            f"{expected_python}"
        )
    expected_projection = (
        MODEL_ROOT / "Data" / "Databases" / "Projection_V2.sqlite3"
    ).resolve()
    expected_simulation = (
        MODEL_ROOT / "Data" / "Databases" / "Simulation.sqlite3"
    ).resolve()
    if paths.projection_v2_db.resolve() != expected_projection:
        raise HistoricalDataError(
            "Projection_V2 path does not belong to the imported model repository"
        )
    if paths.simulation_db.resolve() != expected_simulation:
        raise HistoricalDataError(
            "Simulation path does not belong to the imported model repository"
        )
    expected_modules = (
        (
            "weekly template builder",
            weekly_builder,
            MODEL_ROOT / "Scripts" / "Modeling" / "s4_Best_Ball_Weekly.py",
        ),
        (
            "governed outcome builder",
            governed_outcomes,
            MODEL_ROOT / "Scripts" / "V2" / "build_player_outcomes.py",
        ),
        (
            "weekly bare config",
            weekly_config,
            MODEL_ROOT / "Scripts" / "config.py",
        ),
        (
            "Scripts.config",
            scripts_config,
            MODEL_ROOT / "Scripts" / "config.py",
        ),
        (
            "Scripts.V2.config",
            v2_config,
            MODEL_ROOT / "Scripts" / "V2" / "config.py",
        ),
        (
            "Scripts.V2.contracts",
            v2_contracts,
            MODEL_ROOT / "Scripts" / "V2" / "contracts.py",
        ),
    )
    for label, module, expected_path in expected_modules:
        module_path = Path(module.__file__).resolve()
        if module_path != expected_path.resolve():
            raise HistoricalDataError(
                f"Imported {label} path differs from the canonical contract: "
                f"expected {expected_path.resolve()}, loaded {module_path}"
            )


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise HistoricalDataError(f"{label} lacks required columns: {missing}")


def _allowed_adp_coverage_sha256(audit: pd.DataFrame) -> str:
    return _frame_sha256(
        audit,
        [
            "market_original_player_key",
            "resolved_player_key",
            "expected_resolved_player_key",
            "source",
            "source_table",
            "market_position",
            "target_position",
            "adp",
            "provisional_redirect_applied",
            "resolution_semantics",
            "target_identity_status",
            "target_match_count",
            "position_mismatch_diagnostic",
            "original_target_position",
            "original_identity_status",
            "original_target_match_count",
            "coverage_status",
        ],
    )


def _audit_allowed_adp_universe_coverage(
    market: pd.DataFrame,
    joined_universe: pd.DataFrame,
) -> pd.DataFrame:
    """Require every allowed positive market row to survive origin assembly."""

    _require_columns(
        market,
        [
            "market_original_player_key",
            "player_key",
            "source",
            "source_table",
            "position",
            "adp",
            "provisional_redirect_applied",
        ],
        "market coverage input",
    )
    _require_columns(
        joined_universe,
        ["player_key", "position", "identity_is_confirmed"],
        "joined handoff/feature coverage universe",
    )

    allowed = market.copy()
    allowed["source"] = allowed["source"].astype("string").str.strip().str.lower()
    allowed["adp"] = pd.to_numeric(allowed["adp"], errors="coerce")
    allowed = allowed[
        allowed["source"].isin(ADP_SOURCE_PRIORITY)
        & np.isfinite(allowed["adp"])
        & allowed["adp"].gt(0)
    ].copy()
    allowed["market_original_player_key"] = allowed[
        "market_original_player_key"
    ].astype("string").str.strip().fillna("")
    allowed["resolved_player_key"] = allowed["player_key"].astype(
        "string"
    ).str.strip().fillna("")
    allowed["market_position"] = allowed["position"].astype(
        "string"
    ).str.strip().str.upper().fillna("")
    allowed["source_table"] = allowed["source_table"].astype(
        "string"
    ).fillna("")
    allowed["provisional_redirect_applied"] = pd.to_numeric(
        allowed["provisional_redirect_applied"], errors="coerce"
    ).fillna(-1).astype(int)
    allowed["expected_resolved_player_key"] = allowed[
        "market_original_player_key"
    ].replace(GOVERNED_PROVISIONAL_KEY_REDIRECTS)

    targets = joined_universe[
        ["player_key", "position", "identity_is_confirmed"]
    ].copy()
    targets["target_player_key"] = targets["player_key"].astype(
        "string"
    ).str.strip().fillna("")
    targets["target_position"] = targets["position"].astype(
        "string"
    ).str.strip().str.upper().fillna("")
    confirmed = pd.to_numeric(
        targets["identity_is_confirmed"], errors="coerce"
    ).eq(1)
    targets["target_identity_status"] = np.where(
        confirmed,
        "confirmed",
        "provisional",
    )
    target_summary = (
        targets.groupby("target_player_key", as_index=False, sort=True)
        .agg(
            target_match_count=("target_player_key", "size"),
            target_position=(
                "target_position",
                lambda values: "|".join(sorted(set(values.astype(str)))),
            ),
            target_identity_status=(
                "target_identity_status",
                lambda values: "|".join(sorted(set(values.astype(str)))),
            ),
        )
    )
    original_summary = target_summary.rename(
        columns={
            "target_player_key": "market_original_player_key",
            "target_match_count": "original_target_match_count",
            "target_position": "original_target_position",
            "target_identity_status": "original_identity_status",
        }
    )
    audit = allowed.merge(
        target_summary,
        left_on="resolved_player_key",
        right_on="target_player_key",
        how="left",
        validate="many_to_one",
    ).merge(
        original_summary,
        on="market_original_player_key",
        how="left",
        validate="many_to_one",
    )
    audit["target_match_count"] = pd.to_numeric(
        audit["target_match_count"], errors="coerce"
    ).fillna(0).astype(int)
    audit["original_target_match_count"] = pd.to_numeric(
        audit["original_target_match_count"], errors="coerce"
    ).fillna(0).astype(int)
    for column in (
        "target_position",
        "target_identity_status",
        "original_target_position",
        "original_identity_status",
    ):
        audit[column] = audit[column].astype("string").fillna("")

    redirected = audit["market_original_player_key"].ne(
        audit["expected_resolved_player_key"]
    )
    audit["resolution_semantics"] = np.select(
        [
            redirected,
            audit["target_identity_status"].eq("confirmed"),
            audit["target_match_count"].eq(1),
        ],
        [
            "governed_key_redirect",
            "direct_confirmed",
            "direct_provisional",
        ],
        default="direct_unresolved",
    )
    audit["coverage_status"] = "covered"
    redirect_contract_mismatch = (
        audit["resolved_player_key"].ne(audit["expected_resolved_player_key"])
        | audit["provisional_redirect_applied"].ne(redirected.astype(int))
    )
    missing_target = audit["target_match_count"].eq(0)
    ambiguous_target = audit["target_match_count"].gt(1)
    position_mismatch = (
        audit["target_match_count"].eq(1)
        & (
            audit["market_position"].eq("")
            | audit["target_position"].eq("")
            | audit["market_position"].ne(audit["target_position"])
        )
    )
    # Historical providers sometimes carry a stale position label for the
    # same confirmed identity (for example WR versus TE).  Key coverage is the
    # survivorship contract; retain the position disagreement as a sealed
    # diagnostic rather than deleting or rejecting the keyed observation.
    audit["position_mismatch_diagnostic"] = position_mismatch.astype(int)
    redirect_position_mismatch = redirected & position_mismatch
    audit.loc[
        redirect_position_mismatch, "coverage_status"
    ] = "redirect_position_mismatch"
    audit.loc[ambiguous_target, "coverage_status"] = "ambiguous_resolved_target"
    audit.loc[missing_target, "coverage_status"] = "missing_resolved_target"
    audit.loc[
        redirect_contract_mismatch, "coverage_status"
    ] = "redirect_contract_mismatch"

    audit = audit[
        [
            "market_original_player_key",
            "resolved_player_key",
            "expected_resolved_player_key",
            "source",
            "source_table",
            "market_position",
            "target_position",
            "adp",
            "provisional_redirect_applied",
            "resolution_semantics",
            "target_identity_status",
            "target_match_count",
            "position_mismatch_diagnostic",
            "original_target_position",
            "original_identity_status",
            "original_target_match_count",
            "coverage_status",
        ]
    ].sort_values(
        [
            "market_original_player_key",
            "resolved_player_key",
            "source",
            "source_table",
            "market_position",
            "adp",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    audit_sha256 = _allowed_adp_coverage_sha256(audit)
    failures = audit[audit["coverage_status"].ne("covered")]
    if not failures.empty:
        failure_counts = {
            str(key): int(value)
            for key, value in failures["coverage_status"]
            .value_counts()
            .sort_index()
            .items()
        }
        preview = failures[
            [
                "market_original_player_key",
                "resolved_player_key",
                "source",
                "market_position",
                "target_position",
                "adp",
                "resolution_semantics",
                "coverage_status",
            ]
        ].head(20)
        raise HistoricalDataError(
            "Allowed-source positive ADP universe coverage failed: "
            f"failure_count={len(failures)}, failure_status_counts={failure_counts}, "
            f"missing_unique_original_keys="
            f"{failures.loc[failures['coverage_status'].eq('missing_resolved_target'), 'market_original_player_key'].nunique()}, "
            f"audit_sha256={audit_sha256}, examples={preview.to_dict('records')}"
        )
    return audit


def _load_origin_universe(
    paths: HistoricalSourcePaths,
    origin_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_columns = [
        "player_key",
        "display_name",
        "season",
        "position",
        "team",
        "year_exp",
        "rookie_season",
        "expert_points_median",
        "expert_ppg_team_game_median",
        "expert_ppg_team_game_std",
        "projected_pass_point_share",
        "projected_rush_point_share",
        "projected_receiving_point_share",
        "team_qb1_ppg",
        "adp_median",
        "scoring_hash",
        "run_id",
        "feature_cutoff_season",
        "preseason_source_season",
        "identity_is_confirmed",
    ]
    with _connect_read_only(paths.projection_v2_db) as connection:
        # Explicit allow-list: this table also stores target-season actuals and
        # training residuals for model audit.  They are evaluation fields and
        # must not even be selected while decisions are being assembled.
        handoff = pd.read_sql_query(
            """
            SELECT lock_version,
                   model_run_id,
                   player_key,
                   gsis_id,
                   display_name,
                   season,
                   position,
                   team,
                   historical_pred_fp_per_game,
                   participation_lightgbm,
                   point_center_source,
                   joint_template_draw_required,
                   independent_model_residual_draw_allowed,
                   template_active_ppg_resid_recompute_required,
                   template_center_available
            FROM locked_template_handoff
            WHERE season=?
            """,
            connection,
            params=(origin_year,),
        )
        available_features = {
            str(row[1])
            for row in connection.execute('PRAGMA table_info("player_season_features")')
        }
        missing_features = sorted(set(feature_columns) - available_features)
        if missing_features:
            raise HistoricalDataError(
                "player_season_features schema drift: " + str(missing_features)
            )
        features = pd.read_sql_query(
            "SELECT "
            + ", ".join(f'"{column}"' for column in feature_columns)
            + ' FROM "player_season_features" WHERE season=?',
            connection,
            params=(origin_year,),
        )
        market = pd.read_sql_query(
            """
            SELECT player_key, season, source, source_table, position, adp
            FROM player_season_market_values
            WHERE season=? AND adp IS NOT NULL
            """,
            connection,
            params=(origin_year,),
        )

    _require_columns(
        handoff,
        [
            "player_key",
            "season",
            "position",
            "historical_pred_fp_per_game",
            "participation_lightgbm",
            "point_center_source",
            "joint_template_draw_required",
            "independent_model_residual_draw_allowed",
            "template_center_available",
        ],
        "locked_template_handoff",
    )
    for label, frame in (("handoff", handoff), ("features", features)):
        if frame.empty:
            raise HistoricalDataError(f"No {label} rows for origin {origin_year}")
        if frame["player_key"].isna().any() or frame["player_key"].duplicated().any():
            raise HistoricalDataError(f"{label} has blank or duplicate player keys")
    handoff_keys = set(handoff["player_key"].astype(str))
    feature_keys = set(features["player_key"].astype(str))
    if handoff_keys != feature_keys:
        raise HistoricalDataError(
            "locked handoff and feature universe disagree: "
            f"handoff_only={len(handoff_keys - feature_keys)}, "
            f"feature_only={len(feature_keys - handoff_keys)}"
        )
    if not handoff["joint_template_draw_required"].eq(1).all():
        raise HistoricalDataError("Historical handoff does not require a joint template draw")
    if not handoff["independent_model_residual_draw_allowed"].eq(0).all():
        raise HistoricalDataError("Historical handoff permits an independent residual draw")
    if not features["feature_cutoff_season"].eq(origin_year - 1).all():
        raise HistoricalDataError(
            "Feature cutoff must equal the immediately prior season"
        )
    if not features["preseason_source_season"].eq(origin_year).all():
        raise HistoricalDataError("Feature rows are not origin-season preseason rows")

    universe = handoff.merge(
        features,
        on=["player_key", "season", "position"],
        how="inner",
        validate="one_to_one",
        suffixes=("_handoff", "_feature"),
    )
    joined_keys = set(universe["player_key"].astype(str))
    if len(universe) != len(handoff) or joined_keys != handoff_keys:
        join_diagnostic = handoff[
            ["player_key", "season", "position"]
        ].merge(
            features[["player_key", "season", "position"]],
            on="player_key",
            how="outer",
            validate="one_to_one",
            suffixes=("_handoff", "_feature"),
            indicator=True,
        )
        join_diagnostic["season_matches"] = join_diagnostic[
            "season_handoff"
        ].eq(join_diagnostic["season_feature"])
        join_diagnostic["position_matches"] = join_diagnostic[
            "position_handoff"
        ].eq(join_diagnostic["position_feature"])
        mismatch = join_diagnostic[
            join_diagnostic["_merge"].ne("both")
            | ~join_diagnostic["season_matches"]
            | ~join_diagnostic["position_matches"]
        ]
        raise HistoricalDataError(
            "Joined handoff/feature universe did not preserve the exact raw key "
            f"set and row count: handoff_rows={len(handoff)}, "
            f"feature_rows={len(features)}, joined_rows={len(universe)}, "
            f"joined_missing_keys={len(handoff_keys - joined_keys)}, "
            f"examples={mismatch.head(20).to_dict('records')}"
        )
    universe["display_name"] = universe["display_name_handoff"].where(
        universe["display_name_handoff"].notna(),
        universe["display_name_feature"],
    )
    # Historical mutable provider team labels have already been governed in
    # the V2 feature mart.  Prefer that field to the convenience handoff copy.
    universe["team"] = universe["team_feature"].where(
        universe["team_feature"].notna(),
        universe["team_handoff"],
    )
    universe["player_key"] = universe["player_key"].astype(str)
    universe["position"] = universe["position"].astype(str).str.upper()
    universe["identity_is_confirmed"] = pd.to_numeric(
        universe["identity_is_confirmed"], errors="coerce"
    ).fillna(0).astype(int)
    coverage_universe = universe.copy()
    universe = universe[universe["position"].isin(SUPPORTED_POSITIONS)].copy()
    provisional_audit = universe[universe["identity_is_confirmed"].ne(1)].copy()
    universe = universe[universe["identity_is_confirmed"].eq(1)].copy()
    if universe.empty:
        raise HistoricalDataError("No confirmed canonical identities remain at the origin")
    if universe["player_key"].duplicated().any():
        raise HistoricalDataError("Confirmed origin universe has duplicate player keys")

    confirmed_positions = universe.set_index("player_key")["position"].astype(str).to_dict()
    provisional_audit["resolved_player_key"] = provisional_audit["player_key"].map(
        GOVERNED_PROVISIONAL_KEY_REDIRECTS
    )
    provisional_audit["provisional_resolution"] = np.where(
        provisional_audit["resolved_player_key"].notna(),
        "governed_key_redirect",
        "unresolved",
    )
    for row in provisional_audit[
        provisional_audit["resolved_player_key"].notna()
    ].itertuples(index=False):
        target_key = str(row.resolved_player_key)
        if target_key not in confirmed_positions:
            raise HistoricalDataError(
                f"Governed provisional redirect target is absent at {origin_year}: "
                f"{row.player_key} -> {target_key}"
            )
        if str(row.position) != str(confirmed_positions[target_key]):
            raise HistoricalDataError(
                f"Governed provisional redirect changes position: "
                f"{row.player_key} -> {target_key}"
            )

    market["source"] = market["source"].astype(
        "string"
    ).str.strip().str.lower()
    market["adp"] = pd.to_numeric(market["adp"], errors="coerce")
    market = market[np.isfinite(market["adp"]) & market["adp"].gt(0)].copy()
    market["market_original_player_key"] = market["player_key"].astype(
        "string"
    ).str.strip().fillna("")
    duplicate_market = market.duplicated(
        ["market_original_player_key", "source"], keep=False
    )
    if duplicate_market.any():
        preview = market.loc[
            duplicate_market, ["market_original_player_key", "source"]
        ].head(20)
        raise HistoricalDataError(
            "Keyed market source has duplicate player/source rows: "
            f"{preview.to_dict('records')}"
        )
    market["player_key"] = market["market_original_player_key"].replace(
        GOVERNED_PROVISIONAL_KEY_REDIRECTS
    )
    market["provisional_redirect_applied"] = (
        market["player_key"] != market["market_original_player_key"]
    ).astype(int)
    allowed_adp_coverage_audit = _audit_allowed_adp_universe_coverage(
        market,
        coverage_universe,
    )
    return (
        universe.reset_index(drop=True),
        market.reset_index(drop=True),
        provisional_audit.reset_index(drop=True),
        allowed_adp_coverage_audit,
    )


def _attach_frozen_adp(
    universe: pd.DataFrame,
    market: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    priority = {source: index for index, source in enumerate(ADP_SOURCE_PRIORITY)}
    # Unknown/generic sources are not fallbacks.  In particular, an NFFC row
    # must never enter a DK historical board merely because it exists in the
    # shared market table.
    market = market[
        market["player_key"].isin(universe["player_key"])
        & market["source"].isin(ADP_SOURCE_PRIORITY)
    ].copy()
    market["source_priority"] = market["source"].map(priority)
    market = market.sort_values(
        [
            "player_key",
            "source",
            "provisional_redirect_applied",
            "market_original_player_key",
            "adp",
        ],
        kind="mergesort",
    ).drop_duplicates(["player_key", "source"], keep="first")
    market = market.sort_values(
        ["player_key", "source_priority", "source", "source_table", "adp"],
        kind="mergesort",
    )
    chosen = market.drop_duplicates("player_key", keep="first")[
        [
            "player_key",
            "adp",
            "source",
            "source_table",
            "source_priority",
            "market_original_player_key",
            "provisional_redirect_applied",
        ]
    ].rename(
        columns={
            "adp": "selected_adp",
            "source": "selected_adp_source",
            "source_table": "selected_adp_source_table",
        }
    )
    output = universe.merge(chosen, on="player_key", how="left", validate="one_to_one")
    output["selected_adp"] = pd.to_numeric(output["selected_adp"], errors="coerce")
    output = output[np.isfinite(output["selected_adp"]) & output["selected_adp"].gt(0)].copy()
    output = output.sort_values(
        ["selected_adp", "position", "player_key"], kind="mergesort"
    ).reset_index(drop=True)
    output["board_rank"] = np.arange(1, len(output) + 1)
    audit = output[
        [
            "player_key",
            "display_name",
            "position",
            "selected_adp",
            "selected_adp_source",
            "selected_adp_source_table",
            "market_original_player_key",
            "provisional_redirect_applied",
            "board_rank",
        ]
    ].copy()
    return output, audit


def _audit_provisional_board_boundary(
    provisional: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    audit = provisional.copy()
    audit["known_provisional_adp_regression_fixture"] = audit[
        "player_key"
    ].astype(str).isin(KNOWN_PROVISIONAL_ADP_REGRESSION_KEYS)
    if audit.empty:
        audit["provisional_allowed_adp_rows"] = pd.Series(dtype="Int64")
        audit["provisional_allowed_adp_sources"] = pd.Series(dtype=str)
        audit["provisional_selected_adp"] = pd.Series(dtype=float)
        audit["provisional_adp_min_pick"] = pd.Series(dtype=float)
        return audit
    priority = {source: index for index, source in enumerate(ADP_SOURCE_PRIORITY)}
    allowed_market = market[
        market["market_original_player_key"].isin(audit["player_key"].astype(str))
        & market["source"].isin(ADP_SOURCE_PRIORITY)
    ].copy()
    allowed_market["source_priority"] = allowed_market["source"].map(priority)
    allowed_summary = (
        allowed_market.groupby("market_original_player_key", as_index=False, sort=True)
        .agg(
            provisional_allowed_adp_rows=("source", "size"),
            provisional_allowed_adp_sources=(
                "source",
                lambda values: ",".join(sorted(set(values))),
            ),
        )
        .rename(columns={"market_original_player_key": "player_key"})
    )
    allowed_selected = (
        allowed_market.sort_values(
            [
                "market_original_player_key",
                "source_priority",
                "source",
                "source_table",
                "adp",
            ],
            kind="mergesort",
        )
        .drop_duplicates("market_original_player_key", keep="first")
        [["market_original_player_key", "adp", "source", "source_table"]]
        .rename(
            columns={
                "market_original_player_key": "player_key",
                "adp": "provisional_selected_adp",
                "source": "provisional_selected_adp_source",
                "source_table": "provisional_selected_adp_source_table",
            }
        )
    )
    audit = audit.merge(
        allowed_summary,
        on="player_key",
        how="left",
        validate="one_to_one",
    ).merge(
        allowed_selected,
        on="player_key",
        how="left",
        validate="one_to_one",
    )
    audit["provisional_allowed_adp_rows"] = (
        pd.to_numeric(audit["provisional_allowed_adp_rows"], errors="coerce")
        .fillna(0)
        .astype("Int64")
    )
    audit["provisional_allowed_adp_sources"] = audit[
        "provisional_allowed_adp_sources"
    ].fillna("")
    adp_std, adp_min, adp_max = _adp_noise_columns(
        audit["provisional_selected_adp"]
    )
    audit["provisional_adp_std_dev"] = adp_std
    audit["provisional_adp_min_pick"] = adp_min
    audit["provisional_adp_max_pick"] = adp_max
    # The room samples from ADP bounds rather than drafting strictly by mean
    # rank.  More importantly, a population cap is not an identity rule: any
    # unresolved player with an allowed-source market observation is a
    # potentially draftable candidate and therefore fails closed.
    blocked = audit[
        audit["provisional_resolution"].eq("unresolved")
        & audit["provisional_allowed_adp_rows"].gt(0)
    ]
    if not blocked.empty:
        preview = blocked[
            [
                "player_key",
                "display_name",
                "position",
                "provisional_selected_adp",
                "provisional_selected_adp_source",
                "provisional_adp_min_pick",
                "provisional_allowed_adp_sources",
            ]
        ].head(20)
        raise HistoricalDataError(
            "Unresolved provisional identities have a DK-compatible ADP and "
            "could enter the sampled draft room: "
            f"{preview.to_dict('records')}"
        )
    audit["provisional_resolution"] = np.select(
        [
            audit["provisional_resolution"].eq("governed_key_redirect")
            & audit["provisional_allowed_adp_rows"].gt(0),
            audit["provisional_resolution"].eq("governed_key_redirect"),
        ],
        [
            "governed_key_redirect_allowed_market_merged",
            "governed_key_redirect_no_allowed_market",
        ],
        default="excluded_unresolved_no_allowed_adp",
    )
    return audit


def _impute_missing_centers(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = candidates.copy()
    candidates["pred_fp_per_game"] = pd.to_numeric(
        candidates["historical_pred_fp_per_game"], errors="coerce"
    )
    candidates["replay_center_source"] = "locked_template_handoff"
    records: list[dict[str, Any]] = []
    missing_indices = candidates.index[candidates["pred_fp_per_game"].isna()]
    for index in missing_indices:
        row = candidates.loc[index]
        donors = candidates[
            candidates["position"].eq(row["position"])
            # Never let one invented center become evidence for another.
            & pd.to_numeric(
                candidates["historical_pred_fp_per_game"], errors="coerce"
            ).notna()
            & candidates["selected_adp"].notna()
        ].copy()
        if len(donors) < CENTER_NEIGHBORS:
            raise HistoricalDataError(
                f"Not enough same-position preseason centers to cover {row['player_key']}"
            )
        target_log_adp = math.log1p(float(row["selected_adp"]))
        donors["center_adp_distance"] = np.abs(
            np.log1p(donors["selected_adp"].astype(float)) - target_log_adp
        )
        donors = donors.sort_values(
            ["center_adp_distance", "selected_adp", "player_key"],
            kind="mergesort",
        ).head(CENTER_NEIGHBORS)
        weights = 1.0 / (0.05 + donors["center_adp_distance"].to_numpy(dtype=float))
        imputed = float(
            np.average(donors["pred_fp_per_game"].to_numpy(dtype=float), weights=weights)
        )
        candidates.at[index, "pred_fp_per_game"] = imputed
        candidates.at[index, "replay_center_source"] = CENTER_POLICY_VERSION
        records.append(
            {
                "player_key": row["player_key"],
                "player": row["display_name"],
                "pos": row["position"],
                "board_rank": int(row["board_rank"]),
                "selected_adp": float(row["selected_adp"]),
                "imputed_pred_fp_per_game": imputed,
                "donor_keys": ",".join(donors["player_key"].astype(str)),
                "policy": CENTER_POLICY_VERSION,
            }
        )
    audit = pd.DataFrame.from_records(records)
    top240_count = int((audit.get("board_rank", pd.Series(dtype=int)) <= BOARD_SIZE).sum())
    if top240_count > MAX_TOP240_CENTER_IMPUTATIONS:
        raise HistoricalDataError(
            f"Origin requires {top240_count} top-{BOARD_SIZE} center imputations; "
            f"gate permits at most {MAX_TOP240_CENTER_IMPUTATIONS}"
        )
    if candidates["pred_fp_per_game"].isna().any():
        raise HistoricalDataError("Unresolved historical projection centers remain")
    return candidates, audit


def _assign_runtime_player_labels(
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Give duplicate display names unique deterministic runtime labels."""

    output = candidates.copy()
    display = output["display_name"].astype("string").str.strip()
    if display.isna().any() or display.eq("").any():
        raise HistoricalDataError("A retained candidate has a blank display name")
    normalized = display.str.casefold()
    duplicate = normalized.duplicated(keep=False)
    output["player"] = display.astype(str)
    output.loc[duplicate, "player"] = [
        f"{name} [{position} {str(player_key)[:8]}]"
        for name, position, player_key in output.loc[
            duplicate, ["display_name", "position", "player_key"]
        ].itertuples(index=False, name=None)
    ]
    if output["player"].duplicated().any():
        preview = output.loc[
            output["player"].duplicated(keep=False),
            ["player_key", "display_name", "position", "player"],
        ].head(20)
        raise HistoricalDataError(
            "Runtime player labels remain non-unique after disambiguation: "
            f"{preview.to_dict('records')}"
        )
    audit = output.loc[
        duplicate, ["player_key", "display_name", "position", "player"]
    ].copy()
    audit["runtime_name_policy"] = RUNTIME_NAME_POLICY_VERSION
    return output, audit.reset_index(drop=True)


def _adp_noise_columns(avg_pick: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    avg_pick = pd.to_numeric(avg_pick, errors="raise").astype(float)
    std = (ADP_STD_FRACTION * avg_pick).clip(ADP_STD_MIN, ADP_STD_MAX)
    minimum = (avg_pick - ADP_BOUND_SIGMAS * std).clip(lower=1.0)
    maximum = avg_pick + ADP_BOUND_SIGMAS * std
    return std, minimum, maximum


def _build_decision_frames(
    paths: HistoricalSourcePaths,
    origin_year: int,
    candidate_limit: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
]:
    (
        universe,
        market,
        provisional_audit,
        allowed_adp_coverage_audit,
    ) = _load_origin_universe(paths, origin_year)
    ranked, adp_audit = _attach_frozen_adp(universe, market)
    provisional_audit = _audit_provisional_board_boundary(provisional_audit, market)
    admitted_regression_keys = sorted(
        set(ranked["player_key"].astype(str)).intersection(
            KNOWN_PROVISIONAL_ADP_REGRESSION_KEYS
        )
    )
    if admitted_regression_keys:
        raise HistoricalDataError(
            "Known provisional ADP regression keys entered the candidate population: "
            f"{admitted_regression_keys}"
        )
    if len(ranked) < BOARD_SIZE:
        raise HistoricalDataError(
            f"Only {len(ranked)} keyed ADP rows; {BOARD_SIZE} are required for a full board"
        )
    # Retain the entire confirmed DK-compatible candidate population.  A
    # mean-rank or arbitrary row cap can exclude a player whose sampled ADP
    # lower bound reaches the room, and candidate_limit is kept only as the
    # caller compatibility/floor check above.
    candidates = ranked.copy()
    if len(candidates) < BOARD_SIZE:
        raise HistoricalDataError("Candidate limit cannot be smaller than the 240-pick board")
    candidates, center_audit = _impute_missing_centers(candidates)
    candidates, runtime_name_audit = _assign_runtime_player_labels(candidates)

    candidates["pos"] = candidates["position"].astype(str)
    candidates["year"] = int(origin_year)
    candidates["version"] = LEAGUE
    candidates["dataset"] = PRED_VERSION
    candidates["team"] = candidates["team"].fillna("").astype(str)
    year_exp = pd.to_numeric(candidates["year_exp"], errors="coerce")
    derived_exp = origin_year - pd.to_numeric(candidates["rookie_season"], errors="coerce")
    candidates["year_exp"] = year_exp.where(year_exp.notna(), derived_exp).clip(lower=0).fillna(0)

    participation = pd.to_numeric(
        candidates["participation_lightgbm"], errors="coerce"
    ).astype(float)
    invalid_participation = (
        ~np.isfinite(participation)
        | participation.lt(0.0)
        | participation.gt(1.0)
    )
    if invalid_participation.any():
        preview = candidates.loc[
            invalid_participation,
            ["player_key", "display_name", "position", "participation_lightgbm"],
        ].head(20)
        raise HistoricalDataError(
            "Retained candidates have invalid participation probabilities: "
            f"{preview.to_dict('records')}"
        )
    candidates["pred_appear_current"] = participation
    candidates["pred_appear_ny"] = candidates["pred_appear_current"]
    candidates["pred_fp_per_game_ny"] = candidates["pred_fp_per_game"]
    if not np.isfinite(candidates["pred_fp_per_game"].astype(float)).all():
        raise HistoricalDataError("Retained candidates have non-finite point centers")
    point_center_source = candidates["point_center_source"].astype("string").str.strip()
    imputed_center = candidates["replay_center_source"].eq(CENTER_POLICY_VERSION)
    missing_locked_provenance = ~imputed_center & (
        point_center_source.isna() | point_center_source.eq("")
    )
    if missing_locked_provenance.any():
        preview = candidates.loc[
            missing_locked_provenance,
            ["player_key", "display_name", "position", "point_center_source"],
        ].head(20)
        raise HistoricalDataError(
            "Locked point centers have blank provenance: "
            f"{preview.to_dict('records')}"
        )
    candidates["resolved_point_center_provenance"] = point_center_source
    candidates.loc[
        imputed_center, "resolved_point_center_provenance"
    ] = CENTER_POLICY_VERSION
    if candidates["resolved_point_center_provenance"].isna().any() or candidates[
        "resolved_point_center_provenance"
    ].eq("").any():
        raise HistoricalDataError("Point-center provenance remains unresolved")
    scoring_hash = candidates["scoring_hash"].astype("string").str.strip()
    if scoring_hash.isna().any() or scoring_hash.eq("").any():
        raise HistoricalDataError("Retained candidates have a blank scoring hash")
    if scoring_hash.nunique(dropna=True) != 1:
        raise HistoricalDataError("Retained candidates have multiple scoring hashes")
    candidates["scoring_hash"] = scoring_hash.astype(str)
    candidates["current_projection_model_version"] = candidates[
        "resolved_point_center_provenance"
    ].astype(str)
    candidates["next_projection_model_version"] = "not_used_historical_replay"
    candidates["production_handoff_version"] = "historical_replay_v1"
    candidates["current_projection_source"] = candidates[
        "resolved_point_center_provenance"
    ].astype(str)
    candidates["current_uncertainty_source"] = "joint_weekly_template_only"
    candidates["independent_current_residual_draw_allowed"] = 0
    candidates["next_projection_source"] = "not_used_historical_replay"
    candidates["next_uncertainty_source"] = "zero_not_used_historical_replay"
    candidates["v2_scoring_hash"] = candidates["scoring_hash"]
    resid_columns = [
        "pred_resid_5",
        "pred_resid_10",
        "pred_resid_25",
        "pred_resid_75",
        "pred_resid_90",
        "pred_resid_95",
    ]
    for column in resid_columns:
        candidates[column] = 0.0
        candidates[f"{column}_ny"] = 0.0

    prediction_columns = [
        "player_key",
        "player",
        "pos",
        "pred_fp_per_game",
        "pred_fp_per_game_ny",
        "dataset",
        "version",
        "year",
        "current_projection_model_version",
        "next_projection_model_version",
        "v2_scoring_hash",
        "pred_appear_current",
        "pred_appear_ny",
        *resid_columns,
        *[f"{column}_ny" for column in resid_columns],
        "production_handoff_version",
        "current_projection_source",
        "current_uncertainty_source",
        "independent_current_residual_draw_allowed",
        "next_projection_source",
        "next_uncertainty_source",
    ]
    predictions = candidates[prediction_columns].copy()

    std, minimum, maximum = _adp_noise_columns(candidates["selected_adp"])
    avg_adps = pd.DataFrame(
        {
            "player_key": candidates["player_key"],
            "draft_entity_key": candidates["player_key"],
            "player": candidates["player"],
            "pos": candidates["pos"],
            "team": candidates["team"],
            "Years_of_Experience": candidates["year_exp"],
            "avg_pick": candidates["selected_adp"].astype(float),
            "year": origin_year,
            "league": LEAGUE,
            "std_dev": std,
            "min_pick": minimum,
            "max_pick": maximum,
            "source_table": candidates["selected_adp_source_table"],
            "source_metric": candidates["selected_adp_source"],
            "publication_version": ADP_POLICY_VERSION,
        }
    )

    current_points = pd.to_numeric(candidates["expert_points_median"], errors="coerce")
    expert_ppg = pd.to_numeric(
        candidates["expert_ppg_team_game_median"], errors="coerce"
    )
    current_points = current_points.where(current_points.notna(), expert_ppg * WEEK_COUNT)
    current_points = current_points.where(
        current_points.notna(), candidates["pred_fp_per_game"] * WEEK_COUNT
    )
    player_map = candidates.copy()
    player_map["current_avg_proj_points"] = current_points
    player_map["avg_proj_points"] = current_points
    player_map["model_input_avg_pick"] = candidates["selected_adp"]
    player_map["adp_avg_pick"] = candidates["selected_adp"]
    player_map["avg_pick"] = candidates["selected_adp"]
    player_map["adp_year_exp"] = candidates["year_exp"]
    player_map["source_year_exp"] = candidates["year_exp"]
    player_map["year_exp_source"] = "v2_player_season_features"
    player_map["year_exp_uncapped_delta"] = 0.0
    player_map["projection_context_scoring_hash"] = candidates["scoring_hash"]
    player_map["projection_context_run_id"] = candidates["run_id"]
    for destination, source in (
        ("avg_proj_pass_points", "projected_pass_point_share"),
        ("avg_proj_rush_points", "projected_rush_point_share"),
        ("avg_proj_rec_points", "projected_receiving_point_share"),
    ):
        player_map[destination] = (
            pd.to_numeric(candidates[source], errors="coerce").fillna(0).clip(0, 1)
            * current_points
        )
    player_map["qb_avg_proj_pass_points"] = player_map["avg_proj_pass_points"]
    player_map["std_proj_points"] = (
        pd.to_numeric(candidates["expert_ppg_team_game_std"], errors="coerce")
        * WEEK_COUNT
    )
    player_map["std_pos_rank"] = np.nan
    player_map = weekly_builder.add_projection_buckets(
        player_map,
        value_col="pred_fp_per_game",
        group_cols=["year", "version", "dataset", "pos"],
        pct_col="prediction_rank_pct",
    )
    player_map = weekly_builder.add_exp_fields(player_map)
    player_map = weekly_builder.recompute_selected_universe_match_features(player_map)
    player_map["match_projection_rank_pct"] = player_map["prediction_rank_pct"]
    player_map["match_projection_ppg_scaled"] = (
        player_map["pred_fp_per_game"].clip(lower=0) / weekly_builder.PROJECTION_PPG_SCALE
    )
    player_map["projection_x_exp"] = (
        player_map["match_projection_rank_pct"] * player_map["year_exp_scaled"]
    )
    player_map["market_projection_gap"] = (
        player_map["adp_rank_pct"] - player_map["match_projection_rank_pct"]
    )
    player_map["template_pool_key"] = player_map.apply(
        lambda row: (
            f"{origin_year}|{LEAGUE}|{PRED_VERSION}|{row['pos']}|{row['player_key']}"
        ),
        axis=1,
    )
    player_map["current_context_source"] = "historical_v2_feature_context"
    player_map["current_context_match_method"] = "player_key"
    player_map["current_team_source"] = "v2_player_season_features"
    player_map["current_adp_source"] = candidates["selected_adp_source"].to_numpy()
    player_map["current_context_fallback_fields"] = np.where(
        pd.to_numeric(candidates["expert_points_median"], errors="coerce").isna(),
        "current_avg_proj_points",
        "",
    )
    player_map["current_context_missing_fields"] = ""
    player_map["current_context_missing_optional_fields"] = ""
    player_map["player_key_match_method"] = "historical_player_key"

    with _connect_read_only(paths.simulation_db) as connection:
        templates = pd.read_sql_query(
            """
            SELECT *
            FROM Best_Ball_Weekly_Templates
            WHERE league=? AND season<?
            """,
            connection,
            params=(LEAGUE, origin_year),
        )
    if templates.empty:
        raise HistoricalDataError(f"No strictly prior DK templates for {origin_year}")
    templates["season"] = pd.to_numeric(templates["season"], errors="raise").astype(int)
    if int(templates["season"].max()) >= origin_year:
        raise HistoricalDataError("Future template reached the decision surface")
    eligible_counts = (
        templates[templates["template_eligible"].eq(1)]
        .groupby("pos")["template_id"]
        .nunique()
    )
    for position in SUPPORTED_POSITIONS:
        if int(eligible_counts.get(position, 0)) < weekly_builder.MIN_TEMPLATE_POOL_SIZE:
            raise HistoricalDataError(
                f"Only {int(eligible_counts.get(position, 0))} prior {position} templates"
            )

    pool_members, pool_summary = weekly_builder.build_pool_tables(templates, player_map)
    if pd.to_numeric(pool_members["season"], errors="raise").ge(origin_year).any():
        raise HistoricalDataError("A pool member is not strictly prior to its origin")
    player_map = weekly_builder.finalize_player_map(player_map, pool_summary)
    used_template_ids = set(pool_members["template_id"].astype(int))
    templates = templates[templates["template_id"].astype(int).isin(used_template_ids)].copy()
    if len(used_template_ids) != templates["template_id"].nunique():
        raise HistoricalDataError("Template materialization lost a selected donor")

    audits = {
        "adp": adp_audit,
        "allowed_adp_coverage": allowed_adp_coverage_audit,
        "center": center_audit,
        "runtime_name": runtime_name_audit,
        "full_universe": universe,
        "excluded_provisional": provisional_audit,
    }
    return predictions, avg_adps, player_map, pool_members, templates, audits


def _governed_fantasy_last_week(origin_year: int) -> int:
    """Mirror the V2 outcome contract's completed fantasy-season horizon."""

    return 17 if int(origin_year) >= 2021 else 16


def _canonicalize_target_rows(
    frame: pd.DataFrame,
    candidate_keys: set[str],
    *,
    label: str,
    require_all_candidates: bool,
) -> pd.DataFrame:
    """Apply reviewed redirects but retain only canonical target rows."""

    output = frame.copy()
    if output.empty:
        if require_all_candidates:
            raise HistoricalDataError(f"{label} has no candidate rows")
        output["_original_player_key"] = pd.Series(dtype=str)
        output["_redirect_applied"] = pd.Series(dtype=int)
        output["_canonical_source_row"] = pd.Series(dtype=int)
        return output
    if output["player_key"].isna().any() or output["player_key"].duplicated().any():
        raise HistoricalDataError(f"{label} has blank or duplicate source player keys")
    output["_original_player_key"] = output["player_key"].astype(str)
    output["player_key"] = output["_original_player_key"].replace(
        GOVERNED_PROVISIONAL_KEY_REDIRECTS
    )
    output["_redirect_applied"] = (
        output["player_key"] != output["_original_player_key"]
    ).astype(int)
    output = output[output["player_key"].isin(candidate_keys)].copy()
    output["_canonical_source_row"] = (
        output["_original_player_key"] == output["player_key"]
    ).astype(int)
    output = output.sort_values(
        ["player_key", "_canonical_source_row", "_original_player_key"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    selected = output.drop_duplicates("player_key", keep="first").copy()
    provisional_only = selected[selected["_canonical_source_row"].ne(1)]
    if not provisional_only.empty:
        raise HistoricalDataError(
            f"{label} has only a provisional row for a canonical candidate: "
            f"{provisional_only[['player_key', '_original_player_key']].head(20).to_dict('records')}"
        )
    if require_all_candidates:
        missing = sorted(candidate_keys - set(selected["player_key"].astype(str)))
        if missing:
            raise HistoricalDataError(
                f"{label} is missing canonical candidates: {missing[:20]}"
            )
    return selected.reset_index(drop=True)


def _validate_candidate_identity_mapping(
    identities: pd.DataFrame,
    handoff_universe: pd.DataFrame,
    candidate_keys: set[str],
) -> pd.DataFrame:
    """Validate the exact handoff-key -> GSIS -> canonical-key contract."""

    full_identity = identities.copy()
    raw_full_gsis = full_identity["gsis_id"].astype("string")
    full_identity["gsis_id"] = raw_full_gsis.str.strip()
    if (
        raw_full_gsis.notna()
        & raw_full_gsis.ne(full_identity["gsis_id"])
    ).any():
        raise HistoricalDataError(
            "Full player_identity GSIS IDs contain surrounding whitespace"
        )
    identity = _canonicalize_target_rows(
        identities,
        candidate_keys,
        label="player_identity",
        require_all_candidates=True,
    )
    if not identity["identity_status"].astype(str).eq("confirmed").all():
        raise HistoricalDataError("A target-scored candidate identity is not confirmed")
    identity["gsis_id"] = identity["gsis_id"].astype("string").str.strip()
    if identity["gsis_id"].isna().any() or identity["gsis_id"].eq("").any():
        preview = identity.loc[
            identity["gsis_id"].isna() | identity["gsis_id"].eq(""),
            ["player_key", "identity_status"],
        ].head(20)
        raise HistoricalDataError(
            "Candidate identities have blank GSIS IDs: "
            f"{preview.to_dict('records')}"
        )
    duplicate_gsis = identity["gsis_id"].duplicated(keep=False)
    if duplicate_gsis.any():
        preview = identity.loc[
            duplicate_gsis, ["player_key", "gsis_id"]
        ].head(20)
        raise HistoricalDataError(
            "Candidate GSIS IDs resolve to multiple canonical keys: "
            f"{preview.to_dict('records')}"
        )
    nonblank_full_gsis = full_identity["gsis_id"].notna() & full_identity[
        "gsis_id"
    ].ne("")
    duplicate_full_gsis = nonblank_full_gsis & full_identity[
        "gsis_id"
    ].duplicated(keep=False)
    if duplicate_full_gsis.any():
        preview = full_identity.loc[
            duplicate_full_gsis,
            ["player_key", "gsis_id", "identity_status"],
        ].head(20)
        raise HistoricalDataError(
            "Full player_identity has a nonunique nonblank GSIS mapping: "
            f"{preview.to_dict('records')}"
        )

    handoff_identity = handoff_universe[
        handoff_universe["player_key"].astype(str).isin(candidate_keys)
    ][["player_key", "gsis_id"]].copy()
    handoff_identity["player_key"] = handoff_identity["player_key"].astype(str)
    if handoff_identity["player_key"].duplicated().any():
        raise HistoricalDataError("Candidate handoff has duplicate player keys")
    handoff_identity["handoff_gsis_id"] = handoff_identity["gsis_id"].astype(
        "string"
    ).str.strip()
    handoff_identity = handoff_identity.drop(columns="gsis_id")
    identity_check = identity.merge(
        handoff_identity,
        on="player_key",
        how="left",
        validate="one_to_one",
    )
    if identity_check["handoff_gsis_id"].isna().any() or identity_check[
        "handoff_gsis_id"
    ].eq("").any():
        raise HistoricalDataError("Candidate handoff rows have blank GSIS IDs")
    mismatch = identity_check["gsis_id"].ne(identity_check["handoff_gsis_id"])
    if mismatch.any():
        preview = identity_check.loc[
            mismatch, ["player_key", "gsis_id", "handoff_gsis_id"]
        ].head(20)
        raise HistoricalDataError(
            "Candidate handoff GSIS IDs disagree with player_identity: "
            f"{preview.to_dict('records')}"
        )
    return identity.reset_index(drop=True)


def _validate_exact_outcome_values(
    seasonal: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Validate exact governed labels before any numeric reconciliation."""

    output = seasonal.copy()
    observed = output["outcome_observed"].eq(1)
    row_present = pd.to_numeric(
        output["exact_outcome_row_present"], errors="coerce"
    ).fillna(0)
    if not row_present.isin([0, 1]).all():
        raise HistoricalDataError("Exact outcome row-presence flags are invalid")
    row_present = row_present.astype(int).eq(1)
    if observed.ne(row_present).any():
        preview = output.loc[
            observed.ne(row_present),
            [
                "player_key",
                "display_name",
                "outcome_observed",
                "exact_outcome_row_present",
            ],
        ].head(20)
        raise HistoricalDataError(
            "Spine observation flags disagree with exact player_season_outcomes: "
            f"{preview.to_dict('records')}"
        )
    output["exact_season_points"] = pd.to_numeric(
        output["exact_season_points"], errors="coerce"
    )
    exact_points = output.loc[observed, "exact_season_points"]
    if exact_points.isna().any() or not np.isfinite(
        exact_points.astype(float)
    ).all():
        raise HistoricalDataError(
            "Observed exact season_points must be numeric and finite"
        )
    output["unconditional_season_points"] = pd.to_numeric(
        output["unconditional_season_points"], errors="coerce"
    )
    spine_points = output.loc[observed, "unconditional_season_points"]
    if spine_points.isna().any() or not np.isfinite(
        spine_points.astype(float)
    ).all():
        raise HistoricalDataError(
            "Observed spine unconditional points must be numeric and finite"
        )

    spine_opportunities = pd.to_numeric(
        output["opportunity_games"], errors="coerce"
    )
    if spine_opportunities.isna().any() or not np.isfinite(
        spine_opportunities.astype(float)
    ).all():
        raise HistoricalDataError(
            "Spine opportunity_games must be numeric and finite"
        )
    if spine_opportunities.lt(0).any() or not np.isclose(
        spine_opportunities.to_numpy(dtype=float),
        np.rint(spine_opportunities.to_numpy(dtype=float)),
        rtol=0.0,
        atol=1e-9,
    ).all():
        raise HistoricalDataError(
            "Spine opportunity_games must be nonnegative integers"
        )
    output["opportunity_games"] = spine_opportunities.astype(int)
    expected_spine_appeared = output["opportunity_games"].gt(0).astype(int)
    spine_appeared = pd.to_numeric(output["appeared"], errors="coerce")
    if spine_appeared.isna().any() or not spine_appeared.isin([0, 1]).all():
        raise HistoricalDataError("Spine appeared flags must be 0 or 1")
    output["appeared"] = spine_appeared.astype(int)
    if not np.array_equal(
        expected_spine_appeared.to_numpy(dtype=int),
        output["appeared"].to_numpy(dtype=int),
    ):
        raise HistoricalDataError(
            "Spine appeared disagrees with opportunity_games"
        )

    exact_appeared = pd.to_numeric(
        output.loc[observed, "exact_outcome_appeared"], errors="coerce"
    )
    if exact_appeared.isna().any() or not exact_appeared.isin([0, 1]).all():
        raise HistoricalDataError(
            "Observed exact outcome appeared flags must be 0 or 1"
        )
    exact_opportunities = pd.to_numeric(
        output.loc[observed, "exact_opportunity_games"], errors="coerce"
    )
    if exact_opportunities.isna().any() or not np.isfinite(
        exact_opportunities.astype(float)
    ).all():
        raise HistoricalDataError(
            "Observed exact opportunity_games must be numeric and finite"
        )
    if exact_opportunities.le(0).any() or not np.isclose(
        exact_opportunities.to_numpy(dtype=float),
        np.rint(exact_opportunities.to_numpy(dtype=float)),
        rtol=0.0,
        atol=1e-9,
    ).all():
        raise HistoricalDataError(
            "Observed exact opportunity_games must be positive integers"
        )
    output.loc[observed, "exact_outcome_appeared"] = exact_appeared.astype(int)
    output.loc[observed, "exact_opportunity_games"] = exact_opportunities.astype(int)
    if not np.array_equal(
        exact_appeared.to_numpy(dtype=int),
        output.loc[observed, "appeared"].to_numpy(dtype=int),
    ):
        raise HistoricalDataError(
            "Exact outcome appeared disagrees with the spine"
        )
    if not np.array_equal(
        exact_opportunities.to_numpy(dtype=int),
        output.loc[observed, "opportunity_games"].to_numpy(dtype=int),
    ):
        raise HistoricalDataError(
            "Exact opportunity_games disagrees with the spine"
        )
    if not exact_appeared.astype(int).eq(
        exact_opportunities.gt(0).astype(int)
    ).all():
        raise HistoricalDataError(
            "Exact appeared disagrees with exact opportunity_games"
        )

    internal_delta = np.abs(
        output.loc[observed, "unconditional_season_points"]
        - output.loc[observed, "exact_season_points"]
    )
    if not np.isfinite(internal_delta.astype(float)).all() or internal_delta.gt(
        OUTCOME_RECONCILIATION_ATOL
    ).any():
        raise HistoricalDataError(
            "Spine unconditional points differ from exact governed season points"
        )
    return output, observed


def _load_governed_target_context(
    data: "HistoricalOriginData",
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load pinned target metadata, seasonal labels, and exact GSIS identities."""

    spine_columns = (
        "player_key",
        "display_name",
        "season",
        "position",
        "outcome_complete",
        "outcome_observed",
        "active_target_available",
        "appeared",
        "opportunity_games",
        "unconditional_season_points",
        "scoring_hash",
        "foundation_run_id",
    )
    outcome_columns = (
        "player_key",
        "season",
        "position",
        "season_points",
        "appeared",
        "opportunity_games",
        "scoring_hash",
        "run_id",
    )
    identity_columns = ("player_key", "gsis_id", "identity_status")
    with _connect_read_only(data.paths.projection_v2_db) as connection:
        for table, required in (
            ("player_season_spine", spine_columns),
            ("player_season_outcomes", outcome_columns),
            ("player_identity", identity_columns),
        ):
            available = {
                str(row[1])
                for row in connection.execute(f'PRAGMA table_info("{table}")')
            }
            missing = sorted(set(required) - available)
            if missing:
                raise HistoricalDataError(
                    f"{table} target schema drift: {missing}"
                )
        spine = pd.read_sql_query(
            "SELECT "
            + ", ".join(f'"{column}"' for column in spine_columns)
            + ' FROM "player_season_spine" WHERE season=? AND league=?',
            connection,
            params=(data.origin_year, LEAGUE),
        )
        exact_outcomes = pd.read_sql_query(
            "SELECT "
            + ", ".join(f'"{column}"' for column in outcome_columns)
            + ' FROM "player_season_outcomes" WHERE season=? AND league=?',
            connection,
            params=(data.origin_year, LEAGUE),
        )
        identities = pd.read_sql_query(
            "SELECT "
            + ", ".join(f'"{column}"' for column in identity_columns)
            + ' FROM "player_identity"',
            connection,
        )
        aliases = pd.read_sql_query(
            """
            SELECT player_key, season, source, position
            FROM player_aliases
            """,
            connection,
        )

    candidate_keys = set(data.player_map["player_key"].astype(str))
    seasonal = _canonicalize_target_rows(
        spine,
        candidate_keys,
        label="player_season_spine",
        require_all_candidates=True,
    )
    if not pd.to_numeric(seasonal["outcome_complete"], errors="coerce").eq(1).all():
        raise HistoricalDataError("A replay origin has an incomplete governed outcome")
    for flag_column in (
        "outcome_observed",
        "active_target_available",
        "appeared",
    ):
        flag = pd.to_numeric(seasonal[flag_column], errors="coerce")
        if flag.isna().any() or not flag.isin([0, 1]).all():
            raise HistoricalDataError(
                f"Governed seasonal outcome has an invalid {flag_column} flag"
            )
        seasonal[flag_column] = flag.astype(int)
    seasonal["unconditional_season_points"] = pd.to_numeric(
        seasonal["unconditional_season_points"], errors="coerce"
    )
    missing_points_with_outcome = (
        seasonal["unconditional_season_points"].isna()
        & (
            seasonal["outcome_observed"].eq(1)
            | seasonal["appeared"].eq(1)
        )
    )
    if missing_points_with_outcome.any():
        raise HistoricalDataError(
            "An observed/appeared candidate has no unconditional-season target"
        )

    foundation_values = seasonal["foundation_run_id"].astype("string").str.strip()
    if foundation_values.isna().any() or foundation_values.eq("").any():
        raise HistoricalDataError(
            "Candidate spine rows have blank outcomes foundation run IDs"
        )
    foundation_ids = set(foundation_values.astype(str))
    if len(foundation_ids) != 1:
        raise HistoricalDataError(
            "Origin candidates do not pin exactly one V2 outcomes foundation run"
        )
    foundation_run_id = next(iter(foundation_ids))

    exact = _canonicalize_target_rows(
        exact_outcomes,
        candidate_keys,
        label="player_season_outcomes",
        require_all_candidates=False,
    )
    exact = exact.rename(
        columns={
            "position": "exact_outcome_position",
            "season_points": "exact_season_points",
            "appeared": "exact_outcome_appeared",
            "opportunity_games": "exact_opportunity_games",
            "scoring_hash": "exact_outcome_scoring_hash",
            "run_id": "exact_outcome_run_id",
        }
    )
    exact["exact_outcome_row_present"] = 1
    exact_keep = [
        "player_key",
        "exact_outcome_position",
        "exact_season_points",
        "exact_outcome_appeared",
        "exact_opportunity_games",
        "exact_outcome_scoring_hash",
        "exact_outcome_run_id",
        "exact_outcome_row_present",
    ]
    seasonal = seasonal.merge(
        exact[exact_keep],
        on="player_key",
        how="left",
        validate="one_to_one",
    )
    seasonal, observed = _validate_exact_outcome_values(seasonal)
    exact_run_ids = seasonal.loc[
        observed, "exact_outcome_run_id"
    ].astype("string").str.strip()
    if exact_run_ids.isna().any() or exact_run_ids.eq("").any():
        raise HistoricalDataError(
            "Observed exact outcomes have blank foundation run IDs"
        )
    if not exact_run_ids.astype(str).eq(foundation_run_id).all():
        raise HistoricalDataError(
            "Exact player_season_outcomes rows do not belong to the pinned foundation"
        )

    decision_scoring = data.predictions["v2_scoring_hash"].astype(
        "string"
    ).str.strip()
    spine_scoring = seasonal["scoring_hash"].astype("string").str.strip()
    exact_scoring = seasonal.loc[
        observed, "exact_outcome_scoring_hash"
    ].astype("string").str.strip()
    for label, values in (
        ("decision", decision_scoring),
        ("spine", spine_scoring),
        ("exact outcome", exact_scoring),
    ):
        if values.isna().any() or values.eq("").any():
            raise HistoricalDataError(f"{label} scoring hashes contain blanks")
    decision_hashes = set(decision_scoring.astype(str))
    spine_hashes = set(spine_scoring.astype(str))
    exact_hashes = set(exact_scoring.astype(str))
    if len(decision_hashes) != 1 or len(spine_hashes) != 1:
        raise HistoricalDataError(
            "Decision and seasonal target surfaces must each have one scoring hash"
        )
    if decision_hashes != spine_hashes or (exact_hashes and exact_hashes != spine_hashes):
        raise HistoricalDataError(
            "Decision, spine, and exact outcome scoring hashes disagree"
        )

    nonzero_points = ~np.isclose(
        seasonal["unconditional_season_points"].fillna(0.0).to_numpy(dtype=float),
        0.0,
        rtol=0.0,
        atol=OUTCOME_RECONCILIATION_ATOL,
    )
    seasonal["requires_governed_weekly_mapping"] = (
        seasonal["outcome_observed"].eq(1)
        | seasonal["appeared"].eq(1)
        | pd.Series(nonzero_points, index=seasonal.index)
    )

    identity = _validate_candidate_identity_mapping(
        identities,
        data.audits["full_universe"],
        candidate_keys,
    )

    source_name = f"nflverse_weekly_stats_{data.origin_year}"
    with _connect_read_only(data.paths.projection_v2_db) as connection:
        manifest = pd.read_sql_query(
            """
            SELECT run_id, component, source_name, source_uri, source_sha256, row_count
            FROM source_manifest
            WHERE run_id=? AND component='outcomes' AND source_name=?
            """,
            connection,
            params=(foundation_run_id, source_name),
        )
    if len(manifest) != 1:
        raise HistoricalDataError(
            "Pinned outcomes source_manifest row is not unique for "
            f"{foundation_run_id} / {source_name}"
        )
    manifest_row = manifest.iloc[0].to_dict()
    if str(manifest_row.get("component", "")).strip() != "outcomes":
        raise HistoricalDataError("Pinned weekly manifest row is not an outcomes source")
    source_uri = str(manifest_row.get("source_uri", "")).strip()
    source_sha256 = str(manifest_row.get("source_sha256", "")).strip().lower()
    try:
        row_count = int(manifest_row.get("row_count"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise HistoricalDataError("Pinned weekly manifest row_count is invalid") from exc
    if not source_uri.startswith("https://"):
        raise HistoricalDataError("Pinned weekly target URI must be a nonblank HTTPS URI")
    if len(source_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in source_sha256
    ):
        raise HistoricalDataError("Pinned weekly target SHA256 is invalid")
    if row_count <= 0:
        raise HistoricalDataError("Pinned weekly target row_count must be positive")

    target_manifest = {
        "source_name": source_name,
        "source_uri": source_uri,
        "source_sha256": source_sha256,
        "row_count": row_count,
        "foundation_run_id": foundation_run_id,
        "scoring_hash": next(iter(decision_hashes)),
    }
    return seasonal, identity, target_manifest, identities, aliases


def _load_actual_weekly_outcomes(data: "HistoricalOriginData") -> pd.DataFrame:
    """Load, authenticate, map, and score the exact governed nflverse payload."""

    current_adapter_sha256 = file_sha256(Path(__file__).resolve())
    if current_adapter_sha256 != data.receipt["historical_data_code_sha256"]:
        raise HistoricalDataError(
            "Historical target adapter changed after decision assembly"
        )
    _verify_model_import_contract(data.paths)
    current_decision_hashes = _decision_frame_hashes(
        data.predictions,
        data.avg_adps,
        data.player_map,
        data.template_pools,
        data.templates,
    )
    if current_decision_hashes != data.receipt["table_sha256"]:
        raise HistoricalDataError(
            "A selected decision frame changed after the decision seal"
        )

    (
        seasonal,
        identity,
        manifest,
        full_identity,
        player_aliases,
    ) = _load_governed_target_context(data)
    try:
        raw, downloaded_sha256 = governed_outcomes._read_csv_payload(
            manifest["source_uri"]
        )
    except Exception as exc:
        raise HistoricalDataError(
            f"Could not read pinned weekly target {manifest['source_uri']}: {exc}"
        ) from exc
    downloaded_sha256 = str(downloaded_sha256).lower()
    if downloaded_sha256 != manifest["source_sha256"]:
        raise HistoricalDataError(
            "Downloaded weekly target SHA256 differs from source_manifest"
        )
    if len(raw) != int(manifest["row_count"]):
        raise HistoricalDataError(
            "Downloaded weekly target row count differs from source_manifest: "
            f"expected {manifest['row_count']}, received {len(raw)}"
        )
    _require_columns(
        raw,
        governed_outcomes.WEEKLY_REQUIRED_COLUMNS,
        "pinned nflverse weekly target",
    )
    raw_seasons = set(
        pd.to_numeric(raw["season"], errors="coerce").dropna().astype(int)
    )
    if raw_seasons != {data.origin_year}:
        raise HistoricalDataError(
            f"Pinned weekly payload has unexpected seasons: {sorted(raw_seasons)}"
        )

    scoring_code_path = Path(governed_outcomes.__file__).resolve()
    dependency_paths = {
        "build_player_outcomes": scoring_code_path,
        "contracts": Path(v2_contracts.__file__).resolve(),
        "v2_config": Path(v2_config.__file__).resolve(),
        "scripts_config": Path(scripts_config.__file__).resolve(),
        "weekly_config": Path(weekly_config.__file__).resolve(),
        "weekly_builder": Path(weekly_builder.__file__).resolve(),
    }
    dependency_hashes = {
        label: {
            "path": str(path.resolve()),
            "sha256": file_sha256(path.resolve()),
        }
        for label, path in dependency_paths.items()
    }
    scoring_code_sha256 = dependency_hashes["build_player_outcomes"]["sha256"]

    scored = governed_outcomes.score_weekly_stats(raw, LEAGUE)
    scored["season"] = pd.to_numeric(scored["season"], errors="coerce").astype("Int64")
    scored["week"] = pd.to_numeric(scored["week"], errors="coerce").astype("Int64")
    scored["season_type"] = scored["season_type"].astype("string").str.upper()
    scored["configured_fantasy_points"] = pd.to_numeric(
        scored["fantasy_points_configured"], errors="coerce"
    )
    if scored["configured_fantasy_points"].isna().any() or not np.isfinite(
        scored["configured_fantasy_points"].astype(float)
    ).all():
        raise HistoricalDataError("Configured weekly scoring produced non-finite points")

    origin_regular = scored[
        scored["season"].eq(data.origin_year)
        & scored["season_type"].eq("REG")
        & scored["week"].ge(1)
    ].copy()
    governed_last_week = _governed_fantasy_last_week(data.origin_year)
    full_horizon = origin_regular[
        origin_regular["week"].le(governed_last_week)
    ].copy()

    identity_map = identity[["player_key", "gsis_id"]].copy()
    identity_map["gsis_id"] = identity_map["gsis_id"].astype(str)
    full_horizon["raw_player_id"] = full_horizon["player_id"].astype("string").str.strip()
    mapped = full_horizon.merge(
        identity_map,
        left_on="raw_player_id",
        right_on="gsis_id",
        how="left",
        validate="many_to_one",
        indicator="_candidate_identity_join",
    )
    mapped_rows = mapped["_candidate_identity_join"].eq("both")
    unmapped_full_horizon = mapped[~mapped_rows].copy()
    mapped = mapped[mapped_rows].copy()
    if mapped["player_key"].isna().any():
        raise HistoricalDataError("An exact candidate GSIS join produced a blank player key")
    raw_id_key_counts = mapped.groupby("raw_player_id")["player_key"].nunique()
    if raw_id_key_counts.gt(1).any():
        raise HistoricalDataError(
            "A raw nflverse player_id collided across candidate player keys"
        )

    candidate_positions = (
        data.player_map[["player_key", "pos"]]
        .drop_duplicates("player_key")
        .set_index("player_key")["pos"]
        .astype(str)
        .to_dict()
    )
    mapped["candidate_position"] = mapped["player_key"].map(candidate_positions)
    if mapped["candidate_position"].isna().any():
        raise HistoricalDataError("Mapped weekly target lacks a candidate position")
    mapped["raw_position"] = mapped["position_group"].astype("string").str.upper()
    position_mismatch = (
        mapped["raw_position"].notna()
        & mapped["raw_position"].ne("")
        & mapped["raw_position"].ne(mapped["candidate_position"])
    )
    position_mismatch_examples = (
        mapped.loc[
            position_mismatch,
            [
                "player_key",
                "player_display_name",
                "raw_position",
                "candidate_position",
            ],
        ]
        .drop_duplicates()
        .head(10)
        .to_dict("records")
    )

    all_played = (
        mapped.groupby(["player_key", "week"], as_index=False, sort=True)
        .agg(fantasy_pts=("configured_fantasy_points", "sum"))
    )

    # Rebuild the governed seasonal population through the exact production
    # function.  This preserves its raw-position fallback and opportunity-mask
    # semantics.  It is deliberately separate from the all-played roster
    # stream above, which retains low-opportunity cameo points.
    try:
        governed_aggregate = governed_outcomes.aggregate_player_outcomes(
            raw,
            full_identity,
            league=LEAGUE,
            run_id=manifest["foundation_run_id"],
            player_aliases=player_aliases,
            completed_through_season=data.origin_year,
        )
    except Exception as exc:
        raise HistoricalDataError(
            f"Could not rebuild governed seasonal outcomes: {exc}"
        ) from exc
    governed_aggregate = governed_aggregate[
        governed_aggregate["season"].eq(data.origin_year)
        & governed_aggregate["league"].eq(LEAGUE)
    ].copy()
    governed_aggregate = _canonicalize_target_rows(
        governed_aggregate,
        set(candidate_positions),
        label="rebuilt governed player outcomes",
        require_all_candidates=False,
    )
    governed_aggregate = governed_aggregate.rename(
        columns={
            "season_points": "mapped_governed_season_points",
            "opportunity_games": "mapped_governed_opportunity_games",
        }
    )
    governed_keep = [
        "player_key",
        "mapped_governed_season_points",
        "mapped_governed_opportunity_games",
    ]
    reconciliation = seasonal.merge(
        governed_aggregate[governed_keep],
        on="player_key",
        how="left",
        validate="one_to_one",
    )
    reconciliation["mapped_governed_season_points"] = pd.to_numeric(
        reconciliation["mapped_governed_season_points"], errors="coerce"
    ).fillna(0.0)
    reconciliation["mapped_governed_opportunity_games"] = pd.to_numeric(
        reconciliation["mapped_governed_opportunity_games"], errors="coerce"
    ).fillna(0).astype(int)

    required = reconciliation[
        reconciliation["requires_governed_weekly_mapping"]
    ].copy()
    missing_required = required[
        required["mapped_governed_opportunity_games"].eq(0)
    ]
    if not missing_required.empty:
        preview = missing_required[
            [
                "player_key",
                "display_name",
                "outcome_observed",
                "appeared",
                "unconditional_season_points",
            ]
        ].head(20)
        raise HistoricalDataError(
            "Governed observed/appeared/nonzero candidates have no exact GSIS "
            f"weekly opportunity mapping through Week {governed_last_week}: "
            f"{preview.to_dict('records')}"
        )

    comparable = reconciliation["unconditional_season_points"].notna()
    reconciliation["season_points_abs_delta"] = np.abs(
        reconciliation["mapped_governed_season_points"]
        - reconciliation["unconditional_season_points"].fillna(0.0)
    )
    mismatched = reconciliation[
        comparable
        & reconciliation["season_points_abs_delta"].gt(
            OUTCOME_RECONCILIATION_ATOL
        )
    ]
    if not mismatched.empty:
        preview = mismatched[
            [
                "player_key",
                "display_name",
                "unconditional_season_points",
                "mapped_governed_season_points",
                "mapped_governed_opportunity_games",
                "season_points_abs_delta",
            ]
        ].sort_values("season_points_abs_delta", ascending=False).head(20)
        raise HistoricalDataError(
            "Pinned GSIS weekly points do not reconcile to the governed seasonal "
            f"target within {OUTCOME_RECONCILIATION_ATOL}: "
            f"{preview.to_dict('records')}"
        )

    governed_candidate_points = float(
        reconciliation["mapped_governed_season_points"].sum()
    )
    governed_candidate_rows = int(
        reconciliation["mapped_governed_opportunity_games"].sum()
    )
    all_played_points = float(mapped["configured_fantasy_points"].sum())
    all_played_rows = int(len(mapped))
    opportunity_excluded_rows = all_played_rows - governed_candidate_rows
    opportunity_excluded_points = all_played_points - governed_candidate_points
    if opportunity_excluded_rows < 0:
        raise HistoricalDataError(
            "Governed opportunity rows exceed mapped all-played rows"
        )

    regression_records: list[dict[str, Any]] = []
    all_played_totals_by_key = (
        all_played.groupby("player_key")["fantasy_pts"].sum().to_dict()
    )
    for fixture in KNOWN_WEEKLY_MAPPING_REGRESSIONS:
        if int(fixture["origin_year"]) != data.origin_year:
            continue
        fixture_names = {
            str(fixture["display_name"]).casefold(),
        }
        fixture_names.update(
            str(value).casefold()
            for value in fixture.get("alternate_display_names", ())
        )
        fixture_match = reconciliation[
            reconciliation["display_name"].astype("string").str.casefold().isin(
                fixture_names
            )
            & reconciliation["position"].astype(str).eq(str(fixture["position"]))
        ]
        if fixture_match.empty:
            continue
        if len(fixture_match) != 1:
            raise HistoricalDataError(
                f"Known weekly regression identity is ambiguous: {fixture}"
            )
        fixture_row = fixture_match.iloc[0]
        population = str(fixture["population"])
        if population == "governed":
            actual_points = float(fixture_row["mapped_governed_season_points"])
        elif population == "all_played":
            actual_points = float(
                all_played_totals_by_key.get(str(fixture_row["player_key"]), 0.0)
            )
        else:
            raise HistoricalDataError(
                f"Unknown weekly regression population: {population}"
            )
        expected_points = float(fixture["points"])
        if not np.isclose(
            actual_points,
            expected_points,
            rtol=0.0,
            atol=OUTCOME_RECONCILIATION_ATOL,
        ):
            raise HistoricalDataError(
                "Known weekly mapping regression did not retain its configured "
                f"points: {fixture['display_name']} expected {expected_points}, "
                f"mapped {actual_points} in {population}"
            )
        regression_records.append(
            {
                **fixture,
                "player_key": str(fixture_row["player_key"]),
                "mapped_points": actual_points,
            }
        )

    scoring_mapped = all_played[all_played["week"].between(1, WEEK_COUNT)].copy()
    candidate = (
        data.player_map[["player_key", "player", "pos"]]
        .drop_duplicates("player_key")
        .copy()
    )
    grid = pd.MultiIndex.from_product(
        [candidate["player_key"].astype(str), range(1, WEEK_COUNT + 1)],
        names=["player_key", "week"],
    ).to_frame(index=False)
    grid = grid.merge(
        scoring_mapped,
        on=["player_key", "week"],
        how="left",
        validate="one_to_one",
    )
    grid["fantasy_pts"] = pd.to_numeric(
        grid["fantasy_pts"], errors="coerce"
    ).fillna(0.0)
    grid = grid.merge(candidate, on="player_key", how="left", validate="many_to_one")
    wide = grid.pivot(
        index=["player_key", "player", "pos"],
        columns="week",
        values="fantasy_pts",
    )
    wide = wide.reindex(
        columns=range(1, WEEK_COUNT + 1), fill_value=0.0
    ).reset_index()
    wide.columns = [
        f"week_{column}" if isinstance(column, int) else column
        for column in wide.columns
    ]
    for week in range(1, WEEK_COUNT + 1):
        wide[f"week_{week}"] = pd.to_numeric(
            wide[f"week_{week}"], errors="coerce"
        ).fillna(0.0)

    configured_limitation = (
        "configured-DK scorer; current dictionaries zero two-point and "
        "individual return/special-teams touchdown components"
    )
    seasonal_sha256 = _frame_sha256(
        reconciliation[
            [
                "player_key",
                "outcome_observed",
                "appeared",
                "opportunity_games",
                "unconditional_season_points",
                "exact_season_points",
                "exact_outcome_appeared",
                "exact_opportunity_games",
                "exact_outcome_row_present",
                "scoring_hash",
                "requires_governed_weekly_mapping",
            ]
        ],
        ["player_key"],
    )
    week17_excluded = mapped[mapped["week"].gt(WEEK_COUNT)]
    scoring_horizon_raw = mapped[mapped["week"].between(1, WEEK_COUNT)]
    reconciliation_max_abs_delta = float(
        reconciliation.loc[comparable, "season_points_abs_delta"].max()
        if comparable.any()
        else 0.0
    )
    raw_schema_columns = [
        {"column": str(column), "dtype": str(raw[column].dtype)}
        for column in raw.columns
    ]
    raw_schema_sha256 = _json_sha256(raw_schema_columns)
    scoring_outcome_frame_sha256 = _frame_sha256(wide, ["player_key"])
    candidate_identity_sha256 = _frame_sha256(
        identity[["player_key", "gsis_id", "identity_status"]],
        ["player_key"],
    )
    full_player_identity_sha256 = _frame_sha256(
        full_identity[["player_key", "gsis_id", "identity_status"]],
        ["player_key"],
    )
    target_alias_position_sha256 = _frame_sha256(
        player_aliases[["player_key", "season", "source", "position"]],
        ["player_key", "season", "source", "position"],
    )
    evaluation_audit: dict[str, Any] = {
        "raw_schema_columns": raw_schema_columns,
        "raw_schema_sha256": raw_schema_sha256,
        "outcome_rows": int(len(wide)),
        "raw_payload_rows": int(len(raw)),
        "origin_regular_rows": int(len(origin_regular)),
        "origin_regular_points": float(
            origin_regular["configured_fantasy_points"].sum()
        ),
        "raw_all_player_full_governed_horizon_rows": int(len(full_horizon)),
        "raw_all_player_full_governed_horizon_points": float(
            full_horizon["configured_fantasy_points"].sum()
        ),
        "mapped_full_horizon_rows": int(len(mapped)),
        "mapped_full_horizon_points": all_played_points,
        "unmapped_full_horizon_rows": int(len(unmapped_full_horizon)),
        "unmapped_full_horizon_points": float(
            unmapped_full_horizon["configured_fantasy_points"].sum()
        ),
        "mapped_candidate_count": int(mapped["player_key"].nunique()),
        "all_played_full_horizon_raw_rows": all_played_rows,
        "all_played_full_horizon_player_week_rows": int(len(all_played)),
        "all_played_full_horizon_points": all_played_points,
        "all_played_week1_16_raw_rows": int(len(scoring_horizon_raw)),
        "all_played_week1_16_player_week_rows": int(len(scoring_mapped)),
        "all_played_week1_16_points": float(
            scoring_horizon_raw["configured_fantasy_points"].sum()
        ),
        "configured_dk_week1_16_outcome_frame_sha256": (
            scoring_outcome_frame_sha256
        ),
        "candidate_identity_sha256": candidate_identity_sha256,
        "full_player_identity_sha256": full_player_identity_sha256,
        "target_alias_position_sha256": target_alias_position_sha256,
        "decision_frame_sha256_revalidated": current_decision_hashes,
        "governed_reconciliation_rows": governed_candidate_rows,
        "governed_reconciliation_points": governed_candidate_points,
        "opportunity_excluded_rows": opportunity_excluded_rows,
        "opportunity_excluded_points": opportunity_excluded_points,
        "week17_excluded_rows": int(len(week17_excluded)),
        "week17_excluded_points": float(
            week17_excluded["configured_fantasy_points"].sum()
        ),
        "governed_last_week": governed_last_week,
        "governed_seasonal_rows": int(len(seasonal)),
        "governed_seasonal_sha256": seasonal_sha256,
        "exact_outcome_rows": int(
            reconciliation["outcome_observed"].eq(1).sum()
        ),
        "required_weekly_mapping_rows": int(len(required)),
        "missing_required_weekly_mapping_rows": 0,
        "outcome_reconciliation_atol": OUTCOME_RECONCILIATION_ATOL,
        "outcome_reconciliation_compared_rows": int(comparable.sum()),
        "outcome_reconciliation_max_abs_delta": (
            reconciliation_max_abs_delta
        ),
        "outcome_reconciliation_mismatch_rows": 0,
        "known_mapping_regression_rows_checked": int(
            len(regression_records)
        ),
        "known_mapping_regression_records": regression_records,
        "raw_candidate_position_mismatch_rows": int(
            position_mismatch.sum()
        ),
        "raw_candidate_position_mismatch_players": int(
            mapped.loc[position_mismatch, "player_key"].nunique()
        ),
        "raw_candidate_position_mismatch_examples": (
            position_mismatch_examples
        ),
        "configured_scoring_limitation": configured_limitation,
    }
    evaluation_metrics = {
        f"evaluation_{key}": value
        for key, value in evaluation_audit.items()
    }
    target_source_receipt = {
        "source_name": manifest["source_name"],
        "source_uri": manifest["source_uri"],
        "source_sha256": manifest["source_sha256"],
        "row_count": int(manifest["row_count"]),
        "origin_year": data.origin_year,
        "foundation_run_id": manifest["foundation_run_id"],
        "scoring_hash": manifest["scoring_hash"],
        "scoring_code_path": str(scoring_code_path),
        "scoring_code_sha256": scoring_code_sha256,
        "governed_last_week": governed_last_week,
        "scoring_week_count": WEEK_COUNT,
        "roster_scoring_population": (
            "configured_dk_all_regular_season_played_rows"
        ),
        "roster_position_attribution": "candidate_preseason_position",
        "reconciliation_population": (
            "build_player_outcomes.aggregate_player_outcomes"
        ),
        "configured_scoring_limitation": configured_limitation,
        "mask_code_path": str(scoring_code_path),
        "mask_code_sha256": scoring_code_sha256,
        "scoring_dependency_sha256": dependency_hashes,
        "evaluation_audit": evaluation_audit,
    }
    data.target_outcome_fingerprint = _json_sha256(
        {
            "target_source_receipt": target_source_receipt,
            "governed_seasonal_outcome_sha256": seasonal_sha256,
            "outcome_policy": OUTCOME_POLICY_VERSION,
            "historical_data_code_sha256": current_adapter_sha256,
        }
    )

    data.receipt["target_source_receipt"] = target_source_receipt
    data.receipt["target_scoring_dependency_sha256"] = dependency_hashes
    data.receipt["target_outcome_fingerprint"] = data.target_outcome_fingerprint
    data.receipt["target_outcomes_loaded"] = True
    data.receipt.update(evaluation_metrics)
    return wide.sort_values(
        ["pos", "player_key"], kind="mergesort"
    ).reset_index(drop=True)


def _normalize_rosters(
    rosters: Mapping[str, Sequence[str]] | Sequence[Sequence[str]] | Sequence[str],
) -> list[tuple[str, list[str]]]:
    if isinstance(rosters, Mapping):
        return [(str(roster_id), [str(key) for key in keys]) for roster_id, keys in rosters.items()]
    roster_list = list(rosters)
    if not roster_list:
        return []
    if isinstance(roster_list[0], str):
        return [("0", [str(key) for key in roster_list])]
    return [(str(index), [str(key) for key in keys]) for index, keys in enumerate(roster_list)]


def _score_one_roster(
    roster_keys: Sequence[str],
    outcome_by_key: Mapping[str, tuple[str, np.ndarray]],
) -> tuple[float, list[float]]:
    if len(roster_keys) != len(set(roster_keys)):
        raise HistoricalDataError("A scored roster contains duplicate player keys")
    unknown = sorted(set(roster_keys) - set(outcome_by_key))
    if unknown:
        raise HistoricalDataError(f"Scored roster contains unknown canonical keys: {unknown[:10]}")
    weekly_scores: list[float] = []
    for week_index in range(WEEK_COUNT):
        values_by_pos: dict[str, list[float]] = {position: [] for position in SUPPORTED_POSITIONS}
        for key in roster_keys:
            position, points = outcome_by_key[key]
            values_by_pos[position].append(float(points[week_index]))
        used: dict[str, int] = {}
        total = 0.0
        for position, required in STARTERS.items():
            ordered = sorted(values_by_pos[position], reverse=True)
            total += sum(ordered[:required])
            used[position] = min(required, len(ordered))
            values_by_pos[position] = ordered
        flex = []
        for position in FLEX_POSITIONS:
            flex.extend(values_by_pos[position][used[position] :])
        if flex:
            total += max(flex)
        weekly_scores.append(float(total))
    return float(sum(weekly_scores)), weekly_scores


@dataclass
class HistoricalOriginData:
    """Assembled decision surface with lazily isolated target outcomes."""

    origin_year: int
    paths: HistoricalSourcePaths
    predictions: pd.DataFrame
    avg_adps: pd.DataFrame
    player_map: pd.DataFrame
    template_pools: pd.DataFrame
    templates: pd.DataFrame
    donor_years: tuple[int, ...]
    receipt: dict[str, Any]
    source_fingerprint: str
    audits: dict[str, pd.DataFrame] = field(repr=False)
    db_path: Path | None = None
    target_outcome_fingerprint: str | None = field(default=None, init=False)
    set_year: int = field(init=False)
    pred_vers: str = PRED_VERSION
    league: str = LEAGUE
    _actual_weekly_outcomes: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _target_outcomes_read: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.set_year = self.origin_year

    def assert_target_outcomes_unread(self) -> None:
        if self._target_outcomes_read or self._actual_weekly_outcomes is not None:
            raise HistoricalDataError(
                "Origin-season target outcomes were read before the decision freeze"
            )

    def assert_decision_inputs_clean(self) -> None:
        self.assert_target_outcomes_unread()
        if max(self.donor_years) >= self.origin_year:
            raise HistoricalDataError("Decision donor years are not strictly prior")
        if self.template_pools["season"].ge(self.origin_year).any():
            raise HistoricalDataError("Decision pool contains an origin/future donor")
        forbidden = {"fantasy_pts", "unconditional_season_points", "observed_season_points"}
        for label, frame in (
            ("predictions", self.predictions),
            ("avg_adps", self.avg_adps),
            ("player_map", self.player_map),
            ("template_pools", self.template_pools),
        ):
            overlap = forbidden.intersection(frame.columns)
            if overlap:
                raise HistoricalDataError(f"{label} exposes target outcome columns: {overlap}")
        if len(self.predictions) < BOARD_SIZE or len(self.avg_adps) < BOARD_SIZE:
            raise HistoricalDataError("Decision population cannot cover a 240-pick board")
        key_sets = {
            label: set(frame["player_key"].astype(str))
            for label, frame in (
                ("predictions", self.predictions),
                ("avg_adps", self.avg_adps),
                ("player_map", self.player_map),
            )
        }
        if not (
            key_sets["predictions"]
            == key_sets["avg_adps"]
            == key_sets["player_map"]
        ):
            raise HistoricalDataError("Decision tables expose different candidate keys")
        for label, frame in (
            ("predictions", self.predictions),
            ("avg_adps", self.avg_adps),
            ("player_map", self.player_map),
        ):
            if frame["player_key"].duplicated().any():
                raise HistoricalDataError(f"{label} has duplicate canonical keys")
            if frame["player"].astype("string").str.strip().duplicated().any():
                raise HistoricalDataError(f"{label} has duplicate runtime player labels")
        participation = pd.to_numeric(
            self.predictions["pred_appear_current"], errors="coerce"
        ).astype(float)
        if (
            ~np.isfinite(participation)
            | participation.lt(0.0)
            | participation.gt(1.0)
        ).any():
            raise HistoricalDataError(
                "Decision population has invalid participation probabilities"
            )
        scoring_hashes = self.predictions["v2_scoring_hash"].astype("string").str.strip()
        if (
            scoring_hashes.isna().any()
            or scoring_hashes.eq("").any()
            or scoring_hashes.nunique(dropna=True) != 1
        ):
            raise HistoricalDataError("Decision population has invalid scoring hashes")
        point_sources = self.predictions["current_projection_source"].astype(
            "string"
        ).str.strip()
        if point_sources.isna().any() or point_sources.eq("").any():
            raise HistoricalDataError("Decision population has blank point-center provenance")
        expected_hashes = self.receipt.get("table_sha256")
        if expected_hashes is not None:
            actual_hashes = _decision_frame_hashes(
                self.predictions,
                self.avg_adps,
                self.player_map,
                self.template_pools,
                self.templates,
            )
            if actual_hashes != expected_hashes:
                raise HistoricalDataError(
                    "A selected decision frame changed after freeze authentication"
                )
        if "identity_is_confirmed" not in self.player_map or not self.player_map[
            "identity_is_confirmed"
        ].eq(1).all():
            raise HistoricalDataError("Decision database contains a provisional identity")
        if self.db_path is not None:
            with closing(sqlite3.connect(self.db_path)) as connection:
                table_names = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
            if any("outcome" in name.lower() or "actual" in name.lower() for name in table_names):
                raise HistoricalDataError("Decision database contains an outcome/actual table")

    @property
    def actual_weekly_outcomes(self) -> pd.DataFrame:
        self._target_outcomes_read = True
        if self._actual_weekly_outcomes is None:
            self._actual_weekly_outcomes = _load_actual_weekly_outcomes(self)
        return self._actual_weekly_outcomes.copy()

    def score_rosters(
        self,
        rosters: Mapping[str, Sequence[str]] | Sequence[Sequence[str]] | Sequence[str],
    ) -> pd.DataFrame:
        outcomes = self.actual_weekly_outcomes
        week_columns = [f"week_{week}" for week in range(1, WEEK_COUNT + 1)]
        outcome_by_key = {
            str(row.player_key): (
                str(row.pos),
                np.asarray([getattr(row, column) for column in week_columns], dtype=float),
            )
            for row in outcomes.itertuples(index=False)
        }
        records = []
        for roster_id, roster_keys in _normalize_rosters(rosters):
            score, weekly_scores = _score_one_roster(roster_keys, outcome_by_key)
            record: dict[str, Any] = {
                "roster_id": roster_id,
                "origin_year": self.origin_year,
                "roster_size": len(roster_keys),
                "best_ball_points": score,
                "roster_sha256": _json_sha256(sorted(roster_keys)),
            }
            record.update(
                {f"week_{week}": weekly_scores[week - 1] for week in range(1, WEEK_COUNT + 1)}
            )
            records.append(record)
        return pd.DataFrame.from_records(records)


def assemble_historical_origin(
    origin_year: int,
    *,
    paths: HistoricalSourcePaths | None = None,
    league: str = LEAGUE,
    strict_prior: bool = True,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
) -> HistoricalOriginData:
    """Assemble one origin without reading its realized weekly outcomes."""

    if int(origin_year) not in SUPPORTED_ORIGINS:
        raise HistoricalDataError(
            f"Origin {origin_year} is outside the supported 2017-2025 window"
        )
    if str(league).lower() != LEAGUE:
        raise HistoricalDataError("This replay builder is intentionally DK-only")
    if not strict_prior:
        raise HistoricalDataError("strict_prior=False is prohibited for this study")
    if int(candidate_limit) < BOARD_SIZE:
        raise HistoricalDataError("candidate_limit must cover all 240 room picks")
    paths = (paths or HistoricalSourcePaths()).resolved()
    paths.validate()
    _verify_model_import_contract(paths)
    source_entries, decision_policies = _decision_source_context(paths)
    predictions, avg_adps, player_map, pools, templates, audits = _build_decision_frames(
        paths, int(origin_year), int(candidate_limit)
    )
    donor_years = tuple(
        int(value)
        for value in sorted(
            pd.to_numeric(templates["season"], errors="raise").astype(int).unique()
        )
    )
    table_hashes = _decision_frame_hashes(
        predictions, avg_adps, player_map, pools, templates
    )
    allowed_adp_coverage_audit = audits["allowed_adp_coverage"]
    allowed_adp_coverage_sha256 = _allowed_adp_coverage_sha256(
        allowed_adp_coverage_audit
    )
    decision_audit_hashes = {
        "allowed_adp_coverage": allowed_adp_coverage_sha256,
    }
    source_fingerprint = _json_sha256(
        {
            "source_paths": source_entries,
            "decision_policies": decision_policies,
            "selected_decision_frame_sha256": table_hashes,
            "selected_decision_audit_sha256": decision_audit_hashes,
        }
    )
    center_audit = audits["center"]
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "origin_year": int(origin_year),
        "league": LEAGUE,
        "pred_vers": PRED_VERSION,
        "strict_prior": True,
        "source_fingerprint": source_fingerprint,
        "historical_data_code_sha256": file_sha256(Path(__file__).resolve()),
        "source_files": source_entries,
        "decision_policy_context": decision_policies,
        "decision_authentication_policy": (
            "selected_decision_frame_and_allowed_adp_boundary_sha256_v3"
        ),
        "candidate_limit_requested_compatibility_floor": int(candidate_limit),
        "candidate_population_policy": "all_confirmed_with_allowed_adp",
        "board_size_required": BOARD_SIZE,
        "candidate_rows": int(len(predictions)),
        "excluded_provisional_rows": int(len(audits["excluded_provisional"])),
        "provisional_resolution_policy": PROVISIONAL_RESOLUTION_POLICY_VERSION,
        "provisional_redirect_rows": int(
            audits["excluded_provisional"]["provisional_resolution"]
            .isin(
                [
                    "governed_key_redirect_allowed_market_merged",
                    "governed_key_redirect_no_allowed_market",
                ]
            )
            .sum()
        ),
        "provisional_redirect_allowed_market_rows": int(
            audits["excluded_provisional"]["provisional_resolution"]
            .eq("governed_key_redirect_allowed_market_merged")
            .sum()
        ),
        "unresolved_provisional_excluded_rows": int(
            audits["excluded_provisional"]["provisional_resolution"]
            .eq("excluded_unresolved_no_allowed_adp")
            .sum()
        ),
        "unresolved_provisional_allowed_adp_rows": 0,
        "known_provisional_adp_regression_fixture_rows_checked": int(
            audits["excluded_provisional"][
                "known_provisional_adp_regression_fixture"
            ].sum()
        ),
        "known_provisional_adp_regression_fixture_candidates_admitted": 0,
        "donor_years": list(donor_years),
        "max_donor_year": int(max(donor_years)),
        "pool_rows": int(len(pools)),
        "template_rows": int(len(templates)),
        "adp_policy": ADP_POLICY_VERSION,
        "adp_source_priority": list(ADP_SOURCE_PRIORITY),
        "adp_coverage_policy": ADP_COVERAGE_POLICY_VERSION,
        "allowed_adp_coverage_rows": int(len(allowed_adp_coverage_audit)),
        "allowed_adp_coverage_sha256": allowed_adp_coverage_sha256,
        "allowed_adp_coverage_missing_rows": 0,
        "allowed_adp_coverage_ambiguous_rows": 0,
        "allowed_adp_coverage_position_mismatch_rows": int(
            allowed_adp_coverage_audit["position_mismatch_diagnostic"].sum()
        ),
        "selected_decision_audit_sha256": decision_audit_hashes,
        "adp_noise_policy": ADP_NOISE_POLICY_VERSION,
        "adp_noise_parameters": {
            "std_fraction": ADP_STD_FRACTION,
            "std_min": ADP_STD_MIN,
            "std_max": ADP_STD_MAX,
            "bound_sigmas": ADP_BOUND_SIGMAS,
        },
        "center_policy": CENTER_POLICY_VERSION,
        "center_imputation_rows": int(len(center_audit)),
        "top240_center_imputation_rows": int(
            (center_audit.get("board_rank", pd.Series(dtype=int)) <= BOARD_SIZE).sum()
        ),
        "top240_center_imputation_gate": MAX_TOP240_CENTER_IMPUTATIONS,
        "runtime_name_policy": RUNTIME_NAME_POLICY_VERSION,
        "runtime_disambiguated_player_rows": int(len(audits["runtime_name"])),
        "pool_policy": POOL_POLICY_VERSION,
        "outcome_policy": OUTCOME_POLICY_VERSION,
        "outcome_reconciliation_atol": OUTCOME_RECONCILIATION_ATOL,
        "target_outcomes_loaded": False,
        "table_sha256": table_hashes,
    }
    assembled = HistoricalOriginData(
        origin_year=int(origin_year),
        paths=paths,
        predictions=predictions,
        avg_adps=avg_adps,
        player_map=player_map,
        template_pools=pools,
        templates=templates,
        donor_years=donor_years,
        receipt=receipt,
        source_fingerprint=source_fingerprint,
        audits=audits,
    )
    assembled.assert_decision_inputs_clean()
    return assembled


def materialize_origin_simulation_db(
    origin: HistoricalOriginData,
    destination: Path | str,
) -> Path:
    """Create a runtime-compatible disposable Simulation database."""

    origin.assert_decision_inputs_clean()
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise HistoricalDataError(f"Refusing to overwrite existing study database: {destination}")
    decision_audit = pd.DataFrame(
        [
            {
                "origin_year": origin.origin_year,
                "source_fingerprint": origin.source_fingerprint,
                "receipt_json": json.dumps(origin.receipt, sort_keys=True, default=str),
            }
        ]
    )
    with closing(sqlite3.connect(destination)) as connection:
        origin.predictions.to_sql("Final_Predictions_Resid", connection, index=False)
        origin.avg_adps.to_sql("Avg_ADPs", connection, index=False)
        origin.player_map.to_sql("Best_Ball_Weekly_Player_Map", connection, index=False)
        origin.template_pools.to_sql(
            "Best_Ball_Weekly_Template_Pools", connection, index=False
        )
        origin.templates.to_sql("Best_Ball_Weekly_Templates", connection, index=False)
        decision_audit.to_sql("Historical_Decision_Input_Audit", connection, index=False)
        connection.execute(
            "CREATE UNIQUE INDEX idx_hist_predictions_key "
            "ON Final_Predictions_Resid(player_key)"
        )
        connection.execute(
            "CREATE UNIQUE INDEX idx_hist_adp_key ON Avg_ADPs(player_key)"
        )
        connection.execute(
            "CREATE UNIQUE INDEX idx_hist_map_key "
            "ON Best_Ball_Weekly_Player_Map(player_key)"
        )
        connection.execute(
            "CREATE UNIQUE INDEX idx_hist_pool_rank "
            "ON Best_Ball_Weekly_Template_Pools(template_pool_key, match_rank)"
        )
        connection.commit()
        result = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if result != "ok":
            raise HistoricalDataError(f"Disposable database integrity check failed: {result}")
    origin.db_path = destination
    origin.assert_decision_inputs_clean()
    return destination


def _coerce_source_paths(
    source_db: HistoricalSourcePaths | Path | str | None,
) -> HistoricalSourcePaths:
    if source_db is None:
        return HistoricalSourcePaths()
    if isinstance(source_db, HistoricalSourcePaths):
        return source_db
    source = Path(source_db).expanduser().resolve()
    if source.is_dir():
        if (source / "Data" / "Databases").is_dir():
            model_root = source
        else:
            model_root = MODEL_ROOT
        return HistoricalSourcePaths(
            projection_v2_db=model_root / "Data" / "Databases" / "Projection_V2.sqlite3",
            simulation_db=model_root / "Data" / "Databases" / "Simulation.sqlite3",
        )
    if source.name.lower() == "projection_v2.sqlite3":
        return HistoricalSourcePaths(
            projection_v2_db=source,
            simulation_db=source.with_name("Simulation.sqlite3"),
        )
    if source.name.lower() == "simulation.sqlite3":
        return HistoricalSourcePaths(
            projection_v2_db=source.with_name("Projection_V2.sqlite3"),
            simulation_db=source,
        )
    raise HistoricalDataError(
        "source_db must be HistoricalSourcePaths, a model repo/database directory, "
        "Projection_V2.sqlite3, or Simulation.sqlite3"
    )


@contextmanager
def open_origin(
    source_db: HistoricalSourcePaths | Path | str | None,
    origin_year: int,
    work_dir: Path | str,
    league: str = LEAGUE,
    strict_prior: bool = True,
    smoke: bool = False,
) -> Iterator[HistoricalOriginData]:
    """Yield a disposable FootballSimulation-compatible historical origin.

    ``smoke`` intentionally does not relax data-integrity rules or truncate the
    draftable population.  The compatibility floor differs, but every
    confirmed candidate with an allowed-source ADP is retained in both modes.
    """

    paths = _coerce_source_paths(source_db)
    candidate_limit = BOARD_SIZE if smoke else DEFAULT_CANDIDATE_LIMIT
    origin = assemble_historical_origin(
        origin_year,
        paths=paths,
        league=league,
        strict_prior=strict_prior,
        candidate_limit=candidate_limit,
    )
    work_dir = Path(work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"historical_origin_{origin_year}_", dir=work_dir
    ) as temporary_directory:
        db_path = Path(temporary_directory) / "Simulation.sqlite3"
        materialize_origin_simulation_db(origin, db_path)
        yield origin


__all__ = [
    "HistoricalDataError",
    "HistoricalOriginData",
    "HistoricalSourcePaths",
    "assemble_historical_origin",
    "materialize_origin_simulation_db",
    "open_origin",
]
