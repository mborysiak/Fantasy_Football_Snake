"""Focused fail-closed regressions for historical_data.

These tests are intentionally fixture-only: they do not open the production
databases or download target payloads.  The end-to-end pinned-source checks run
inside HistoricalOriginData scoring and are sealed in target_source_receipt.
"""

from __future__ import annotations

import pandas as pd
import pytest

import historical_data as subject


def test_final_roster_state_contract_preserves_prefix_and_excludes_opponents() -> None:
    roster = subject.validate_final_roster_state_contract(
        ["mine-a", "mine-b", "forced", "later"],
        to_add=["mine-a", "mine-b"],
        to_drop=["opponent-a", "opponent-b"],
        depth=2,
        action_key="forced",
        rounds=4,
    )

    assert roster == ["mine-a", "mine-b", "forced", "later"]


@pytest.mark.parametrize(
    ("roster", "message"),
    [
        (["mine-b", "mine-a", "forced", "later"], "user-pick prefix"),
        (["mine-a", "mine-b", "forced", "opponent-a"], "opponent-drafted"),
    ],
)
def test_final_roster_state_contract_rejects_state_leakage(
    roster: list[str], message: str
) -> None:
    with pytest.raises(subject.HistoricalDataError, match=message):
        subject.validate_final_roster_state_contract(
            roster,
            to_add=["mine-a", "mine-b"],
            to_drop=["opponent-a", "opponent-b"],
            depth=2,
            action_key="forced",
            rounds=4,
        )


def _identity_frames() -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    identities = pd.DataFrame(
        {
            "player_key": ["key-a", "key-b"],
            "gsis_id": ["00-001", "00-002"],
            "identity_status": ["confirmed", "confirmed"],
        }
    )
    handoff = identities[["player_key", "gsis_id"]].copy()
    return identities, handoff, {"key-a", "key-b"}


def test_exact_gsis_contract_accepts_one_to_one_mapping() -> None:
    identities, handoff, keys = _identity_frames()

    result = subject._validate_candidate_identity_mapping(
        identities, handoff, keys
    )

    assert set(result["player_key"]) == keys


@pytest.mark.parametrize("surface", ["identity", "handoff"])
def test_exact_gsis_contract_rejects_blank_ids(surface: str) -> None:
    identities, handoff, keys = _identity_frames()
    target = identities if surface == "identity" else handoff
    target.loc[target["player_key"].eq("key-a"), "gsis_id"] = ""

    with pytest.raises(subject.HistoricalDataError, match="blank GSIS"):
        subject._validate_candidate_identity_mapping(identities, handoff, keys)


def test_exact_gsis_contract_rejects_duplicate_ids() -> None:
    identities, handoff, keys = _identity_frames()
    identities.loc[identities["player_key"].eq("key-b"), "gsis_id"] = "00-001"

    with pytest.raises(subject.HistoricalDataError, match="multiple canonical"):
        subject._validate_candidate_identity_mapping(identities, handoff, keys)


def test_exact_gsis_contract_rejects_noncandidate_collision() -> None:
    identities, handoff, keys = _identity_frames()
    identities = pd.concat(
        [
            identities,
            pd.DataFrame(
                {
                    "player_key": ["key-c"],
                    "gsis_id": ["00-001"],
                    "identity_status": ["confirmed"],
                }
            ),
        ],
        ignore_index=True,
    )

    with pytest.raises(subject.HistoricalDataError, match="Full player_identity"):
        subject._validate_candidate_identity_mapping(identities, handoff, keys)


def test_exact_gsis_contract_rejects_handoff_mismatch() -> None:
    identities, handoff, keys = _identity_frames()
    handoff.loc[handoff["player_key"].eq("key-b"), "gsis_id"] = "00-999"

    with pytest.raises(subject.HistoricalDataError, match="disagree"):
        subject._validate_candidate_identity_mapping(identities, handoff, keys)


def _exact_outcome_frame(**updates: object) -> pd.DataFrame:
    row: dict[str, object] = {
        "player_key": "key-a",
        "display_name": "Player A",
        "outcome_observed": 1,
        "appeared": 1,
        "opportunity_games": 2,
        "unconditional_season_points": 12.5,
        "exact_outcome_row_present": 1,
        "exact_season_points": 12.5,
        "exact_outcome_appeared": 1,
        "exact_opportunity_games": 2,
    }
    row.update(updates)
    return pd.DataFrame([row])


def test_exact_outcome_value_contract_accepts_consistent_row() -> None:
    output, observed = subject._validate_exact_outcome_values(
        _exact_outcome_frame()
    )

    assert observed.tolist() == [True]
    assert output.loc[0, "exact_season_points"] == 12.5


@pytest.mark.parametrize("value", ["not-a-number", float("inf"), float("-inf")])
def test_exact_outcome_value_contract_rejects_nonfinite_points(value: object) -> None:
    with pytest.raises(subject.HistoricalDataError, match="numeric and finite"):
        subject._validate_exact_outcome_values(
            _exact_outcome_frame(exact_season_points=value)
        )


@pytest.mark.parametrize("value", ["not-a-number", float("inf"), float("-inf")])
def test_exact_outcome_value_contract_rejects_nonfinite_spine_points(
    value: object,
) -> None:
    with pytest.raises(subject.HistoricalDataError, match="numeric and finite"):
        subject._validate_exact_outcome_values(
            _exact_outcome_frame(unconditional_season_points=value)
        )


def test_exact_outcome_value_contract_rejects_presence_mismatch() -> None:
    with pytest.raises(subject.HistoricalDataError, match="observation flags"):
        subject._validate_exact_outcome_values(
            _exact_outcome_frame(exact_outcome_row_present=0)
        )


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"exact_outcome_appeared": 0}, "appeared disagrees"),
        ({"exact_outcome_appeared": 2}, "must be 0 or 1"),
        ({"exact_opportunity_games": 1}, "opportunity_games disagrees"),
        ({"exact_opportunity_games": -1}, "positive integers"),
        ({"exact_opportunity_games": 1.5}, "positive integers"),
        ({"appeared": 2}, "must be 0 or 1"),
        ({"opportunity_games": -1}, "nonnegative integers"),
        ({"opportunity_games": 1.5}, "nonnegative integers"),
    ],
)
def test_exact_outcome_value_contract_rejects_inconsistent_participation(
    updates: dict[str, object], message: str
) -> None:
    with pytest.raises(subject.HistoricalDataError, match=message):
        subject._validate_exact_outcome_values(_exact_outcome_frame(**updates))


def _provisional_fixture(source: str, adp: float = 999.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = subject.KNOWN_PROVISIONAL_ADP_REGRESSION_KEYS[0]
    provisional = pd.DataFrame(
        {
            "player_key": [key],
            "display_name": ["Truncated"],
            "position": ["WR"],
            "provisional_resolution": ["unresolved"],
        }
    )
    market = pd.DataFrame(
        {
            "player_key": [key],
            "market_original_player_key": [key],
            "source": [source],
            "source_table": ["fixture"],
            "adp": [adp],
        }
    )
    return provisional, market


def test_any_allowed_adp_for_unresolved_identity_fails_without_rank_exemption() -> None:
    provisional, market = _provisional_fixture(
        "fantasypros_best_ball_adp", adp=999.0
    )

    with pytest.raises(subject.HistoricalDataError, match="could enter"):
        subject._audit_provisional_board_boundary(provisional, market)


def test_generic_adp_source_cannot_admit_provisional_identity() -> None:
    provisional, market = _provisional_fixture("adp_fpros", adp=120.0)

    audit = subject._audit_provisional_board_boundary(provisional, market)

    assert audit.loc[0, "provisional_allowed_adp_rows"] == 0
    assert audit.loc[0, "provisional_resolution"] == "excluded_unresolved_no_allowed_adp"


def _coverage_market(
    *,
    original_key: str = "missing-key",
    resolved_key: str | None = None,
    source: str = "fantasypros_best_ball_adp",
    position: str = "WR",
    redirected: int = 0,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_original_player_key": [original_key],
            "player_key": [resolved_key or original_key],
            "source": [source],
            "source_table": ["fixture"],
            "position": [position],
            "adp": [42.0],
            "provisional_redirect_applied": [redirected],
        }
    )


def _coverage_universe(
    key: str = "key-a",
    position: str = "WR",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_key": [key],
            "position": [position],
            "identity_is_confirmed": [1],
        }
    )


def test_allowed_adp_coverage_rejects_key_absent_from_joined_universe() -> None:
    with pytest.raises(
        subject.HistoricalDataError,
        match="missing_resolved_target.*audit_sha256",
    ):
        subject._audit_allowed_adp_universe_coverage(
            _coverage_market(),
            _coverage_universe(),
        )


def test_allowed_adp_coverage_ignores_absent_generic_source_key() -> None:
    audit = subject._audit_allowed_adp_universe_coverage(
        _coverage_market(source="adp_fpros"),
        _coverage_universe(),
    )

    assert audit.empty


def test_allowed_adp_coverage_accepts_governed_redirect_target() -> None:
    original_key, resolved_key = next(
        iter(subject.GOVERNED_PROVISIONAL_KEY_REDIRECTS.items())
    )
    audit = subject._audit_allowed_adp_universe_coverage(
        _coverage_market(
            original_key=original_key,
            resolved_key=resolved_key,
            redirected=1,
        ),
        _coverage_universe(key=resolved_key),
    )

    assert audit.loc[0, "coverage_status"] == "covered"
    assert audit.loc[0, "resolution_semantics"] == "governed_key_redirect"


def test_allowed_adp_coverage_records_position_mismatch_diagnostic() -> None:
    audit = subject._audit_allowed_adp_universe_coverage(
        _coverage_market(original_key="key-a", position="RB"),
        _coverage_universe(key="key-a", position="WR"),
    )

    assert audit.loc[0, "coverage_status"] == "covered"
    assert audit.loc[0, "position_mismatch_diagnostic"] == 1


def test_allowed_adp_coverage_rejects_redirect_position_change() -> None:
    original_key, resolved_key = next(
        iter(subject.GOVERNED_PROVISIONAL_KEY_REDIRECTS.items())
    )

    with pytest.raises(
        subject.HistoricalDataError,
        match="redirect_position_mismatch.*audit_sha256",
    ):
        subject._audit_allowed_adp_universe_coverage(
            _coverage_market(
                original_key=original_key,
                resolved_key=resolved_key,
                position="RB",
                redirected=1,
            ),
            _coverage_universe(key=resolved_key, position="WR"),
        )


def test_allowed_adp_coverage_hash_seals_diagnostics() -> None:
    direct = subject._audit_allowed_adp_universe_coverage(
        _coverage_market(original_key="key-a", position="WR"),
        _coverage_universe(key="key-a", position="WR"),
    )
    mismatch = subject._audit_allowed_adp_universe_coverage(
        _coverage_market(original_key="key-a", position="RB"),
        _coverage_universe(key="key-a", position="WR"),
    )

    assert subject._allowed_adp_coverage_sha256(direct) != subject._allowed_adp_coverage_sha256(
        mismatch
    )


def test_known_material_weekly_regression_fixture_contract() -> None:
    observed = {
        (
            int(row["origin_year"]),
            str(row["display_name"]),
            str(row["population"]),
        ): float(row["points"])
        for row in subject.KNOWN_WEEKLY_MAPPING_REGRESSIONS
    }
    expected = {
        (2018, "Tyler Ervin", "all_played"): 6.5,
        (2019, "Robbie Chosen", "governed"): 164.5,
        (2019, "Deonte Harty", "governed"): 10.2,
        (2020, "Lamar Miller", "all_played"): 2.6,
        (2020, "Chris Herndon", "governed"): 56.4,
        (2020, "Deonte Harty", "governed"): 48.7,
    }

    assert observed == expected
    robbie = next(
        row
        for row in subject.KNOWN_WEEKLY_MAPPING_REGRESSIONS
        if row["display_name"] == "Robbie Chosen"
    )
    assert set(robbie["alternate_display_names"]) == {
        "Robby Anderson",
        "Robbie Anderson",
    }


def test_all_enumerated_provisional_regressions_are_unique() -> None:
    keys = subject.KNOWN_PROVISIONAL_ADP_REGRESSION_KEYS

    assert len(keys) == 11
    assert len(set(keys)) == len(keys)
