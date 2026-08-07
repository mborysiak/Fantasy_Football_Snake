import io
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.zSim_Helper import (
    BASE_PRED_COL,
    PREDICTION_HORIZON_COL,
    FootballSimulation,
)


APP_DIR = Path(__file__).parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from snake_draft_app import (
    apply_loaded_state,
    get_prediction_options,
    get_player_data,
    load_draft_state,
    run_simulation,
    save_draft_state,
)
import snake_draft_app as snake_app


def _simulation_with_conn(conn):
    sim = FootballSimulation.__new__(FootballSimulation)
    sim.conn = conn
    sim.set_year = 2026
    sim.league = "dk"
    sim.pred_vers = "final_ensemble"
    sim.uses_v2_joint_template = False
    sim.template_resid_blend = 1.0
    return sim


def test_prediction_options_require_matching_current_weekly_artifacts(tmp_path):
    database = tmp_path / "Simulation.sqlite3"
    with sqlite3.connect(database) as conn:
        conn.execute(
            """
            CREATE TABLE Final_Predictions_Resid (
                year INTEGER,
                version TEXT,
                dataset TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO Final_Predictions_Resid VALUES (?, ?, ?)",
            [
                (2025, "nffc", "final_ensemble"),
                (2026, "nffc", "final_ensemble"),
                (2026, "dk", "final_ensemble"),
            ],
        )
        conn.execute(
            """
            CREATE TABLE Best_Ball_Weekly_Player_Map (
                year INTEGER,
                version TEXT,
                dataset TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO Best_Ball_Weekly_Player_Map VALUES (?, ?, ?)",
            [
                (2026, "nffc", "final_ensemble"),
                (2026, "dk", "final_ensemble"),
            ],
        )

    options = get_prediction_options(
        str(database),
        "final_ensemble",
    )

    assert options.to_dict("records") == [
        {"year": 2026, "version": "dk"},
        {"year": 2026, "version": "nffc"},
    ]


def _create_adp_table(conn, include_player_key):
    key_column = "player_key TEXT," if include_player_key else ""
    conn.execute(
        f"""
        CREATE TABLE Avg_ADPs (
            {key_column}
            player TEXT,
            Years_of_Experience REAL,
            avg_pick REAL,
            year INTEGER,
            league TEXT,
            std_dev REAL,
            min_pick REAL,
            max_pick REAL
        )
        """
    )


def _create_entity_adp_table(conn):
    conn.execute(
        """
        CREATE TABLE Avg_ADPs (
            player_key TEXT,
            draft_entity_key TEXT,
            player TEXT,
            pos TEXT,
            Years_of_Experience REAL,
            avg_pick REAL,
            year INTEGER,
            league TEXT,
            std_dev REAL,
            min_pick REAL,
            max_pick REAL
        )
        """
    )


def _projection_frame(player_key="key-1", player="Canonical Name"):
    return pd.DataFrame(
        {
            "player_key": [player_key],
            "player": [player],
            "pos": ["WR"],
            "team": ["AAA"],
            "model_input_avg_pick": [99.0],
            "model_input_year_exp": [2.0],
        }
    )


def test_adp_join_prefers_player_key_when_names_disagree():
    conn = sqlite3.connect(":memory:")
    _create_adp_table(conn, include_player_key=True)
    conn.execute(
        """
        INSERT INTO Avg_ADPs
        VALUES ('key-1', 'Old Display Name', 3, 17, 2026, 'dk', 2, 12, 24)
        """
    )
    sim = _simulation_with_conn(conn)

    result = sim.join_adp(_projection_frame())

    assert sim.adp_join_method == "player_key"
    assert result.loc[0, "player"] == "Canonical Name"
    assert result.loc[0, "avg_pick"] == 17


def test_canonical_adp_join_rejects_a_keyless_table():
    conn = sqlite3.connect(":memory:")
    _create_adp_table(conn, include_player_key=False)
    conn.execute(
        """
        INSERT INTO Avg_ADPs
        VALUES ('Canonical Name', 3, 17, 2026, 'dk', 2, 12, 24)
        """
    )
    sim = _simulation_with_conn(conn)

    with pytest.raises(
        ValueError,
        match=r"Canonical projections require Avg_ADPs\.player_key",
    ):
        sim.join_adp(_projection_frame())


def test_adp_legacy_name_fallback_is_explicit_and_collision_checked():
    conn = sqlite3.connect(":memory:")
    _create_adp_table(conn, include_player_key=False)
    conn.execute(
        """
        INSERT INTO Avg_ADPs
        VALUES ('Amon Ra St Brown', 3, 21, 2026, 'dk', 2, 16, 26)
        """
    )
    sim = _simulation_with_conn(conn)

    result = sim.join_adp(
        _projection_frame(player="Amon-Ra St. Brown").drop(
            columns=["player_key"]
        )
    )

    assert sim.adp_join_method == "legacy_normalized_name"
    assert result.loc[0, "avg_pick"] == 21

    duplicate_projection_names = pd.concat(
        [
            _projection_frame("key-1", "Amon-Ra St. Brown").drop(
                columns=["player_key"]
            ),
            _projection_frame("key-2", "Amon Ra St Brown").drop(
                columns=["player_key"]
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="duplicate identities"):
        sim.join_adp(duplicate_projection_names)


@pytest.mark.parametrize(
    ("rows", "error_match"),
    [
        (
            [
                (
                    "key-1",
                    "Canonical Name",
                    3,
                    17,
                    2026,
                    "dk",
                    2,
                    12,
                    24,
                ),
                (
                    None,
                    "Unkeyed ADP Row",
                    2,
                    50,
                    2026,
                    "dk",
                    5,
                    40,
                    60,
                ),
            ],
            "blank identities",
        ),
        (
            [
                (
                    "key-1",
                    "Canonical Name",
                    3,
                    17,
                    2026,
                    "dk",
                    2,
                    12,
                    24,
                ),
                (
                    "key-1",
                    "Duplicate Alias",
                    3,
                    18,
                    2026,
                    "dk",
                    2,
                    13,
                    25,
                ),
            ],
            "duplicate identities",
        ),
    ],
)
def test_keyed_adp_join_rejects_incomplete_or_duplicate_slice(
    rows,
    error_match,
):
    conn = sqlite3.connect(":memory:")
    _create_adp_table(conn, include_player_key=True)
    conn.executemany(
        "INSERT INTO Avg_ADPs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    sim = _simulation_with_conn(conn)

    with pytest.raises(ValueError, match=error_match):
        sim.join_adp(_projection_frame())


def test_keyed_alias_adp_flows_to_runtime_and_app_display():
    conn = sqlite3.connect(":memory:")
    _create_adp_table(conn, include_player_key=True)
    conn.executemany(
        "INSERT INTO Avg_ADPs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "tet-key",
                "Tet Mcmillan",
                1,
                43.669216,
                2026,
                "dk",
                6.5,
                35.0,
                53.0,
            ),
            (
                "amon-key",
                "Amon Ra St Brown",
                5,
                7.185248,
                2026,
                "dk",
                1.25,
                4.0,
                11.0,
            ),
        ],
    )
    projection = pd.DataFrame(
        {
            "player_key": ["tet-key", "amon-key"],
            "player": ["Tetairoa McMillan", "Amon-Ra St. Brown"],
            "pos": ["WR", "WR"],
            "team": ["CAR", "DET"],
            "model_input_avg_pick": [39.56, 7.05],
            "model_input_year_exp": [1.0, 5.0],
            "pred_fp_per_game": [16.0, 19.0],
            "pred_p10": [12.0, 15.0],
            "pred_p90": [20.0, 23.0],
        }
    )
    sim = _simulation_with_conn(conn)

    runtime = sim.join_adp(projection)
    sim.player_data = runtime
    displayed = get_player_data(sim).set_index("PlayerKey")

    assert sim.adp_join_method == "player_key"
    assert runtime.set_index("player_key").loc["tet-key", "avg_pick"] == pytest.approx(
        43.669216
    )
    assert runtime.set_index("player_key").loc["amon-key", "avg_pick"] == pytest.approx(
        7.185248
    )
    assert displayed.loc["tet-key", "Player"] == "Tetairoa McMillan"
    assert displayed.loc["amon-key", "Player"] == "Amon-Ra St. Brown"
    assert displayed.loc["tet-key", "ADP"] == pytest.approx(43.7)
    assert displayed.loc["amon-key", "ADP"] == pytest.approx(7.2)


def test_keyed_adp_ignores_unkeyed_non_offensive_draft_entities():
    conn = sqlite3.connect(":memory:")
    _create_entity_adp_table(conn)
    conn.executemany(
        "INSERT INTO Avg_ADPs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "key-1",
                "player:key-1",
                "Canonical Name",
                "WR",
                3,
                17,
                2026,
                "nffc",
                2,
                12,
                24,
            ),
            (
                None,
                "team-unit:NYJ:TK",
                "New York Jets Kicker",
                "TK",
                None,
                250,
                2026,
                "nffc",
                15,
                220,
                280,
            ),
            (
                None,
                "team-unit:NYJ:TDSP",
                "New York Jets Defense",
                "TDSP",
                None,
                260,
                2026,
                "nffc",
                15,
                230,
                290,
            ),
        ],
    )
    sim = _simulation_with_conn(conn)
    sim.league = "nffc"

    result = sim.join_adp(_projection_frame())

    assert sim.adp_join_method == "player_key"
    assert result.loc[0, "avg_pick"] == 17


def test_keyed_adp_rejects_an_unkeyed_offensive_entity_with_position():
    conn = sqlite3.connect(":memory:")
    _create_entity_adp_table(conn)
    conn.executemany(
        "INSERT INTO Avg_ADPs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "key-1",
                "player:key-1",
                "Canonical Name",
                "WR",
                3,
                17,
                2026,
                "dk",
                2,
                12,
                24,
            ),
            (
                None,
                "player:missing-key",
                "Unkeyed Running Back",
                "RB",
                2,
                50,
                2026,
                "dk",
                5,
                40,
                60,
            ),
        ],
    )
    sim = _simulation_with_conn(conn)

    with pytest.raises(ValueError, match="blank identities"):
        sim.join_adp(_projection_frame())


def test_keyed_adp_join_rejects_an_ungoverned_default_pick():
    conn = sqlite3.connect(":memory:")
    _create_adp_table(conn, include_player_key=True)
    conn.execute(
        """
        INSERT INTO Avg_ADPs
        VALUES ('other-key', 'Other Player', 3, 17, 2026, 'dk', 2, 12, 24)
        """
    )
    sim = _simulation_with_conn(conn)
    projection = _projection_frame()
    projection["model_input_avg_pick"] = np.nan

    with pytest.raises(ValueError, match="lack both current ADP"):
        sim.join_adp(projection)


def _create_projection_and_map_tables(conn):
    residual_columns = ",\n".join(
        f"pred_resid_{suffix} REAL"
        for suffix in (
            "5",
            "10",
            "25",
            "75",
            "90",
            "95",
            "5_ny",
            "10_ny",
            "25_ny",
            "75_ny",
            "90_ny",
            "95_ny",
        )
    )
    conn.execute(
        f"""
        CREATE TABLE Final_Predictions_Resid (
            player_key TEXT,
            player TEXT,
            pos TEXT,
            pred_fp_per_game REAL,
            pred_fp_per_game_ny REAL,
            {residual_columns},
            year INTEGER,
            dataset TEXT,
            version TEXT
        )
        """
    )
    values = [
        "key-1",
        "Projection Display",
        "WR",
        12.0,
        13.0,
        *([0.0] * 12),
        2026,
        "final_ensemble",
        "dk",
    ]
    conn.execute(
        f"INSERT INTO Final_Predictions_Resid VALUES "
        f"({','.join(['?'] * len(values))})",
        values,
    )
    conn.execute(
        """
        CREATE TABLE Best_Ball_Weekly_Player_Map (
            player_key TEXT,
            player TEXT,
            pos TEXT,
            team TEXT,
            avg_pick REAL,
            year_exp REAL,
            year INTEGER,
            dataset TEXT,
            version TEXT,
            template_pool_key TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO Best_Ball_Weekly_Player_Map
        VALUES (
            'key-1', 'Map Display Alias', 'WR', 'AAA', 30, 2,
            2026, 'final_ensemble', 'dk', 'pool-1'
        )
        """
    )


def test_projection_player_map_alignment_uses_player_key_not_display_name():
    conn = sqlite3.connect(":memory:")
    _create_projection_and_map_tables(conn)
    sim = _simulation_with_conn(conn)

    result = sim.get_model_predictions()

    assert sim.player_map_join_method == "player_key"
    assert result.loc[0, "player"] == "Projection Display"
    assert result.loc[0, "team"] == "AAA"


def _create_template_tables(conn):
    conn.execute(
        """
        CREATE TABLE Best_Ball_Weekly_Templates (
            template_id INTEGER,
            league TEXT,
            active_ppg_resid REAL,
            week_1 REAL,
            week_2 REAL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO Best_Ball_Weekly_Templates
        VALUES (1, 'dk', 0, 1.0, 0.5)
        """
    )
    conn.execute(
        """
        CREATE TABLE Best_Ball_Weekly_Template_Pools (
            template_pool_key TEXT,
            template_id INTEGER,
            pool_version TEXT,
            template_sample_prob REAL,
            match_rank INTEGER
        )
        """
    )
    conn.execute(
        """
        INSERT INTO Best_Ball_Weekly_Template_Pools
        VALUES ('pool-1', 1, 'dk', 1.0, 1)
        """
    )
    conn.execute(
        """
        CREATE TABLE Best_Ball_Weekly_Player_Map (
            player_key TEXT,
            player TEXT,
            template_pool_key TEXT,
            year INTEGER,
            version TEXT,
            dataset TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO Best_Ball_Weekly_Player_Map
        VALUES (
            'key-1', 'Old Template Display', 'pool-1',
            2026, 'dk', 'final_ensemble'
        )
        """
    )


def test_template_cache_and_sampling_use_player_key_across_name_changes():
    conn = sqlite3.connect(":memory:")
    _create_template_tables(conn)
    cache = FootballSimulation.read_weekly_template_profile_cache(
        conn,
        2026,
        "dk",
        "final_ensemble",
    )
    assert cache[-1] == "player_key"
    sim = _simulation_with_conn(conn)
    sim.weekly_template_profiles = None
    sim.weekly_template_week_cols = None
    sim.weekly_template_cum_probs = None
    sim.weekly_template_active_ppg_resids = None
    sim.weekly_template_centered_active_ppg_resids = None
    sim.weekly_template_active_ppg_resid_sds = None
    sim.weekly_template_identity_column = None
    sim.weekly_template_tensor_cache = {}
    sim.uses_v2_joint_template = True

    predictions = pd.DataFrame(
        {
            "player_key": ["key-1"],
            "player": ["Renamed Display"],
            "pos": ["WR"],
            "team": ["AAA"],
            0: [12.0],
            BASE_PRED_COL: [12.0],
            PREDICTION_HORIZON_COL: ["pred_fp_per_game"],
        }
    )

    scores = sim.sample_template_weekly_score_bank(
        predictions,
        num_scenarios=2,
        num_weeks=2,
        seed=7,
    )

    assert sim.weekly_template_identity_column == "player_key"
    assert set(sim.weekly_template_profiles) == {"key-1"}
    assert np.array_equal(scores, np.array([[[12.0, 6.0]]] * 2))
    assert (("key-1",), 2) in sim.weekly_template_tensor_cache


def _runtime_simulation(player_count=80):
    sim = FootballSimulation.__new__(FootballSimulation)
    sim.num_rounds = 20
    sim.num_teams = 12
    sim.position_ranges = {"RB": (5, 7)}
    sim.player_data = pd.DataFrame(
        {
            "player_key": [f"key-{idx}" for idx in range(player_count)],
            "player": [f"Player {idx}" for idx in range(player_count)],
            "pos": ["RB"] * player_count,
            "adp_min_pick": np.arange(player_count) + 1,
            "adp_max_pick": np.arange(player_count) + 20,
        }
    )
    return sim


def test_selection_coverage_fails_closed_and_ilp_pruning_retains_selected_key():
    sim = _runtime_simulation()

    selected, drafted = sim.validate_selection_coverage(
        ["key-79"],
        ["key-0"],
    )
    assert selected == ["key-79"]
    assert drafted == ["key-0"]
    with pytest.raises(ValueError, match="absent from the active player population"):
        sim.validate_selection_coverage(["missing-key"], [])

    ppg = sim.player_data[["player_key", "player", "pos"]].copy()
    ppg[0] = np.linspace(30, 1, len(ppg))
    adp = sim.player_data[["player_key", "player", "pos"]].copy()
    adp[0] = np.arange(len(adp)) + 1

    filtered, _, filtered_adp = sim.filter_best_ball_ilp_pool(
        ppg,
        None,
        adp,
        {"key-79"},
    )

    assert "key-79" in set(filtered.player_key)
    assert np.array_equal(filtered.player_key, filtered_adp.player_key)
    assert len(filtered) <= 57


def _app_player_data():
    return pd.DataFrame(
        {
            "PlayerKey": ["key-1", "key-2"],
            "Player": ["New Display", "Other Player"],
            "Pos": ["WR", "RB"],
            "ADP": [10.0, 20.0],
            "PredPPG": [15.0, 12.0],
            "MyTeam": [False, False],
            "OtherTeam": [False, False],
        }
    )


def _draft_settings(league):
    return {
        "year": 2026,
        "league": league,
        "num_teams": 12,
        "my_pick_position": 1,
        "num_rounds": 20,
        "scoring_mode": "best_ball_policy",
        "weekly_score_mode": "template",
        "pos_require": {"QB": 3, "RB": 6, "WR": 8, "TE": 3},
        "num_iters": 24,
    }


def _csv_buffer(frame):
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer


def test_saved_csv_round_trips_league_and_old_csv_defaults_to_dk():
    selected = _app_player_data()
    selected.loc[0, "MyTeam"] = True
    draft_data, settings_data = save_draft_state(
        selected,
        _draft_settings("nffc"),
    )
    assert settings_data.loc[0, "League"] == "nffc"

    combined = pd.concat([draft_data, settings_data], ignore_index=True)
    _, loaded_settings, error = load_draft_state(_csv_buffer(combined))
    assert error is None
    assert loaded_settings["League"] == "nffc"

    legacy_combined = combined.drop(columns=["League"])
    _, legacy_settings, legacy_error = load_draft_state(
        _csv_buffer(legacy_combined)
    )
    assert legacy_error is None
    assert legacy_settings["League"] == "dk"

    blank_combined = combined.copy()
    blank_combined.loc[blank_combined.Type == "Settings", "League"] = ""
    _, blank_settings, blank_error = load_draft_state(
        _csv_buffer(blank_combined)
    )
    assert blank_error is None
    assert blank_settings["League"] == "dk"


def test_csv_upload_applies_saved_league_and_legacy_upload_restores_dk():
    app = AppTest.from_file(
        str(APP_DIR / "snake_draft_app.py"),
        default_timeout=30,
    )
    app.run()
    nffc_csv = (
        "Type,Team,League,Year,NumTeams,MyPickPosition,NumRounds,"
        "ScoringMode,WeeklyScoreMode,QB,RB,WR,TE,NumIters\n"
        "Settings,,nffc,2026,12,3,20,best_ball_policy,template,3,6,8,3,24\n"
    ).encode("utf-8")
    app.file_uploader[0].set_value(
        ("nffc.csv", nffc_csv, "text/csv")
    ).run(timeout=30)
    assert len(app.exception) == 0
    assert app.selectbox[0].value == "nffc"

    legacy_csv = (
        "Type,Team,Year,NumTeams,MyPickPosition,NumRounds,"
        "ScoringMode,WeeklyScoreMode,QB,RB,WR,TE,NumIters\n"
        "Settings,,2026,12,3,20,best_ball_policy,template,3,6,8,3,24\n"
    ).encode("utf-8")
    app.file_uploader[0].set_value(
        ("legacy.csv", legacy_csv, "text/csv")
    ).run(timeout=30)
    assert len(app.exception) == 0
    assert app.selectbox[0].value == "dk"


def test_uploaded_player_selections_survive_the_next_widget_rerun():
    app = AppTest.from_file(
        str(APP_DIR / "snake_draft_app.py"),
        default_timeout=30,
    )
    app.run()
    selected = app.dataframe[0].value.copy()
    selected.loc[selected.index[0], "MyTeam"] = True
    selected.loc[selected.index[1], "OtherTeam"] = True
    expected_my_key = selected.iloc[0].PlayerKey
    expected_other_key = selected.iloc[1].PlayerKey
    draft_data, settings_data = save_draft_state(
        selected,
        _draft_settings("dk"),
    )
    payload = pd.concat(
        [draft_data, settings_data],
        ignore_index=True,
    ).to_csv(index=False).encode("utf-8")

    app.file_uploader[0].set_value(
        ("selected-draft.csv", payload, "text/csv")
    ).run(timeout=30)
    loaded_grid = app.dataframe[0].value
    assert set(loaded_grid.loc[loaded_grid.MyTeam, "PlayerKey"]) == {
        expected_my_key
    }
    assert set(loaded_grid.loc[loaded_grid.OtherTeam, "PlayerKey"]) == {
        expected_other_key
    }

    # Any widget interaction causes the same rerun as a data-editor checkbox.
    app.text_input[0].set_value("force-rerun").run(timeout=30)
    rerun_grid = app.dataframe[0].value
    assert len(app.exception) == 0
    assert set(rerun_grid.loc[rerun_grid.MyTeam, "PlayerKey"]) == {
        expected_my_key
    }
    assert set(rerun_grid.loc[rerun_grid.OtherTeam, "PlayerKey"]) == {
        expected_other_key
    }


def test_saved_state_and_simulation_use_player_key(monkeypatch):
    loaded = pd.DataFrame(
        {
            "PlayerKey": ["key-1"],
            "Player": ["Old Display"],
            "Team": ["MyTeam"],
        }
    )
    applied = apply_loaded_state(_app_player_data(), loaded)
    assert applied.loc[applied.PlayerKey == "key-1", "MyTeam"].item()

    settings = _draft_settings("dk")
    settings["scoring_mode"] = "best_ball_ilp"
    settings["num_iters"] = 1
    draft_data, _ = save_draft_state(applied, settings)
    assert draft_data.loc[0, "PlayerKey"] == "key-1"

    class RecordingSimulation:
        pass

    recording_sim = RecordingSimulation()
    captured = {}

    def fake_isolated(sim, to_add, to_drop, **kwargs):
        captured.update({"to_add": to_add, "to_drop": to_drop, **kwargs})
        return pd.DataFrame()

    monkeypatch.setattr(snake_app, "run_isolated_simulation", fake_isolated)
    run_simulation(
        recording_sim,
        applied,
        num_iters=1,
        scoring_mode="best_ball_ilp",
    )
    assert captured["to_add"] == ["key-1"]

    with pytest.raises(ValueError, match="absent from the active player population"):
        apply_loaded_state(
            _app_player_data(),
            pd.DataFrame(
                {
                    "PlayerKey": ["missing-key"],
                    "Player": ["Old Display"],
                    "Team": ["MyTeam"],
                }
            ),
        )
