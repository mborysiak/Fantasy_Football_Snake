#%%
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from zSim_Helper import FootballSimulation
import io
import os
from datetime import datetime

# Configuration
db_name = 'Simulation.sqlite3'
pred_vers = 'final_ensemble'

#-----------------
# Helper Functions
#-----------------

def get_db_path(filename):
    from pathlib import Path
    return Path(__file__).parents[0] / filename

def get_conn(filename):
    conn = sqlite3.connect(get_db_path(filename))
    return conn

@st.cache_data(show_spinner=False)
def load_cached_weekly_template_profiles(filename, year, league, pred_vers, db_mtime):
    conn = get_conn(filename)
    try:
        return FootballSimulation.read_weekly_template_profile_cache(
            conn,
            year,
            league,
            pred_vers,
        )
    finally:
        conn.close()

def hydrate_weekly_template_profiles(sim, filename, year, league, pred_vers):
    db_mtime = os.path.getmtime(get_db_path(filename))
    cache = load_cached_weekly_template_profiles(
        filename,
        year,
        league,
        pred_vers,
        db_mtime,
    )
    return sim.set_weekly_template_profile_cache(*cache)

def get_prediction_options(filename, pred_vers):
    """Return available year/league combinations from residual predictions."""
    conn = get_conn(filename)
    options = pd.read_sql_query(f'''
        SELECT year, version
        FROM Final_Predictions_Resid
        WHERE dataset='{pred_vers}'
        GROUP BY year, version
        ORDER BY year DESC, version
    ''', conn)
    conn.close()

    if len(options) == 0:
        raise ValueError("No Final_Predictions_Resid rows found in the app database.")

    return options

def init_sim(
    filename,
    pred_vers,
    year,
    league,
    num_teams,
    num_rounds,
    my_pick_position,
    pos_require,
    position_ranges=None,
    use_stack_bonus=False,
    stack_bonus_pct=0.25,
    stack_pair_cap=12.0,
    stack_team_cap=18.0,
):
    conn = get_conn(filename)
    sim = FootballSimulation(
        conn=conn, 
        set_year=year,
        pos_require_start=pos_require, 
        num_teams=num_teams,
        num_rounds=num_rounds,
        my_pick_position=my_pick_position,
        pred_vers=pred_vers, 
        league=league, 
        use_ownership=0,
        position_ranges=position_ranges,
        use_stack_bonus=use_stack_bonus,
        stack_bonus_pct=stack_bonus_pct,
        stack_pair_cap=stack_pair_cap,
        stack_team_cap=stack_team_cap,
    )
    return sim

def get_player_data(sim):
    """Get player data formatted for the app"""
    player_data = sim.player_data.reset_index()
    
    # Select and rename relevant columns
    player_data = player_data[[
        'player', 'pos', 'team', 'years_of_experience', 'pred_fp_per_game', 'pred_p10', 'pred_p90',
        'avg_pick', 'adp_std_dev', 'adp_min_pick', 'adp_max_pick'
    ]]
    
    # Rename columns for display
    player_data = player_data.rename(columns={
        'player': 'Player',
        'pos': 'Pos', 
        'team': 'NFLTeam',
        'avg_pick': 'ADP',
        'pred_fp_per_game': 'PredPPG',
        'pred_p10': 'PredP10',
        'pred_p90': 'PredP90',
        'years_of_experience': 'Years_Exp',
        'adp_std_dev': 'ADP_StdDev',
        'adp_min_pick': 'ADP_Min',
        'adp_max_pick': 'ADP_Max',
    })
    
    # Round values for display
    player_data['PredPPG'] = player_data['PredPPG'].round(1)
    player_data['PredP10'] = player_data['PredP10'].round(1)
    player_data['PredP90'] = player_data['PredP90'].round(1)
    player_data['ADP'] = player_data['ADP'].round(1)
    player_data['ADP_StdDev'] = player_data['ADP_StdDev'].round(1)
    player_data['ADP_Min'] = player_data['ADP_Min'].round(1)
    player_data['ADP_Max'] = player_data['ADP_Max'].round(1)

    # Add selection columns
    player_data['MyTeam'] = False
    player_data['OtherTeam'] = False
    
    # Sort by ADP (best players first)
    player_data = player_data.sort_values(by='ADP')
    
    # Reorder columns
    player_data = player_data[[
        'Player', 'Pos', 'NFLTeam', 'Years_Exp', 'MyTeam', 'OtherTeam', 'ADP',
        'PredPPG', 'PredP10', 'PredP90', 'ADP_StdDev', 'ADP_Min', 'ADP_Max'
    ]]
    
    return player_data

def create_interactive_grid(data, key_suffix=""):
    """Create the interactive data editor for player selection"""
    selected = st.data_editor(
        data,
        key=f"player_grid_{key_suffix}",
        column_config={
            "MyTeam": st.column_config.CheckboxColumn(
                "My Team",
                help="Check to add player to your team",
                default=False,
            ),
            "OtherTeam": st.column_config.CheckboxColumn(
                "Other Team",
                help="Check to exclude player (drafted by others)",
                default=False,
            ),
            "ADP": st.column_config.NumberColumn(
                "ADP",
                help="Average Draft Position",
                format="%.1f"
            ),
            "PredPPG": st.column_config.NumberColumn(
                "Pred PPG",
                help="Predicted Points Per Game",
                format="%.1f"
            ),
            "PredP10": st.column_config.NumberColumn(
                "P10",
                help="10th percentile predicted points per game",
                format="%.1f"
            ),
            "PredP90": st.column_config.NumberColumn(
                "P90",
                help="90th percentile predicted points per game",
                format="%.1f"
            ),

        },
        use_container_width=True,
        disabled=["Player", "Pos", "NFLTeam", "ADP", "PredPPG", "PredP10", "PredP90"],
        hide_index=True,
        height=500
    )
    return selected

def run_simulation(
    sim,
    selected_data,
    num_iters,
    scoring_mode,
    current_pick_ev=False,
    ev_shortlist_size=8,
    weekly_score_mode='residual',
    parallel_workers=1,
):
    """Run the snake draft simulation"""
    
    # Get selected players
    my_team = selected_data[selected_data['MyTeam'] == True]['Player'].tolist()
    other_teams = selected_data[selected_data['OtherTeam'] == True]['Player'].tolist()
    
    # Run simulation
    results = sim.run_sim(
        to_add=my_team, 
        to_drop=other_teams, 
        num_iters=num_iters, 
        num_avg_pts=3, 
        scoring_mode=scoring_mode,
        current_pick_ev=current_pick_ev,
        ev_shortlist_size=ev_shortlist_size,
        weekly_score_mode=weekly_score_mode,
        parallel_workers=parallel_workers,
    )
    
    # Keep all the round-specific columns
    return results

def render_timing_summary(results):
    """Display timing diagnostics attached by the simulation helper."""
    timings = getattr(results, 'attrs', {}).get('timings')
    if not timings:
        return

    sections = timings.get('sections', {})
    total_sec = sections.get('total', sum(sections.values()))
    requested_iters = max(int(timings.get('requested_iters', 1)), 1)
    success_trials = int(timings.get('success_trials', 0))
    model = timings.get('model', {})

    section_labels = {
        'prediction_adp_generation': 'Prediction + ADP samples',
        'template_profile_load': 'Template profile load',
        'pool_filter': 'Pool filter',
        'model_build': 'ILP model build',
        'parallel_base_sims': 'Parallel base simulations',
        'parallel_fallback': 'Parallel fallback overhead',
        'weekly_score_sampling': 'Weekly score sampling',
        'opponent_draft_availability': 'Opponent draft availability',
        'ilp_solve': 'ILP solve',
        'result_accounting': 'Result accounting',
        'current_pick_ev': 'Current pick EV solves',
        'final_results': 'Final result table',
        'iteration_total': 'Iteration total',
    }

    timing_rows = []
    for key, label in section_labels.items():
        sec = float(sections.get(key, 0))
        if sec <= 0:
            continue
        timing_rows.append({
            'Section': label,
            'Total Sec': sec,
            'Ms / Iter': (sec / requested_iters) * 1000,
            'Pct Total': (sec / total_sec) * 100 if total_sec > 0 else 0,
        })

    timing_df = pd.DataFrame(timing_rows)
    if len(timing_df) > 0:
        timing_df = timing_df.sort_values('Total Sec', ascending=False)

    with st.expander("Simulation timing", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total seconds", f"{total_sec:.1f}")
        col2.metric("Successful sims", f"{success_trials}/{requested_iters}")
        col3.metric("ILP vars", f"{int(model.get('num_vars', 0)):,}")
        col4.metric("Binary vars", f"{int(model.get('binary_vars', 0)):,}")

        st.caption(
            f"Mode: {timings.get('mode')} | Weekly scores: {timings.get('weekly_score_mode')} | "
            f"Workers: {timings.get('parallel_workers', 1)} | "
            f"Status counts: {timings.get('status_counts', {})} | "
            f"Exceptions: {timings.get('failed_exception_count', 0)}"
        )
        if model.get('full_x_count', 0):
            st.caption(
                f"Player-round vars: {int(model.get('x_count', 0)):,}/"
                f"{int(model.get('full_x_count', 0)):,} "
                f"(pruned {int(model.get('x_pruned_count', 0)):,}, "
                f"buffer +/-{int(model.get('x_pick_buffer', 0))} picks)"
            )
        if model.get('start_var_count', 0):
            st.caption(f"Weekly start vars: {int(model.get('start_var_count', 0)):,}")
        if model.get('stack_bonus_pct', 0) > 0:
            st.caption(
                f"Stack bonus: {model.get('stack_bonus_pct', 0) * 100:.0f}% "
                f"(pairs {int(model.get('stack_pair_count', 0)):,}, "
                f"capped score vars {int(model.get('stack_score_count', 0)):,}, "
                f"pair cap {model.get('stack_pair_cap', 0):.1f}, "
                f"QB/team cap {model.get('stack_team_cap', 0):.1f})"
            )
        if len(timing_df) > 0:
            st.dataframe(
                timing_df,
                column_config={
                    'Total Sec': st.column_config.NumberColumn('Total Sec', format='%.3f'),
                    'Ms / Iter': st.column_config.NumberColumn('Ms / Iter', format='%.1f'),
                    'Pct Total': st.column_config.ProgressColumn(
                        'Pct Total',
                        format='%.1f%%',
                        min_value=0,
                        max_value=100,
                    ),
                },
                use_container_width=True,
                hide_index=True,
            )

def get_round_recommendations(results, sim, my_team_count, max_rounds=3):
    """Get top recommendations for the current and next few rounds"""
    
    # Calculate which rounds to show
    adjusted_picks = sim.calculate_adjusted_picks(my_team_count)
    
    if len(adjusted_picks) == 0:
        return None, []
    
    # Determine the rounds we care about
    current_round_num = my_team_count + 1
    rounds_to_show = []
    
    for i in range(min(max_rounds, len(adjusted_picks))):
        round_num = current_round_num + i
        rounds_to_show.append(round_num)
    
    # Create round-specific recommendations
    round_data = {}
    
    for round_num in rounds_to_show:
        count_col = f'Round{round_num}Count'
        available_col = f'Round{round_num}Available'
        
        # Check if the required columns exist
        if all(col in results.columns for col in [count_col, available_col]) and 'player' in results.columns:
            # Filter for players who were available for this round
            round_results = results[results[available_col] > 0].copy()
            
            if len(round_results) > 0:
                # Calculate percentage selected when available for this round
                round_results[f'Round{round_num}PctAvailable'] = (
                    round_results[count_col] / round_results[available_col] * 100
                ).round(1)
                
                display_cols = ['player', count_col, available_col, f'Round{round_num}PctAvailable']
                ev_cols = ['CurrentPickEV', 'CurrentPickEVVsBest', 'CurrentPickEVSamples']
                include_ev = round_num == current_round_num and all(col in round_results.columns for col in ev_cols)
                if include_ev:
                    display_cols.extend(ev_cols)
                    top_picks = (
                        round_results[round_results.CurrentPickEV.notna()]
                        .sort_values('CurrentPickEV', ascending=False)
                        .head(10)[display_cols]
                        .copy()
                    )
                else:
                    # Sort by actual selection count for this round (not percentage) and take top 10
                    top_picks = round_results.nlargest(10, count_col)[display_cols].copy()
                
                # Rename columns for display
                top_picks = top_picks.rename(columns={
                    'player': 'Player',
                    count_col: 'Selected',
                    available_col: 'Available', 
                    f'Round{round_num}PctAvailable': 'Pct Selected'
                })
                if include_ev:
                    top_picks = top_picks.rename(columns={
                        'CurrentPickEV': 'EV',
                        'CurrentPickEVVsBest': 'EV vs Best',
                        'CurrentPickEVSamples': 'EV Samples',
                    })
                
                round_data[round_num] = top_picks
    
    return adjusted_picks, round_data

def create_my_team_display(selected_data, pos_require, position_ranges=None):
    """Create display of current team with position requirements"""
    my_team = selected_data[selected_data['MyTeam'] == True]
    
    # Count positions
    pos_counts = my_team['Pos'].value_counts() if len(my_team) > 0 else {}
    
    team_display = []
    for pos, required in pos_require.items():
        if pos == 'FLEX':
            continue  # Skip FLEX for now
        current_count = pos_counts.get(pos, 0)
        if position_ranges and pos in position_ranges:
            min_count, max_count = position_ranges[pos]
            label = f"{pos} ({current_count}/{min_count}-{max_count})"
            required_slots = max_count
        else:
            label = f"{pos} ({current_count}/{required})"
            required_slots = required
        
        # Add header row for each position
        team_display.append([label, ""])
        
        # Add filled positions
        pos_players = my_team[my_team['Pos'] == pos]['Player'].tolist()
        for i, player in enumerate(pos_players):
            team_display.append(["", player])
        
        # Add empty slots
        for i in range(max(0, required_slots - current_count)):
            team_display.append(["", "-- Empty --"])
    
    return pd.DataFrame(team_display, columns=['Position', 'Player'])

def save_draft_state(selected_data, settings):
    """Save current draft state to CSV format"""
    # Get only the players that have selections
    my_team = selected_data[selected_data['MyTeam'] == True][['Player', 'Pos', 'ADP', 'PredPPG']].copy()
    other_team = selected_data[selected_data['OtherTeam'] == True][['Player', 'Pos', 'ADP', 'PredPPG']].copy()
    
    # Add team designation
    my_team['Team'] = 'MyTeam'
    other_team['Team'] = 'OtherTeam'
    
    # Combine the data
    draft_data = pd.concat([my_team, other_team], ignore_index=True)
    position_ranges = settings.get('position_ranges')
    if position_ranges is None:
        position_ranges = {
            pos: (count, count)
            for pos, count in settings['pos_require'].items()
        }
    
    # Add settings as metadata (we'll store this in the CSV as well)
    settings_data = pd.DataFrame([{
        'Type': 'Settings',
        'Year': settings['year'],
        'League': settings['league'],
        'NumTeams': settings['num_teams'],
        'MyPickPosition': settings['my_pick_position'],
        'NumRounds': settings['num_rounds'],
        'ScoringMode': settings['scoring_mode'],
        'WeeklyScoreMode': settings.get('weekly_score_mode', 'residual'),
        'QB': settings['pos_require']['QB'],
        'RB': settings['pos_require']['RB'],
        'WR': settings['pos_require']['WR'],
        'TE': settings['pos_require']['TE'],
        'QB_Min': position_ranges['QB'][0],
        'QB_Max': position_ranges['QB'][1],
        'RB_Min': position_ranges['RB'][0],
        'RB_Max': position_ranges['RB'][1],
        'WR_Min': position_ranges['WR'][0],
        'WR_Max': position_ranges['WR'][1],
        'TE_Min': position_ranges['TE'][0],
        'TE_Max': position_ranges['TE'][1],
        'NumIters': settings['num_iters'],
        'ParallelWorkers': settings.get('parallel_workers', 1),
        'CurrentPickEV': settings.get('current_pick_ev', False),
        'EVShortlistSize': settings.get('ev_shortlist_size', 8),
        'UseStackBonus': settings.get('use_stack_bonus', False),
        'StackBonusPct': settings.get('stack_bonus_pct', 0.25) * 100,
        'StackPairCap': settings.get('stack_pair_cap', 12.0),
        'StackTeamCap': settings.get('stack_team_cap', 18.0),
        'SavedDate': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }])
    
    return draft_data, settings_data

def load_draft_state(uploaded_file):
    """Load draft state from uploaded CSV file"""
    try:
        # Read the CSV
        df = pd.read_csv(uploaded_file)
        
        # Check if this is a valid draft state file
        if 'Team' not in df.columns:
            return None, None, "Invalid file format. Please upload a valid draft state CSV."
        
        # Extract settings if they exist (for backward compatibility)
        settings_data = None
        if 'Type' in df.columns:
            settings_rows = df[df['Type'] == 'Settings']
            if len(settings_rows) > 0:
                settings_data = settings_rows.iloc[0]
                # Remove settings rows from main data
                df = df[df['Type'] != 'Settings']
        
        return df, settings_data, None
        
    except Exception as e:
        return None, None, f"Error loading file: {str(e)}"

def apply_loaded_state(player_data, loaded_data):
    """Apply loaded draft state to current player data"""
    if loaded_data is None or len(loaded_data) == 0:
        return player_data
    
    # Reset all selections
    player_data['MyTeam'] = False
    player_data['OtherTeam'] = False
    
    # Apply loaded selections
    for _, row in loaded_data.iterrows():
        player_name = row['Player']
        team = row['Team']
        
        # Find the player in current data
        player_idx = player_data[player_data['Player'] == player_name].index
        
        if len(player_idx) > 0:
            if team == 'MyTeam':
                player_data.loc[player_idx, 'MyTeam'] = True
            elif team == 'OtherTeam':
                player_data.loc[player_idx, 'OtherTeam'] = True
    
    return player_data

def get_setting_int(settings, key, default):
    """Read an integer setting from a loaded CSV row with backward-compatible defaults."""
    if settings is None or key not in settings:
        return default

    value = settings.get(key, default)
    if pd.isna(value):
        return default

    return int(value)

def get_setting_bool(settings, key, default):
    """Read a boolean setting from a loaded CSV row with backward-compatible defaults."""
    if settings is None or key not in settings:
        return default

    value = settings.get(key, default)
    if pd.isna(value):
        return default
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'y')

    return bool(value)

def get_setting_float(settings, key, default):
    """Read a float setting from a loaded CSV row with backward-compatible defaults."""
    if settings is None or key not in settings:
        return default

    value = settings.get(key, default)
    if pd.isna(value):
        return default

    return float(value)

#------------------
# App Components
#------------------

def render_save_button(selected_data, settings, placeholder):
    """Render the Save & Download button in the sidebar placeholder"""
    
    # Count current selections to determine if button should be enabled
    my_team_count = len(selected_data[selected_data['MyTeam'] == True])
    other_team_count = len(selected_data[selected_data['OtherTeam'] == True])
    has_selections = my_team_count > 0 or other_team_count > 0
    
    if has_selections:
        # Generate the CSV data
        draft_data, settings_data = save_draft_state(selected_data, settings)
        
        if len(draft_data) > 0:
            # Combine draft data and settings
            combined_data = pd.concat([draft_data, settings_data], ignore_index=True)
            
            # Create CSV data
            csv_buffer = io.StringIO()
            combined_data.to_csv(csv_buffer, index=False)
            csv_string = csv_buffer.getvalue()
            
            # Generate filename with custom name and timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = st.session_state.save_file_name
            if file_name.strip():
                # Clean the filename (remove invalid characters)
                clean_name = "".join(c for c in file_name.strip() if c.isalnum() or c in (' ', '-', '_')).strip()
                clean_name = clean_name.replace(' ', '_')
                filename = f"{clean_name}_{timestamp}.csv"
            else:
                filename = f"draft_state_{timestamp}.csv"
            
            # Direct download button
            placeholder.download_button(
                label="💾 Save & Download Draft",
                data=csv_string,
                file_name=filename,
                mime="text/csv",
                help="Save and download current draft selections as CSV",
                key="save_draft_button"
            )
        else:
            placeholder.button("💾 Save & Download Draft", disabled=True, help="No players selected to save")
    else:
        placeholder.button("💾 Save & Download Draft", disabled=True, help="No players selected to save")

def sidebar_controls(prediction_options):
    """Create sidebar controls"""
    st.sidebar.header("Draft Settings")
    
    # Get loaded settings if available
    loaded_settings = st.session_state.get('loaded_settings_data', None)
    
    # Debug info
    if loaded_settings is not None:
        st.sidebar.write(f"🔧 Using loaded settings")
    
    # Determine default values from loaded settings or use defaults
    default_ilp_ranges = FootballSimulation.default_best_ball_position_ranges()
    if loaded_settings is not None:
        default_year = get_setting_int(loaded_settings, 'Year', int(prediction_options.year.max()))
        default_league = str(loaded_settings.get('League', 'dk')).lower()
        default_num_teams = get_setting_int(loaded_settings, 'NumTeams', 12)
        default_my_pick = get_setting_int(loaded_settings, 'MyPickPosition', 1)
        default_qb = get_setting_int(loaded_settings, 'QB', 3)
        default_rb = get_setting_int(loaded_settings, 'RB', 6)
        default_wr = get_setting_int(loaded_settings, 'WR', 8)
        default_te = get_setting_int(loaded_settings, 'TE', 3)
        default_num_rounds = get_setting_int(loaded_settings, 'NumRounds', 20)
        default_num_iters = get_setting_int(loaded_settings, 'NumIters', 200)
        default_parallel_workers = get_setting_int(loaded_settings, 'ParallelWorkers', min(4, max(1, (os.cpu_count() or 2) - 1)))
        default_current_pick_ev = get_setting_bool(loaded_settings, 'CurrentPickEV', False)
        default_ev_shortlist_size = get_setting_int(loaded_settings, 'EVShortlistSize', 8)
        default_use_stack_bonus = get_setting_bool(loaded_settings, 'UseStackBonus', True)
        default_stack_bonus_pct = get_setting_float(loaded_settings, 'StackBonusPct', 25.0)
        default_stack_pair_cap = get_setting_float(loaded_settings, 'StackPairCap', 12.0)
        default_stack_team_cap = get_setting_float(loaded_settings, 'StackTeamCap', 18.0)
        default_scoring_mode = str(loaded_settings.get('ScoringMode', 'best_ball_ilp'))
        default_weekly_score_mode = str(loaded_settings.get('WeeklyScoreMode', 'template'))
        if default_scoring_mode == 'best_ball_marginal':
            default_scoring_mode = 'best_ball_lookahead'
        default_position_ranges = {
            pos: (
                get_setting_int(loaded_settings, f'{pos}_Min', min_count),
                get_setting_int(loaded_settings, f'{pos}_Max', max_count),
            )
            for pos, (min_count, max_count) in default_ilp_ranges.items()
        }
    else:
        default_year = int(prediction_options.year.max())
        default_league = 'dk'
        default_num_teams = 12
        default_my_pick = 1
        default_qb = 3
        default_rb = 6
        default_wr = 8
        default_te = 3
        default_num_rounds = 20
        default_num_iters = 50
        default_parallel_workers = min(4, max(1, (os.cpu_count() or 2) - 1))
        default_current_pick_ev = False
        default_ev_shortlist_size = 8
        default_use_stack_bonus = True
        default_stack_bonus_pct = 25.0
        default_stack_pair_cap = 12.0
        default_stack_team_cap = 18.0
        default_scoring_mode = 'best_ball_ilp'
        default_weekly_score_mode = 'template'
        default_position_ranges = default_ilp_ranges
    
    year_options = sorted(prediction_options.year.unique(), reverse=True)
    year_index = year_options.index(default_year) if default_year in year_options else 0
    year = st.sidebar.selectbox(
        'Prediction Year',
        options=year_options,
        index=year_index,
        help="Select the prediction year from Final_Predictions_Resid"
    )

    # League selection
    league_options = sorted(prediction_options[prediction_options.year == year].version.unique())
    league_index = league_options.index(default_league) if default_league in league_options else 0
    league = st.sidebar.selectbox(
        'League Type',
        options=league_options,
        index=league_index,
        help="Select the league type for predictions and ADP data"
    )
    
    # League settings
    num_teams = st.sidebar.number_input(
        'Number of Teams', 
        min_value=8, 
        max_value=16, 
        value=default_num_teams, 
        step=1
    )
    
    my_pick_position = st.sidebar.number_input(
        'My Pick Position', 
        min_value=1, 
        max_value=num_teams, 
        value=min(default_my_pick, num_teams),  # Ensure it doesn't exceed num_teams
        step=1
    )
    
    st.sidebar.header("Simulation Settings")

    num_iters = st.sidebar.number_input(
        'Number of Simulations', 
        min_value=10, 
        max_value=500, 
        value=default_num_iters, 
        step=10,
        help='Best-ball ILP is more expensive than lookahead; 50-100 simulations is a practical starting range.'
    )

    max_workers = max(1, min(12, os.cpu_count() or 1))
    parallel_workers = st.sidebar.number_input(
        'Parallel Workers',
        min_value=1,
        max_value=max_workers,
        value=min(max(default_parallel_workers, 1), max_workers),
        step=1,
        help='Best-ball ILP simulations can run across processes. Use 1 to disable parallel execution.',
    )

    scoring_options = {
        'Best-ball ILP': 'best_ball_ilp',
        'Best-ball lookahead': 'best_ball_lookahead',
        'Total roster points': 'total_points',
    }
    scoring_labels = list(scoring_options.keys())
    scoring_values = list(scoring_options.values())
    scoring_index = scoring_values.index(default_scoring_mode) if default_scoring_mode in scoring_values else 0
    scoring_label = st.sidebar.selectbox(
        'Optimization Scoring',
        options=scoring_labels,
        index=scoring_index,
        help=(
            'Best-ball ILP solves the draft and weekly best-ball lineup simultaneously. '
            'Best-ball lookahead is a faster heuristic that compares current picks by greedy full-draft completion. '
            'Total roster points preserves the older full-roster sum objective.'
        )
    )
    scoring_mode = scoring_options[scoring_label]
    weekly_score_mode = 'residual'
    current_pick_ev = False
    ev_shortlist_size = default_ev_shortlist_size
    use_stack_bonus = False
    stack_bonus_pct = min(max(default_stack_bonus_pct, 0.0), 50.0) / 100.0
    stack_pair_cap = default_stack_pair_cap
    stack_team_cap = default_stack_team_cap
    if scoring_mode == 'best_ball_ilp':
        weekly_score_options = {
            'Weekly templates': 'template',
            'Residual weeks': 'residual',
        }
        weekly_score_labels = list(weekly_score_options.keys())
        weekly_score_values = list(weekly_score_options.values())
        weekly_score_index = (
            weekly_score_values.index(default_weekly_score_mode)
            if default_weekly_score_mode in weekly_score_values
            else 0
        )
        weekly_score_label = st.sidebar.selectbox(
            'ILP Weekly Scores',
            options=weekly_score_labels,
            index=weekly_score_index,
            help=(
                'Weekly templates draw a season PPG from residuals, then apply a sampled '
                'historical 16-week best-ball profile, weighted by template similarity, '
                'with a centered, variance-preserving 0.30 template residual blend. '
                'Residual weeks preserves the prior independent weekly sampling behavior.'
            ),
        )
        weekly_score_mode = weekly_score_options[weekly_score_label]

        use_stack_bonus = st.sidebar.checkbox(
            'QB Stack Bonus',
            value=default_use_stack_bonus,
            help='Adds a capped roster-level bonus for same-team QB plus WR/TE stacks.',
        )
        if use_stack_bonus:
            stack_bonus_pct = (
                st.sidebar.slider(
                    'Stack Bonus %',
                    min_value=0.0,
                    max_value=50.0,
                    value=min(max(default_stack_bonus_pct, 0.0), 50.0),
                    step=5.0,
                    help='Pair bonus is this percent of combined QB plus WR/TE projected PPG, capped per pair and per QB/team.',
                )
                / 100.0
            )

        current_pick_ev = st.sidebar.checkbox(
            'Current Pick EV Ranking',
            value=default_current_pick_ev,
            help='After the normal ILP run, force each shortlisted current-pick candidate and rank by expected final roster score.',
        )
        if current_pick_ev:
            ev_shortlist_size = st.sidebar.number_input(
                'EV Shortlist Size',
                min_value=3,
                max_value=10,
                value=min(max(default_ev_shortlist_size, 3), 10),
                step=1,
            )

    st.sidebar.header("Roster Construction")
    if scoring_mode == 'best_ball_ilp':
        position_ranges = {}
        for pos in ['QB', 'RB', 'WR', 'TE']:
            default_min, default_max = default_position_ranges[pos]
            col_min, col_max = st.sidebar.columns(2)
            with col_min:
                min_count = st.number_input(
                    f'{pos} Min',
                    min_value=0,
                    max_value=15,
                    value=min(default_min, default_max),
                    step=1,
                    key=f'{pos.lower()}_min',
                )
            with col_max:
                max_count = st.number_input(
                    f'{pos} Max',
                    min_value=min_count,
                    max_value=15,
                    value=max(default_min, default_max),
                    step=1,
                    key=f'{pos.lower()}_max',
                )
            position_ranges[pos] = (min_count, max_count)

        min_total = sum(min_count for min_count, _ in position_ranges.values())
        max_total = sum(max_count for _, max_count in position_ranges.values())
        default_total = min(max(default_num_rounds, min_total), max_total)
        num_rounds = st.sidebar.number_input(
            'Total Roster Size',
            min_value=min_total,
            max_value=max_total,
            value=default_total,
            step=1,
            help='Best-ball ILP selects this many total players while staying inside the position ranges.',
        )
        st.sidebar.caption(f"Range capacity: {min_total}-{max_total} players.")
        pos_require = {
            pos: max_count
            for pos, (_, max_count) in position_ranges.items()
        }
    else:
        position_ranges = None
        pos_require = {}
        pos_require['QB'] = st.sidebar.number_input('QB', min_value=1, max_value=5, value=default_qb, step=1)
        pos_require['RB'] = st.sidebar.number_input('RB', min_value=2, max_value=15, value=default_rb, step=1)
        pos_require['WR'] = st.sidebar.number_input('WR', min_value=3, max_value=15, value=default_wr, step=1)
        pos_require['TE'] = st.sidebar.number_input('TE', min_value=1, max_value=5, value=default_te, step=1)
        num_rounds = sum(pos_require.values())
        st.sidebar.write(f"**Total Rounds:** {num_rounds}")
    
    st.sidebar.header("Save/Load Draft")
    
    # File upload for loading draft state
    uploaded_file = st.sidebar.file_uploader(
        "Load Draft State",
        type=['csv'],
        help="Upload a previously saved draft state CSV file",
        key=f"file_uploader_{st.session_state.file_uploader_key}"
    )
    
    if uploaded_file is not None:
        loaded_data, loaded_settings, error = load_draft_state(uploaded_file)
        if error:
            st.sidebar.error(error)
        else:
            st.session_state.loaded_draft_data = loaded_data
            st.session_state.loaded_settings_data = loaded_settings
            st.session_state.data_loaded_applied = False  # Reset flag to allow reapplication
            st.session_state.settings_applied_to_ui = False  # Reset settings application flag
            st.sidebar.success("Draft state loaded successfully!")
            
            # Show loaded info
            if loaded_settings is not None:
                st.sidebar.write(f"**Loaded:** {loaded_settings.get('SavedDate', 'Unknown date')}")
                st.sidebar.write(f"**League:** {loaded_settings.get('League', 'Unknown').upper()}")
                st.sidebar.write(f"**Draft Position:** {loaded_settings.get('MyPickPosition', 'Unknown')}")
                st.sidebar.write(f"**Team Size:** {loaded_settings.get('NumTeams', 'Unknown')}")
            
            st.sidebar.write(f"**Players loaded:** {len(loaded_data) if loaded_data is not None else 0}")
    
    # Save draft state section
    st.sidebar.subheader("Save Current Draft")
    
    # File name input
    file_name = st.sidebar.text_input(
        "Draft Name (optional)",
        placeholder="e.g., my_draft, league_name",
        help="Enter a custom name for your draft file"
    )
    
    # Update session state with file name
    st.session_state.save_file_name = file_name
    
    # Create a placeholder for the save button that will be populated later
    save_button_placeholder = st.sidebar.empty()
    
    # Clear selections section  
    st.sidebar.subheader("Clear Draft")
    
    # Clear all selections button with confirmation
    if st.sidebar.button("🗑️ Clear All Selections", help="Remove all player selections"):
        st.session_state.confirm_clear = True
    
    # Handle confirmation dialog
    if 'confirm_clear' in st.session_state and st.session_state.confirm_clear:
        with st.sidebar:
            st.warning("⚠️ Are you sure you want to clear all selections?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, Clear All", key="confirm_clear_yes"):
                    st.session_state.clear_selections = True
                    st.session_state.confirm_clear = False
                    # Clear any loaded data to ensure it doesn't interfere
                    st.session_state.loaded_draft_data = None
                    st.session_state.loaded_settings_data = None
                    st.session_state.data_loaded_applied = False
                    st.session_state.settings_applied_to_ui = False
                    # Reset file uploader by changing its key
                    st.session_state.file_uploader_key += 1
            with col2:
                if st.button("❌ Cancel", key="confirm_clear_no"):
                    st.session_state.confirm_clear = False
    
    # Clear loaded settings after they've been used for default values
    # This prevents them from overriding user changes on subsequent runs
    if loaded_settings is not None:
        # Only clear if this is not the first run after loading (to allow the values to be applied)
        if st.session_state.get('settings_applied_to_ui', False):
            st.session_state.loaded_settings_data = None
            st.session_state.settings_applied_to_ui = False
        else:
            st.session_state.settings_applied_to_ui = True
    
    return {
        'year': year,
        'league': league,
        'num_teams': num_teams,
        'num_rounds': num_rounds,
        'my_pick_position': my_pick_position,
        'pos_require': pos_require,
        'position_ranges': position_ranges,
        'num_iters': num_iters,
        'parallel_workers': parallel_workers,
        'scoring_mode': scoring_mode,
        'weekly_score_mode': weekly_score_mode,
        'use_stack_bonus': use_stack_bonus,
        'stack_bonus_pct': stack_bonus_pct,
        'stack_pair_cap': stack_pair_cap,
        'stack_team_cap': stack_team_cap,
        'current_pick_ev': current_pick_ev,
        'ev_shortlist_size': ev_shortlist_size,
        'save_button_placeholder': save_button_placeholder
    }

#======================
# Main App
#======================

def main():
    st.set_page_config(page_title="Snake Draft Optimizer", layout="wide")
    
    # Initialize all session state variables at the start
    if 'loaded_draft_data' not in st.session_state:
        st.session_state.loaded_draft_data = None
    if 'loaded_settings_data' not in st.session_state:
        st.session_state.loaded_settings_data = None
    if 'confirm_clear' not in st.session_state:
        st.session_state.confirm_clear = False
    if 'file_uploader_key' not in st.session_state:
        st.session_state.file_uploader_key = 0
    if 'save_file_name' not in st.session_state:
        st.session_state.save_file_name = ""
    if 'clear_selections' not in st.session_state:
        st.session_state.clear_selections = False
    if 'grid_key_counter' not in st.session_state:
        st.session_state.grid_key_counter = 0
    if 'league_changed' not in st.session_state:
        st.session_state.league_changed = False
    if 'data_loaded_applied' not in st.session_state:
        st.session_state.data_loaded_applied = False
    if 'settings_applied_to_ui' not in st.session_state:
        st.session_state.settings_applied_to_ui = False
    
    # Custom CSS to reduce padding
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("🐍 Snake Draft Optimizer")
    
    prediction_options = get_prediction_options(db_name, pred_vers)

    # Get settings from sidebar
    settings = sidebar_controls(prediction_options)
    
    try:
        # Initialize simulation (will be fresh if refresh button was clicked)
        sim = init_sim(
            db_name, 
            pred_vers, 
            settings['year'], 
            settings['league'],
            settings['num_teams'],
            settings['num_rounds'],
            settings['my_pick_position'],
            settings['pos_require'],
            settings['position_ranges'],
            settings['use_stack_bonus'],
            settings['stack_bonus_pct'],
            settings['stack_pair_cap'],
            settings['stack_team_cap'],
        )
        
        # Reset the flag after initialization
        if st.session_state.league_changed:
            st.session_state.league_changed = False
        
        # Get player data
        player_data = get_player_data(sim)
        
        # Apply loaded draft state if available and not yet applied
        if (st.session_state.loaded_draft_data is not None and 
            not st.session_state.data_loaded_applied):
            player_data = apply_loaded_state(player_data, st.session_state.loaded_draft_data)
            # Mark as applied but don't clear the data yet
            st.session_state.data_loaded_applied = True
        
        # Handle clear selections if requested (before creating the grid)
        if st.session_state.clear_selections:
            # Reset player data by clearing selections
            player_data['MyTeam'] = False
            player_data['OtherTeam'] = False
            st.session_state.clear_selections = False
            # Force grid to reset by changing the key
            st.session_state.grid_key_counter += 1
            st.success("All selections cleared!")
        
        # Get grid key for forcing reset when needed
        grid_key = f"main_{st.session_state.grid_key_counter}"
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("1. Select Players")
            
            # Player selection grid
            selected_data = create_interactive_grid(player_data, key_suffix=grid_key)
            
            # Render the Save & Download button in the sidebar placeholder
            # This ensures we capture the latest grid state
            render_save_button(selected_data, settings, settings['save_button_placeholder'])
            
            # Get current selections for validation
            current_my_team = selected_data[selected_data['MyTeam'] == True]
            current_other_team = selected_data[selected_data['OtherTeam'] == True]
            
            # Check for conflicts
            conflicts = set(current_my_team['Player']) & set(current_other_team['Player'])
            if conflicts:
                st.error(f"Players cannot be on both your team and other teams: {', '.join(conflicts)}")
                return
            
            # Calculate draft status for simulation button
            my_team_players = selected_data[selected_data['MyTeam'] == True]
            num_selected = len(my_team_players)
            adjusted_picks = sim.calculate_adjusted_picks(num_selected)
            
            # Run simulation section
            st.header("2. Optimal Recommendations")
            
            if st.button("🚀 Run Simulation", type="primary"):
                if len(adjusted_picks) == 0:
                    st.info("All picks completed! No more recommendations needed.")
                else:
                    with st.spinner("Running simulation..."):
                        if (
                            settings['scoring_mode'] == 'best_ball_ilp'
                            and settings['weekly_score_mode'] == 'template'
                        ):
                            hydrate_weekly_template_profiles(
                                sim,
                                db_name,
                                settings['year'],
                                settings['league'],
                                pred_vers,
                            )

                        results = run_simulation(
                            sim, 
                            selected_data, 
                            settings['num_iters'], 
                            settings['scoring_mode'],
                            settings['current_pick_ev'],
                            settings['ev_shortlist_size'],
                            settings['weekly_score_mode'],
                            settings['parallel_workers'],
                        )

                    render_timing_summary(results)
                    
                    # Get round-specific recommendations
                    picks_remaining, round_recommendations = get_round_recommendations(
                        results, sim, num_selected, max_rounds=3
                    )
                    
                    if round_recommendations:
                        st.write("**Top recommendations by round** (based on optimal draft simulations)")
                        
                        # Add explanation
                        st.info(
                            "🎯 **Current Round**: Available players are compared by full draft-path outcome\n\n"
                            "📊 **Future Rounds**: Simulated from typical ADP patterns and availability"
                        )
                        
                        # Create tabs for each round
                        if len(round_recommendations) == 1:
                            # Single round - no tabs needed
                            round_num = list(round_recommendations.keys())[0]
                            pick_num = picks_remaining[0] if picks_remaining else "Unknown"
                            st.subheader(f"Round {round_num} (Pick #{pick_num})")
                            
                            round_data = round_recommendations[round_num]
                            st.dataframe(
                                round_data,
                                column_config={
                                    "Pct Selected": st.column_config.ProgressColumn(
                                        "Pct Selected",
                                        help="Percentage selected when available in this round",
                                        format="%.1f%%",
                                        min_value=0,
                                        max_value=100,
                                    ),
                                    "EV": st.column_config.NumberColumn(
                                        "EV",
                                        help="Average final roster score when this player is forced as the current pick",
                                        format="%.1f",
                                    ),
                                    "EV vs Best": st.column_config.NumberColumn(
                                        "EV vs Best",
                                        help="Expected score gap versus the best current-pick candidate",
                                        format="%.1f",
                                    ),
                                    "EV Samples": st.column_config.NumberColumn(
                                        "EV Samples",
                                        help="Number of scenario solves used for this EV estimate",
                                        format="%d",
                                    ),
                                },
                                use_container_width=True,
                                hide_index=True,
                                height=400
                            )
                        else:
                            # Multiple rounds - use tabs
                            tab_labels = []
                            for i, round_num in enumerate(round_recommendations.keys()):
                                pick_num = picks_remaining[i] if i < len(picks_remaining) else "TBD"
                                if i == 0:
                                    tab_labels.append(f"🎯 Round {round_num} (Pick #{pick_num})")
                                else:
                                    tab_labels.append(f"Round {round_num} (Pick #{pick_num})")
                            
                            tabs = st.tabs(tab_labels)
                            
                            for i, (round_num, round_data) in enumerate(round_recommendations.items()):
                                with tabs[i]:
                                    if i == 0:
                                        st.write("**Current round - compared by full draft-path outcome**")
                                    else:
                                        st.write("**Future round - based on typical ADP availability**")
                                    
                                    st.dataframe(
                                        round_data,
                                        column_config={
                                            "Pct Selected": st.column_config.ProgressColumn(
                                                "Pct Selected",
                                                help="Percentage selected when available in this round",
                                                format="%.1f%%",
                                                min_value=0,
                                                max_value=100,
                                            ),
                                            "EV": st.column_config.NumberColumn(
                                                "EV",
                                                help="Average final roster score when this player is forced as the current pick",
                                                format="%.1f",
                                            ),
                                            "EV vs Best": st.column_config.NumberColumn(
                                                "EV vs Best",
                                                help="Expected score gap versus the best current-pick candidate",
                                                format="%.1f",
                                            ),
                                            "EV Samples": st.column_config.NumberColumn(
                                                "EV Samples",
                                                help="Number of scenario solves used for this EV estimate",
                                                format="%d",
                                            ),
                                        },
                                        use_container_width=True,
                                        hide_index=True,
                                        height=350
                                    )
                    else:
                        st.warning("No recommendations available for upcoming rounds.")
        
        with col2:
            st.header("📋 My Team")
            
            # Display current team
            team_display = create_my_team_display(
                selected_data,
                settings['pos_require'],
                settings['position_ranges'],
            )
            st.dataframe(team_display, use_container_width=True, hide_index=True)
            
            # Calculate draft status based on current selections
            my_team_players = selected_data[selected_data['MyTeam'] == True]
            num_selected = len(my_team_players)
            adjusted_picks = sim.calculate_adjusted_picks(num_selected)
            
            st.subheader("📊 Draft Status")
            st.metric("Players Selected", num_selected)
            st.metric("Remaining Picks", len(adjusted_picks))
            
            if len(adjusted_picks) > 0:
                st.write(f"**Next Pick:** {adjusted_picks[0]}")
                if len(adjusted_picks) > 1:
                    st.write(f"**Future Picks:** {adjusted_picks[1:3]}..." if len(adjusted_picks) > 3 else f"**Future Picks:** {adjusted_picks[1:]}")
                
                # Show pick timing info
                total_picks = settings['num_teams'] * settings['num_rounds']
                pick_progress = adjusted_picks[0] / total_picks * 100
                st.progress(pick_progress / 100, text=f"Draft Progress: {pick_progress:.1f}%")
                
            else:
                st.success("All picks completed!")
                
            # Position requirements status
            if len(my_team_players) > 0:
                st.subheader("📋 Position Status")
                pos_status = {}
                for pos, required in settings['pos_require'].items():
                    if pos != 'FLEX':
                        current = len(my_team_players[my_team_players['Pos'] == pos])
                        pos_status[pos] = f"{current}/{required}"
                
                for pos, status in pos_status.items():
                    current, required = map(int, status.split('/'))
                    if current >= required:
                        st.success(f"{pos}: {status} ✓")
                    elif current > 0:
                        st.warning(f"{pos}: {status}")
                    else:
                        st.error(f"{pos}: {status}")
                
    except Exception as e:
        st.error(f"Error initializing simulation: {str(e)}")
        st.write("Please check your database connection and settings.")

if __name__ == '__main__':
    main()
# %%
