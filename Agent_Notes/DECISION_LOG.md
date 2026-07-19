# Decision Log

| Date | Area | Decision | Status | Source |
| --- | --- | --- | --- | --- |
| 2026-07-06 | Project memory | Use lightweight `Agent_Notes/` plus focused app contracts, runbooks, and research studies | active | `Agent_Notes/Session_Notes/2026-07.md` |
| 2026-07-06 | Cross-repo contract | Treat `app/Simulation.sqlite3` as a generated app copy owned by the modeling repo source build | active | `Agent_Notes/CROSS_REPO_CONTEXT.md` |
| 2026-07-06 | Weekly template sampling | App should use `template_sample_prob` from `Best_Ball_Weekly_Template_Pools` when present | active | `docs/data_contracts/simulation_sqlite_app_contract.md` |
| 2026-07-06 | Template residual blend | Keep template residual strength at 0.30, with centered/scaled template residuals to avoid mean shift and preserve calibrated residual variance | active | `docs/runbooks/best_ball_ilp_validation.md` |
| 2026-07-06 | X-pruning | Use a wider max-side pick buffer to reduce deterministic-rank availability leakage | active | `research/studies/2026-07-05_best_ball_template_calibration/` |
| 2026-07-06 | Residual calibration | Use the model residual draw and weekly template together: the model residual controls calibrated season distribution, while the centered template residual adds player-type/template context | active | `docs/runbooks/best_ball_ilp_validation.md` |
| 2026-07-06 | Streamlit caching | Cache weekly template profile data by DB modified time so repeated ILP runs do not repeatedly reload the same template tables | active | `app/snake_draft_app.py` |
| 2026-07-06 | ADP fallback | Keep app ADP loading tolerant of missing ADP joins by using canonical names and model/player-map fallback context where available, then rely on audits to flag suspicious draftable misses | active | `docs/data_contracts/simulation_sqlite_app_contract.md` |
| 2026-07-06 | Query helpers | Store reusable DB Browser/SQLite inspection queries under `docs/runbooks/queries/`, not `app/` | active | `docs/runbooks/queries/README.md` |
| 2026-07-07 | League-aware templates | When copied DBs include `Best_Ball_Weekly_Templates.league`, join template pools to templates on both `template_id` and league context to avoid cross-league template matches | active | `app/zSim_Helper.py` |
| 2026-07-19 | Sequential best-ball policy | Add a Beta policy that separates construction from evaluation outcomes and removes all user/opponent picks from candidate-specific draft-room state; keep legacy ILP frozen as default/fallback | beta | `research/studies/2026-07-19_sequential_best_ball_policy/` |
| 2026-07-19 | Policy complexity gate | Keep greedy plus scarcity and a 16-player broad screen; do not add beam search or adaptive allocation because 16 versus 32 candidates had zero observed omission regret in the physical slot-six and derived mid-draft fixtures | active | `research/studies/2026-07-19_sequential_best_ball_policy/results/milestone_a_summary.json` |
| 2026-07-19 | Best-ball horizons | Label policy/template comparisons as 16-week targets and do not silently compare them with the legacy 17-week residual target | active | `docs/runbooks/best_ball_sequential_policy.md` |
| 2026-07-19 | Policy bank isolation | Draw construction and evaluation from unique, disjoint prediction columns; expose a separate evaluation seed and require rollout-path invariance when it changes | active | `research/studies/2026-07-19_sequential_best_ball_policy/results/verification.json` |
| 2026-07-19 | Policy runtime gate | Precompute conditional player survival by turn and vectorize legality; five matched physical-fixture runs produced sequential p50 3.26s versus legacy p50 7.09s | active | `research/studies/2026-07-19_sequential_best_ball_policy/results/milestone_a_summary.json` |
