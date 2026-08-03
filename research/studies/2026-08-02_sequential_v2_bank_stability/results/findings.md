# Sequential V2 Decision-Bank Stability Findings

## Outcome

The exact frozen 27-state current-V2 study completed with zero errors. All
24-room completion, legality, physical-state, pool-coverage, bank-disjointness,
decision-nesting, and rollout-path-invariance checks passed.

Growing the operational decision bank from the production-equivalent D128 to
D256 materially reduced recommendation instability on these reused
compatibility seeds. D256 passes the frozen methodological compatibility gates
and advances to fresh-seed confirmation. It is not ready for production
promotion because Sequential still fails the runtime gate, and fresh-seed plus
current-V2 24-versus-32 shortlist validation have not run.

## Recommendation Quality

| Metric | D128 control | D256 challenger |
| --- | ---: | ---: |
| Cross-fitted reference score gap, mean | 1.4001 | 0.1074 |
| Cross-fitted reference score gap, maximum | 10.8227 | 4.3025 |
| Cross-fitted gaps above 10 | 2 | 0 |
| Full-R512 regret, mean | 2.2560 | 0.9633 |
| Full-R512 regret, maximum | 13.3429 | 5.1322 |
| Exact full-R512 winner agreement | 11/27 (40.74%) | 17/27 (62.96%) |
| Mean rank correlation with R512 | 0.9161 | 0.9375 |

The artifact retains the field name `crossfit_regret`, but this quantity is a
cross-fitted out-of-half score gap: one R256 half selects the reference action
and the other scores it, then the halves swap. It may be negative in an
individual state and should not be described as literal non-negative oracle
regret. The full-R512 diagnostic regret is the conventional in-sample oracle
gap.

D128 and D256 selected the same action in 19 of 27 states. Among the eight
changes, seven improved and one worsened on the common policy-inert R512 bank:

| Slot | Seed | Round | D128 | D256 | D256 - D128 |
| ---: | ---: | ---: | --- | --- | ---: |
| 1 | 17 | 15 | Jacoby Brissett | Aaron Rodgers | +1.221 |
| 1 | 2017 | 8 | Kyle Pitts | Jaxson Dart | +4.661 |
| 1 | 2017 | 15 | Adonai Mitchell | Jalen Nailor | +0.973 |
| 6 | 2017 | 8 | Makai Lemon | Jordan Addison | +0.408 |
| 6 | 2017 | 15 | Denzel Boston | Jauan Jennings | +7.168 |
| 12 | 17 | 8 | Michael Pittman | Alec Pierce | +13.343 |
| 12 | 2017 | 1 | Saquon Barkley | De'Von Achane | +7.667 |
| 12 | 2017 | 15 | Denzel Boston | Jerry Jeudy | -0.539 |

Across all 27 states, the paired conditional point estimate was +1.2927
combined-objective points, or +0.0579% of the control reference mean. The
10,000-draw bootstrap clustered the full slot/depth trajectory by base seed.
With only three reused seed clusters, its 95% interval of [0, +2.2599] points
([0, +0.1015%]) is compatibility evidence only. It passes the frozen -0.25%
noninferiority gate, but the interval touches zero and does not demonstrate
general superiority or a realized-scoring gain.

## Runtime

Weekly-template profiles were hydrated before the timed policy calls, and the
policy-inert R512 scoring work was excluded.

| Arm | Median policy time | Versus D128 | Versus Legacy |
| --- | ---: | ---: | ---: |
| Legacy | 3.614 s | 0.277x | 1.000x |
| D128 control | 13.038 s | 1.000x | 3.607x |
| D256 challenger | 13.452 s | 1.032x | 3.722x |

D256 adds only 3.18% to the D128 median, so doubling the decision bank is not
the main latency problem. Candidate rollout work dominates. The app also
initializes a fresh Preview with 50 rooms while the helper constant, UI help,
and this study use the intended 24; that mismatch must be resolved before an
app-equivalent confirmation.

## Frozen Gates

| Gate | Result |
| --- | --- |
| Exact frozen configuration | Pass |
| All 27 states complete | Pass |
| Legality, room, bank, and path contracts | Pass |
| D256 cross-fitted gap at most 10 in every state | Pass |
| D256 mean and maximum gap no worse than D128 | Pass |
| Paired reference value noninferior at -0.25% | Pass |
| D256 median no slower than Legacy | **Fail** |

`compatibility_methodology_pass` is therefore true, while
`frozen_study_all_gates_pass` and `promotion_ready` are false.

## Next Sequence

1. Resolve the fresh-Preview 50-room default versus the tested and intended 24.
2. Profile and exact-output-optimize candidate rollouts while retaining D256 as
   the confirmation candidate.
3. Preregister an untouched 27-state confirmation with fresh base seeds (for
   example 3017/4017/5017), base-seed clustering, and the same bank-isolation
   contracts.
4. Restore the current-V2 24-versus-32 shortlist coverage gate in that run.
5. If those gates pass, run rolling-origin historical forced-pick and opponent-
   sensitivity validation before removing Preview status.

The test held draft rooms at 24. It shows that decision-bank sampling was a
material source of the observed instability; it does not establish that adding
more draft rooms improves recommendations. More rooms should be tested only as
a separate opponent-path sensitivity after the current configuration and
runtime are under control.

## Integrity Receipt

- Source database SHA-256:
  `8e89f0d24561c0989af3bf7d661d399ada0f335d59a5b40783bbffaefa1ef029`
- Database size: 59,453,440 bytes; `quick_check` and `integrity_check` both
  `ok`; freelist count 0 before and after the run.
- Model contract: DK, 350 unique canonical `player_key` rows,
  `joint_centered_template_v2_v1`, 16 weeks.
- Study script SHA-256:
  `a77127fb1be45007283337d8e9debc68126ee23d602bbd34295dbadabb0578ae`
- Focused runtime/V2 handoff tests: 17 passed.
- Independent artifact and methodology reviews found no remaining P0-P2 issue.

See `summary.json` for the aggregate receipt and `state_metrics.csv` for the
state-level evidence.
