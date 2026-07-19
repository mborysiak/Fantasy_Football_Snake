WITH base AS (
    SELECT p.template_pool_key,
           p.template_distance,
           p.match_rank,
           t.template_id,
           t.player,
           t.season,
           t.avg_proj_points,
           t.year_exp,
		   t.season_points,
           t.active_games,
           t.profile_total,
           t.week_1, t.week_2, t.week_3, t.week_4,
           t.week_5, t.week_6, t.week_7, t.week_8,
           t.week_9, t.week_10, t.week_11, t.week_12,
           t.week_13, t.week_14, t.week_15, t.week_16
    FROM Best_Ball_Weekly_Template_Pools p
    JOIN Best_Ball_Weekly_Templates t
      ON p.template_id = t.template_id
    JOIN Best_Ball_Weekly_Player_Map m
      ON p.template_pool_key = m.template_pool_key
    WHERE m.player = 'Drake Maye'
),
weeks AS (
    SELECT template_id, week_1 AS mult FROM base UNION ALL
    SELECT template_id, week_2 FROM base UNION ALL
    SELECT template_id, week_3 FROM base UNION ALL
    SELECT template_id, week_4 FROM base UNION ALL
    SELECT template_id, week_5 FROM base UNION ALL
    SELECT template_id, week_6 FROM base UNION ALL
    SELECT template_id, week_7 FROM base UNION ALL
    SELECT template_id, week_8 FROM base UNION ALL
    SELECT template_id, week_9 FROM base UNION ALL
    SELECT template_id, week_10 FROM base UNION ALL
    SELECT template_id, week_11 FROM base UNION ALL
    SELECT template_id, week_12 FROM base UNION ALL
    SELECT template_id, week_13 FROM base UNION ALL
    SELECT template_id, week_14 FROM base UNION ALL
    SELECT template_id, week_15 FROM base UNION ALL
    SELECT template_id, week_16 FROM base
),
ranked AS (
    SELECT template_id,
           mult,
           ROW_NUMBER() OVER (PARTITION BY template_id ORDER BY mult) AS rn
    FROM weeks
),
pct AS (
    SELECT template_id,
           MAX(CASE WHEN rn = 2 THEN mult END) AS p10_mult,
           MAX(CASE WHEN rn = 15 THEN mult END) AS p90_mult
    FROM ranked
    GROUP BY template_id
)
SELECT b.player,
       b.season,
       ROUND(b.avg_proj_points, 1) AS proj_pts,
       b.year_exp,
       b.active_games,
	   b.season_points,
       ROUND(b.profile_total / 16.0, 3) AS full_season_avg,
       ROUND((b.week_1 + b.week_2 + b.week_3 + b.week_4) / 4.0, 3) AS q1_avg,
       ROUND((b.week_5 + b.week_6 + b.week_7 + b.week_8) / 4.0, 3) AS q2_avg,
       ROUND((b.week_9 + b.week_10 + b.week_11 + b.week_12) / 4.0, 3) AS q3_avg,
       ROUND((b.week_13 + b.week_14 + b.week_15 + b.week_16) / 4.0, 3) AS q4_avg,
       ROUND(pct.p10_mult, 3) AS p10_mult,
       ROUND(pct.p90_mult, 3) AS p90_mult,
       ROUND(b.template_distance, 4) AS template_distance,
       b.match_rank
FROM base b
JOIN pct
  ON b.template_id = pct.template_id
ORDER BY b.match_rank;