WITH comp_win_team AS (
    SELECT DISTINCT
      season::INTEGER,
      daynum::INTEGER,
      wteamid::INTEGER AS teamid,
      lteamid::INTEGER AS opponent_teamid,
      wscore  AS score,
      lscore  AS opponent_score,
      --     CASE wloc WHEN 'H' THEN 1 ELSE 0 END AS home_flg,
      wloc AS location,
      numot,
      1 AS result
    FROM m_compact_2018
  ),
    comp_lose_team AS (
    SELECT DISTINCT
      season::INTEGER,
      daynum::INTEGER,
      lteamid::INTEGER AS teamid,
      wteamid::INTEGER AS opponent_teamid,
      lscore  AS score,
      wscore  AS opponent_score,
      --       CASE wloc WHEN 'H' THEN 1 ELSE 0 END AS home_flg,
      wloc AS location,
      numot,
      0 AS result
    FROM m_compact_2018
  ),
    compact_union AS (
    SELECT *
    FROM comp_win_team
    UNION
    SELECT *
    FROM comp_lose_team
  ),
compact_result AS (
SELECT DISTINCT
  t1.teamid,
  t1.firstd1season,
  t2.season,
  daynum,
  score,
  result,
  dense_rank() OVER (PARTITION BY t1.teamid, t2.season ORDER BY daynum) AS match_no,
  dense_rank() OVER (PARTITION BY t2.opponent_teamid, t2.season ORDER BY t2.daynum) AS match_no_2,
--   CASE WHEN t3.seed IS NOT NULL THEN 1 ELSE 0 END AS seed_flg,
--   CASE
--   WHEN substring(t3.seed FROM 2 FOR 2)='0'  THEN substring(t3.seed FROM 3 FOR 1)::INT
--   WHEN substring(t3.seed FROM 2 FOR 2)!='0' THEN substring(t3.seed FROM 2 FOR 2)::INT
--   ELSE 30
--   END AS seed_num,
  t6.coachname,
  location,
  numot,
  opponent_teamid AS teamid_2,
  opponent_score AS score_2,
  1-result AS result_2,
--   CASE WHEN t5.seed IS NOT NULL THEN 1 ELSE 0 END AS seed_flg_2,
--   CASE
--   WHEN substring(t5.seed FROM 2 FOR 2)='0'  THEN substring(t5.seed FROM 3 FOR 1)::INT
--   WHEN substring(t5.seed FROM 2 FOR 2)!='0' THEN substring(t5.seed FROM 2 FOR 2)::INT
--   ELSE 30
--   END AS seed_num_2,
  t7.coachname AS coachname_2
FROM m_teams AS t1
INNER JOIN compact_union AS t2
  ON t1.teamid::INTEGER = t2.teamid::INTEGER
-- LEFT OUTER JOIN m_ncaa_tourney_seeds AS t3
--   ON t1.teamid::INTEGER = t3.teamid::INTEGER AND t2.season::INTEGER = t3.season::INTEGER
-- LEFT OUTER JOIN m_teams AS t4
--   ON opponent_teamid::INTEGER = t4.teamid::INTEGER
--   LEFT OUTER JOIN m_ncaa_tourney_seeds AS t5
--     ON t2.opponent_teamid = t5.teamid AND t2.season = t5.season
  LEFT OUTER JOIN m_team_coaches AS t6
    ON t2.teamid::INTEGER = t6.teamid::INTEGER AND t2.season::INTEGER = t6.season::INTEGER AND t2.daynum::INTEGER >= firstdaynum::INTEGER AND t2.daynum::INTEGER <= lastdaynum::INTEGER
  LEFT OUTER JOIN m_team_coaches AS t7
    ON t2.opponent_teamid::INTEGER = t7.teamid::INTEGER AND t2.season::INTEGER = t7.season::INTEGER AND t2.daynum::INTEGER >= t7.firstdaynum::INTEGER AND t2.daynum::INTEGER <= t7.lastdaynum::INTEGER
)
-- SELECT count(DISTINCT teamid)
SELECT *
FROM compact_result
-- WHERE season=2018
;
  ,
--     detail_win_team AS (
--     SELECT DISTINCT
--       season,
--       daynum::INTEGER,
--       wteamid AS teamid,
--       lteamid AS opponent_teamid,
--       wscore  AS score,
--       wloc AS location,
--       numot,
--       1 AS result,
--       wfgm::INTEGER   AS fgm      ,
--       wfga   AS fga      ,
--       wfgm3  AS fgm3      ,
--       wfga3  AS fga3      ,
--       wftm   AS ftm      ,
--       wfta   AS fta      ,
--       wor    AS or_      ,
--       wdr    AS dr      ,
--       wast   AS ast      ,
--       wto    AS to_      ,
--       wstl   AS stl      ,
--       wblk   AS blk      ,
--       wpf    AS pf
--     FROM w_ncaa_detail_1985_2018
--   ),
--     detail_lose_team AS (
--     SELECT DISTINCT
--       season,
--       daynum::INTEGER,
--       lteamid AS teamid,
--       wteamid AS opponent_teamid,
--       lscore  AS score,
--       wloc AS location,
--       numot,
--       0 AS result,
--       lfgm    AS fgm       ,
--       lfga    AS fga       ,
--       lfgm3   AS fgm3       ,
--       lfga3   AS fga3       ,
--       lftm    AS ftm       ,
--       lfta    AS fta       ,
--       lor     AS or_       ,
--       ldr     AS dr       ,
--       last    AS ast       ,
--       lto     AS to_      ,
--       lstl    AS stl       ,
--       lblk    AS blk       ,
--       lpf     AS pf
--     FROM w_ncaa_detail_1985_2018
--   ),
--     detail_union AS (
--     SELECT DISTINCT *
--     FROM detail_win_team
--     UNION
--     SELECT DISTINCT *
--     FROM detail_lose_team
--   ),
--     ncaa_seeds AS (
--     SELECT DISTINCT *
--     FROM m_ncaa_tourney_seeds
--   ),
--     detail_result AS (
--     SELECT
--       t1.teamid::INTEGER,
--       firstd1season,
--       t2.season::INTEGER,
--       t2.daynum::INTEGER,
--       dense_rank() OVER (PARTITION BY t1.teamid, t2.season ORDER BY t2.daynum) AS match_no,
--       t2.score,
--       t2.location,
--       t2.numot,
--       t2.result,
--       t2.fgm,
--       t2.fga,
--       t2.fgm3,
--       t2.fga3,
--       t2.ftm,
--       t2.fta,
--       t2.or_,
--       t2.dr,
--       t2.ast,
--       t2.to_,
--       t2.stl,
--       t2.blk,
--       t2.pf,
--       t6.coachname,
--       t2.opponent_teamid::INTEGER AS teamid_2,
--       dense_rank() OVER (PARTITION BY t2.opponent_teamid, t2.season ORDER BY t2.daynum) AS match_no_2,
--       1-t2.result AS result_2,
--       t4.score        AS score_2       ,
--       t4.fgm         AS fgm_2      ,
--       t4.fga         AS fga_2      ,
--       t4.fgm3        AS fgm3_2       ,
--       t4.fga3        AS fga3_2       ,
--       t4.ftm         AS ftm_2      ,
--       t4.fta         AS fta_2      ,
--       t4.or_         AS or_2     ,
--       t4.dr          AS dr_2     ,
--       t4.ast         AS ast_2      ,
--       t4.to_         AS to_2     ,
--       t4.stl         AS stl_2      ,
--       t4.blk         AS blk_2      ,
--       t4.pf          AS pf_2     ,
--       t7.coachname AS coachname_2,
--       CASE WHEN t3.seed IS NOT NULL THEN 1 ELSE 0 END AS seed_flg,
--       CASE
--       WHEN substring(t3.seed FROM 2 FOR 2)='0'  THEN substring(t3.seed FROM 3 FOR 1)::INT
--       WHEN substring(t3.seed FROM 2 FOR 2)!='0' THEN substring(t3.seed FROM 2 FOR 2)::INT
--       ELSE 30
--       END AS seed_num,
--       CASE WHEN t5.seed IS NOT NULL THEN 1 ELSE 0 END AS seed_flg_2,
--       CASE
--       WHEN substring(t5.seed FROM 2 FOR 2)='0'  THEN substring(t5.seed FROM 3 FOR 1)::INT
--       WHEN substring(t5.seed FROM 2 FOR 2)!='0' THEN substring(t5.seed FROM 2 FOR 2)::INT
--       ELSE 30
--       END AS seed_num_2
--     FROM m_teams AS t1
--       INNER JOIN detail_union AS t2
--         ON t1.teamid = t2.teamid
--       LEFT OUTER JOIN ncaa_seeds AS t3
--         ON t1.teamid = t3.teamid AND t2.season = t3.season
--       LEFT OUTER JOIN detail_union AS t4
--         ON t2.opponent_teamid = t4.teamid AND t2.season = t4.season AND t2.daynum = t4.daynum
--       LEFT OUTER JOIN ncaa_seeds AS t5
--         ON t2.opponent_teamid = t5.teamid AND t2.season = t5.season
--       LEFT OUTER JOIN (SELECT DISTINCT * FROM m_team_coaches) AS t6 ON t2.teamid = t6.teamid AND t2.season = t6.season AND t2.daynum >= firstdaynum AND t2.daynum <= lastdaynum
--       LEFT OUTER JOIN (SELECT DISTINCT * FROM m_team_coaches) AS t7 ON t2.opponent_teamid = t7.teamid AND t2.season = t7.season AND t2.daynum >= t7.firstdaynum AND t2.daynum <= t7.lastdaynum
--   ),
  ranking AS (
  SELECT DISTINCT
    teamid,
    season,
    rankingdaynum,
    avg(ordinalrank) AS ordinalrank
  FROM m_massey_ordinals
  GROUP BY
    teamid,
    season,
    rankingdaynum
),
    season_base AS (
    SELECT *
    FROM (SELECT DISTINCT teamid FROM compact_result WHERE season::INTEGER >=2003) AS t1
      CROSS JOIN (SELECT DISTINCT season::INTEGER FROM compact_result WHERE season::INTEGER >=2003) AS t2
  ),
    day_base AS (
    SELECT *
    FROM (SELECT DISTINCT * FROM season_base WHERE season::INTEGER >=2003) AS t1
      CROSS JOIN (SELECT DISTINCT row_number() OVER (ORDER BY daynum) AS daynum_all FROM m_compact_2018 ORDER BY daynum_all LIMIT 156) AS t2
  ),
    tmp_rank AS (
    SELECT
      t1.teamid,
      t1.season::INTEGER,
      daynum_all,
      CASE WHEN rankingdaynum IS NULL THEN NULL ELSE daynum_all END AS null_no,
      rankingdaynum,
      ordinalrank
    FROM day_base AS t1
      LEFT OUTER JOIN ranking AS t2
        ON t1.teamid::INTEGER = t2.teamid::INTEGER AND t1.season::INTEGER = t2.season::INTEGER AND t1.daynum_all::INTEGER = t2.rankingdaynum::INTEGER
  ),
    rank_fill AS (
    SELECT
      t1.teamid::INTEGER,
      t1.season::INTEGER,
      t1.daynum_all::INTEGER,
      ordinalrank
    FROM (
        SELECT
          teamid,
          season,
          daynum_all,
          max(null_no)OVER (PARTITION BY teamid, season ORDER BY daynum_all ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW ) AS fill_na
        FROM tmp_rank
      ) AS t1
    INNER JOIN tmp_rank AS t2
      ON t1.teamid::INTEGER = t2.teamid::INTEGER AND t1.season::INTEGER = t2.season::INTEGER AND t1.fill_na::INTEGER = t2.rankingdaynum::INTEGER
  ),
  compact_all_join AS (
    SELECT DISTINCT
      t1.teamid,
      firstd1season,
      t1.season,
      daynum,
      match_no,
      score,
      location,
      numot,
      result,
--       fgm,
--       fga,
--       fgm3,
--       fga3,
--       ftm,
--       fta,
--       or_,
--       dr,
--       ast,
--       to_,
--       stl,
--       blk,
--       pf,
      coachname,
      teamid_2,
      match_no_2,
      result_2,
      score_2,
--       fgm_2,
--       fga_2,
--       fgm3_2,
--       fga3_2,
--       ftm_2,
--       fta_2,
--       or_2,
--       dr_2,
--       ast_2,
--       to_2,
--       stl_2,
--       blk_2,
--       pf_2,
      coachname_2
      ,
--       seed_flg,
--       seed_num,
--       seed_flg_2,
--       seed_num_2,
      CASE WHEN t2.ordinalrank IS NULL THEN -1 ELSE t2.ordinalrank END AS ordinalrank,
      CASE WHEN t3.ordinalrank IS NULL THEN -1 ELSE t3.ordinalrank END AS ordinalrank_2,
      CASE WHEN t4.coach_change_flg IS NULL THEN 0 ELSE 1 END AS coach_change_flg,
      CASE WHEN t5.coach_change_flg IS NULL THEN 0 ELSE 1 END AS coach_change_flg_2
    FROM compact_result AS t1
--     FROM detail_result AS t1
    LEFT OUTER JOIN rank_fill AS t2
      ON t1.teamid::INTEGER = t2.teamid::INTEGER AND t1.season::INTEGER = t2.season::INTEGER AND t1.daynum::INTEGER = t2.daynum_all::INTEGER
    LEFT OUTER JOIN rank_fill AS t3
      ON t1.teamid_2::INTEGER = t3.teamid::INTEGER AND t1.season::INTEGER = t3.season::INTEGER AND t1.daynum::INTEGER = t3.daynum_all::INTEGER
    LEFT OUTER JOIN coach_change AS t4
      ON t1.teamid::INTEGER = t4.teamid::INTEGER AND t1.season::INTEGER = t4.season::INTEGER
      LEFT OUTER JOIN coach_change AS t5
        ON t1.teamid_2::INTEGER = t5.teamid::INTEGER AND t1.season::INTEGER = t5.season::INTEGER
--     WHERE
--       t1.season::INTEGER =2015 AND
--           t1.teamid::INTEGER = 1140
  )
-- SELECT DISTINCT season
-- SELECT count(1)
SELECT *
FROM compact_all_join
-- FROM detail_all_join
-- ORDER BY teamid, daynum
-- LIMIT 500
;

SELECT DISTINCT season
FROM m_team_coaches
;

SELECT teamid, season, count(DISTINCT playername) AS members
FROM m_players_2010_17
WHERE teamid::INTEGER=1111
GROUP BY teamid, season
;
















WITH
    detail_win_team AS (
    SELECT DISTINCT
      season::INTEGER,
      daynum::INTEGER,
      wteamid AS teamid,
      lteamid AS opponent_teamid,
      wscore  AS score,
      wloc AS location,
      numot,
      1 AS result,
      wfgm::INTEGER   AS fgm      ,
      wfga   AS fga      ,
      wfgm3  AS fgm3      ,
      wfga3  AS fga3      ,
      wftm   AS ftm      ,
      wfta   AS fta      ,
      wor    AS or_      ,
      wdr    AS dr      ,
      wast   AS ast      ,
      wto    AS to_      ,
      wstl   AS stl      ,
      wblk   AS blk      ,
      wpf    AS pf
    FROM m_detail_2018
    WHERE season=2018
  )
  ,
    detail_lose_team AS (
    SELECT DISTINCT
      season::INTEGER,
      daynum::INTEGER,
      lteamid AS teamid,
      wteamid AS opponent_teamid,
      lscore  AS score,
      wloc AS location,
      numot,
      0 AS result,
      lfgm    AS fgm       ,
      lfga    AS fga       ,
      lfgm3   AS fgm3       ,
      lfga3   AS fga3       ,
      lftm    AS ftm       ,
      lfta    AS fta       ,
      lor     AS or_       ,
      ldr     AS dr       ,
      last    AS ast       ,
      lto     AS to_      ,
      lstl    AS stl       ,
      lblk    AS blk       ,
      lpf     AS pf
    FROM m_detail_2018
    WHERE season=2018
  ),
    detail_union AS (
    SELECT DISTINCT *
    FROM detail_win_team
    UNION
    SELECT DISTINCT *
    FROM detail_lose_team
  )
  ,
    detail_result AS (
    SELECT
      t1.teamid::INTEGER,
      firstd1season,
      t2.season::INTEGER,
      t2.daynum::INTEGER,
      dense_rank() OVER (PARTITION BY t1.teamid, t2.season ORDER BY t2.daynum) AS match_no,
      t2.score,
      t2.location,
      t2.numot,
      t2.result,
      t2.fgm,
      t2.fga,
      t2.fgm3,
      t2.fga3,
      t2.ftm,
      t2.fta,
      t2.or_,
      t2.dr,
      t2.ast,
      t2.to_,
      t2.stl,
      t2.blk,
      t2.pf,
      t6.coachname,
      t2.opponent_teamid::INTEGER AS teamid_2,
      dense_rank() OVER (PARTITION BY t2.opponent_teamid, t2.season ORDER BY t2.daynum) AS match_no_2,
      1-t2.result AS result_2,
      t4.score        AS score_2       ,
      t4.fgm         AS fgm_2      ,
      t4.fga         AS fga_2      ,
      t4.fgm3        AS fgm3_2       ,
      t4.fga3        AS fga3_2       ,
      t4.ftm         AS ftm_2      ,
      t4.fta         AS fta_2      ,
      t4.or_         AS or_2     ,
      t4.dr          AS dr_2     ,
      t4.ast         AS ast_2      ,
      t4.to_         AS to_2     ,
      t4.stl         AS stl_2      ,
      t4.blk         AS blk_2      ,
      t4.pf          AS pf_2     ,
      t7.coachname AS coachname_2
    FROM m_teams AS t1
      INNER JOIN detail_union AS t2
        ON t1.teamid::INTEGER = t2.teamid
      LEFT OUTER JOIN detail_union AS t4
        ON t2.opponent_teamid::INTEGER = t4.teamid AND t2.season::INTEGER = t4.season AND t2.daynum::INTEGER = t4.daynum
      LEFT OUTER JOIN (SELECT DISTINCT * FROM m_team_coaches_2018) AS t6 ON t2.teamid::INTEGER = t6.teamid AND t2.season::INTEGER = t6.season AND t2.daynum::INTEGER >= firstdaynum AND t2.daynum <= lastdaynum
      LEFT OUTER JOIN (SELECT DISTINCT * FROM m_team_coaches_2018) AS t7 ON t2.opponent_teamid::INTEGER = t7.teamid AND t2.season::INTEGER = t7.season AND t2.daynum >= t7.firstdaynum AND t2.daynum <= t7.lastdaynum
  ),
    ranking AS (
    SELECT DISTINCT
      teamid::INTEGER,
      season::INTEGER,
      rankingdaynum,
      avg(ordinalrank) AS ordinalrank
    FROM m_massey_ordinals
    WHERE season::INTEGER=2018
    GROUP BY
      teamid,
      season,
      rankingdaynum
  )
  ,
    season_base AS (
    SELECT *
    FROM (SELECT DISTINCT teamid FROM detail_result WHERE season::INTEGER >=2003) AS t1
      CROSS JOIN (SELECT DISTINCT season::INTEGER FROM detail_result WHERE season::INTEGER >=2003) AS t2
    WHERE season::INTEGER=2018
  ),
    day_base AS (
    SELECT *
    FROM (SELECT DISTINCT * FROM season_base WHERE season::INTEGER >=2003) AS t1
      CROSS JOIN (SELECT DISTINCT row_number() OVER (ORDER BY daynum) AS daynum_all FROM m_detail_2018 ORDER BY daynum_all LIMIT 156) AS t2
    WHERE season::INTEGER=2018
  ),
    tmp_rank AS (
    SELECT
      t1.teamid,
      t1.season::INTEGER,
      daynum_all,
      CASE WHEN rankingdaynum IS NULL THEN NULL ELSE daynum_all END AS null_no,
      rankingdaynum,
      ordinalrank
    FROM day_base AS t1
      LEFT OUTER JOIN ranking AS t2
        ON t1.teamid::INTEGER = t2.teamid::INTEGER AND t1.season::INTEGER = t2.season::INTEGER AND t1.daynum_all::INTEGER = t2.rankingdaynum::INTEGER
  ),
    rank_fill AS (
    SELECT
      t1.teamid::INTEGER,
      t1.season::INTEGER,
      t1.daynum_all::INTEGER,
      ordinalrank
    FROM (
           SELECT
             teamid,
             season,
             daynum_all,
             max(null_no)OVER (PARTITION BY teamid, season ORDER BY daynum_all ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW ) AS fill_na
           FROM tmp_rank
         ) AS t1
      INNER JOIN tmp_rank AS t2
        ON t1.teamid::INTEGER = t2.teamid::INTEGER AND t1.season::INTEGER = t2.season::INTEGER AND t1.fill_na::INTEGER = t2.rankingdaynum::INTEGER
  ),
    detail_all_join AS (
    SELECT DISTINCT
      t1.teamid,
      firstd1season,
      t1.season,
      daynum,
      match_no,
      score,
      location,
      numot,
      result,
            fgm,
            fga,
            fgm3,
            fga3,
            ftm,
            fta,
            or_,
            dr,
            ast,
            to_,
            stl,
            blk,
            pf,
      coachname,
      teamid_2,
      match_no_2,
      result_2,
      score_2,
            fgm_2,
            fga_2,
            fgm3_2,
            fga3_2,
            ftm_2,
            fta_2,
            or_2,
            dr_2,
            ast_2,
            to_2,
            stl_2,
            blk_2,
            pf_2,
      coachname_2
      ,
      --       seed_flg,
      --       seed_num,
      --       seed_flg_2,
      --       seed_num_2,
      CASE WHEN t2.ordinalrank IS NULL THEN -1 ELSE t2.ordinalrank END AS ordinalrank,
      CASE WHEN t3.ordinalrank IS NULL THEN -1 ELSE t3.ordinalrank END AS ordinalrank_2
    FROM detail_result AS t1
      --     FROM detail_result AS t1
      LEFT OUTER JOIN rank_fill AS t2
        ON t1.teamid::INTEGER = t2.teamid::INTEGER AND t1.season::INTEGER = t2.season::INTEGER AND t1.daynum::INTEGER = t2.daynum_all::INTEGER
      LEFT OUTER JOIN rank_fill AS t3
        ON t1.teamid_2::INTEGER = t3.teamid::INTEGER AND t1.season::INTEGER = t3.season::INTEGER AND t1.daynum::INTEGER = t3.daynum_all::INTEGER
    WHERE t1.season::INTEGER=2018
    --     WHERE
    --       t1.season::INTEGER =2015 AND
    --           t1.teamid::INTEGER = 1140
  )
-- SELECT DISTINCT season
-- SELECT count(1)
SELECT *
FROM detail_all_join
-- ORDER BY teamid, daynum
-- LIMIT 500
;


