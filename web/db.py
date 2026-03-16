"""DuckDB query engine for Silver Parquet data."""

import os
import threading
from pathlib import Path

import duckdb
import pandas as pd

from src.upcoming_matches import load_upcoming_matches

# ── Persistent connection with MotherDuck ──
_lock = threading.Lock()
_con = None
_upcoming_cache_lock = threading.Lock()
_upcoming_cache_df = None
_upcoming_cache_meta = None
_upcoming_cache_mtime = None


def _get_persistent_con():
    """Return a persistent MotherDuck connection."""
    global _con
    if _con is not None:
        return _con
    with _lock:
        if _con is not None:
            return _con

        token = os.environ.get("MOTHERDUCK_TOKEN")
        db_name = os.environ.get("MOTHERDUCK_DB", "lolesports")

        if token:
            print(f"Connecting to MotherDuck ({db_name})...")
            # Connect to MotherDuck
            con = duckdb.connect(f"md:{db_name}?motherduck_token={token}")
        else:
            print(
                "WARNING: MOTHERDUCK_TOKEN not found. Using local in-memory DB (data may be missing)."
            )
            # Fallback to local memory if token is missing
            con = duckdb.connect(":memory:")

        _con = con
    return _con


def reload_data():
    """Force reconnect to MotherDuck."""
    global _con
    with _lock:
        if _con:
            _con.close()
        _con = None
    _get_persistent_con()


def query_df(sql: str, params: dict | None = None):
    """Execute query and return pandas DataFrame."""
    con = _get_persistent_con()

    # Replace old parquet path constants with actual table names
    sql = sql.replace("games", "games")
    sql = sql.replace("players", "players")
    sql = sql.replace("series", "series")
    sql = sql.replace("champions", "champions")

    if params:
        for k, v in params.items():
            sql = sql.replace(f"${k}", str(v))
    return con.execute(sql).fetchdf()


def query_one(sql: str, params: dict | None = None):
    """Execute query and return single row as dict."""
    df = query_df(sql, params)
    if len(df) == 0:
        return None
    return df.iloc[0].to_dict()


def _load_local_upcoming_matches_cached():
    """Load normalized upcoming matches from the local Leaguepedia payload with mtime caching."""
    global _upcoming_cache_df, _upcoming_cache_meta, _upcoming_cache_mtime

    upcoming_path = Path("data/bronze/leaguepedia/match_results.json")
    try:
        current_mtime = (
            upcoming_path.stat().st_mtime if upcoming_path.exists() else None
        )
    except OSError:
        current_mtime = None

    if _upcoming_cache_df is not None and _upcoming_cache_mtime == current_mtime:
        return _upcoming_cache_df.copy(), dict(_upcoming_cache_meta or {})

    with _upcoming_cache_lock:
        if _upcoming_cache_df is not None and _upcoming_cache_mtime == current_mtime:
            return _upcoming_cache_df.copy(), dict(_upcoming_cache_meta or {})

        upcoming_df, upcoming_meta = load_upcoming_matches(upcoming_path)
        _upcoming_cache_df = upcoming_df
        _upcoming_cache_meta = upcoming_meta
        _upcoming_cache_mtime = current_mtime
        return upcoming_df.copy(), dict(upcoming_meta)


def get_upcoming_matches(limit: int = 20, league: str | None = None):
    """Return normalized upcoming matches, preferring MotherDuck and falling back to local Leaguepedia bronze."""
    safe_league = league.replace("'", "''") if league else None
    con = _get_persistent_con()
    where_clause = f"WHERE league = '{safe_league}'" if safe_league else ""

    try:
        remote_df = con.execute(
            f"""
            SELECT
                match_id,
                match_time,
                match_date,
                league,
                event_name,
                phase_label,
                team1,
                team2,
                best_of,
                patch,
                overview_page,
                source
            FROM web_upcoming_matches
            {where_clause}
            ORDER BY match_time ASC, league, team1, team2
            LIMIT {limit}
            """
        ).fetchdf()
        if len(remote_df) > 0:
            return remote_df
    except duckdb.Error:
        pass

    local_df, _ = _load_local_upcoming_matches_cached()
    if safe_league:
        local_df = local_df.loc[local_df["league"] == safe_league].copy()
    return (
        local_df.sort_values(["match_time", "league", "team1", "team2"], kind="stable")
        .head(limit)
        .reset_index(drop=True)
    )


def get_upcoming_match_leagues():
    """Return distinct leagues available in upcoming matches."""
    con = _get_persistent_con()
    try:
        remote_df = con.execute(
            """
            SELECT league, COUNT(*) AS total_matches, MIN(match_time) AS next_match_time
            FROM web_upcoming_matches
            GROUP BY league
            ORDER BY next_match_time ASC, league
            """
        ).fetchdf()
        if len(remote_df) > 0:
            return remote_df
    except duckdb.Error:
        pass

    local_df, _ = _load_local_upcoming_matches_cached()
    if local_df.empty:
        return pd.DataFrame(columns=["league", "total_matches", "next_match_time"])
    return (
        local_df.groupby("league", as_index=False)
        .agg(total_matches=("match_id", "count"), next_match_time=("match_time", "min"))
        .sort_values(["next_match_time", "league"], kind="stable")
        .reset_index(drop=True)
    )


def get_upcoming_matches_meta():
    """Return metadata about the upcoming schedule source."""
    con = _get_persistent_con()
    try:
        remote_meta = con.execute(
            """
            SELECT fetched_at, row_count, source_path
            FROM web_upcoming_matches_meta
            LIMIT 1
            """
        ).fetchdf()
        if len(remote_meta) > 0:
            row = remote_meta.iloc[0].to_dict()
            row["source"] = "motherduck"
            return row
    except duckdb.Error:
        pass

    _, local_meta = _load_local_upcoming_matches_cached()
    return {
        "fetched_at": local_meta.get("fetched_at"),
        "row_count": int(local_meta.get("row_count", 0)),
        "source_path": local_meta.get("path"),
        "source": "local_bronze",
    }


# =============================================================
# HOME
# =============================================================


def get_recent_series(limit: int = 20):
    return query_df(
        f"""
        SELECT match_date, league, team1, team2, score, series_winner,
               series_format, tournament_phase
        FROM series
        ORDER BY match_date DESC
        LIMIT {limit}
    """
    )


def get_active_leagues():
    return query_df(
        """
        SELECT league, MAX(match_date) as last_match, COUNT(*) as total_series
        FROM series
        WHERE match_date >= CURRENT_DATE - INTERVAL '30 days'
        GROUP BY league
        ORDER BY last_match DESC
    """
    )


def search_players(term: str):
    safe = term.replace("'", "''")
    return query_df(
        f"""
        WITH ranked AS (
            SELECT playername, teamname, position, league,
                   MAX(game_date) as last_game,
                   ROW_NUMBER() OVER (
                       PARTITION BY LOWER(playername)
                       ORDER BY MAX(game_date) DESC
                   ) as rn
            FROM players
            WHERE LOWER(playername) LIKE LOWER('%{safe}%')
            GROUP BY playername, teamname, position, league
        )
        SELECT playername, teamname, position, league, last_game
        FROM ranked WHERE rn = 1
        ORDER BY last_game DESC
        LIMIT 20
    """
    )


def search_teams(term: str):
    safe = term.replace("'", "''")
    return query_df(
        f"""
        WITH ranked AS (
            SELECT teamname, league,
                   MAX(game_date) as last_game,
                   ROW_NUMBER() OVER (
                       PARTITION BY LOWER(teamname)
                       ORDER BY MAX(game_date) DESC
                   ) as rn
            FROM players
            WHERE LOWER(teamname) LIKE LOWER('%{safe}%')
            GROUP BY teamname, league
        )
        SELECT teamname, league, last_game
        FROM ranked WHERE rn = 1
        ORDER BY last_game DESC
        LIMIT 20
    """
    )


# =============================================================
# PLAYER
# =============================================================


def get_player_info(name: str):
    safe = name.replace("'", "''")
    return query_one(
        f"""
        SELECT playername, teamname, position, league
        FROM players
        WHERE LOWER(playername) = LOWER('{safe}')
        ORDER BY game_date DESC
        LIMIT 1
    """
    )


def get_player_career_stats(name: str, year: int = None, split: str = None):
    safe = name.replace("'", "''")
    conditions = [f"LOWER(playername) = LOWER('{safe}')"]
    if year:
        conditions.append(f"year = {year}")
    if split:
        split_safe = split.replace("'", "''")
        conditions.append(f"split = '{split_safe}'")
    where = " AND ".join(conditions)
    return query_one(
        f"""
        SELECT
            COUNT(*) as games,
            SUM(result) as wins,
            ROUND(100.0 * SUM(result) / COUNT(*), 1) as winrate,
            ROUND(AVG(kills), 1) as avg_kills,
            ROUND(AVG(deaths), 1) as avg_deaths,
            ROUND(AVG(assists), 1) as avg_assists,
            ROUND(AVG(CASE WHEN deaths = 0 THEN kills + assists
                       ELSE (kills + assists) * 1.0 / deaths END), 2) as kda,
            ROUND(AVG(dpm), 0) as avg_dpm,
            ROUND(AVG(cspm), 1) as avg_cspm,
            ROUND(AVG(golddiffat15), 0) as avg_gd15,
            ROUND(AVG(xpdiffat15), 0) as avg_xd15,
            ROUND(AVG(visionscore), 1) as avg_vs
        FROM players
        WHERE {where}
    """
    )


def get_player_by_year(name: str):
    safe = name.replace("'", "''")
    return query_df(
        f"""
        SELECT
            year,
            split,
            teamname,
            COUNT(*) as games,
            SUM(result) as wins,
            ROUND(100.0 * SUM(result) / COUNT(*), 1) as winrate,
            ROUND(AVG(kills), 1) as avg_k,
            ROUND(AVG(deaths), 1) as avg_d,
            ROUND(AVG(assists), 1) as avg_a,
            ROUND(AVG(dpm), 0) as avg_dpm
        FROM players
        WHERE LOWER(playername) = LOWER('{safe}')
        GROUP BY year, split, teamname
        ORDER BY year DESC, split
    """
    )


def get_player_champions(
    name: str, year: int = None, split: str = None, limit: int = 15
):
    safe = name.replace("'", "''")
    conditions = [f"LOWER(playername) = LOWER('{safe}')"]
    if year:
        conditions.append(f"year = {year}")
    if split:
        split_safe = split.replace("'", "''")
        conditions.append(f"split = '{split_safe}'")
    where = " AND ".join(conditions)
    return query_df(
        f"""
        SELECT
            champion,
            COUNT(*) as games,
            SUM(result) as wins,
            ROUND(100.0 * SUM(result) / COUNT(*), 1) as winrate,
            ROUND(AVG(kills), 1) as avg_k,
            ROUND(AVG(deaths), 1) as avg_d,
            ROUND(AVG(assists), 1) as avg_a,
            ROUND(AVG(dpm), 0) as avg_dpm
        FROM players
        WHERE {where}
        GROUP BY champion
        HAVING COUNT(*) >= 1
        ORDER BY games DESC
        LIMIT {limit}
    """
    )


def get_player_recent_games(
    name: str, year: int = None, split: str = None, limit: int = 20
):
    safe = name.replace("'", "''")
    conditions = [f"LOWER(playername) = LOWER('{safe}')"]
    if year:
        conditions.append(f"year = {year}")
    if split:
        split_safe = split.replace("'", "''")
        conditions.append(f"split = '{split_safe}'")
    where = " AND ".join(conditions)
    return query_df(
        f"""
        SELECT
            game_date, league, split, teamname, champion, position,
            kills, deaths, assists, dpm, cspm, result
        FROM players
        WHERE {where}
        ORDER BY game_date DESC, game DESC
        LIMIT {limit}
    """
    )


# =============================================================
# TEAM
# =============================================================


def get_team_info(name: str):
    safe = name.replace("'", "''")
    return query_one(
        f"""
        SELECT teamname, league
        FROM games
        WHERE LOWER(teamname) = LOWER('{safe}')
           OR LOWER(teamname) LIKE LOWER('%{safe}%')
        ORDER BY game_date DESC
        LIMIT 1
    """
    )


def get_team_roster(name: str):
    safe = name.replace("'", "''")
    return query_df(
        f"""
        WITH recent AS (
            SELECT playername, position, champion, result, game_date,
                   ROW_NUMBER() OVER (PARTITION BY playername ORDER BY game_date DESC) as rn
            FROM players
            WHERE LOWER(teamname) = LOWER('{safe}')
              AND game_date >= CURRENT_DATE - INTERVAL '90 days'
        )
        SELECT playername, position,
               COUNT(*) as games,
               SUM(result) as wins,
               ROUND(100.0 * SUM(result) / COUNT(*), 1) as winrate
        FROM players
        WHERE LOWER(teamname) = LOWER('{safe}')
          AND game_date >= CURRENT_DATE - INTERVAL '90 days'
        GROUP BY playername, position
        ORDER BY CASE position
            WHEN 'top' THEN 1 WHEN 'jng' THEN 2 WHEN 'mid' THEN 3
            WHEN 'bot' THEN 4 WHEN 'sup' THEN 5 ELSE 6 END
    """
    )


def get_team_stats_by_split(name: str):
    safe = name.replace("'", "''")
    return query_df(
        f"""
        SELECT
            year, league, split,
            COUNT(*) as games,
            SUM(result) as wins,
            ROUND(100.0 * SUM(result) / COUNT(*), 1) as winrate,
            ROUND(AVG(gamelength), 0) as avg_length
        FROM games
        WHERE LOWER(teamname) = LOWER('{safe}')
        GROUP BY year, league, split
        ORDER BY year DESC, split
    """
    )


def get_team_titles(name: str):
    safe = name.replace("'", "''")
    return query_df(
        f"""
        SELECT year, league, split, runner_up, final_score
        FROM champions
        WHERE LOWER(champion) = LOWER('{safe}')
        ORDER BY final_date DESC
    """
    )


def get_team_recent_series(name: str, limit: int = 20):
    safe = name.replace("'", "''")
    return query_df(
        f"""
        SELECT match_date, league, team1, team2, score,
               series_winner, series_format, tournament_phase
        FROM series
        WHERE LOWER(team1) LIKE LOWER('%{safe}%') OR LOWER(team2) LIKE LOWER('%{safe}%')
        ORDER BY match_date DESC
        LIMIT {limit}
    """
    )


def get_team_betting_stats(
    name: str,
    year: int = None,
    split: str = None,
    playoffs: int = None,
    league: str = None,
    h2h_opponent: str = None,
):
    """Full betting stats for a single team (overall, per-split, or H2H)."""
    safe = name.replace("'", "''")
    conditions = [f"LOWER(teamname) = LOWER('{safe}')"]
    if year:
        conditions.append(f"year = {year}")
    if league:
        league_safe = league.replace("'", "''")
        conditions.append(f"league = '{league_safe}'")
    if split:
        split_safe = split.replace("'", "''")
        if split == "N/A":
            conditions.append("(split IS NULL OR split = '')")
        else:
            conditions.append(f"split = '{split_safe}'")
    if playoffs is not None:
        conditions.append(f"COALESCE(playoffs, 0) = {playoffs}")
    if h2h_opponent:
        opp_safe = h2h_opponent.replace("'", "''")
        conditions.append(
            f"""gameid IN (
            SELECT gameid FROM games
            WHERE LOWER(teamname) = LOWER('{opp_safe}')
        )"""
        )

    where = "WHERE " + " AND ".join(conditions)

    return query_one(
        f"""
        WITH base AS (
            SELECT
                CAST(result AS DOUBLE)                                        AS result,
                CAST(gamelength AS DOUBLE) / 60.0                            AS game_minutes,
                CAST(COALESCE(firstblood,  0) AS DOUBLE)                     AS firstblood,
                CAST(COALESCE(firsttower,  0) AS DOUBLE)                     AS firsttower,
                CAST(COALESCE(firstdragon, 0) AS DOUBLE)                     AS firstdragon,
                COALESCE(TRY_CAST(firstherald AS DOUBLE), 0.0)               AS firstherald,
                CAST(COALESCE(firstbaron,  0) AS DOUBLE)                     AS firstbaron,
                CAST(COALESCE(teamkills,   0) AS DOUBLE)                     AS teamkills,
                CAST(COALESCE(teamkills, 0) + COALESCE(teamdeaths, 0) AS DOUBLE) AS total_kills,
                CAST(COALESCE(towers,  0) + COALESCE(opp_towers,   0) AS DOUBLE) AS total_towers,
                CAST(COALESCE(dragons, 0) + COALESCE(opp_dragons,  0) AS DOUBLE) AS total_dragons,
                CAST(COALESCE(barons,  0) + COALESCE(opp_barons,   0) AS DOUBLE) AS total_nashors,
                CAST(COALESCE(inhibitors, 0) + COALESCE(opp_inhibitors, 0) AS DOUBLE) AS total_inhibitors
            FROM games
            {where}
        )
        SELECT
            COUNT(*)                                                              AS games,
            ROUND(100.0 * SUM(result)       / NULLIF(COUNT(*), 0), 1)                       AS winrate,
            ROUND(100.0 * AVG(firstblood),  1)                                   AS first_blood_pct,
            ROUND(100.0 * AVG(firsttower),  1)                                   AS first_tower_pct,
            ROUND(100.0 * AVG(firstdragon), 1)                                   AS first_dragon_pct,
            ROUND(100.0 * AVG(firstherald), 1)                                   AS first_herald_pct,
            ROUND(100.0 * AVG(firstbaron),  1)                                   AS first_baron_pct,
            ROUND(AVG(total_kills),         1)                                   AS avg_total_kills,
            ROUND(AVG(total_towers),        1)                                   AS avg_total_towers,
            ROUND(AVG(total_dragons),       1)                                   AS avg_total_dragons,
            ROUND(AVG(total_nashors),       1)                                   AS avg_total_nashors,
            ROUND(AVG(total_inhibitors),    1)                                   AS avg_total_inhibitors,
            ROUND(AVG(game_minutes),        1)                                   AS avg_game_minutes,
            ROUND(100.0 * SUM(CASE WHEN total_kills  > 25  THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) AS kills_over_25_pct,
            ROUND(100.0 * SUM(CASE WHEN total_towers > 10  THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) AS towers_over_10_pct,
            ROUND(100.0 * SUM(CASE WHEN total_nashors >= 2 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) AS nashors_over_1_5_pct,
            CAST(MAX(teamkills) AS INTEGER)                                      AS most_kills_game
        FROM base
    """
    )


def get_team_winrate_by_split(name: str):
    """Win rate evolution per split for charting."""
    safe = name.replace("'", "''")
    return query_df(
        f"""
        SELECT
            CAST(year AS INTEGER) as year,
            split,
            COUNT(*) as games,
            ROUND(100.0 * SUM(CAST(result AS DOUBLE)) / COUNT(*), 1) as winrate,
            ROUND(AVG(CAST(gamelength AS DOUBLE)) / 60.0, 1) as avg_minutes,
            ROUND(AVG(CAST(COALESCE(teamkills,0) + COALESCE(teamdeaths,0) AS DOUBLE)), 1) as avg_kills
        FROM games
        WHERE LOWER(teamname) = LOWER('{safe}')
        GROUP BY year, split
        ORDER BY year ASC, split ASC
    """
    )


def get_team_form(name: str, limit: int = 10):
    """Last N games for form display."""
    safe = name.replace("'", "''")
    return query_df(
        f"""
        SELECT
            game_date,
            CAST(result AS INTEGER) as result,
            CAST(COALESCE(teamkills, 0) + COALESCE(teamdeaths, 0) AS INTEGER) as total_kills,
            CAST(gamelength AS INTEGER) / 60 as game_minutes,
            opp_teamname
        FROM (
            SELECT g.game_date, g.result, g.teamkills, g.teamdeaths, g.gamelength,
                   opp.teamname as opp_teamname,
                   ROW_NUMBER() OVER (ORDER BY g.game_date DESC, g.gameid DESC) as rn
            FROM games g
            LEFT JOIN games opp
                   ON g.gameid = opp.gameid AND opp.teamname != g.teamname
            WHERE LOWER(g.teamname) = LOWER('{safe}')
        ) t
        WHERE rn <= {limit}
        ORDER BY rn DESC
    """
    )


# =============================================================
# TOURNAMENT
# =============================================================


def get_tournament_leagues():
    return query_df(
        """
        SELECT DISTINCT league, MIN(year) as from_year, MAX(year) as to_year,
               COUNT(DISTINCT year) as years
        FROM series
        GROUP BY league
        ORDER BY COUNT(*) DESC
    """
    )


def get_tournament_years(league: str):
    safe = league.replace("'", "''")
    return query_df(
        f"""
        SELECT DISTINCT year
        FROM series
        WHERE league = '{safe}'
        ORDER BY year DESC
    """
    )


def get_tournament_results(league: str, year: int):
    safe = league.replace("'", "''")
    return query_df(
        f"""
        SELECT match_date, team1, team2, score, series_winner,
               series_format, tournament_phase
        FROM series
        WHERE league = '{safe}' AND year = {year}
        ORDER BY match_date
    """
    )


def get_tournament_champion(league: str, year: int):
    safe = league.replace("'", "''")
    return query_df(
        f"""
        SELECT champion, runner_up, final_score, split
        FROM champions
        WHERE league = '{safe}' AND year = {year}
        ORDER BY final_date
    """
    )


# =============================================================
# GAME
# =============================================================


def get_game_players(gameid: str):
    safe = gameid.replace("'", "''")
    return query_df(
        f"""
        SELECT
            teamname, playername, position, champion, side, result,
            kills, deaths, assists, dpm, damageshare,
            totalgold, cspm, visionscore,
            golddiffat15, xpdiffat15, csdiffat15
        FROM players
        WHERE gameid = '{safe}'
        ORDER BY teamname,
            CASE position
                WHEN 'top' THEN 1 WHEN 'jng' THEN 2 WHEN 'mid' THEN 3
                WHEN 'bot' THEN 4 WHEN 'sup' THEN 5 ELSE 6 END
    """
    )


def get_game_teams(gameid: str):
    safe = gameid.replace("'", "''")
    return query_df(
        f"""
        SELECT
            teamname, side, result, gamelength,
            teamkills, teamdeaths,
            pick1, pick2, pick3, pick4, pick5,
            ban1, ban2, ban3, ban4, ban5,
            towers, dragons, barons, heralds, elders, void_grubs,
            firstblood, firsttower, firstdragon, firstbaron
        FROM games
        WHERE gameid = '{safe}'
        ORDER BY side
    """
    )


def search_games(team: str = None, league: str = None, limit: int = 50):
    conditions = []
    if team:
        safe = team.replace("'", "''")
        conditions.append(f"LOWER(teamname) = LOWER('{safe}')")
    if league:
        safe = league.replace("'", "''")
        conditions.append(f"league = '{safe}'")
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    return query_df(
        f"""
        SELECT DISTINCT gameid, game_date, league, teamname, side, result,
               pick1, pick2, pick3, pick4, pick5, gamelength
        FROM games
        {where}
        ORDER BY game_date DESC
        LIMIT {limit}
    """
    )


# =============================================================
# COMPARE
# =============================================================


def get_player_comparison(p1: str, p2: str):
    safe1 = p1.replace("'", "''")
    safe2 = p2.replace("'", "''")
    return query_df(
        f"""
        SELECT
            playername,
            COUNT(*) as games,
            SUM(result) as wins,
            ROUND(100.0 * SUM(result) / COUNT(*), 1) as winrate,
            ROUND(AVG(kills), 1) as avg_k,
            ROUND(AVG(deaths), 1) as avg_d,
            ROUND(AVG(assists), 1) as avg_a,
            ROUND(AVG(CASE WHEN deaths = 0 THEN kills + assists
                       ELSE (kills + assists) * 1.0 / deaths END), 2) as kda,
            ROUND(AVG(dpm), 0) as avg_dpm,
            ROUND(AVG(cspm), 1) as avg_cspm,
            ROUND(AVG(golddiffat15), 0) as avg_gd15,
            ROUND(AVG(visionscore), 1) as avg_vs
        FROM players
        WHERE LOWER(playername) IN (LOWER('{safe1}'), LOWER('{safe2}'))
        GROUP BY playername
    """
    )


def get_head_to_head(p1: str, p2: str):
    safe1 = p1.replace("'", "''")
    safe2 = p2.replace("'", "''")
    return query_df(
        f"""
        WITH p1_games AS (
            SELECT gameid, teamname as p1_team, result as p1_result,
                   champion as p1_champ, kills as p1_k, deaths as p1_d, assists as p1_a
            FROM players
            WHERE LOWER(playername) = LOWER('{safe1}')
        ),
        p2_games AS (
            SELECT gameid, teamname as p2_team, result as p2_result,
                   champion as p2_champ, kills as p2_k, deaths as p2_d, assists as p2_a
            FROM players
            WHERE LOWER(playername) = LOWER('{safe2}')
        )
        SELECT
            COUNT(*) as games,
            SUM(CASE WHEN p1.p1_result = 1 THEN 1 ELSE 0 END) as p1_wins,
            SUM(CASE WHEN p2.p2_result = 1 THEN 1 ELSE 0 END) as p2_wins,
            ROUND(AVG(p1.p1_k), 1) as p1_avg_k,
            ROUND(AVG(p1.p1_d), 1) as p1_avg_d,
            ROUND(AVG(p1.p1_a), 1) as p1_avg_a,
            ROUND(AVG(p2.p2_k), 1) as p2_avg_k,
            ROUND(AVG(p2.p2_d), 1) as p2_avg_d,
            ROUND(AVG(p2.p2_a), 1) as p2_avg_a
        FROM p1_games p1
        JOIN p2_games p2 ON p1.gameid = p2.gameid
        WHERE p1.p1_team != p2.p2_team
    """
    )


# =============================================================
# SERIES GAMES (expandable details)
# =============================================================


def get_series_games(team1: str, team2: str, date: str) -> list[dict]:
    """Get all games in a series with full player stats."""
    safe1 = team1.replace("'", "''")
    safe2 = team2.replace("'", "''")
    safe_date = date.replace("'", "''")

    games_df = query_df(
        f"""
        SELECT DISTINCT g1.gameid, g1.game_date, g1.game,
               g1.teamname as team1_name, g1.side as team1_side, g1.result as team1_result,
               g1.gamelength,
               g1.pick1 as t1_pick1, g1.pick2 as t1_pick2, g1.pick3 as t1_pick3, g1.pick4 as t1_pick4, g1.pick5 as t1_pick5,
               g1.ban1 as t1_ban1, g1.ban2 as t1_ban2, g1.ban3 as t1_ban3, g1.ban4 as t1_ban4, g1.ban5 as t1_ban5,
               g1.teamkills as t1_kills, g1.towers as t1_towers, g1.dragons as t1_dragons, g1.barons as t1_barons,
               g2.teamname as team2_name, g2.side as team2_side, g2.result as team2_result,
               g2.pick1 as t2_pick1, g2.pick2 as t2_pick2, g2.pick3 as t2_pick3, g2.pick4 as t2_pick4, g2.pick5 as t2_pick5,
               g2.ban1 as t2_ban1, g2.ban2 as t2_ban2, g2.ban3 as t2_ban3, g2.ban4 as t2_ban4, g2.ban5 as t2_ban5,
               g2.teamkills as t2_kills, g2.towers as t2_towers, g2.dragons as t2_dragons, g2.barons as t2_barons
        FROM games g1
        JOIN games g2 ON g1.gameid = g2.gameid AND g1.teamname != g2.teamname
        WHERE CAST(g1.game_date AS DATE) = '{safe_date}'
          AND (
              (LOWER(g1.teamname) = LOWER('{safe1}') AND LOWER(g2.teamname) = LOWER('{safe2}'))
              OR (LOWER(g1.teamname) LIKE LOWER('%{safe1}%') AND LOWER(g2.teamname) LIKE LOWER('%{safe2}%'))
          )
        ORDER BY g1.game
    """
    )

    if len(games_df) == 0:
        return []

    result = []
    for _, g in games_df.iterrows():
        gameid = g["gameid"]
        players_df = query_df(
            f"""
            SELECT playername, teamname, position, champion, side,
                   kills, deaths, assists,
                   COALESCE(dpm, 0) as dpm,
                   COALESCE(cspm, 0) as cspm,
                   COALESCE(totalgold, 0) as gold,
                   COALESCE(visionscore, 0) as vision,
                   COALESCE(golddiffat15, 0) as gd15,
                   COALESCE(damageshare, 0) as dmg_share,
                   result
            FROM players
            WHERE gameid = '{gameid}'
            ORDER BY teamname,
                CASE position
                    WHEN 'top' THEN 1 WHEN 'jng' THEN 2 WHEN 'mid' THEN 3
                    WHEN 'bot' THEN 4 WHEN 'sup' THEN 5 ELSE 6 END
        """
        )

        def _safe(v):
            if v is None:
                return None
            try:
                import numpy as np

                if isinstance(v, (np.integer,)):
                    return int(v)
                if isinstance(v, (np.floating,)):
                    return round(float(v), 1)
            except ImportError:
                pass
            if isinstance(v, float):
                return round(v, 1)
            return v

        teams = {}
        for _, p in players_df.iterrows():
            tn = p["teamname"]
            if tn not in teams:
                teams[tn] = []
            teams[tn].append(
                {
                    "player": p["playername"],
                    "pos": p["position"],
                    "champion": p["champion"],
                    "k": _safe(p["kills"]),
                    "d": _safe(p["deaths"]),
                    "a": _safe(p["assists"]),
                    "dpm": _safe(p["dpm"]),
                    "cspm": _safe(p["cspm"]),
                    "gold": _safe(p["gold"]),
                    "vision": _safe(p["vision"]),
                    "gd15": _safe(p["gd15"]),
                    "dmg_pct": _safe(p["dmg_share"] * 100 if p["dmg_share"] else 0),
                    "win": int(p["result"]) if p["result"] is not None else 0,
                }
            )

        game_data = {
            "gameid": gameid,
            "game": _safe(g["game"]),
            "length": _safe(g["gamelength"]),
            "team1": {
                "name": g["team1_name"],
                "side": g["team1_side"],
                "win": int(g["team1_result"]) if g["team1_result"] is not None else 0,
                "kills": _safe(g["t1_kills"]),
                "towers": _safe(g["t1_towers"]),
                "dragons": _safe(g["t1_dragons"]),
                "barons": _safe(g["t1_barons"]),
                "picks": [
                    g[f"t1_pick{i}"] for i in range(1, 6) if g.get(f"t1_pick{i}")
                ],
                "bans": [g[f"t1_ban{i}"] for i in range(1, 6) if g.get(f"t1_ban{i}")],
                "players": teams.get(g["team1_name"], []),
            },
            "team2": {
                "name": g["team2_name"],
                "side": g["team2_side"],
                "win": int(g["team2_result"]) if g["team2_result"] is not None else 0,
                "kills": _safe(g["t2_kills"]),
                "towers": _safe(g["t2_towers"]),
                "dragons": _safe(g["t2_dragons"]),
                "barons": _safe(g["t2_barons"]),
                "picks": [
                    g[f"t2_pick{i}"] for i in range(1, 6) if g.get(f"t2_pick{i}")
                ],
                "bans": [g[f"t2_ban{i}"] for i in range(1, 6) if g.get(f"t2_ban{i}")],
                "players": teams.get(g["team2_name"], []),
            },
        }
        result.append(game_data)

    return result


# =============================================================
# COMPARE TEAMS
# =============================================================


def _resolve_team_name(name: str) -> str | None:
    """Resolve a partial team name to the exact name in data, preferring exact match."""
    safe = name.replace("'", "''")
    exact = query_one(
        f"""
        SELECT teamname FROM games
        WHERE LOWER(teamname) = LOWER('{safe}')
        ORDER BY game_date DESC LIMIT 1
    """
    )
    if exact:
        return exact["teamname"]
    fuzzy = query_one(
        f"""
        SELECT teamname FROM games
        WHERE LOWER(teamname) LIKE LOWER('%{safe}%')
        ORDER BY game_date DESC LIMIT 1
    """
    )
    return fuzzy["teamname"] if fuzzy else None


def get_team_comparison(
    t1: str,
    t2: str,
    year: int = None,
    split: str = None,
    playoffs: int = None,
    league: str = None,
):
    name1 = _resolve_team_name(t1)
    name2 = _resolve_team_name(t2)
    if not name1 or not name2:
        return query_df("SELECT 1 WHERE false")
    safe1 = name1.replace("'", "''")
    safe2 = name2.replace("'", "''")
    conditions = [f"LOWER(teamname) IN (LOWER('{safe1}'), LOWER('{safe2}'))"]
    conditions.append(
        f"""gameid IN (
        SELECT gameid FROM games WHERE LOWER(teamname) = LOWER('{safe1}')
        INTERSECT
        SELECT gameid FROM games WHERE LOWER(teamname) = LOWER('{safe2}')
    )"""
    )
    if year:
        conditions.append(f"year = {year}")
    if league:
        league_safe = league.replace("'", "''")
        conditions.append(f"league = '{league_safe}'")
    if split:
        split_safe = split.replace("'", "''")
        if split == "N/A":
            conditions.append("(split IS NULL OR split = '')")
        else:
            conditions.append(f"COALESCE(split, '') = '{split_safe}'")
    if playoffs is not None:
        conditions.append(f"COALESCE(playoffs, 0) = {playoffs}")
    where = "WHERE " + " AND ".join(conditions)

    return query_df(
        f"""
        SELECT
            teamname,
            COUNT(*) as games,
            SUM(result) as wins,
            ROUND(100.0 * SUM(result) / NULLIF(COUNT(*), 0), 1) as winrate,
            ROUND(AVG(gamelength), 0) as avg_length,
            ROUND(AVG(teamkills), 1) as avg_kills,
            ROUND(AVG(teamdeaths), 1) as avg_deaths,
            SUM(CASE WHEN firstblood = 1 THEN 1 ELSE 0 END) as fb_count,
            ROUND(100.0 * SUM(CASE WHEN firstblood = 1 THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0), 1) as fb_pct,
            SUM(CASE WHEN firsttower = 1 THEN 1 ELSE 0 END) as ft_count,
            ROUND(100.0 * SUM(CASE WHEN firsttower = 1 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as ft_pct,
            ROUND(AVG(towers), 1) as avg_towers,
            ROUND(AVG(dragons), 1) as avg_dragons,
            ROUND(AVG(barons), 1) as avg_barons
        FROM games
        {where}
        GROUP BY teamname
    """
    )


def get_team_head_to_head(
    t1: str,
    t2: str,
    year: int = None,
    split: str = None,
    playoffs: int = None,
    league: str = None,
):
    name1 = _resolve_team_name(t1)
    name2 = _resolve_team_name(t2)
    if not name1 or not name2:
        return query_df("SELECT 1 WHERE false")
    safe1 = name1.replace("'", "''")
    safe2 = name2.replace("'", "''")

    conditions = [
        f"((team1 = '{safe1}' AND team2 = '{safe2}') OR (team1 = '{safe2}' AND team2 = '{safe1}'))"
    ]
    if year:
        conditions.append(f"year = {year}")
    if league:
        league_safe = league.replace("'", "''")
        conditions.append(f"league = '{league_safe}'")
    if split:
        split_safe = split.replace("'", "''")
        if split == "N/A":
            conditions.append("(split IS NULL OR split = '')")
        else:
            conditions.append(f"COALESCE(split, '') = '{split_safe}'")
    if playoffs is not None:
        conditions.append(f"COALESCE(playoffs, 0) = {playoffs}")
    where = "WHERE " + " AND ".join(conditions)

    return query_df(
        f"""
        SELECT
            match_date,
            CAST(year AS INTEGER) as year,
            COALESCE(split, '') as split,
            CAST(COALESCE(playoffs, 0) AS INTEGER) as playoffs,
            league,
            team1,
            team2,
            score,
            series_winner,
            series_format,
            tournament_phase
        FROM series
        {where}
        ORDER BY match_date DESC
    """
    )


def get_team_h2h_by_split(t1: str, t2: str):
    """Returns H2H series wins grouped by split for comparison charts."""
    name1 = _resolve_team_name(t1)
    name2 = _resolve_team_name(t2)
    if not name1 or not name2:
        return []
    safe1 = name1.replace("'", "''")
    safe2 = name2.replace("'", "''")
    df = query_df(
        f"""
        SELECT
            CAST(year AS INTEGER) as year,
            league,
            COALESCE(split, 'N/A') as split,
            CAST(COALESCE(playoffs, 0) AS INTEGER) as playoffs,
            COUNT(*) as total,
            SUM(CASE WHEN series_winner = '{safe1}' THEN 1 ELSE 0 END) as t1_wins,
            SUM(CASE WHEN series_winner = '{safe2}' THEN 1 ELSE 0 END) as t2_wins
        FROM series
        WHERE (team1 = '{safe1}' AND team2 = '{safe2}')
           OR (team1 = '{safe2}' AND team2 = '{safe1}')
        GROUP BY year, league, split, playoffs
        ORDER BY year DESC, league, split, playoffs
    """
    )
    return df.to_dict("records")


def get_team_h2h_summary(
    t1: str,
    t2: str,
    year: int = None,
    split: str = None,
    playoffs: int = None,
    league: str = None,
):
    name1 = _resolve_team_name(t1)
    name2 = _resolve_team_name(t2)
    if not name1 or not name2:
        return {"total_series": 0, "t1_wins": 0, "t2_wins": 0}
    safe1 = name1.replace("'", "''")
    safe2 = name2.replace("'", "''")

    conditions = [
        f"((team1 = '{safe1}' AND team2 = '{safe2}') OR (team1 = '{safe2}' AND team2 = '{safe1}'))"
    ]
    if year:
        conditions.append(f"year = {year}")
    if league:
        league_safe = league.replace("'", "''")
        conditions.append(f"league = '{league_safe}'")
    if split:
        split_safe = split.replace("'", "''")
        if split == "N/A":
            conditions.append("(split IS NULL OR split = '')")
        else:
            conditions.append(f"COALESCE(split, '') = '{split_safe}'")
    if playoffs is not None:
        conditions.append(f"COALESCE(playoffs, 0) = {playoffs}")
    where = "WHERE " + " AND ".join(conditions)

    row = query_one(
        f"""
        WITH h2h AS (
            SELECT series_winner
            FROM series
            {where}
        )
        SELECT
            COUNT(*) as total_series,
            SUM(CASE WHEN series_winner = '{safe1}' THEN 1 ELSE 0 END) as t1_wins,
            SUM(CASE WHEN series_winner = '{safe2}' THEN 1 ELSE 0 END) as t2_wins
        FROM h2h
    """
    )
    if not row:
        return {"total_series": 0, "t1_wins": 0, "t2_wins": 0}
    return {k: int(v) if v is not None else 0 for k, v in row.items()}


# =============================================================
# RANKINGS
# =============================================================


def get_player_rankings(
    stat: str = "kda",
    position: str = None,
    league: str = None,
    year: int = None,
    split: str = None,
    min_games: int = 30,
    limit: int = 50,
):
    conditions = ["champion IS NOT NULL"]
    if position:
        conditions.append(f"position = '{position}'")
    if league:
        safe = league.replace("'", "''")
        conditions.append(f"league = '{safe}'")
    if year:
        conditions.append(f"year = {year}")
    if split:
        split_safe = split.replace("'", "''")
        conditions.append(f"split = '{split_safe}'")

    where = "WHERE " + " AND ".join(conditions)

    stat_cols = {
        "kda": "ROUND(AVG(CASE WHEN deaths=0 THEN kills+assists ELSE (kills+assists)*1.0/deaths END), 2) as value",
        "winrate": "ROUND(100.0 * SUM(result) / COUNT(*), 1) as value",
        "dpm": "ROUND(AVG(dpm), 0) as value",
        "cspm": "ROUND(AVG(cspm), 1) as value",
        "gd15": "ROUND(AVG(golddiffat15), 0) as value",
        "kills": "ROUND(AVG(kills), 1) as value",
        "vision": "ROUND(AVG(visionscore), 1) as value",
    }

    stat_sql = stat_cols.get(stat, stat_cols["kda"])

    return query_df(
        f"""
        SELECT
            playername, position, teamname, league,
            COUNT(*) as games,
            SUM(result) as wins,
            ROUND(100.0 * SUM(result) / COUNT(*), 1) as winrate,
            {stat_sql}
        FROM players
        {where}
        GROUP BY playername, position, teamname, league
        HAVING COUNT(*) >= {min_games}
        ORDER BY value DESC
        LIMIT {limit}
    """
    )


def get_player_splits(name: str):
    safe = name.replace("'", "''")
    return query_df(
        f"""
        SELECT DISTINCT year, split
        FROM players
        WHERE LOWER(playername) = LOWER('{safe}')
        ORDER BY year DESC, split
    """
    )


def get_available_years():
    return query_df(
        """
        SELECT DISTINCT year FROM players
        ORDER BY year DESC
    """
    )["year"].tolist()


def get_available_leagues():
    return query_df(
        """
        SELECT DISTINCT league, COUNT(*) as c
        FROM players
        GROUP BY league ORDER BY c DESC
    """
    )["league"].tolist()


# =============================================================
# BETTING STATS
# =============================================================


def get_betting_stats(
    team: str = None,
    league: str = None,
    year: int = None,
    split: str = None,
    playoffs: int = None,
    limit: int = 30,
):
    """
    Returns per-team betting-relevant stats aggregated from the games table.
    Total kills = teamkills + teamdeaths (both sides combined).
    """
    conditions = ["teamname IS NOT NULL"]
    if team:
        safe = team.replace("'", "''")
        conditions.append(f"LOWER(teamname) LIKE LOWER('%{safe}%')")
    if league:
        safe = league.replace("'", "''")
        conditions.append(f"league = '{safe}'")
    if year:
        conditions.append(f"year = {year}")
    if split:
        safe = split.replace("'", "''")
        conditions.append(f"split = '{safe}'")
    if playoffs is not None:
        conditions.append(f"playoffs = {playoffs}")

    where = "WHERE " + " AND ".join(conditions)

    return query_df(
        f"""
        WITH base AS (
            SELECT
                teamname,
                league,
                CAST(result         AS DOUBLE) AS result,
                CAST(gamelength     AS DOUBLE) / 60.0  AS game_minutes,
                CAST(COALESCE(firstblood,  0) AS DOUBLE) AS firstblood,
                CAST(COALESCE(firsttower,  0) AS DOUBLE) AS firsttower,
                CAST(COALESCE(firstdragon, 0) AS DOUBLE) AS firstdragon,
                COALESCE(TRY_CAST(firstherald AS DOUBLE), 0.0)  AS firstherald,
                CAST(COALESCE(firstbaron,  0) AS DOUBLE) AS firstbaron,
                CAST(COALESCE(teamkills,   0) AS DOUBLE) AS teamkills,
                CAST(COALESCE(teamkills,   0) + COALESCE(teamdeaths, 0) AS DOUBLE) AS total_kills,
                CAST(COALESCE(towers,    0) + COALESCE(opp_towers,    0) AS DOUBLE) AS total_towers,
                CAST(COALESCE(dragons,   0) + COALESCE(opp_dragons,   0) AS DOUBLE) AS total_dragons,
                CAST(COALESCE(barons,    0) + COALESCE(opp_barons,    0) AS DOUBLE) AS total_nashors,
                CAST(COALESCE(inhibitors,0) + COALESCE(opp_inhibitors, 0) AS DOUBLE) AS total_inhibitors
            FROM games
            {where}
        )
        SELECT
            teamname,
            league,
            COUNT(*)                                                          AS games,
            ROUND(100.0 * SUM(result)       / COUNT(*), 1)                   AS winrate,
            ROUND(100.0 * AVG(firstblood),  1)                               AS first_blood_pct,
            ROUND(100.0 * AVG(firsttower),  1)                               AS first_tower_pct,
            ROUND(100.0 * AVG(firstdragon), 1)                               AS first_dragon_pct,
            ROUND(100.0 * AVG(firstherald), 1)                               AS first_herald_pct,
            ROUND(100.0 * AVG(firstbaron),  1)                               AS first_baron_pct,
            ROUND(AVG(total_kills),         1)                               AS avg_total_kills,
            ROUND(AVG(total_towers),        1)                               AS avg_total_towers,
            ROUND(AVG(total_dragons),       1)                               AS avg_total_dragons,
            ROUND(AVG(total_nashors),       1)                               AS avg_total_nashors,
            ROUND(AVG(total_inhibitors),    1)                               AS avg_total_inhibitors,
            ROUND(AVG(game_minutes),        1)                               AS avg_game_minutes,
            ROUND(100.0 * SUM(CASE WHEN total_kills  > 25  THEN 1 ELSE 0 END) / COUNT(*), 1) AS kills_over_25_pct,
            ROUND(100.0 * SUM(CASE WHEN total_towers > 10  THEN 1 ELSE 0 END) / COUNT(*), 1) AS towers_over_10_pct,
            ROUND(100.0 * SUM(CASE WHEN total_nashors >= 2 THEN 1 ELSE 0 END) / COUNT(*), 1) AS nashors_over_1_5_pct,
            CAST(MAX(teamkills) AS INTEGER)                                  AS most_kills_game
        FROM base
        GROUP BY teamname, league
        HAVING COUNT(*) >= 3
        ORDER BY games DESC
        LIMIT {limit}
    """
    )


def get_betting_filters():
    """Leagues, years, splits available for betting page filters."""
    leagues = query_df(
        """
        SELECT DISTINCT league, COUNT(*) as c
        FROM games
        GROUP BY league ORDER BY c DESC
    """
    )["league"].tolist()

    years = query_df(
        """
        SELECT DISTINCT year FROM games ORDER BY year DESC
    """
    )["year"].tolist()

    splits = query_df(
        """
        SELECT DISTINCT split FROM games
        WHERE split IS NOT NULL ORDER BY split
    """
    )["split"].tolist()

    return leagues, years, splits


def get_available_splits():
    return query_df(
        """
        SELECT DISTINCT split
        FROM players
        WHERE split IS NOT NULL
        ORDER BY split
    """
    )["split"].tolist()
