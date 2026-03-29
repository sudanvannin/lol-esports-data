"""DuckDB query engine for Silver Parquet data."""

import json
import os
import threading
from pathlib import Path

import duckdb
import pandas as pd

from src.ml.odds_snapshots import (
    DEFAULT_ODDS_SNAPSHOTS_SCORED_PATH,
    build_edge_board,
    build_edge_board_meta,
    load_snapshot_table,
)
from .official_schedule import load_official_schedule_snapshot
from .upcoming_matches import load_upcoming_matches

# ── Persistent connection with MotherDuck ──
_lock = threading.Lock()
_con = None
_upcoming_cache_lock = threading.Lock()
_upcoming_cache_df = None
_upcoming_cache_meta = None
_upcoming_cache_mtime = None
_official_schedule_cache_lock = threading.Lock()
_recent_cache_df = None
_official_upcoming_cache_df = None
_official_schedule_cache_meta = None
_official_schedule_cache_mtime = None
_edge_board_cache_lock = threading.Lock()
_edge_board_cache_df = None
_edge_board_cache_meta = None
_edge_board_cache_mtime = None
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOCAL_SILVER_DIR = DATA_DIR / "silver"

LEAGUE_DISPLAY_NAMES = {
    "FST": "First Stand",
    "LAS": "LCK Academy Series",
    "LCKC": "LCK Challengers",
    "LTA N": "LTA North",
    "LTA S": "LTA South",
    "WLDs": "Worlds",
}


def _league_label(value):
    if value is None:
        return None
    text = str(value)
    return LEAGUE_DISPLAY_NAMES.get(text, text)


def _with_league_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()
    for source_col, label_col in (
        ("league", "league_label"),
        ("league_code", "league_code_label"),
    ):
        if source_col in out.columns and label_col not in out.columns:
            out[label_col] = out[source_col].map(_league_label)
    return out


def _with_league_labels_row(row: dict | None) -> dict | None:
    if not row:
        return row

    out = dict(row)
    if "league" in out and "league_label" not in out:
        out["league_label"] = _league_label(out["league"])
    if "league_code" in out and "league_code_label" not in out:
        out["league_code_label"] = _league_label(out["league_code"])
    return out


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ordinal_label(value: float) -> str:
    integer = int(round(_safe_float(value)))
    if 10 <= integer % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(integer % 10, "th")
    return f"{integer}{suffix}"


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
                "WARNING: MOTHERDUCK_TOKEN not found. Using local parquet fallback from data/silver."
            )
            con = duckdb.connect(":memory:")
            _bootstrap_local_silver_views(con)

        _con = con
    return _con


def _bootstrap_local_silver_views(con):
    """Register local silver parquet datasets as DuckDB views."""
    view_specs = {
        "games": (
            f"{(LOCAL_SILVER_DIR / 'games').as_posix()}/**/*.parquet",
            True,
            (LOCAL_SILVER_DIR / "games").exists(),
        ),
        "players": (
            f"{(LOCAL_SILVER_DIR / 'players').as_posix()}/**/*.parquet",
            True,
            (LOCAL_SILVER_DIR / "players").exists(),
        ),
        "series": (
            (LOCAL_SILVER_DIR / "series.parquet").as_posix(),
            False,
            (LOCAL_SILVER_DIR / "series.parquet").exists(),
        ),
        "champions": (
            (LOCAL_SILVER_DIR / "champions.parquet").as_posix(),
            False,
            (LOCAL_SILVER_DIR / "champions.parquet").exists(),
        ),
    }

    for view_name, (source_path, hive_partitioning, exists) in view_specs.items():
        if not exists:
            continue
        if hive_partitioning:
            con.execute(
                f"""
                CREATE OR REPLACE VIEW {view_name} AS
                SELECT * FROM read_parquet('{source_path}', hive_partitioning=1)
                """
            )
        else:
            con.execute(
                f"""
                CREATE OR REPLACE VIEW {view_name} AS
                SELECT * FROM read_parquet('{source_path}')
                """
            )


def _empty_upcoming_matches_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "match_id",
            "match_time",
            "match_date",
            "league",
            "event_name",
            "phase_label",
            "team1",
            "team2",
            "best_of",
            "patch",
            "overview_page",
            "source",
        ]
    )


def _empty_upcoming_leagues_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["league", "total_matches", "next_match_time"])


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
    return _with_league_labels_row(df.iloc[0].to_dict())


def _load_local_upcoming_matches_cached():
    """Load normalized upcoming matches from the local Leaguepedia payload with mtime caching."""
    global _upcoming_cache_df, _upcoming_cache_meta, _upcoming_cache_mtime

    upcoming_path = DATA_DIR / "bronze" / "leaguepedia" / "match_results.json"
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


def _load_local_official_schedule_cached():
    """Load compact official schedule snapshot with mtime caching."""
    global _recent_cache_df, _official_upcoming_cache_df
    global _official_schedule_cache_meta, _official_schedule_cache_mtime

    schedule_path = DATA_DIR / "bronze" / "official" / "web_schedule.json"
    try:
        current_mtime = (
            schedule_path.stat().st_mtime if schedule_path.exists() else None
        )
    except OSError:
        current_mtime = None

    if _recent_cache_df is not None and _official_schedule_cache_mtime == current_mtime:
        return (
            _recent_cache_df.copy(),
            _official_upcoming_cache_df.copy(),
            dict(_official_schedule_cache_meta or {}),
        )

    with _official_schedule_cache_lock:
        if (
            _recent_cache_df is not None
            and _official_schedule_cache_mtime == current_mtime
        ):
            return (
                _recent_cache_df.copy(),
                _official_upcoming_cache_df.copy(),
                dict(_official_schedule_cache_meta or {}),
            )

        recent_df, upcoming_df, metadata = load_official_schedule_snapshot(
            schedule_path
        )
        _recent_cache_df = recent_df
        _official_upcoming_cache_df = upcoming_df
        _official_schedule_cache_meta = metadata
        _official_schedule_cache_mtime = current_mtime
        return recent_df.copy(), upcoming_df.copy(), dict(metadata)


def _load_local_edge_board_cached():
    """Load local scored odds snapshots and derive the latest edge board."""
    global _edge_board_cache_df, _edge_board_cache_meta, _edge_board_cache_mtime

    try:
        current_mtime = (
            DEFAULT_ODDS_SNAPSHOTS_SCORED_PATH.stat().st_mtime
            if DEFAULT_ODDS_SNAPSHOTS_SCORED_PATH.exists()
            else None
        )
    except OSError:
        current_mtime = None

    if _edge_board_cache_df is not None and _edge_board_cache_mtime == current_mtime:
        return _edge_board_cache_df.copy(), dict(_edge_board_cache_meta or {})

    with _edge_board_cache_lock:
        if _edge_board_cache_df is not None and _edge_board_cache_mtime == current_mtime:
            return _edge_board_cache_df.copy(), dict(_edge_board_cache_meta or {})

        scored_df = load_snapshot_table(DEFAULT_ODDS_SNAPSHOTS_SCORED_PATH)
        edge_board_df = build_edge_board(scored_df, recommended_only=False)
        edge_board_df = _with_league_labels(edge_board_df)
        meta = build_edge_board_meta(scored_df)
        meta["source_path"] = str(DEFAULT_ODDS_SNAPSHOTS_SCORED_PATH)

        _edge_board_cache_df = edge_board_df
        _edge_board_cache_meta = meta
        _edge_board_cache_mtime = current_mtime
        return edge_board_df.copy(), dict(meta)


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
            FROM web_upcoming_matches_live
            {where_clause}
            ORDER BY match_time ASC, league, team1, team2
            LIMIT {limit}
            """
        ).fetchdf()
        if len(remote_df) > 0:
            return _with_league_labels(remote_df)
    except duckdb.Error:
        pass

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
            return _with_league_labels(remote_df)
    except duckdb.Error:
        pass

    _, official_upcoming_df, _ = _load_local_official_schedule_cached()
    if safe_league:
        official_upcoming_df = official_upcoming_df.loc[
            official_upcoming_df["league"] == safe_league
        ].copy()
    if not official_upcoming_df.empty:
        return _with_league_labels(
            official_upcoming_df.sort_values(
                ["match_time", "league", "team1", "team2"],
                kind="stable",
            )
            .head(limit)
            .reset_index(drop=True)
        )

    if not os.environ.get("MOTHERDUCK_TOKEN"):
        return _with_league_labels(_empty_upcoming_matches_df())

    local_df, _ = _load_local_upcoming_matches_cached()
    if safe_league:
        local_df = local_df.loc[local_df["league"] == safe_league].copy()
    return _with_league_labels(
        local_df.sort_values(["match_time", "league", "team1", "team2"], kind="stable")
        .head(limit)
        .reset_index(drop=True)
    )


def get_upcoming_match(match_id: str):
    """Return a single upcoming match by id across the fast-lane sources."""
    safe_match_id = match_id.replace("'", "''")
    con = _get_persistent_con()

    for table_name in ("web_upcoming_matches_live", "web_upcoming_matches"):
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
                FROM {table_name}
                WHERE match_id = '{safe_match_id}'
                LIMIT 1
                """
            ).fetchdf()
            if len(remote_df) > 0:
                return _with_league_labels_row(remote_df.iloc[0].to_dict())
        except duckdb.Error:
            pass

    _, official_upcoming_df, _ = _load_local_official_schedule_cached()
    if not official_upcoming_df.empty:
        match_df = official_upcoming_df.loc[
            official_upcoming_df["match_id"] == match_id
        ].head(1)
        if not match_df.empty:
            return _with_league_labels_row(match_df.iloc[0].to_dict())

    if not os.environ.get("MOTHERDUCK_TOKEN"):
        return None

    local_df, _ = _load_local_upcoming_matches_cached()
    match_df = local_df.loc[local_df["match_id"] == match_id].head(1)
    if match_df.empty:
        return None
    return _with_league_labels_row(match_df.iloc[0].to_dict())


def get_prob_win_matches(match_ids: list[str] | None = None):
    """Return precomputed winner probabilities for upcoming matches when available."""
    con = _get_persistent_con()
    where_clause = ""
    if match_ids:
        safe_ids = [item.replace("'", "''") for item in match_ids if item]
        if safe_ids:
            values = ", ".join(f"'{value}'" for value in safe_ids)
            where_clause = f"WHERE match_id IN ({values})"

    try:
        remote_df = con.execute(
            f"""
            SELECT
                match_id,
                winner_available,
                winner_error,
                team1_win_prob,
                team2_win_prob,
                team1_win_pct,
                team2_win_pct,
                team1_fair_odds,
                team2_fair_odds,
                favorite_name,
                favorite_prob_pct,
                warnings_json
            FROM web_prob_win_matches
            {where_clause}
            """
        ).fetchdf()
        return remote_df
    except duckdb.Error:
        return pd.DataFrame(
            columns=[
                "match_id",
                "winner_available",
                "winner_error",
                "team1_win_prob",
                "team2_win_prob",
                "team1_win_pct",
                "team2_win_pct",
                "team1_fair_odds",
                "team2_fair_odds",
                "favorite_name",
                "favorite_prob_pct",
                "warnings_json",
            ]
        )


def get_prob_win_detail(match_id: str):
    """Return precomputed winner and totals markets for one upcoming match."""
    safe_match_id = match_id.replace("'", "''")
    con = _get_persistent_con()

    try:
        match_df = con.execute(
            f"""
            SELECT
                match_id,
                match_time,
                match_date,
                league,
                league_label,
                event_name,
                phase_label,
                team1,
                team2,
                best_of,
                patch,
                league_code,
                split_name,
                playoffs,
                winner_available,
                winner_error,
                team1_win_prob,
                team2_win_prob,
                team1_win_pct,
                team2_win_pct,
                team1_fair_odds,
                team2_fair_odds,
                favorite_name,
                favorite_prob_pct,
                totals_available,
                totals_error,
                warnings_json
            FROM web_prob_win_matches
            WHERE match_id = '{safe_match_id}'
            LIMIT 1
            """
        ).fetchdf()
    except duckdb.Error:
        return None

    if match_df.empty:
        return None

    match_row = match_df.iloc[0].to_dict()
    try:
        totals_df = con.execute(
            f"""
            SELECT
                match_id,
                market_key,
                market_label,
                line,
                predicted_mean,
                distribution,
                over_prob,
                under_prob,
                over_prob_pct,
                under_prob_pct,
                over_fair_odds,
                under_fair_odds,
                team1_sample,
                team2_sample,
                baseline_sample
            FROM web_prob_win_totals
            WHERE match_id = '{safe_match_id}'
            ORDER BY market_key
            """
        ).fetchdf()
    except duckdb.Error:
        totals_df = pd.DataFrame()

    warnings = []
    warnings_json = match_row.get("warnings_json")
    if warnings_json:
        try:
            warnings = json.loads(warnings_json)
        except json.JSONDecodeError:
            warnings = [_clean for _clean in [warnings_json] if _clean]

    return {
        "match": {
            "match_id": match_row.get("match_id"),
            "match_time": match_row.get("match_time"),
            "match_date": match_row.get("match_date"),
            "league": match_row.get("league"),
            "league_label": match_row.get("league_label"),
            "event_name": match_row.get("event_name"),
            "phase_label": match_row.get("phase_label"),
            "team1": match_row.get("team1"),
            "team2": match_row.get("team2"),
            "best_of": match_row.get("best_of"),
            "patch": match_row.get("patch"),
        },
        "context": {
            "league_code": match_row.get("league_code"),
            "split_name": match_row.get("split_name"),
            "patch_version": match_row.get("patch"),
            "best_of": int(match_row.get("best_of") or 0),
            "playoffs": bool(match_row.get("playoffs")),
        },
        "winner_market": {
            "available": bool(match_row.get("winner_available")),
            "error": match_row.get("winner_error"),
            "team1_name": match_row.get("team1"),
            "team2_name": match_row.get("team2"),
            "team1_win_prob": match_row.get("team1_win_prob"),
            "team2_win_prob": match_row.get("team2_win_prob"),
            "team1_win_pct": match_row.get("team1_win_pct"),
            "team2_win_pct": match_row.get("team2_win_pct"),
            "team1_fair_odds": match_row.get("team1_fair_odds"),
            "team2_fair_odds": match_row.get("team2_fair_odds"),
            "favorite_name": match_row.get("favorite_name"),
            "favorite_prob_pct": match_row.get("favorite_prob_pct"),
        },
        "totals_market": {
            "available": bool(match_row.get("totals_available")),
            "error": match_row.get("totals_error"),
            "markets": totals_df.to_dict("records"),
        },
        "warnings": warnings,
        "has_any_market": bool(
            match_row.get("winner_available") or match_row.get("totals_available")
        ),
        "confidence_gap_pct": abs(
            float(match_row.get("team1_win_pct") or 0.0)
            - float(match_row.get("team2_win_pct") or 0.0)
        ),
    }


def get_edge_board(
    limit: int = 80,
    league: str | None = None,
    bookmaker: str | None = None,
    recommended_only: bool = True,
):
    """Return the latest scored odds board, preferring MotherDuck and falling back to local tracking."""
    con = _get_persistent_con()
    filters: list[str] = []
    if league:
        safe_league = league.replace("'", "''")
        filters.append(f"league_code = '{safe_league}'")
    if bookmaker:
        safe_bookmaker = bookmaker.replace("'", "''")
        filters.append(f"bookmaker = '{safe_bookmaker}'")
    if recommended_only:
        filters.append("recommend_bet = TRUE")
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

    try:
        remote_df = con.execute(
            f"""
            SELECT
                captured_at,
                match_time,
                league_code,
                team1_name,
                team2_name,
                bookmaker,
                recommend_bet,
                recommendation_reason,
                recommended_selection,
                recommended_odds,
                recommended_model_probability,
                recommended_fair_odds,
                recommended_edge_pct,
                recommended_ev_pct,
                recommended_kelly_pct,
                team1_odds,
                team2_odds,
                model_name,
                model_run_id,
                snapshot_id,
                warnings_json
            FROM web_edge_board
            {where_clause}
            ORDER BY recommend_bet DESC, recommended_ev_pct DESC, recommended_edge_pct DESC, match_time ASC
            LIMIT {limit}
            """
        ).fetchdf()
        if len(remote_df) > 0:
            return _with_league_labels(remote_df)
    except duckdb.Error:
        pass

    local_df, _ = _load_local_edge_board_cached()
    if league:
        local_df = local_df.loc[local_df["league_code"] == league].copy()
    if bookmaker:
        local_df = local_df.loc[local_df["bookmaker"] == bookmaker].copy()
    if recommended_only and "recommend_bet" in local_df.columns:
        local_df = local_df.loc[local_df["recommend_bet"].fillna(False)].copy()
    if local_df.empty:
        return pd.DataFrame(
            columns=[
                "captured_at",
                "match_time",
                "league_code",
                "league_code_label",
                "team1_name",
                "team2_name",
                "bookmaker",
                "recommend_bet",
                "recommendation_reason",
                "recommended_selection",
                "recommended_odds",
                "recommended_model_probability",
                "recommended_fair_odds",
                "recommended_edge_pct",
                "recommended_ev_pct",
                "recommended_kelly_pct",
                "team1_odds",
                "team2_odds",
                "model_name",
                "model_run_id",
                "snapshot_id",
                "warnings_json",
            ]
        )
    return local_df.head(limit).reset_index(drop=True)


def get_edge_board_bookmakers():
    """Return distinct bookmakers available in the edge board."""
    con = _get_persistent_con()
    try:
        remote_df = con.execute(
            """
            SELECT bookmaker, COUNT(*) AS total_rows, MAX(captured_at) AS latest_captured_at
            FROM web_edge_board
            GROUP BY bookmaker
            ORDER BY latest_captured_at DESC, bookmaker
            """
        ).fetchdf()
        if len(remote_df) > 0:
            return remote_df
    except duckdb.Error:
        pass

    local_df, _ = _load_local_edge_board_cached()
    if local_df.empty:
        return pd.DataFrame(columns=["bookmaker", "total_rows", "latest_captured_at"])
    return (
        local_df.groupby("bookmaker", as_index=False)
        .agg(total_rows=("bookmaker", "count"), latest_captured_at=("captured_at", "max"))
        .sort_values(["latest_captured_at", "bookmaker"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )


def get_edge_board_meta():
    """Return metadata about the latest edge board snapshot source."""
    con = _get_persistent_con()
    try:
        remote_df = con.execute(
            """
            SELECT source, row_count, recommendation_count, latest_captured_at, bookmaker_count, model_run_ids_json
            FROM web_edge_board_meta
            LIMIT 1
            """
        ).fetchdf()
        if len(remote_df) > 0:
            row = remote_df.iloc[0].to_dict()
            model_run_ids_json = row.get("model_run_ids_json")
            if model_run_ids_json:
                try:
                    row["model_run_ids"] = json.loads(model_run_ids_json)
                except json.JSONDecodeError:
                    row["model_run_ids"] = [model_run_ids_json]
            else:
                row["model_run_ids"] = []
            return row
    except duckdb.Error:
        pass

    _, meta = _load_local_edge_board_cached()
    return meta


def get_upcoming_match_leagues():
    """Return distinct leagues available in upcoming matches."""
    con = _get_persistent_con()
    try:
        remote_df = con.execute(
            """
            SELECT league, COUNT(*) AS total_matches, MIN(match_time) AS next_match_time
            FROM web_upcoming_matches_live
            GROUP BY league
            ORDER BY next_match_time ASC, league
            """
        ).fetchdf()
        if len(remote_df) > 0:
            return _with_league_labels(remote_df)
    except duckdb.Error:
        pass

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
            return _with_league_labels(remote_df)
    except duckdb.Error:
        pass

    _, official_upcoming_df, _ = _load_local_official_schedule_cached()
    if not official_upcoming_df.empty:
        return _with_league_labels(
            official_upcoming_df.groupby("league", as_index=False)
            .agg(
                total_matches=("match_id", "count"),
                next_match_time=("match_time", "min"),
            )
            .sort_values(["next_match_time", "league"], kind="stable")
            .reset_index(drop=True)
        )

    if not os.environ.get("MOTHERDUCK_TOKEN"):
        return _with_league_labels(_empty_upcoming_leagues_df())

    local_df, _ = _load_local_upcoming_matches_cached()
    if local_df.empty:
        return _with_league_labels(
            pd.DataFrame(columns=["league", "total_matches", "next_match_time"])
        )
    return _with_league_labels(
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
            SELECT fetched_at, row_count, event_count, pages_fetched, lookback_days, source_path, source
            FROM web_upcoming_matches_live_meta
            LIMIT 1
            """
        ).fetchdf()
        if len(remote_meta) > 0:
            row = remote_meta.iloc[0].to_dict()
            row["source"] = row.get("source") or "motherduck_official_schedule"
            return row
    except duckdb.Error:
        pass

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

    _, official_upcoming_df, official_meta = _load_local_official_schedule_cached()
    if not official_upcoming_df.empty:
        return {
            "fetched_at": official_meta.get("fetched_at"),
            "row_count": int(
                official_meta.get("upcoming_row_count", len(official_upcoming_df))
            ),
            "event_count": int(official_meta.get("event_count", 0)),
            "pages_fetched": int(official_meta.get("pages_fetched", 0)),
            "lookback_days": int(official_meta.get("recent_lookback_days", 0)),
            "source_path": official_meta.get("path"),
            "source": official_meta.get("source", "riot_official_schedule"),
        }

    if not os.environ.get("MOTHERDUCK_TOKEN"):
        return {
            "fetched_at": None,
            "row_count": 0,
            "source_path": str(DATA_DIR / "bronze" / "official" / "web_schedule.json"),
            "source": "local_official_schedule_empty",
        }

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


def get_home_recent_series(limit: int = 20):
    """Return recent matches for the home dashboard, preferring the fast-lane feed."""
    con = _get_persistent_con()
    try:
        remote_df = con.execute(
            f"""
            SELECT
                match_date,
                league,
                team1,
                team2,
                score,
                series_winner,
                series_format,
                tournament_phase
            FROM web_recent_matches_live
            ORDER BY match_time DESC, league, team1, team2
            LIMIT {limit}
            """
        ).fetchdf()
        if len(remote_df) > 0:
            return _with_league_labels(remote_df)
    except duckdb.Error:
        pass

    recent_df, _, _ = _load_local_official_schedule_cached()
    if not recent_df.empty:
        local_recent = recent_df.rename(columns={"match_time": "match_date"})
        return _with_league_labels(
            local_recent[
                [
                    "match_date",
                    "league",
                    "team1",
                    "team2",
                    "score",
                    "series_winner",
                    "series_format",
                    "tournament_phase",
                ]
            ]
            .sort_values(
                ["match_date", "league", "team1", "team2"],
                ascending=[False, True, True, True],
                kind="stable",
            )
            .head(limit)
            .reset_index(drop=True)
        )

    return get_recent_series(limit=limit)


def get_home_analytics_summary():
    """Return top-line dashboard metrics for the home page."""
    summary = {
        "completed_series_7d": 0,
        "completed_series_30d": 0,
        "active_leagues_30d": 0,
        "tracked_games_30d": 0,
        "tracked_teams_90d": 0,
        "avg_game_minutes_30d": 0.0,
        "avg_total_kills_30d": 0.0,
        "upcoming_matches": 0,
        "recommended_edges": 0,
    }

    try:
        series_row = query_one(
            """
            SELECT
                SUM(
                    CASE
                        WHEN match_date >= CURRENT_DATE - INTERVAL '7 days' THEN 1
                        ELSE 0
                    END
                ) AS completed_series_7d,
                SUM(
                    CASE
                        WHEN match_date >= CURRENT_DATE - INTERVAL '30 days' THEN 1
                        ELSE 0
                    END
                ) AS completed_series_30d,
                COUNT(
                    DISTINCT CASE
                        WHEN match_date >= CURRENT_DATE - INTERVAL '30 days' THEN league
                        ELSE NULL
                    END
                ) AS active_leagues_30d
            FROM series
            """
        )
        if series_row:
            for key in (
                "completed_series_7d",
                "completed_series_30d",
                "active_leagues_30d",
            ):
                summary[key] = int(series_row.get(key) or 0)
    except duckdb.Error:
        pass

    try:
        games_row = query_one(
            """
            SELECT
                COUNT(*) AS tracked_games_30d,
                COUNT(DISTINCT teamname) AS tracked_teams_90d,
                ROUND(AVG(CAST(gamelength AS DOUBLE)) / 60.0, 1) AS avg_game_minutes_30d,
                ROUND(
                    AVG(CAST(COALESCE(teamkills, 0) + COALESCE(teamdeaths, 0) AS DOUBLE)),
                    1
                ) AS avg_total_kills_30d
            FROM games
            WHERE game_date >= CURRENT_DATE - INTERVAL '30 days'
            """
        )
        if games_row:
            summary["tracked_games_30d"] = int(games_row.get("tracked_games_30d") or 0)
            summary["tracked_teams_90d"] = int(games_row.get("tracked_teams_90d") or 0)
            summary["avg_game_minutes_30d"] = float(
                games_row.get("avg_game_minutes_30d") or 0.0
            )
            summary["avg_total_kills_30d"] = float(
                games_row.get("avg_total_kills_30d") or 0.0
            )
    except duckdb.Error:
        pass

    try:
        upcoming_leagues = get_upcoming_match_leagues()
        if upcoming_leagues is not None and not upcoming_leagues.empty:
            summary["upcoming_matches"] = int(
                pd.to_numeric(
                    upcoming_leagues["total_matches"],
                    errors="coerce",
                )
                .fillna(0)
                .sum()
            )
    except Exception:
        pass

    try:
        edge_meta = get_edge_board_meta()
        summary["recommended_edges"] = int(
            edge_meta.get("recommendation_count", 0) or 0
        )
    except Exception:
        pass

    return summary


def get_power_rankings(limit: int = 10, days: int = 45):
    """Return a compact team power ranking based on recent form and control metrics."""
    empty = pd.DataFrame(
        columns=[
            "rank",
            "teamname",
            "league",
            "league_label",
            "games",
            "wins",
            "winrate",
            "last5_games",
            "last5_wins",
            "last5_winrate",
            "avg_kill_diff",
            "avg_tower_diff",
            "avg_dragon_diff",
            "first_blood_pct",
            "first_tower_pct",
            "power_score",
            "trend_label",
        ]
    )

    try:
        rankings_df = query_df(
            f"""
            WITH recent AS (
                SELECT
                    teamname,
                    league,
                    game_date,
                    gameid,
                    CAST(result AS DOUBLE) AS result,
                    CAST(COALESCE(teamkills, 0) - COALESCE(teamdeaths, 0) AS DOUBLE) AS kill_diff,
                    CAST(COALESCE(towers, 0) - COALESCE(opp_towers, 0) AS DOUBLE) AS tower_diff,
                    CAST(COALESCE(dragons, 0) - COALESCE(opp_dragons, 0) AS DOUBLE) AS dragon_diff,
                    CAST(COALESCE(firstblood, 0) AS DOUBLE) AS first_blood,
                    CAST(COALESCE(firsttower, 0) AS DOUBLE) AS first_tower,
                    ROW_NUMBER() OVER (
                        PARTITION BY teamname
                        ORDER BY game_date DESC, gameid DESC
                    ) AS rn
                FROM games
                WHERE game_date >= CURRENT_DATE - INTERVAL '{days} days'
            ),
            aggregated AS (
                SELECT
                    teamname,
                    MAX(league) AS league,
                    COUNT(*) AS games,
                    CAST(SUM(result) AS INTEGER) AS wins,
                    ROUND(100.0 * AVG(result), 1) AS winrate,
                    COUNT(CASE WHEN rn <= 5 THEN 1 END) AS last5_games,
                    CAST(SUM(CASE WHEN rn <= 5 THEN result ELSE 0 END) AS INTEGER) AS last5_wins,
                    ROUND(
                        100.0 * SUM(CASE WHEN rn <= 5 THEN result ELSE 0 END)
                        / NULLIF(COUNT(CASE WHEN rn <= 5 THEN 1 END), 0),
                        1
                    ) AS last5_winrate,
                    ROUND(AVG(kill_diff), 1) AS avg_kill_diff,
                    ROUND(AVG(tower_diff), 1) AS avg_tower_diff,
                    ROUND(AVG(dragon_diff), 1) AS avg_dragon_diff,
                    ROUND(100.0 * AVG(first_blood), 1) AS first_blood_pct,
                    ROUND(100.0 * AVG(first_tower), 1) AS first_tower_pct,
                    ROUND(
                        AVG(result) * 100.0 * 0.45
                        + (
                            SUM(CASE WHEN rn <= 5 THEN result ELSE 0 END)
                            / NULLIF(COUNT(CASE WHEN rn <= 5 THEN 1 END), 0)
                        ) * 100.0 * 0.35
                        + AVG(kill_diff) * 6.0
                        + AVG(tower_diff) * 4.0
                        + AVG(dragon_diff) * 3.0
                        + AVG(first_blood) * 8.0
                        + AVG(first_tower) * 8.0,
                        1
                    ) AS power_score
                FROM recent
                GROUP BY teamname
                HAVING COUNT(*) >= 8
            )
            SELECT
                ROW_NUMBER() OVER (
                    ORDER BY power_score DESC, winrate DESC, teamname ASC
                ) AS rank,
                teamname,
                league,
                games,
                wins,
                winrate,
                last5_games,
                last5_wins,
                last5_winrate,
                avg_kill_diff,
                avg_tower_diff,
                avg_dragon_diff,
                first_blood_pct,
                first_tower_pct,
                power_score
            FROM aggregated
            ORDER BY power_score DESC, winrate DESC, teamname ASC
            LIMIT {limit}
            """
        )
    except duckdb.Error:
        return empty

    if rankings_df.empty:
        return empty

    rankings_df = _with_league_labels(rankings_df)

    def _trend_label(row: pd.Series) -> str:
        overall = float(row.get("winrate") or 0.0)
        recent = float(row.get("last5_winrate") or 0.0)
        if recent >= overall + 10.0:
            return "Surging"
        if recent <= overall - 10.0:
            return "Cooling"
        return "Stable"

    rankings_df["trend_label"] = rankings_df.apply(_trend_label, axis=1)
    return rankings_df


def get_league_trends(limit: int = 6, days: int = 30):
    """Return league-level pace and activity trends for the dashboard."""
    empty = pd.DataFrame(
        columns=[
            "league",
            "league_label",
            "games",
            "teams",
            "avg_total_kills",
            "avg_game_minutes",
            "kill_pace",
            "tempo_label",
        ]
    )

    try:
        trends_df = query_df(
            f"""
            SELECT
                league,
                COUNT(*) AS games,
                COUNT(DISTINCT teamname) AS teams,
                ROUND(
                    AVG(CAST(COALESCE(teamkills, 0) + COALESCE(teamdeaths, 0) AS DOUBLE)),
                    1
                ) AS avg_total_kills,
                ROUND(AVG(CAST(gamelength AS DOUBLE)) / 60.0, 1) AS avg_game_minutes,
                ROUND(
                    AVG(CAST(COALESCE(teamkills, 0) + COALESCE(teamdeaths, 0) AS DOUBLE))
                    / NULLIF(AVG(CAST(gamelength AS DOUBLE)) / 60.0, 0),
                    2
                ) AS kill_pace
            FROM games
            WHERE game_date >= CURRENT_DATE - INTERVAL '{days} days'
            GROUP BY league
            HAVING COUNT(*) >= 10
            ORDER BY games DESC, kill_pace DESC, league ASC
            LIMIT {limit}
            """
        )
    except duckdb.Error:
        return empty

    if trends_df.empty:
        return empty

    trends_df = _with_league_labels(trends_df)

    def _tempo_label(value: object) -> str:
        pace = float(value or 0.0)
        if pace >= 1.45:
            return "High tempo"
        if pace >= 1.25:
            return "Up-tempo"
        return "Measured"

    trends_df["tempo_label"] = trends_df["kill_pace"].apply(_tempo_label)
    return trends_df


def get_home_model_watchlist(limit: int = 6):
    """Return the strongest pre-match model convictions for the home dashboard."""
    empty = pd.DataFrame(
        columns=[
            "match_id",
            "match_time",
            "league",
            "league_label",
            "team1",
            "team2",
            "favorite_name",
            "favorite_prob_pct",
            "confidence_gap_pct",
            "team1_fair_odds",
            "team2_fair_odds",
            "angle_label",
        ]
    )

    upcoming_df = get_upcoming_matches(limit=max(limit * 3, 12))
    if upcoming_df is None or upcoming_df.empty:
        return empty

    match_ids = [item for item in upcoming_df["match_id"].tolist() if item]
    predictions_df = get_prob_win_matches(match_ids)
    if predictions_df is None or predictions_df.empty:
        return empty

    merged_df = upcoming_df.merge(predictions_df, on="match_id", how="inner")
    if merged_df.empty:
        return empty

    if "winner_available" in merged_df.columns:
        merged_df = merged_df.loc[merged_df["winner_available"].fillna(False)].copy()
    if merged_df.empty:
        return empty

    merged_df["team1_win_pct"] = pd.to_numeric(
        merged_df.get("team1_win_pct"),
        errors="coerce",
    ).fillna(0.0)
    merged_df["team2_win_pct"] = pd.to_numeric(
        merged_df.get("team2_win_pct"),
        errors="coerce",
    ).fillna(0.0)
    merged_df["confidence_gap_pct"] = (
        merged_df["team1_win_pct"] - merged_df["team2_win_pct"]
    ).abs()
    merged_df["favorite_prob_pct"] = merged_df[
        ["team1_win_pct", "team2_win_pct"]
    ].max(axis=1)
    merged_df["favorite_name"] = merged_df.apply(
        lambda row: row.get("favorite_name")
        or (
            row.get("team1")
            if float(row.get("team1_win_pct") or 0.0)
            >= float(row.get("team2_win_pct") or 0.0)
            else row.get("team2")
        ),
        axis=1,
    )

    def _angle_label(value: object) -> str:
        gap = float(value or 0.0)
        if gap >= 30.0:
            return "Strong favorite"
        if gap <= 8.0:
            return "Coin flip"
        return "Lean"

    merged_df["angle_label"] = merged_df["confidence_gap_pct"].apply(_angle_label)
    merged_df = merged_df.sort_values(
        ["confidence_gap_pct", "match_time"],
        ascending=[False, True],
        kind="stable",
    )

    return _with_league_labels(
        merged_df[
            [
                "match_id",
                "match_time",
                "league",
                "team1",
                "team2",
                "favorite_name",
                "favorite_prob_pct",
                "confidence_gap_pct",
                "team1_fair_odds",
                "team2_fair_odds",
                "angle_label",
            ]
        ]
        .head(limit)
        .reset_index(drop=True)
    )


def get_home_edge_highlights(limit: int = 5):
    """Return the best current betting edges for the dashboard."""
    empty = pd.DataFrame(
        columns=[
            "captured_at",
            "match_time",
            "league_code",
            "league_code_label",
            "team1_name",
            "team2_name",
            "bookmaker",
            "recommended_selection",
            "recommended_odds",
            "recommended_fair_odds",
            "recommended_edge_pct",
            "recommended_ev_pct",
            "recommended_kelly_pct",
            "recommendation_reason",
        ]
    )

    try:
        edge_df = get_edge_board(limit=max(limit * 3, 12), recommended_only=True)
    except Exception:
        return empty

    if edge_df is None or edge_df.empty:
        return empty

    sort_columns = [
        col
        for col in ("recommended_ev_pct", "recommended_edge_pct", "match_time")
        if col in edge_df.columns
    ]
    if sort_columns:
        ascending = [False, False, True][: len(sort_columns)]
        edge_df = edge_df.sort_values(sort_columns, ascending=ascending, kind="stable")

    columns = [col for col in empty.columns if col in edge_df.columns]
    return edge_df[columns].head(limit).reset_index(drop=True)


def get_recent_series(limit: int = 20):
    # Prefer the curated series table because the expandable match detail depends on
    # the corresponding games/players rows existing in the same dataset.
    series_df = _with_league_labels(
        query_df(
            f"""
            SELECT match_date, league, team1, team2, score, series_winner,
                   series_format, tournament_phase
            FROM series
            ORDER BY match_date DESC
            LIMIT {limit}
        """
        )
    )
    if series_df is not None and not series_df.empty:
        return series_df

    con = _get_persistent_con()
    try:
        remote_df = con.execute(
            f"""
            SELECT
                match_date,
                league,
                team1,
                team2,
                score,
                series_winner,
                series_format,
                tournament_phase
            FROM web_recent_matches_live
            ORDER BY match_time DESC, league, team1, team2
            LIMIT {limit}
            """
        ).fetchdf()
        if len(remote_df) > 0:
            return _with_league_labels(remote_df)
    except duckdb.Error:
        pass

    recent_df, _, _ = _load_local_official_schedule_cached()
    if not recent_df.empty:
        local_recent = recent_df.rename(columns={"match_time": "match_date"})
        return _with_league_labels(
            local_recent[
                [
                    "match_date",
                    "league",
                    "team1",
                    "team2",
                    "score",
                    "series_winner",
                    "series_format",
                    "tournament_phase",
                ]
            ]
            .sort_values(
                ["match_date", "league", "team1", "team2"],
                ascending=[False, True, True, True],
                kind="stable",
            )
            .head(limit)
            .reset_index(drop=True)
        )

    return series_df


def get_active_leagues():
    con = _get_persistent_con()
    try:
        remote_df = con.execute(
            """
            SELECT league, MAX(match_time) AS last_match, COUNT(*) AS total_series
            FROM web_recent_matches_live
            GROUP BY league
            ORDER BY last_match DESC
            """
        ).fetchdf()
        if len(remote_df) > 0:
            return _with_league_labels(remote_df)
    except duckdb.Error:
        pass

    recent_df, _, _ = _load_local_official_schedule_cached()
    if not recent_df.empty:
        return _with_league_labels(
            recent_df.groupby("league", as_index=False)
            .agg(last_match=("match_time", "max"), total_series=("match_id", "count"))
            .sort_values("last_match", ascending=False, kind="stable")
            .reset_index(drop=True)
        )

    return _with_league_labels(
        query_df(
            """
            SELECT league, MAX(match_date) as last_match, COUNT(*) as total_series
            FROM series
            WHERE match_date >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY league
            ORDER BY last_match DESC
        """
        )
    )


def search_players(term: str):
    safe = term.replace("'", "''")
    return _with_league_labels(
        query_df(
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
    )


def search_teams(term: str):
    safe = term.replace("'", "''")
    return _with_league_labels(
        query_df(
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
    return _with_league_labels(
        query_df(
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
    )


def get_player_role_benchmark(
    name: str,
    year: int = None,
    split: str = None,
    min_games: int = 8,
):
    """Benchmark a player against peers in the same role."""
    info = get_player_info(name)
    position = str(info.get("position") or "") if info else ""
    if not position:
        return {
            "available": False,
            "position": "",
            "peer_count": 0,
            "scope_label": "",
            "impact_index": 0.0,
            "impact_label": "Unavailable",
            "impact_note": "No role benchmark is available for this player yet.",
            "metrics": [],
        }

    safe_name = name.replace("'", "''")
    safe_position = position.replace("'", "''")
    conditions = [f"LOWER(position) = LOWER('{safe_position}')"]
    if year:
        conditions.append(f"year = {year}")
    if split:
        split_safe = split.replace("'", "''")
        conditions.append(f"split = '{split_safe}'")
    where = " AND ".join(conditions)

    peers = query_df(
        f"""
        WITH peer_pool AS (
            SELECT
                playername,
                COUNT(*) AS games,
                ROUND(AVG(CAST(dpm AS DOUBLE)), 1) AS avg_dpm,
                ROUND(AVG(CAST(damageshare AS DOUBLE)) * 100.0, 1) AS avg_damage_share_pct,
                ROUND(AVG(CAST(earnedgoldshare AS DOUBLE)) * 100.0, 1) AS avg_gold_share_pct,
                ROUND(AVG(CAST(golddiffat15 AS DOUBLE)), 1) AS avg_gd15,
                ROUND(AVG(CAST(visionscore AS DOUBLE)), 1) AS avg_vision,
                ROUND(AVG(CAST(cspm AS DOUBLE)), 2) AS avg_cspm,
                ROUND(
                    AVG(
                        COALESCE(CAST(firstbloodkill AS DOUBLE), 0.0)
                        + COALESCE(CAST(firstbloodassist AS DOUBLE), 0.0)
                    ) * 100.0,
                    1
                ) AS avg_fb_involvement_pct
            FROM players
            WHERE {where}
            GROUP BY playername
            HAVING COUNT(*) >= {min_games} OR LOWER(playername) = LOWER('{safe_name}')
        )
        SELECT *
        FROM peer_pool
        ORDER BY games DESC, playername
    """
    )

    if peers.empty:
        return {
            "available": False,
            "position": position,
            "peer_count": 0,
            "scope_label": f"{year} {split}".strip() if year or split else "All available games",
            "impact_index": 0.0,
            "impact_label": "Unavailable",
            "impact_note": "No role benchmark is available for this slice yet.",
            "metrics": [],
        }

    target_mask = peers["playername"].astype(str).str.lower() == name.lower()
    if not target_mask.any():
        return {
            "available": False,
            "position": position,
            "peer_count": int(len(peers)),
            "scope_label": f"{year} {split}".strip() if year or split else "All available games",
            "impact_index": 0.0,
            "impact_label": "Unavailable",
            "impact_note": "The selected slice does not have enough games for a benchmark.",
            "metrics": [],
        }

    target = peers.loc[target_mask].iloc[0]
    peer_pool = peers.loc[~target_mask].copy()
    if peer_pool.empty:
        peer_pool = peers.copy()

    metrics_config = [
        ("avg_dpm", "DPM", 0),
        ("avg_damage_share_pct", "Damage Share", 1),
        ("avg_gold_share_pct", "Gold Share", 1),
        ("avg_gd15", "GD@15", 0),
        ("avg_cspm", "CS/min", 1),
        ("avg_vision", "Vision", 1),
        ("avg_fb_involvement_pct", "First Blood Involvement", 1),
    ]

    def _percentile(series: pd.Series, player_value: float) -> float:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if clean.empty:
            return 50.0
        if len(clean) == 1:
            return 100.0 if abs(_safe_float(clean.iloc[0]) - player_value) < 1e-9 else 50.0
        ranked = clean.rank(pct=True, method="average")
        target_idx = clean.index[clean == player_value]
        if len(target_idx) > 0:
            return round(float(ranked.loc[target_idx].max() * 100.0), 1)
        below = float((clean < player_value).sum())
        equal = float((clean == player_value).sum())
        return round(((below + 0.5 * equal) / len(clean)) * 100.0, 1)

    metric_rows: list[dict[str, object]] = []
    percentiles_for_index: dict[str, float] = {}
    for column, label, decimals in metrics_config:
        player_value = round(_safe_float(target.get(column)), decimals)
        peer_avg = round(
            float(pd.to_numeric(peer_pool[column], errors="coerce").dropna().mean())
            if column in peer_pool and not peer_pool.empty
            else player_value,
            decimals,
        )
        delta = round(player_value - peer_avg, decimals)
        percentile = _percentile(pd.to_numeric(peers[column], errors="coerce"), player_value)
        percentiles_for_index[column] = percentile

        if percentile >= 80.0:
            strength_label = "Elite"
        elif percentile >= 65.0:
            strength_label = "Strong"
        elif percentile >= 35.0:
            strength_label = "Middle band"
        else:
            strength_label = "Below field"

        delta_threshold = 10.0 if column == "avg_dpm" else (60.0 if column == "avg_gd15" else 1.0)
        if delta > delta_threshold:
            tone = "up"
        elif delta < -delta_threshold:
            tone = "down"
        else:
            tone = "flat"

        metric_rows.append(
            {
                "metric": label,
                "player_value": player_value,
                "peer_avg": peer_avg,
                "delta": delta,
                "delta_text": f"{delta:+.{decimals}f}" if decimals else f"{int(delta):+d}",
                "percentile": percentile,
                "percentile_label": f"{_ordinal_label(percentile)} pct",
                "strength_label": strength_label,
                "tone": tone,
            }
        )

    if position in {"sup", "jng"}:
        index_columns = [
            "avg_vision",
            "avg_fb_involvement_pct",
            "avg_gd15",
            "avg_gold_share_pct",
        ]
    else:
        index_columns = [
            "avg_dpm",
            "avg_damage_share_pct",
            "avg_gd15",
            "avg_cspm",
        ]
    impact_index = round(
        sum(percentiles_for_index.get(column, 50.0) for column in index_columns)
        / len(index_columns),
        1,
    )
    if impact_index >= 80.0:
        impact_label = "Top quartile role impact"
        impact_note = "This player rates near the top of the position on the current analytical mix."
    elif impact_index >= 65.0:
        impact_label = "Strong role profile"
        impact_note = "The player is beating the role baseline on multiple axes."
    elif impact_index >= 45.0:
        impact_label = "On-role baseline"
        impact_note = "The profile sits close to the middle of the position."
    else:
        impact_label = "Below current role baseline"
        impact_note = "This slice trails the positional baseline on several core metrics."

    scope_label = f"{year} {split}".strip() if year or split else "All available games"
    return {
        "available": True,
        "position": position,
        "peer_count": int(len(peer_pool)),
        "scope_label": scope_label,
        "impact_index": impact_index,
        "impact_label": impact_label,
        "impact_note": impact_note,
        "metrics": metric_rows,
    }


def get_player_analytics_summary(
    name: str,
    year: int = None,
    split: str = None,
    recent_window: int = 12,
):
    """Return an analytics-oriented snapshot for one player."""
    safe = name.replace("'", "''")
    conditions = [f"LOWER(playername) = LOWER('{safe}')"]
    if year:
        conditions.append(f"year = {year}")
    if split:
        split_safe = split.replace("'", "''")
        conditions.append(f"split = '{split_safe}'")
    where = " AND ".join(conditions)

    player_df = query_df(
        f"""
        SELECT
            game_date,
            game,
            teamname,
            league,
            position,
            champion,
            result,
            kills,
            deaths,
            assists,
            dpm,
            cspm,
            visionscore,
            vspm,
            damageshare,
            earnedgoldshare,
            earned_gpm,
            golddiffat15,
            xpdiffat15,
            csdiffat15,
            firstbloodkill,
            firstbloodassist
        FROM players
        WHERE {where}
        ORDER BY game_date DESC, game DESC
    """
    )

    if player_df.empty:
        return {
            "available": False,
            "games": 0,
            "wins": 0,
            "winrate": 0.0,
            "recent_games": 0,
            "recent_winrate": 0.0,
            "previous_winrate": 0.0,
            "winrate_delta": 0.0,
            "avg_kda": 0.0,
            "avg_dpm": 0.0,
            "avg_cspm": 0.0,
            "avg_gd15": 0.0,
            "avg_xd15": 0.0,
            "avg_csd15": 0.0,
            "avg_damage_share_pct": 0.0,
            "avg_gold_share_pct": 0.0,
            "avg_earned_gpm": 0.0,
            "avg_vision": 0.0,
            "avg_vspm": 0.0,
            "first_blood_involvement_pct": 0.0,
            "unique_champions": 0,
            "top_champion_share_pct": 0.0,
            "top3_champion_share_pct": 0.0,
            "impact_index": 0.0,
            "trend_label": "Unavailable",
            "trend_note": "No games are available for this player in the selected slice.",
            "lane_label": "Unavailable",
            "lane_note": "Lane-state metrics are not available.",
            "role_label": "Unavailable",
            "role_note": "Role profile is not available.",
            "setup_label": "Unavailable",
            "setup_note": "Setup profile is not available.",
            "pool_label": "Unavailable",
            "pool_note": "Champion-pool read is not available.",
        }

    def _mean(frame: pd.DataFrame, column: str, scale: float = 1.0, digits: int = 1) -> float:
        if frame.empty or column not in frame.columns:
            return 0.0
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            return 0.0
        return round(float(values.mean()) * scale, digits)

    def _frame_kda(frame: pd.DataFrame) -> float:
        if frame.empty:
            return 0.0
        kills = pd.to_numeric(frame["kills"], errors="coerce").fillna(0.0)
        deaths = pd.to_numeric(frame["deaths"], errors="coerce").fillna(0.0)
        assists = pd.to_numeric(frame["assists"], errors="coerce").fillna(0.0)
        series = (kills + assists) / deaths.where(deaths > 0, 1.0)
        return round(float(series.mean()), 2) if not series.empty else 0.0

    games = int(len(player_df))
    recent_games = min(recent_window, games)
    recent_df = player_df.head(recent_games).copy()
    previous_df = player_df.iloc[recent_games : recent_games * 2].copy()

    winrate = round(float(pd.to_numeric(player_df["result"], errors="coerce").fillna(0.0).mean()) * 100.0, 1)
    recent_winrate = round(float(pd.to_numeric(recent_df["result"], errors="coerce").fillna(0.0).mean()) * 100.0, 1)
    previous_winrate = (
        round(float(pd.to_numeric(previous_df["result"], errors="coerce").fillna(0.0).mean()) * 100.0, 1)
        if not previous_df.empty
        else recent_winrate
    )
    winrate_delta = round(recent_winrate - previous_winrate, 1)
    recent_dpm = _mean(recent_df, "dpm", digits=0)
    previous_dpm = _mean(previous_df, "dpm", digits=0) if not previous_df.empty else recent_dpm
    dpm_delta = round(recent_dpm - previous_dpm, 1)

    avg_damage_share_pct = _mean(player_df, "damageshare", scale=100.0, digits=1)
    avg_gold_share_pct = _mean(player_df, "earnedgoldshare", scale=100.0, digits=1)
    avg_vision = _mean(player_df, "visionscore", digits=1)
    avg_vspm = _mean(player_df, "vspm", digits=2)
    first_blood_involvement_pct = round(
        float(
            (
                pd.to_numeric(player_df["firstbloodkill"], errors="coerce").fillna(0.0)
                + pd.to_numeric(player_df["firstbloodassist"], errors="coerce").fillna(0.0)
            ).mean()
        )
        * 100.0,
        1,
    )

    champion_counts = player_df["champion"].dropna().astype(str).value_counts()
    unique_champions = int(champion_counts.size)
    top_champion_share_pct = (
        round(float(champion_counts.iloc[0]) / games * 100.0, 1)
        if not champion_counts.empty
        else 0.0
    )
    top3_champion_share_pct = (
        round(float(champion_counts.head(3).sum()) / games * 100.0, 1)
        if not champion_counts.empty
        else 0.0
    )

    benchmark = get_player_role_benchmark(name, year=year, split=split)
    impact_index = _safe_float(benchmark.get("impact_index"), 50.0)

    position = str(player_df.iloc[0].get("position") or "").lower()
    avg_gd15 = _mean(player_df, "golddiffat15", digits=0)
    avg_xd15 = _mean(player_df, "xpdiffat15", digits=0)
    avg_csd15 = _mean(player_df, "csdiffat15", digits=1)
    avg_kda = _frame_kda(player_df)
    avg_cspm = _mean(player_df, "cspm", digits=1)
    avg_earned_gpm = _mean(player_df, "earned_gpm", digits=0)

    if games < 6:
        trend_label = "Thin sample"
        trend_note = "There are not enough games in this slice for a stable trend read."
    elif winrate_delta >= 12.0 or dpm_delta >= 60.0:
        trend_label = "Heating up"
        trend_note = "Recent output is materially ahead of the previous window."
    elif winrate_delta <= -12.0 or dpm_delta <= -60.0:
        trend_label = "Cooling off"
        trend_note = "Recent results and output have slipped versus the prior window."
    else:
        trend_label = "Stable"
        trend_note = "Recent performance is broadly aligned with the previous run of games."

    if avg_gd15 >= 250 and avg_xd15 >= 150:
        lane_label = "Lane driver"
        lane_note = "The player tends to create measurable lane advantages before 15 minutes."
    elif avg_gd15 <= -150 and avg_xd15 <= -100:
        lane_label = "Under pressure"
        lane_note = "Early deficits are a recurring feature of the lane phase."
    elif avg_csd15 >= 6.0:
        lane_label = "Farm edge"
        lane_note = "The player consistently builds a CS edge even without massive gold gaps."
    else:
        lane_label = "Neutral lane"
        lane_note = "The lane phase is usually played to parity before the game opens."

    if position in {"mid", "bot", "top"} and avg_damage_share_pct >= 27.0 and avg_gold_share_pct >= 22.0:
        role_label = "Primary carry"
        role_note = "The player is taking a large share of team resources and converting them into output."
    elif position in {"jng", "sup"} and first_blood_involvement_pct >= 28.0 and avg_vision >= 40.0:
        role_label = "Setup engine"
        role_note = "The player is heavily involved in openings and map setup for the team."
    elif avg_damage_share_pct <= 19.0 and avg_gold_share_pct <= 20.0:
        role_label = "Low-resource connector"
        role_note = "The profile leans toward enabling the rest of the lineup more than carrying itself."
    else:
        role_label = "Balanced role"
        role_note = "Resource share and output sit in a middle band for the role."

    if avg_vspm >= 2.2 or avg_vision >= 55.0:
        setup_label = "Map anchor"
        setup_note = "Vision volume is a major part of the player's value."
    elif first_blood_involvement_pct >= 30.0:
        setup_label = "Early instigator"
        setup_note = "The player is regularly present in first-blood sequences."
    elif avg_vspm <= 1.2 and avg_vision <= 22.0:
        setup_label = "Light setup load"
        setup_note = "The role allocation is not centered on vision and setup burden."
    else:
        setup_label = "Standard setup"
        setup_note = "Vision and opening involvement are close to the middle of the role."

    if unique_champions >= 10 and top3_champion_share_pct <= 48.0:
        pool_label = "Wide pool"
        pool_note = "The player wins with a broad spread of picks rather than a narrow comfort core."
    elif top_champion_share_pct >= 28.0:
        pool_label = "Comfort specialist"
        pool_note = "A single champion takes a large share of the draft profile."
    elif top3_champion_share_pct >= 65.0:
        pool_label = "Condensed pool"
        pool_note = "The player is leaning on a concentrated cluster of core picks."
    else:
        pool_label = "Flexible core"
        pool_note = "There is a stable comfort set without becoming overly narrow."

    return {
        "available": True,
        "position": position,
        "games": games,
        "wins": int(pd.to_numeric(player_df["result"], errors="coerce").fillna(0.0).sum()),
        "winrate": winrate,
        "recent_games": recent_games,
        "recent_winrate": recent_winrate,
        "previous_winrate": previous_winrate,
        "winrate_delta": winrate_delta,
        "avg_kda": avg_kda,
        "avg_dpm": _mean(player_df, "dpm", digits=0),
        "avg_cspm": avg_cspm,
        "avg_gd15": avg_gd15,
        "avg_xd15": avg_xd15,
        "avg_csd15": avg_csd15,
        "avg_damage_share_pct": avg_damage_share_pct,
        "avg_gold_share_pct": avg_gold_share_pct,
        "avg_earned_gpm": avg_earned_gpm,
        "avg_vision": avg_vision,
        "avg_vspm": avg_vspm,
        "first_blood_involvement_pct": first_blood_involvement_pct,
        "unique_champions": unique_champions,
        "top_champion_share_pct": top_champion_share_pct,
        "top3_champion_share_pct": top3_champion_share_pct,
        "impact_index": impact_index,
        "trend_label": trend_label,
        "trend_note": trend_note,
        "lane_label": lane_label,
        "lane_note": lane_note,
        "role_label": role_label,
        "role_note": role_note,
        "setup_label": setup_label,
        "setup_note": setup_note,
        "pool_label": pool_label,
        "pool_note": pool_note,
    }


def get_player_champion_analytics(
    name: str,
    year: int = None,
    split: str = None,
    limit: int = 10,
):
    """Return champion-signature analytics for one player."""
    safe = name.replace("'", "''")
    conditions = [f"LOWER(playername) = LOWER('{safe}')"]
    if year:
        conditions.append(f"year = {year}")
    if split:
        split_safe = split.replace("'", "''")
        conditions.append(f"split = '{split_safe}'")
    where = " AND ".join(conditions)

    total_row = query_one(
        f"""
        SELECT COUNT(*) AS games
        FROM players
        WHERE {where}
    """
    ) or {"games": 0}
    total_games = int(total_row.get("games") or 0)
    if total_games == 0:
        return pd.DataFrame()

    champions_df = query_df(
        f"""
        SELECT
            champion,
            COUNT(*) AS games,
            SUM(result) AS wins,
            ROUND(100.0 * SUM(result) / COUNT(*), 1) AS winrate,
            ROUND(AVG(kills), 1) AS avg_k,
            ROUND(AVG(deaths), 1) AS avg_d,
            ROUND(AVG(assists), 1) AS avg_a,
            ROUND(AVG(CAST(dpm AS DOUBLE)), 0) AS avg_dpm,
            ROUND(AVG(CAST(golddiffat15 AS DOUBLE)), 0) AS avg_gd15,
            ROUND(AVG(CAST(damageshare AS DOUBLE)) * 100.0, 1) AS avg_damage_share_pct,
            ROUND(AVG(CAST(earnedgoldshare AS DOUBLE)) * 100.0, 1) AS avg_gold_share_pct
        FROM players
        WHERE {where}
        GROUP BY champion
        ORDER BY games DESC, winrate DESC
        LIMIT {limit}
    """
    )
    if champions_df.empty:
        return champions_df

    champions_df = champions_df.copy()
    champions_df["pick_share_pct"] = champions_df["games"].apply(
        lambda value: round(float(value) / total_games * 100.0, 1)
    )

    def _profile_label(row: pd.Series) -> str:
        avg_dpm = _safe_float(row.get("avg_dpm"))
        avg_gd15 = _safe_float(row.get("avg_gd15"))
        damage_share = _safe_float(row.get("avg_damage_share_pct"))
        winrate = _safe_float(row.get("winrate"))
        if damage_share >= 28.0 and avg_dpm >= 600:
            return "Carry pick"
        if avg_gd15 >= 180:
            return "Lane pressure"
        if winrate >= 65.0 and _safe_float(row.get("games")) >= 4:
            return "Winning comfort"
        if damage_share <= 18.0:
            return "Utility look"
        return "Core option"

    champions_df["profile_label"] = champions_df.apply(_profile_label, axis=1)
    return champions_df


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
    return _with_league_labels(
        query_df(
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
    )


def get_team_titles(name: str):
    safe = name.replace("'", "''")
    return _with_league_labels(
        query_df(
            f"""
        SELECT year, league, split, runner_up, final_score
        FROM champions
        WHERE LOWER(champion) = LOWER('{safe}')
        ORDER BY final_date DESC
    """
        )
    )


def get_team_recent_series(name: str, limit: int = 20):
    safe = name.replace("'", "''")
    return _with_league_labels(
        query_df(
            f"""
        SELECT match_date, league, team1, team2, score,
               series_winner, series_format, tournament_phase
        FROM series
        WHERE LOWER(team1) LIKE LOWER('%{safe}%') OR LOWER(team2) LIKE LOWER('%{safe}%')
        ORDER BY match_date DESC
        LIMIT {limit}
    """
        )
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


def get_team_analytics_summary(name: str, recent_days: int = 60):
    """Return a compact analytical summary for one team page."""
    safe = name.replace("'", "''")
    previous_start = recent_days * 2

    recent_row = query_one(
        f"""
        SELECT
            COUNT(*) AS games_recent,
            CAST(SUM(result) AS INTEGER) AS wins_recent,
            ROUND(100.0 * AVG(CAST(result AS DOUBLE)), 1) AS winrate_recent,
            ROUND(AVG(CAST(COALESCE(teamkills, 0) - COALESCE(teamdeaths, 0) AS DOUBLE)), 1) AS avg_kill_diff_recent,
            ROUND(AVG(CAST(COALESCE(towers, 0) - COALESCE(opp_towers, 0) AS DOUBLE)), 1) AS avg_tower_diff_recent,
            ROUND(AVG(CAST(COALESCE(dragons, 0) - COALESCE(opp_dragons, 0) AS DOUBLE)), 1) AS avg_dragon_diff_recent,
            ROUND(AVG(CAST(COALESCE(barons, 0) - COALESCE(opp_barons, 0) AS DOUBLE)), 1) AS avg_baron_diff_recent,
            ROUND(AVG(CAST(COALESCE(teamkills, 0) + COALESCE(teamdeaths, 0) AS DOUBLE)), 1) AS avg_total_kills_recent,
            ROUND(AVG(CAST(gamelength AS DOUBLE)) / 60.0, 1) AS avg_game_minutes_recent,
            ROUND(100.0 * AVG(CAST(COALESCE(firstblood, 0) AS DOUBLE)), 1) AS first_blood_pct,
            ROUND(100.0 * AVG(CAST(COALESCE(firsttower, 0) AS DOUBLE)), 1) AS first_tower_pct,
            ROUND(100.0 * AVG(CAST(COALESCE(firstdragon, 0) AS DOUBLE)), 1) AS first_dragon_pct,
            ROUND(
                100.0 * AVG(
                    CASE WHEN LOWER(COALESCE(side, '')) = 'blue' THEN CAST(result AS DOUBLE) ELSE NULL END
                ),
                1
            ) AS blue_side_winrate,
            COUNT(CASE WHEN LOWER(COALESCE(side, '')) = 'blue' THEN 1 END) AS blue_games,
            ROUND(
                100.0 * AVG(
                    CASE WHEN LOWER(COALESCE(side, '')) = 'red' THEN CAST(result AS DOUBLE) ELSE NULL END
                ),
                1
            ) AS red_side_winrate,
            COUNT(CASE WHEN LOWER(COALESCE(side, '')) = 'red' THEN 1 END) AS red_games
        FROM games
        WHERE LOWER(teamname) = LOWER('{safe}')
          AND game_date >= CURRENT_DATE - INTERVAL '{recent_days} days'
        """
    ) or {}

    previous_row = query_one(
        f"""
        SELECT
            COUNT(*) AS games_previous,
            ROUND(100.0 * AVG(CAST(result AS DOUBLE)), 1) AS winrate_previous
        FROM games
        WHERE LOWER(teamname) = LOWER('{safe}')
          AND game_date < CURRENT_DATE - INTERVAL '{recent_days} days'
          AND game_date >= CURRENT_DATE - INTERVAL '{previous_start} days'
        """
    ) or {}

    player_row = query_one(
        f"""
        SELECT
            ROUND(AVG(CAST(golddiffat15 AS DOUBLE)), 0) AS avg_gd15,
            ROUND(AVG(CAST(xpdiffat15 AS DOUBLE)), 0) AS avg_xd15,
            ROUND(AVG(CAST(csdiffat15 AS DOUBLE)), 1) AS avg_csd15,
            ROUND(AVG(CAST(visionscore AS DOUBLE)), 1) AS avg_vision,
            ROUND(AVG(CAST(damageshare AS DOUBLE)) * 100.0, 1) AS avg_damage_share_pct
        FROM players
        WHERE LOWER(teamname) = LOWER('{safe}')
          AND game_date >= CURRENT_DATE - INTERVAL '{recent_days} days'
        """
    ) or {}

    games_recent = int(recent_row.get("games_recent") or 0)
    winrate_recent = _safe_float(recent_row.get("winrate_recent"))
    winrate_previous = _safe_float(previous_row.get("winrate_previous"))
    winrate_delta = round(winrate_recent - winrate_previous, 1)
    avg_kill_diff_recent = _safe_float(recent_row.get("avg_kill_diff_recent"))
    avg_tower_diff_recent = _safe_float(recent_row.get("avg_tower_diff_recent"))
    avg_dragon_diff_recent = _safe_float(recent_row.get("avg_dragon_diff_recent"))
    avg_total_kills_recent = _safe_float(recent_row.get("avg_total_kills_recent"))
    avg_game_minutes_recent = _safe_float(recent_row.get("avg_game_minutes_recent"))
    first_blood_pct = _safe_float(recent_row.get("first_blood_pct"))
    first_tower_pct = _safe_float(recent_row.get("first_tower_pct"))
    first_dragon_pct = _safe_float(recent_row.get("first_dragon_pct"))
    blue_side_winrate = _safe_float(recent_row.get("blue_side_winrate"))
    red_side_winrate = _safe_float(recent_row.get("red_side_winrate"))
    side_bias_gap = round(blue_side_winrate - red_side_winrate, 1)
    objective_control_pct = round(
        (first_blood_pct + first_tower_pct + first_dragon_pct) / 3.0,
        1,
    )
    avg_gd15 = _safe_float(player_row.get("avg_gd15"))
    avg_xd15 = _safe_float(player_row.get("avg_xd15"))
    avg_csd15 = _safe_float(player_row.get("avg_csd15"))
    avg_vision = _safe_float(player_row.get("avg_vision"))
    avg_damage_share_pct = _safe_float(player_row.get("avg_damage_share_pct"))
    kill_pace = round(
        avg_total_kills_recent / avg_game_minutes_recent,
        2,
    ) if avg_game_minutes_recent > 0 else 0.0
    dominance_score = round(
        winrate_recent * 0.45
        + objective_control_pct * 0.25
        + avg_kill_diff_recent * 4.0
        + avg_tower_diff_recent * 3.0
        + avg_dragon_diff_recent * 2.5,
        1,
    )

    if games_recent == 0:
        return {
            "games_recent": 0,
            "wins_recent": 0,
            "winrate_recent": 0.0,
            "winrate_previous": 0.0,
            "winrate_delta": 0.0,
            "avg_kill_diff_recent": 0.0,
            "avg_tower_diff_recent": 0.0,
            "avg_dragon_diff_recent": 0.0,
            "avg_baron_diff_recent": 0.0,
            "avg_total_kills_recent": 0.0,
            "avg_game_minutes_recent": 0.0,
            "first_blood_pct": 0.0,
            "first_tower_pct": 0.0,
            "first_dragon_pct": 0.0,
            "objective_control_pct": 0.0,
            "blue_side_winrate": 0.0,
            "red_side_winrate": 0.0,
            "side_bias_gap": 0.0,
            "avg_gd15": 0.0,
            "avg_xd15": 0.0,
            "avg_csd15": 0.0,
            "avg_vision": 0.0,
            "avg_damage_share_pct": 0.0,
            "kill_pace": 0.0,
            "dominance_score": 0.0,
            "trend_label": "Insufficient sample",
            "trend_note": "No recent games available for a reliable trend read.",
            "early_game_label": "No read",
            "early_game_note": "Lane-state data is not sufficient yet.",
            "tempo_label": "No read",
            "tempo_note": "Not enough recent games to estimate team pace.",
            "side_bias_label": "No read",
            "side_bias_note": "Side splits need a usable sample on both blue and red.",
            "control_label": "No read",
            "control_note": "Objective-control signals are not available yet.",
        }

    if winrate_delta >= 8.0:
        trend_label = "Heating up"
        trend_note = "Recent win rate is materially above the previous window."
    elif winrate_delta <= -8.0:
        trend_label = "Cooling off"
        trend_note = "Recent results have slipped versus the previous window."
    else:
        trend_label = "Stable"
        trend_note = "Results are broadly in line with the prior window."

    if avg_gd15 >= 500 and avg_xd15 >= 250:
        early_game_label = "Fast starter"
        early_game_note = "The team regularly builds leads before the game opens up."
    elif avg_gd15 <= -250 and avg_xd15 <= -100:
        early_game_label = "Slow starter"
        early_game_note = "Early deficits are common; the team tends to scale back in later."
    else:
        early_game_label = "Balanced opener"
        early_game_note = "Early-game edges are modest and usually earned through execution."

    if kill_pace >= 0.95:
        tempo_label = "Skirmish heavy"
        tempo_note = "Games feature frequent fighting and a faster event rate."
    elif kill_pace <= 0.78:
        tempo_label = "Measured tempo"
        tempo_note = "The team plays lower-event games and leans on cleaner setups."
    else:
        tempo_label = "Controlled pace"
        tempo_note = "The team mixes proactive fights with structure-first setups."

    blue_games = int(recent_row.get("blue_games") or 0)
    red_games = int(recent_row.get("red_games") or 0)
    if blue_games >= 4 and red_games >= 4:
        if side_bias_gap >= 8.0:
            side_bias_label = "Blue leaning"
            side_bias_note = "Results are meaningfully better on blue side in the recent sample."
        elif side_bias_gap <= -8.0:
            side_bias_label = "Red resilient"
            side_bias_note = "The team is outperforming its blue-side record on red."
        else:
            side_bias_label = "Side balanced"
            side_bias_note = "Performance is stable regardless of side assignment."
    else:
        side_bias_label = "Thin sample"
        side_bias_note = "Side split read is still noisy with the current sample."

    if objective_control_pct >= 58.0 and avg_tower_diff_recent >= 1.0:
        control_label = "Objective-led"
        control_note = "Openers and structures are converting into reliable map control."
    elif avg_kill_diff_recent >= 4.0:
        control_label = "Snowballing"
        control_note = "The team creates kill pressure quickly and tends to accelerate ahead."
    elif objective_control_pct <= 46.0 and avg_tower_diff_recent < 0:
        control_label = "Reactive"
        control_note = "Early objectives are often conceded before the team stabilizes."
    else:
        control_label = "Contest-ready"
        control_note = "The team stays competitive across openings without one extreme identity."

    return {
        "games_recent": games_recent,
        "wins_recent": int(recent_row.get("wins_recent") or 0),
        "winrate_recent": winrate_recent,
        "winrate_previous": winrate_previous,
        "winrate_delta": winrate_delta,
        "avg_kill_diff_recent": avg_kill_diff_recent,
        "avg_tower_diff_recent": avg_tower_diff_recent,
        "avg_dragon_diff_recent": avg_dragon_diff_recent,
        "avg_baron_diff_recent": _safe_float(recent_row.get("avg_baron_diff_recent")),
        "avg_total_kills_recent": avg_total_kills_recent,
        "avg_game_minutes_recent": avg_game_minutes_recent,
        "first_blood_pct": first_blood_pct,
        "first_tower_pct": first_tower_pct,
        "first_dragon_pct": first_dragon_pct,
        "objective_control_pct": objective_control_pct,
        "blue_side_winrate": blue_side_winrate,
        "red_side_winrate": red_side_winrate,
        "side_bias_gap": side_bias_gap,
        "avg_gd15": avg_gd15,
        "avg_xd15": avg_xd15,
        "avg_csd15": avg_csd15,
        "avg_vision": avg_vision,
        "avg_damage_share_pct": avg_damage_share_pct,
        "kill_pace": kill_pace,
        "dominance_score": dominance_score,
        "trend_label": trend_label,
        "trend_note": trend_note,
        "early_game_label": early_game_label,
        "early_game_note": early_game_note,
        "tempo_label": tempo_label,
        "tempo_note": tempo_note,
        "side_bias_label": side_bias_label,
        "side_bias_note": side_bias_note,
        "control_label": control_label,
        "control_note": control_note,
    }


def get_team_player_impact(name: str, days: int = 120):
    """Return recent player-level impact metrics for one team."""
    safe = name.replace("'", "''")
    return query_df(
        f"""
        SELECT
            playername,
            position,
            COUNT(*) AS games,
            ROUND(100.0 * AVG(CAST(result AS DOUBLE)), 1) AS winrate,
            ROUND(AVG(CAST(dpm AS DOUBLE)), 0) AS avg_dpm,
            ROUND(AVG(CAST(damageshare AS DOUBLE)) * 100.0, 1) AS avg_damage_share_pct,
            ROUND(AVG(CAST(earnedgoldshare AS DOUBLE)) * 100.0, 1) AS avg_gold_share_pct,
            ROUND(AVG(CAST(golddiffat15 AS DOUBLE)), 0) AS avg_gd15,
            ROUND(AVG(CAST(csdiffat15 AS DOUBLE)), 1) AS avg_csd15,
            ROUND(AVG(CAST(visionscore AS DOUBLE)), 1) AS avg_vision
        FROM players
        WHERE LOWER(teamname) = LOWER('{safe}')
          AND game_date >= CURRENT_DATE - INTERVAL '{days} days'
        GROUP BY playername, position
        HAVING COUNT(*) >= 2
        ORDER BY CASE position
            WHEN 'top' THEN 1 WHEN 'jng' THEN 2 WHEN 'mid' THEN 3
            WHEN 'bot' THEN 4 WHEN 'sup' THEN 5 ELSE 6 END,
            games DESC,
            playername ASC
        """
    )


def get_team_signature_champions(name: str, limit: int = 10, days: int = 180):
    """Return the champions that most define a team's recent identity."""
    safe = name.replace("'", "''")
    return query_df(
        f"""
        SELECT
            champion,
            position,
            COUNT(*) AS games,
            ROUND(100.0 * AVG(CAST(result AS DOUBLE)), 1) AS winrate,
            ROUND(AVG(CAST(dpm AS DOUBLE)), 0) AS avg_dpm,
            ROUND(AVG(CAST(golddiffat15 AS DOUBLE)), 0) AS avg_gd15
        FROM players
        WHERE LOWER(teamname) = LOWER('{safe}')
          AND game_date >= CURRENT_DATE - INTERVAL '{days} days'
        GROUP BY champion, position
        HAVING COUNT(*) >= 3
        ORDER BY games DESC, winrate DESC, champion ASC
        LIMIT {limit}
        """
    )


def get_team_patch_profile(name: str, limit: int = 8):
    """Return patch-level recent performance for one team."""
    safe = name.replace("'", "''")
    return query_df(
        f"""
        SELECT
            CAST(patch AS VARCHAR) AS patch,
            COUNT(*) AS games,
            ROUND(100.0 * AVG(CAST(result AS DOUBLE)), 1) AS winrate,
            ROUND(AVG(CAST(COALESCE(teamkills, 0) - COALESCE(teamdeaths, 0) AS DOUBLE)), 1) AS avg_kill_diff,
            ROUND(AVG(CAST(gamelength AS DOUBLE)) / 60.0, 1) AS avg_minutes,
            ROUND(100.0 * AVG(CAST(COALESCE(firstdragon, 0) AS DOUBLE)), 1) AS first_dragon_pct
        FROM games
        WHERE LOWER(teamname) = LOWER('{safe}')
          AND patch IS NOT NULL
        GROUP BY patch
        HAVING COUNT(*) >= 2
        ORDER BY CAST(patch AS DOUBLE) DESC, games DESC
        LIMIT {limit}
        """
    )


# =============================================================
# TOURNAMENT
# =============================================================


def get_tournament_leagues():
    return _with_league_labels(
        query_df(
            """
        SELECT DISTINCT league, MIN(year) as from_year, MAX(year) as to_year,
               COUNT(DISTINCT year) as years
        FROM series
        GROUP BY league
        ORDER BY COUNT(*) DESC
    """
        )
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
            game_date,
            league,
            patch,
            game,
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


def get_game_team_backdrop(gameid: str, recent_days: int = 60):
    """Return recent profile snapshots for the two teams in one game."""
    teams_df = get_game_teams(gameid)
    if teams_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for team in teams_df.to_dict("records"):
        analytics = get_team_analytics_summary(
            str(team.get("teamname") or ""),
            recent_days=recent_days,
        )
        rows.append(
            {
                "teamname": team.get("teamname"),
                "side": team.get("side"),
                "won_game": bool(team.get("result")),
                "games_recent": int(analytics.get("games_recent") or 0),
                "winrate_recent": _safe_float(analytics.get("winrate_recent")),
                "winrate_delta": _safe_float(analytics.get("winrate_delta")),
                "dominance_score": _safe_float(analytics.get("dominance_score")),
                "avg_gd15": _safe_float(analytics.get("avg_gd15")),
                "objective_control_pct": _safe_float(
                    analytics.get("objective_control_pct")
                ),
                "trend_label": analytics.get("trend_label"),
                "trend_note": analytics.get("trend_note"),
                "tempo_label": analytics.get("tempo_label"),
                "tempo_note": analytics.get("tempo_note"),
                "control_label": analytics.get("control_label"),
                "control_note": analytics.get("control_note"),
            }
        )

    return pd.DataFrame(rows)


def get_game_analytics_summary(gameid: str, recent_days: int = 60):
    """Return a game-level analytic readout blending box score and recent team form."""
    teams_df = get_game_teams(gameid)
    if len(teams_df) < 2:
        return {}

    players_df = get_game_players(gameid)
    team_rows = teams_df.to_dict("records")
    team1 = team_rows[0]
    team2 = team_rows[1]

    winner = team1 if int(team1.get("result") or 0) == 1 else team2
    loser = team2 if winner is team1 else team1

    def _int_value(row: dict[str, object], key: str) -> int:
        return int(_safe_float(row.get(key)))

    def _team_player_sum(team_name: str, column: str) -> float:
        if players_df.empty or column not in players_df.columns:
            return 0.0
        mask = players_df["teamname"] == team_name
        if not mask.any():
            return 0.0
        return round(float(players_df.loc[mask, column].fillna(0).sum()), 1)

    winner_name = str(winner.get("teamname") or "")
    loser_name = str(loser.get("teamname") or "")
    winner_gd15 = _team_player_sum(winner_name, "golddiffat15")
    loser_gd15 = _team_player_sum(loser_name, "golddiffat15")

    kill_diff = _int_value(winner, "teamkills") - _int_value(loser, "teamkills")
    tower_diff = _int_value(winner, "towers") - _int_value(loser, "towers")
    dragon_diff = _int_value(winner, "dragons") - _int_value(loser, "dragons")
    baron_diff = _int_value(winner, "barons") - _int_value(loser, "barons")
    herald_diff = _int_value(winner, "heralds") - _int_value(loser, "heralds")
    elder_diff = _int_value(winner, "elders") - _int_value(loser, "elders")
    grub_diff = _int_value(winner, "void_grubs") - _int_value(loser, "void_grubs")

    total_objectives = (
        _int_value(team1, "towers")
        + _int_value(team2, "towers")
        + _int_value(team1, "dragons")
        + _int_value(team2, "dragons")
        + _int_value(team1, "barons")
        + _int_value(team2, "barons")
        + _int_value(team1, "heralds")
        + _int_value(team2, "heralds")
        + _int_value(team1, "elders")
        + _int_value(team2, "elders")
        + _int_value(team1, "void_grubs")
        + _int_value(team2, "void_grubs")
    )
    winner_objectives = (
        _int_value(winner, "towers")
        + _int_value(winner, "dragons")
        + _int_value(winner, "barons")
        + _int_value(winner, "heralds")
        + _int_value(winner, "elders")
        + _int_value(winner, "void_grubs")
    )
    objective_share_pct = round(
        100.0 * winner_objectives / total_objectives,
        1,
    ) if total_objectives else 50.0

    neutral_objective_diff = dragon_diff + baron_diff + herald_diff + elder_diff + grub_diff
    total_kills = _int_value(team1, "teamkills") + _int_value(team2, "teamkills")
    game_minutes = round(_safe_float(winner.get("gamelength")) / 60.0, 1)

    if winner_gd15 <= -600:
        script_label = "Comeback win"
        script_note = (
            f"{winner_name} recovered from an early deficit and flipped the game later."
        )
    elif winner_gd15 >= 1200 and tower_diff >= 4:
        script_label = "Clean snowball"
        script_note = (
            f"{winner_name} converted lane pressure into structures and never let the game reset."
        )
    elif objective_share_pct >= 62.0 and neutral_objective_diff >= 3:
        script_label = "Objective squeeze"
        script_note = (
            f"{winner_name} controlled neutral setups and used them to close map space."
        )
    elif total_kills >= 35 and kill_diff <= 5:
        script_label = "High-variance scrap"
        script_note = (
            "The game stayed fight-heavy, and the final separation came through sharper execution."
        )
    else:
        script_label = "Mid-game separation"
        script_note = (
            f"{winner_name} created a usable edge, then widened it through cleaner mid-game decisions."
        )

    early_edge_team = winner_name if winner_gd15 >= 0 else loser_name
    early_edge_value = abs(winner_gd15 if winner_gd15 else loser_gd15)
    if early_edge_value >= 900:
        decisive_edge_label = "Early game lead"
        decisive_edge_note = (
            f"{early_edge_team} built roughly {int(round(early_edge_value))} team GD@15."
        )
    elif tower_diff >= 4:
        decisive_edge_label = "Structure control"
        decisive_edge_note = (
            f"{winner_name} finished {tower_diff} towers ahead and denied space consistently."
        )
    elif neutral_objective_diff >= 3:
        decisive_edge_label = "Neutral objective grip"
        decisive_edge_note = (
            f"{winner_name} won the neutral-objective battle by {neutral_objective_diff}."
        )
    elif kill_diff >= 8:
        decisive_edge_label = "Fight control"
        decisive_edge_note = (
            f"{winner_name} ended the game with a {kill_diff}-kill margin."
        )
    else:
        decisive_edge_label = "Thin margins"
        decisive_edge_note = (
            "No single axis dominated; the result came from smaller edges stacking together."
        )

    backdrop_df = get_game_team_backdrop(gameid, recent_days=recent_days)
    if len(backdrop_df) >= 2:
        backdrop_records = backdrop_df.to_dict("records")

        def _profile_score(row: dict[str, object]) -> float:
            return round(
                _safe_float(row.get("dominance_score"))
                + (_safe_float(row.get("winrate_recent")) - 50.0) * 0.8
                + _safe_float(row.get("avg_gd15")) / 40.0,
                1,
            )

        team1_score = _profile_score(backdrop_records[0])
        team2_score = _profile_score(backdrop_records[1])
        form_favorite = (
            str(backdrop_records[0].get("teamname") or "")
            if team1_score >= team2_score
            else str(backdrop_records[1].get("teamname") or "")
        )
        profile_gap = abs(team1_score - team2_score)
        if form_favorite == winner_name:
            backdrop_label = (
                "Result matched form" if profile_gap >= 8.0 else "Favorite held in a live matchup"
            )
        else:
            backdrop_label = (
                "Upset versus recent form"
                if profile_gap >= 8.0
                else "Volatile matchup flipped"
            )
        backdrop_note = (
            f"{form_favorite} entered with the stronger recent profile, but {winner_name} owned this game."
            if form_favorite != winner_name
            else f"{winner_name} entered with the stronger recent profile and played to that level."
        )
    else:
        backdrop_label = "Backdrop unavailable"
        backdrop_note = "Recent team-form context is not available for this matchup."

    h2h_summary = get_team_h2h_summary(winner_name, loser_name)
    h2h_total_series = int(h2h_summary.get("total_series") or 0)
    if h2h_total_series:
        winner_h2h_wins = int(h2h_summary.get("t1_wins") or 0)
        loser_h2h_wins = int(h2h_summary.get("t2_wins") or 0)
        if winner_h2h_wins > loser_h2h_wins:
            h2h_note = (
                f"Tracked series history also leans {winner_name} {winner_h2h_wins}-{loser_h2h_wins}."
            )
        elif winner_h2h_wins < loser_h2h_wins:
            h2h_note = (
                f"{winner_name} won despite trailing the tracked series history {winner_h2h_wins}-{loser_h2h_wins}."
            )
        else:
            h2h_note = (
                f"The tracked series history was level at {winner_h2h_wins}-{loser_h2h_wins}."
            )
    else:
        winner_h2h_wins = 0
        loser_h2h_wins = 0
        h2h_note = "No tracked series history exists for this matchup."

    return _with_league_labels_row(
        {
            "winning_team": winner_name,
            "losing_team": loser_name,
            "winning_side": winner.get("side"),
            "league": winner.get("league"),
            "game_date": winner.get("game_date"),
            "patch": winner.get("patch"),
            "game_number": winner.get("game"),
            "game_minutes": game_minutes,
            "total_kills": total_kills,
            "kill_diff": kill_diff,
            "tower_diff": tower_diff,
            "dragon_diff": dragon_diff,
            "baron_diff": baron_diff,
            "grub_diff": grub_diff,
            "objective_share_pct": objective_share_pct,
            "winner_gd15": round(winner_gd15, 1),
            "loser_gd15": round(loser_gd15, 1),
            "script_label": script_label,
            "script_note": script_note,
            "decisive_edge_label": decisive_edge_label,
            "decisive_edge_note": decisive_edge_note,
            "backdrop_label": backdrop_label,
            "backdrop_note": backdrop_note,
            "h2h_total_series": h2h_total_series,
            "winner_h2h_wins": winner_h2h_wins,
            "loser_h2h_wins": loser_h2h_wins,
            "h2h_note": h2h_note,
        }
    )


def get_game_lane_matchups(gameid: str):
    """Return position-by-position lane reads for one game."""
    teams_df = get_game_teams(gameid)
    players_df = get_game_players(gameid)
    if len(teams_df) < 2 or players_df.empty:
        return pd.DataFrame()

    team1_name = str(teams_df.iloc[0].get("teamname") or "")
    team2_name = str(teams_df.iloc[1].get("teamname") or "")
    positions = [
        ("top", "Top"),
        ("jng", "Jungle"),
        ("mid", "Mid"),
        ("bot", "Bot"),
        ("sup", "Support"),
    ]

    def _kda(row: pd.Series) -> float:
        kills = _safe_float(row.get("kills"))
        deaths = max(_safe_float(row.get("deaths")), 1.0)
        assists = _safe_float(row.get("assists"))
        return round((kills + assists) / deaths, 2)

    rows: list[dict[str, object]] = []
    for position_key, position_label in positions:
        t1_rows = players_df[
            (players_df["teamname"] == team1_name)
            & (players_df["position"] == position_key)
        ]
        t2_rows = players_df[
            (players_df["teamname"] == team2_name)
            & (players_df["position"] == position_key)
        ]
        if t1_rows.empty or t2_rows.empty:
            continue

        t1 = t1_rows.iloc[0]
        t2 = t2_rows.iloc[0]
        t1_kda = _kda(t1)
        t2_kda = _kda(t2)
        gd15_signal = _safe_float(t1.get("golddiffat15"))
        dpm_diff = _safe_float(t1.get("dpm")) - _safe_float(t2.get("dpm"))
        kda_diff = t1_kda - t2_kda
        vision_diff = _safe_float(t1.get("visionscore")) - _safe_float(
            t2.get("visionscore")
        )
        edge_score = (
            gd15_signal / 225.0
            + kda_diff / 1.4
            + dpm_diff / 140.0
            + vision_diff / 25.0
        )

        if abs(edge_score) < 0.75:
            edge_team = "Even lane"
            edge_label = "Even"
        else:
            edge_team = team1_name if edge_score > 0 else team2_name
            if abs(edge_score) >= 2.5:
                edge_label = "Dominant"
            elif abs(edge_score) >= 1.25:
                edge_label = "Clear edge"
            else:
                edge_label = "Slight edge"

        driver_scores = {
            "lane lead": abs(gd15_signal) / 225.0,
            "fight output": abs(dpm_diff) / 140.0,
            "survivability": abs(kda_diff) / 1.4,
            "vision control": abs(vision_diff) / 25.0,
        }
        top_driver = max(driver_scores, key=driver_scores.get)
        if edge_team == "Even lane":
            edge_note = "Both players posted a broadly similar game across lane, fights, and map setup."
        elif top_driver == "lane lead":
            edge_note = f"{edge_team} created the clearest advantage through the lane phase."
        elif top_driver == "fight output":
            edge_note = f"{edge_team} separated through fight damage and conversion."
        elif top_driver == "survivability":
            edge_note = f"{edge_team} won the trade pattern and survived skirmishes more cleanly."
        else:
            edge_note = f"{edge_team} created more map value through vision and setup."

        rows.append(
            {
                "position": position_key,
                "position_label": position_label,
                "team1_name": team1_name,
                "team1_player": t1.get("playername"),
                "team1_champion": t1.get("champion"),
                "team1_kda": t1_kda,
                "team1_dpm": round(_safe_float(t1.get("dpm")), 0),
                "team1_damage_share_pct": round(
                    _safe_float(t1.get("damageshare")) * 100.0,
                    1,
                ),
                "team1_vision": round(_safe_float(t1.get("visionscore")), 1),
                "team1_gd15": round(_safe_float(t1.get("golddiffat15")), 0),
                "team2_name": team2_name,
                "team2_player": t2.get("playername"),
                "team2_champion": t2.get("champion"),
                "team2_kda": t2_kda,
                "team2_dpm": round(_safe_float(t2.get("dpm")), 0),
                "team2_damage_share_pct": round(
                    _safe_float(t2.get("damageshare")) * 100.0,
                    1,
                ),
                "team2_vision": round(_safe_float(t2.get("visionscore")), 1),
                "team2_gd15": round(_safe_float(t2.get("golddiffat15")), 0),
                "edge_team": edge_team,
                "edge_label": edge_label,
                "edge_note": edge_note,
            }
        )

    return pd.DataFrame(rows)


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

    return _with_league_labels(
        query_df(
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
    return _with_league_labels(df).to_dict("records")


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

    return _with_league_labels(
        query_df(
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

    return _with_league_labels(
        query_df(
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
    )


def get_betting_filters():
    """Leagues, years, splits available for betting page filters."""
    leagues = _with_league_labels(
        query_df(
            """
        SELECT DISTINCT league, COUNT(*) as c
        FROM games
        GROUP BY league ORDER BY c DESC
    """
        )
    )

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

    return leagues["league"].tolist(), years, splits


def get_available_splits():
    return query_df(
        """
        SELECT DISTINCT split
        FROM players
        WHERE split IS NOT NULL
        ORDER BY split
    """
    )["split"].tolist()
