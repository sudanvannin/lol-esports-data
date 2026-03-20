"""
Gold layer builder and validation runner.

The Gold layer is built directly from Bronze Oracle/official match sources to keep
source fidelity for fields like patch version, datacompleteness, source IDs, and
official schedule metadata. Every snapshot is immutable and stored under
``data/gold/snapshots/<snapshot_id>`` with a manifest and validation reports.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC)


def _normalize_path(path: Path) -> str:
    """Normalize filesystem paths for DuckDB glob usage."""
    return path.resolve().as_posix()


def _json_default(value: Any) -> Any:
    """Serialize datetimes and paths in manifests/reports."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(slots=True)
class GoldBuildResult:
    """Artifacts produced by a Gold snapshot build."""

    snapshot_id: str
    snapshot_dir: Path
    manifest_path: Path
    validation_report_path: Path
    tables: dict[str, Path]


class GoldLayerBuilder:
    """Build curated Gold snapshot tables and validation artifacts."""

    TABLE_EXPORTS = {
        "dim_league": "dim_league.parquet",
        "dim_team": "dim_team.parquet",
        "dim_player": "dim_player.parquet",
        "external_reconciliation": "external_reconciliation.parquet",
        "fact_game_team": "fact_game_team.parquet",
        "fact_game_player": "fact_game_player.parquet",
        "fact_draft": "fact_draft.parquet",
        "fact_series": "fact_series.parquet",
        "match_features_prematch": "match_features_prematch.parquet",
        "model_core_series": "model_core_series.parquet",
        "quality_issues": "quality_issues.parquet",
        "source_coverage": "source_coverage.parquet",
        "validation_summary": "validation_summary.parquet",
        "dataset_manifest": "dataset_manifest.parquet",
    }
    MODEL_CORE_LEAGUES = (
        "LCK",
        "LEC",
        "LPL",
        "LCS",
        "CBLOL",
        "LTA",
        "LTA S",
        "LTA N",
        "MSI",
        "WLDs",
        "FST",
    )

    def __init__(
        self,
        bronze_path: str = "data/bronze",
        silver_path: str = "data/silver",
        gold_path: str = "data/gold",
        snapshot_id: str | None = None,
    ):
        self.bronze_path = Path(bronze_path)
        self.silver_path = Path(silver_path)
        self.gold_path = Path(gold_path)
        self.snapshots_path = self.gold_path / "snapshots"
        self.snapshots_path.mkdir(parents=True, exist_ok=True)

        self.snapshot_id = snapshot_id or _utc_now().strftime("%Y%m%dT%H%M%SZ")
        self.snapshot_dir = self.snapshots_path / self.snapshot_id
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.con = duckdb.connect(":memory:")
        self.build_started_at = _utc_now()

        self.oracle_pattern = _normalize_path(self.bronze_path / "oracle_elixir" / "*.csv")
        self.official_matches_pattern = _normalize_path(
            self.bronze_path / "matches" / "**" / "*.json"
        )
        self.official_leagues_pattern = _normalize_path(
            self.bronze_path / "leagues" / "**" / "*.json"
        )
        self.official_tournaments_pattern = _normalize_path(
            self.bronze_path / "tournaments" / "**" / "*.json"
        )
        self.leaguepedia_match_results_path = self.bronze_path / "leaguepedia" / "match_results.json"
        self.team_alias_groups_path = (
            self.gold_path / "manual_overrides" / "team_alias_groups.json"
        )
        self.team_alias_lookup = self._load_team_alias_lookup()

    def build(self) -> GoldBuildResult:
        """Build the complete Gold snapshot and validation metadata."""
        logger.info("Building Gold snapshot %s", self.snapshot_id)

        self._assert_sources_exist()
        self._create_source_views()
        self._build_dimensions()
        self._build_facts()
        self._build_external_reconciliation()
        self._build_model_ready_tables()
        validation_summary = self._build_validation_summary()
        quality_issue_count = self.con.execute(
            "SELECT COUNT(*) FROM gold_quality_issues"
        ).fetchone()[0]

        table_counts = self._collect_table_counts()
        manifest = self._build_manifest(
            table_counts=table_counts,
            validation_summary=validation_summary,
            quality_issue_count=quality_issue_count,
        )
        self._write_manifest_tables(manifest, validation_summary)
        exported_tables = self._export_snapshot_tables()
        manifest_path, validation_report_path = self._write_json_reports(
            manifest=manifest,
            validation_summary=validation_summary,
        )
        self._write_latest_pointer(manifest, validation_report_path)

        logger.info("Gold snapshot %s ready at %s", self.snapshot_id, self.snapshot_dir)
        return GoldBuildResult(
            snapshot_id=self.snapshot_id,
            snapshot_dir=self.snapshot_dir,
            manifest_path=manifest_path,
            validation_report_path=validation_report_path,
            tables=exported_tables,
        )

    def _assert_sources_exist(self) -> None:
        if not list((self.bronze_path / "oracle_elixir").glob("*.csv")):
            raise FileNotFoundError(
                f"No Oracle's Elixir CSV files found under {self.bronze_path / 'oracle_elixir'}"
            )

    def _load_team_alias_lookup(self) -> dict[str, str]:
        """Load audited team alias groups into a normalized lookup."""
        if not self.team_alias_groups_path.exists():
            return {}

        payload = json.loads(self.team_alias_groups_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            return {}

        lookup: dict[str, str] = {}
        for item in payload:
            if not isinstance(item, dict):
                continue

            canonical = self._normalize_team_name(item.get("canonical"))
            aliases = item.get("aliases", [])
            if not canonical or not isinstance(aliases, list):
                continue

            lookup[canonical] = canonical
            for alias in aliases:
                alias_norm = self._normalize_team_name(alias)
                if alias_norm:
                    lookup[alias_norm] = canonical

        logger.info("Loaded %s normalized team aliases", len(lookup))
        return lookup

    @staticmethod
    def _normalize_team_name(name: Any) -> str:
        """Normalize a team name to its SQL join key shape."""
        return re.sub(r"[^a-z0-9]+", "", str(name or "").lower())

    def _team_norm_sql(self, column_sql: str) -> str:
        """Build a SQL expression that normalizes a team name with alias overrides."""
        base_expr = (
            f"regexp_replace(lower(coalesce({column_sql}, '')), '[^a-z0-9]+', '', 'g')"
        )
        if not self.team_alias_lookup:
            return base_expr

        when_clauses = " ".join(
            f"WHEN {base_expr} = '{alias}' THEN '{canonical}'"
            for alias, canonical in sorted(self.team_alias_lookup.items())
        )
        return f"(CASE {when_clauses} ELSE {base_expr} END)"

    def _load_leaguepedia_match_rows(self) -> pd.DataFrame:
        """Load Leaguepedia MatchSchedule payload into a normalized DataFrame."""
        expected_columns = [
            "MatchId",
            "Team1",
            "Team2",
            "Winner",
            "Team1Score",
            "Team2Score",
            "DateTime_UTC",
            "BestOf",
            "OverviewPage",
            "ShownName",
            "Patch",
            "Round",
            "Phase",
            "Tab",
            "MatchDay",
        ]
        empty_df = pd.DataFrame(columns=expected_columns)

        if not self.leaguepedia_match_results_path.exists():
            return empty_df

        raw_text = self.leaguepedia_match_results_path.read_text(encoding="utf-8").strip()
        if not raw_text or raw_text in {"[]", "{}"}:
            return empty_df

        payload = json.loads(raw_text)
        if isinstance(payload, dict):
            rows = payload.get("rows", [])
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []

        if not rows:
            return empty_df

        leaguepedia_df = pd.DataFrame(rows)
        leaguepedia_df = leaguepedia_df.rename(columns={"DateTime UTC": "DateTime_UTC"})

        for column in expected_columns:
            if column not in leaguepedia_df.columns:
                leaguepedia_df[column] = None

        return leaguepedia_df[expected_columns]

    def _create_source_views(self) -> None:
        logger.info("Creating Bronze source views")

        self.con.execute(
            f"""
            CREATE OR REPLACE VIEW bronze_oracle_raw AS
            SELECT *
            FROM read_csv_auto(
                '{self.oracle_pattern}',
                all_varchar=true,
                union_by_name=true,
                ignore_errors=true,
                sample_size=-1,
                filename=true
            )
            """
        )

        self.con.execute(
            f"""
            CREATE OR REPLACE VIEW bronze_official_matches AS
            WITH raw AS (
                SELECT *
                FROM read_json_auto(
                    '{self.official_matches_pattern}',
                    maximum_object_size=10000000
                )
            )
            SELECT
                CAST(content.match_id AS VARCHAR) AS official_match_id,
                CAST(content.state AS VARCHAR) AS official_match_state,
                TRY_CAST(content.start_time AS TIMESTAMP) AS official_start_time,
                CAST(TRY_CAST(content.start_time AS TIMESTAMP) AS DATE) AS official_match_date,
                CAST(content.block_name AS VARCHAR) AS official_block_name,
                UPPER(CAST(content.league.name AS VARCHAR)) AS official_league_name,
                LOWER(CAST(content.league.slug AS VARCHAR)) AS official_league_slug,
                CAST(content.strategy.type AS VARCHAR) AS official_strategy_type,
                TRY_CAST(content.strategy.count AS INTEGER) AS official_best_of,
                CAST(list_extract(list_transform(content.teams, x -> x.name), 1) AS VARCHAR)
                    AS official_team1_name,
                CAST(list_extract(list_transform(content.teams, x -> x.name), 2) AS VARCHAR)
                    AS official_team2_name,
                CAST(list_extract(list_transform(content.teams, x -> x.code), 1) AS VARCHAR)
                    AS official_team1_code,
                CAST(list_extract(list_transform(content.teams, x -> x.code), 2) AS VARCHAR)
                    AS official_team2_code,
                CAST(list_extract(list_transform(content.teams, x -> x.result.outcome), 1) AS VARCHAR)
                    AS official_team1_outcome,
                CAST(list_extract(list_transform(content.teams, x -> x.result.outcome), 2) AS VARCHAR)
                    AS official_team2_outcome,
                TRY_CAST(list_extract(list_transform(content.teams, x -> x.result.gameWins), 1) AS INTEGER)
                    AS official_team1_game_wins,
                TRY_CAST(list_extract(list_transform(content.teams, x -> x.result.gameWins), 2) AS INTEGER)
                    AS official_team2_game_wins
            FROM raw
            WHERE content.state = 'completed'
              AND array_length(content.teams) = 2
            """
        )

        self.con.execute(
            f"""
            CREATE OR REPLACE VIEW bronze_official_leagues AS
            SELECT
                CAST(id AS VARCHAR) AS official_league_id,
                LOWER(CAST(slug AS VARCHAR)) AS official_league_slug,
                CAST(name AS VARCHAR) AS official_league_name,
                CAST(region AS VARCHAR) AS official_region,
                CAST(image_url AS VARCHAR) AS official_image_url,
                TRY_CAST(priority AS INTEGER) AS official_priority
            FROM read_json_auto('{self.official_leagues_pattern}', maximum_object_size=10000000)
            """
        )

        self.con.execute(
            f"""
            CREATE OR REPLACE VIEW bronze_official_tournaments AS
            SELECT
                CAST(id AS VARCHAR) AS official_tournament_id,
                CAST(slug AS VARCHAR) AS official_tournament_slug,
                CAST(name AS VARCHAR) AS official_tournament_name,
                CAST(league_id AS VARCHAR) AS official_league_id,
                TRY_CAST(start_date AS TIMESTAMP) AS official_start_time,
                TRY_CAST(end_date AS TIMESTAMP) AS official_end_time
            FROM read_json_auto('{self.official_tournaments_pattern}', maximum_object_size=10000000)
            """
        )

        leaguepedia_df = self._load_leaguepedia_match_rows()
        self.con.register("leaguepedia_match_rows_df", leaguepedia_df)
        leaguepedia_team1_norm = self._team_norm_sql("leaguepedia_team1_name")
        leaguepedia_team2_norm = self._team_norm_sql("leaguepedia_team2_name")
        self.con.execute(
            """
            CREATE OR REPLACE TABLE bronze_leaguepedia_matches AS
            WITH typed AS (
                SELECT
                    NULLIF(CAST(MatchId AS VARCHAR), '') AS leaguepedia_match_id,
                    NULLIF(CAST(Team1 AS VARCHAR), '') AS leaguepedia_team1_name,
                    NULLIF(CAST(Team2 AS VARCHAR), '') AS leaguepedia_team2_name,
                    NULLIF(CAST(Winner AS VARCHAR), '') AS leaguepedia_winner,
                    TRY_CAST(Team1Score AS INTEGER) AS leaguepedia_team1_score,
                    TRY_CAST(Team2Score AS INTEGER) AS leaguepedia_team2_score,
                    TRY_CAST(DateTime_UTC AS TIMESTAMP) AS leaguepedia_start_time,
                    CAST(TRY_CAST(DateTime_UTC AS TIMESTAMP) AS DATE) AS leaguepedia_match_date,
                    TRY_CAST(BestOf AS INTEGER) AS leaguepedia_best_of,
                    NULLIF(CAST(OverviewPage AS VARCHAR), '') AS leaguepedia_overview_page,
                    NULLIF(CAST(ShownName AS VARCHAR), '') AS leaguepedia_shown_name,
                    NULLIF(CAST(Patch AS VARCHAR), '') AS leaguepedia_patch,
                    NULLIF(CAST(Round AS VARCHAR), '') AS leaguepedia_round,
                    NULLIF(CAST(Phase AS VARCHAR), '') AS leaguepedia_phase,
                    NULLIF(CAST(Tab AS VARCHAR), '') AS leaguepedia_tab,
                    NULLIF(CAST(MatchDay AS VARCHAR), '') AS leaguepedia_match_day
                FROM leaguepedia_match_rows_df
            )
            SELECT
                *,
                {leaguepedia_team1_norm} AS leaguepedia_team1_name_norm,
                {leaguepedia_team2_norm} AS leaguepedia_team2_name_norm,
                CASE
                    WHEN {leaguepedia_team1_norm} < {leaguepedia_team2_norm}
                    THEN {leaguepedia_team1_norm}
                    ELSE {leaguepedia_team2_norm}
                END AS team_lo_norm,
                CASE
                    WHEN {leaguepedia_team1_norm} < {leaguepedia_team2_norm}
                    THEN {leaguepedia_team2_norm}
                    ELSE {leaguepedia_team1_norm}
                END AS team_hi_norm,
                CASE
                    WHEN {leaguepedia_team1_norm} < {leaguepedia_team2_norm}
                    THEN leaguepedia_team1_score
                    ELSE leaguepedia_team2_score
                END AS team_lo_score,
                CASE
                    WHEN {leaguepedia_team1_norm} < {leaguepedia_team2_norm}
                    THEN leaguepedia_team2_score
                    ELSE leaguepedia_team1_score
                END AS team_hi_score
            FROM typed
            """.format(
                leaguepedia_team1_norm=leaguepedia_team1_norm,
                leaguepedia_team2_norm=leaguepedia_team2_norm,
            )
        )

        self.con.execute(
            """
            CREATE OR REPLACE VIEW bronze_oracle_team_rows AS
            WITH raw AS (
                SELECT *
                FROM bronze_oracle_raw
                WHERE lower(coalesce(position, '')) = 'team'
            ),
            typed AS (
                SELECT
                    CAST(gameid AS VARCHAR) AS game_id,
                    CAST(datacompleteness AS VARCHAR) AS data_completeness,
                    TRY_CAST(date AS TIMESTAMP) AS game_datetime,
                    CAST(TRY_CAST(date AS TIMESTAMP) AS DATE) AS game_date,
                    TRY_CAST(year AS INTEGER) AS season_year,
                    NULLIF(split, '') AS split_name,
                    COALESCE(TRY_CAST(playoffs AS INTEGER), 0) AS playoffs,
                    CAST(league AS VARCHAR) AS league_code,
                    NULLIF(CAST(patch AS VARCHAR), '') AS patch_version,
                    TRY_CAST(game AS INTEGER) AS game_number,
                    NULLIF(CAST(teamid AS VARCHAR), '') AS source_team_id,
                    NULLIF(CAST(teamname AS VARCHAR), '') AS team_name,
                    lower(CAST(side AS VARCHAR)) AS side,
                    CASE
                        WHEN TRY_CAST(result AS INTEGER) IS NOT NULL THEN TRY_CAST(result AS INTEGER)
                        WHEN lower(CAST(result AS VARCHAR)) = 'win' THEN 1
                        WHEN lower(CAST(result AS VARCHAR)) = 'loss' THEN 0
                        ELSE NULL
                    END AS win,
                    TRY_CAST(gamelength AS DOUBLE) AS game_length_seconds,
                    TRY_CAST(teamkills AS INTEGER) AS team_kills,
                    TRY_CAST(teamdeaths AS INTEGER) AS team_deaths,
                    TRY_CAST(firstPick AS INTEGER) AS first_pick,
                    NULLIF(CAST(pick1 AS VARCHAR), '') AS pick_1,
                    NULLIF(CAST(pick2 AS VARCHAR), '') AS pick_2,
                    NULLIF(CAST(pick3 AS VARCHAR), '') AS pick_3,
                    NULLIF(CAST(pick4 AS VARCHAR), '') AS pick_4,
                    NULLIF(CAST(pick5 AS VARCHAR), '') AS pick_5,
                    NULLIF(CAST(ban1 AS VARCHAR), '') AS ban_1,
                    NULLIF(CAST(ban2 AS VARCHAR), '') AS ban_2,
                    NULLIF(CAST(ban3 AS VARCHAR), '') AS ban_3,
                    NULLIF(CAST(ban4 AS VARCHAR), '') AS ban_4,
                    NULLIF(CAST(ban5 AS VARCHAR), '') AS ban_5,
                    TRY_CAST(firstblood AS INTEGER) AS first_blood,
                    TRY_CAST(firsttower AS INTEGER) AS first_tower,
                    TRY_CAST(firstdragon AS INTEGER) AS first_dragon,
                    TRY_CAST(firstherald AS INTEGER) AS first_herald,
                    TRY_CAST(firstbaron AS INTEGER) AS first_baron,
                    TRY_CAST(towers AS INTEGER) AS towers,
                    TRY_CAST(dragons AS INTEGER) AS dragons,
                    TRY_CAST(heralds AS INTEGER) AS heralds,
                    TRY_CAST(barons AS INTEGER) AS barons,
                    TRY_CAST(elders AS INTEGER) AS elders,
                    TRY_CAST(void_grubs AS INTEGER) AS void_grubs,
                    TRY_CAST(atakhans AS INTEGER) AS atakhans,
                    TRY_CAST(inhibitors AS INTEGER) AS inhibitors,
                    TRY_CAST(opp_towers AS INTEGER) AS opp_towers,
                    TRY_CAST(opp_dragons AS INTEGER) AS opp_dragons,
                    TRY_CAST(opp_heralds AS INTEGER) AS opp_heralds,
                    TRY_CAST(opp_barons AS INTEGER) AS opp_barons,
                    TRY_CAST(opp_elders AS INTEGER) AS opp_elders,
                    TRY_CAST(opp_void_grubs AS INTEGER) AS opp_void_grubs,
                    TRY_CAST(opp_atakhans AS INTEGER) AS opp_atakhans,
                    TRY_CAST(opp_inhibitors AS INTEGER) AS opp_inhibitors,
                    filename AS source_file
                FROM raw
            )
            SELECT
                *,
                CASE
                    WHEN source_team_id IS NOT NULL THEN 'team:' || source_team_id
                    ELSE 'teamname:' || regexp_replace(lower(coalesce(team_name, 'unknown')), '[^a-z0-9]+', '', 'g')
                END AS team_key,
                regexp_replace(lower(coalesce(team_name, 'unknown')), '[^a-z0-9]+', '', 'g') AS team_name_norm
            FROM typed
            """
        )

        self.con.execute(
            """
            CREATE OR REPLACE VIEW bronze_oracle_player_rows AS
            WITH raw AS (
                SELECT *
                FROM bronze_oracle_raw
                WHERE lower(coalesce(position, '')) <> 'team'
            )
            SELECT
                CAST(gameid AS VARCHAR) AS game_id,
                CAST(datacompleteness AS VARCHAR) AS data_completeness,
                TRY_CAST(date AS TIMESTAMP) AS game_datetime,
                CAST(TRY_CAST(date AS TIMESTAMP) AS DATE) AS game_date,
                TRY_CAST(year AS INTEGER) AS season_year,
                NULLIF(split, '') AS split_name,
                COALESCE(TRY_CAST(playoffs AS INTEGER), 0) AS playoffs,
                CAST(league AS VARCHAR) AS league_code,
                NULLIF(CAST(patch AS VARCHAR), '') AS patch_version,
                TRY_CAST(game AS INTEGER) AS game_number,
                NULLIF(CAST(playerid AS VARCHAR), '') AS source_player_id,
                NULLIF(CAST(playername AS VARCHAR), '') AS player_name,
                TRY_CAST(participantid AS INTEGER) AS participant_id,
                NULLIF(CAST(teamid AS VARCHAR), '') AS source_team_id,
                NULLIF(CAST(teamname AS VARCHAR), '') AS team_name,
                lower(CAST(side AS VARCHAR)) AS side,
                lower(CAST(position AS VARCHAR)) AS role,
                NULLIF(CAST(champion AS VARCHAR), '') AS champion_name,
                CASE
                    WHEN TRY_CAST(result AS INTEGER) IS NOT NULL THEN TRY_CAST(result AS INTEGER)
                    WHEN lower(CAST(result AS VARCHAR)) = 'win' THEN 1
                    WHEN lower(CAST(result AS VARCHAR)) = 'loss' THEN 0
                    ELSE NULL
                END AS win,
                TRY_CAST(gamelength AS DOUBLE) AS game_length_seconds,
                TRY_CAST(kills AS INTEGER) AS kills,
                TRY_CAST(deaths AS INTEGER) AS deaths,
                TRY_CAST(assists AS INTEGER) AS assists,
                TRY_CAST(damagetochampions AS DOUBLE) AS damage_to_champions,
                TRY_CAST(dpm AS DOUBLE) AS dpm,
                TRY_CAST(damageshare AS DOUBLE) AS damage_share,
                TRY_CAST(damagetakenperminute AS DOUBLE) AS damage_taken_per_minute,
                TRY_CAST(damagemitigatedperminute AS DOUBLE) AS damage_mitigated_per_minute,
                TRY_CAST(damagetotowers AS DOUBLE) AS damage_to_towers,
                TRY_CAST(totalgold AS DOUBLE) AS total_gold,
                TRY_CAST(earnedgold AS DOUBLE) AS earned_gold,
                TRY_CAST("earned gpm" AS DOUBLE) AS earned_gpm,
                TRY_CAST(earnedgoldshare AS DOUBLE) AS earned_gold_share,
                TRY_CAST(goldspent AS DOUBLE) AS gold_spent,
                TRY_CAST(minionkills AS DOUBLE) AS minion_kills,
                TRY_CAST(monsterkills AS DOUBLE) AS monster_kills,
                TRY_CAST("total cs" AS DOUBLE) AS total_cs,
                TRY_CAST(cspm AS DOUBLE) AS cs_per_minute,
                TRY_CAST(monsterkillsenemyjungle AS DOUBLE) AS enemy_jungle_cs,
                TRY_CAST(monsterkillsownjungle AS DOUBLE) AS own_jungle_cs,
                TRY_CAST(visionscore AS DOUBLE) AS vision_score,
                TRY_CAST(vspm AS DOUBLE) AS vision_score_per_minute,
                TRY_CAST(wardsplaced AS DOUBLE) AS wards_placed,
                TRY_CAST(wardskilled AS DOUBLE) AS wards_killed,
                TRY_CAST(controlwardsbought AS DOUBLE) AS control_wards_bought,
                TRY_CAST(goldat10 AS DOUBLE) AS gold_at_10,
                TRY_CAST(xpat10 AS DOUBLE) AS xp_at_10,
                TRY_CAST(csat10 AS DOUBLE) AS cs_at_10,
                TRY_CAST(golddiffat10 AS DOUBLE) AS gold_diff_at_10,
                TRY_CAST(xpdiffat10 AS DOUBLE) AS xp_diff_at_10,
                TRY_CAST(csdiffat10 AS DOUBLE) AS cs_diff_at_10,
                TRY_CAST(killsat10 AS DOUBLE) AS kills_at_10,
                TRY_CAST(assistsat10 AS DOUBLE) AS assists_at_10,
                TRY_CAST(deathsat10 AS DOUBLE) AS deaths_at_10,
                TRY_CAST(goldat15 AS DOUBLE) AS gold_at_15,
                TRY_CAST(xpat15 AS DOUBLE) AS xp_at_15,
                TRY_CAST(csat15 AS DOUBLE) AS cs_at_15,
                TRY_CAST(golddiffat15 AS DOUBLE) AS gold_diff_at_15,
                TRY_CAST(xpdiffat15 AS DOUBLE) AS xp_diff_at_15,
                TRY_CAST(csdiffat15 AS DOUBLE) AS cs_diff_at_15,
                TRY_CAST(killsat15 AS DOUBLE) AS kills_at_15,
                TRY_CAST(assistsat15 AS DOUBLE) AS assists_at_15,
                TRY_CAST(deathsat15 AS DOUBLE) AS deaths_at_15,
                TRY_CAST(doublekills AS INTEGER) AS double_kills,
                TRY_CAST(triplekills AS INTEGER) AS triple_kills,
                TRY_CAST(quadrakills AS INTEGER) AS quadra_kills,
                TRY_CAST(pentakills AS INTEGER) AS penta_kills,
                TRY_CAST(firstbloodkill AS INTEGER) AS first_blood_kill,
                TRY_CAST(firstbloodassist AS INTEGER) AS first_blood_assist,
                TRY_CAST(firstbloodvictim AS INTEGER) AS first_blood_victim,
                filename AS source_file,
                CASE
                    WHEN NULLIF(CAST(playerid AS VARCHAR), '') IS NOT NULL
                    THEN 'player:' || CAST(playerid AS VARCHAR)
                    ELSE 'playername:' || regexp_replace(lower(coalesce(playername, 'unknown')), '[^a-z0-9]+', '', 'g')
                END AS player_key,
                regexp_replace(lower(coalesce(playername, 'unknown')), '[^a-z0-9]+', '', 'g') AS player_name_norm
            FROM raw
            """
        )

    def _build_dimensions(self) -> None:
        logger.info("Building Gold dimensions")

        self.con.execute(
            """
            CREATE OR REPLACE TABLE gold_dim_league AS
            WITH oracle_leagues AS (
                SELECT
                    league_code,
                    MIN(game_date) AS first_seen_date,
                    MAX(game_date) AS last_seen_date,
                    COUNT(DISTINCT game_id) AS distinct_games,
                    COUNT(DISTINCT season_year) AS seasons_seen
                FROM bronze_oracle_team_rows
                GROUP BY league_code
            ),
            known_map AS (
                SELECT * FROM (
                    VALUES
                        ('CBLOL', 'cblol-brazil'),
                        ('LCK', 'lck'),
                        ('LEC', 'lec'),
                        ('LCS', 'lcs'),
                        ('LPL', 'lpl'),
                        ('MSI', 'msi'),
                        ('WLDs', 'worlds'),
                        ('PCS', 'pcs'),
                        ('VCS', 'vcs'),
                        ('LJL', 'ljl-japan'),
                        ('NACL', 'nacl'),
                        ('LCKC', 'lck_challengers_league')
                ) AS t(league_code, official_league_slug)
            )
            SELECT
                o.league_code AS league_key,
                o.league_code,
                m.official_league_slug,
                l.official_league_id,
                l.official_league_name,
                l.official_region,
                l.official_image_url,
                l.official_priority,
                o.first_seen_date,
                o.last_seen_date,
                o.distinct_games,
                o.seasons_seen,
                CASE
                    WHEN l.official_league_id IS NOT NULL THEN 'matched_official_league'
                    WHEN m.official_league_slug IS NOT NULL THEN 'mapped_slug_without_official_row'
                    ELSE 'oracle_only'
                END AS curation_status,
                '{snapshot_id}' AS snapshot_id
            FROM oracle_leagues o
            LEFT JOIN known_map m USING (league_code)
            LEFT JOIN bronze_official_leagues l
                ON l.official_league_slug = m.official_league_slug
            ORDER BY o.league_code
            """.format(snapshot_id=self.snapshot_id)
        )

        self.con.execute(
            """
            CREATE OR REPLACE TABLE gold_dim_team AS
            SELECT
                team_key,
                MAX(source_team_id) FILTER (WHERE source_team_id IS NOT NULL) AS source_team_id,
                arg_max(team_name, game_datetime) FILTER (WHERE team_name IS NOT NULL) AS canonical_team_name,
                MIN(game_date) AS first_seen_date,
                MAX(game_date) AS last_seen_date,
                COUNT(DISTINCT game_id) AS games_played,
                COUNT(DISTINCT league_code) AS leagues_played_count,
                COUNT(*) FILTER (WHERE source_team_id IS NULL) AS rows_without_source_team_id,
                list_sort(list_distinct(list(team_name) FILTER (WHERE team_name IS NOT NULL))) AS team_name_variants,
                list_sort(list_distinct(list(league_code) FILTER (WHERE league_code IS NOT NULL))) AS leagues_seen,
                CASE
                    WHEN MAX(source_team_id) FILTER (WHERE source_team_id IS NOT NULL) IS NOT NULL
                    THEN 'source_team_id'
                    ELSE 'normalized_team_name'
                END AS canonical_key_source,
                '{snapshot_id}' AS snapshot_id
            FROM bronze_oracle_team_rows
            GROUP BY team_key
            ORDER BY canonical_team_name
            """.format(snapshot_id=self.snapshot_id)
        )

        self.con.execute(
            """
            CREATE OR REPLACE TABLE gold_dim_player AS
            WITH player_base AS (
                SELECT
                    player_key,
                    source_player_id,
                    player_name,
                    role,
                    team_name,
                    league_code,
                    game_id,
                    game_date,
                    game_datetime
                FROM bronze_oracle_player_rows
            ),
            role_counts AS (
                SELECT
                    player_key,
                    role,
                    COUNT(*) AS role_games,
                    ROW_NUMBER() OVER (
                        PARTITION BY player_key
                        ORDER BY COUNT(*) DESC, role
                    ) AS role_rank
                FROM player_base
                WHERE role IS NOT NULL
                GROUP BY player_key, role
            )
            SELECT
                p.player_key,
                MAX(p.source_player_id) FILTER (WHERE p.source_player_id IS NOT NULL) AS source_player_id,
                arg_max(p.player_name, p.game_datetime) FILTER (WHERE p.player_name IS NOT NULL)
                    AS canonical_player_name,
                rc.role AS primary_role,
                MIN(p.game_date) AS first_seen_date,
                MAX(p.game_date) AS last_seen_date,
                COUNT(DISTINCT p.game_id) AS games_played,
                COUNT(DISTINCT p.team_name) FILTER (WHERE p.team_name IS NOT NULL) AS teams_played_count,
                list_sort(list_distinct(list(p.player_name) FILTER (WHERE p.player_name IS NOT NULL)))
                    AS player_name_variants,
                list_sort(list_distinct(list(p.team_name) FILTER (WHERE p.team_name IS NOT NULL)))
                    AS teams_seen,
                list_sort(list_distinct(list(p.league_code) FILTER (WHERE p.league_code IS NOT NULL)))
                    AS leagues_seen,
                CASE
                    WHEN MAX(p.source_player_id) FILTER (WHERE p.source_player_id IS NOT NULL) IS NOT NULL
                    THEN 'source_player_id'
                    ELSE 'normalized_player_name'
                END AS canonical_key_source,
                '{snapshot_id}' AS snapshot_id
            FROM player_base p
            LEFT JOIN role_counts rc
                ON p.player_key = rc.player_key
               AND rc.role_rank = 1
            GROUP BY p.player_key, rc.role
            ORDER BY canonical_player_name
            """.format(snapshot_id=self.snapshot_id)
        )

    def _build_facts(self) -> None:
        logger.info("Building Gold facts")

        self.con.execute(
            """
            CREATE OR REPLACE VIEW gold_team_game_base AS
            SELECT
                t.*,
                o.team_key AS opponent_team_key,
                o.team_name AS opponent_team_name
            FROM bronze_oracle_team_rows t
            LEFT JOIN bronze_oracle_team_rows o
                ON t.game_id = o.game_id
               AND t.team_key <> o.team_key
            """
        )

        self.con.execute(
            """
            CREATE OR REPLACE TABLE gold_fact_game_team AS
            SELECT
                '{snapshot_id}' AS snapshot_id,
                game_id AS game_key,
                game_id || ':' || side AS game_team_key,
                game_id,
                game_date,
                game_datetime,
                season_year,
                league_code,
                split_name,
                playoffs,
                patch_version,
                game_number,
                data_completeness,
                team_key,
                source_team_id,
                team_name,
                team_name_norm,
                opponent_team_key,
                opponent_team_name,
                side,
                win,
                game_length_seconds,
                team_kills,
                team_deaths,
                first_pick,
                first_blood,
                first_tower,
                first_dragon,
                first_herald,
                first_baron,
                towers,
                dragons,
                heralds,
                barons,
                elders,
                void_grubs,
                atakhans,
                inhibitors,
                opp_towers,
                opp_dragons,
                opp_heralds,
                opp_barons,
                opp_elders,
                opp_void_grubs,
                opp_atakhans,
                opp_inhibitors,
                CASE WHEN source_team_id IS NULL THEN TRUE ELSE FALSE END AS has_missing_source_team_id,
                'oracle_elixir' AS source_system,
                source_file
            FROM gold_team_game_base
            """.format(snapshot_id=self.snapshot_id)
        )

    def _build_external_reconciliation(self) -> None:
        """Create multi-source reconciliation and source coverage outputs."""
        logger.info("Building external reconciliation tables")

        self.con.execute(
            """
            CREATE OR REPLACE TABLE gold_fact_series AS
            WITH game_matchups AS (
                SELECT
                    t1.game_id,
                    t1.game_date,
                    t1.game_datetime,
                    t1.season_year,
                    t1.league_code,
                    t1.split_name,
                    t1.playoffs,
                    t1.patch_version,
                    t1.game_number,
                    CASE WHEN t1.team_key < t2.team_key THEN t1.team_key ELSE t2.team_key END AS team1_key,
                    CASE WHEN t1.team_key < t2.team_key THEN t1.team_name ELSE t2.team_name END AS team1_name,
                    CASE WHEN t1.team_key < t2.team_key THEN t1.source_team_id ELSE t2.source_team_id END
                        AS team1_source_team_id,
                    CASE WHEN t1.team_key < t2.team_key THEN t1.win ELSE t2.win END AS team1_win,
                    CASE WHEN t1.team_key < t2.team_key THEN t2.team_key ELSE t1.team_key END AS team2_key,
                    CASE WHEN t1.team_key < t2.team_key THEN t2.team_name ELSE t1.team_name END AS team2_name,
                    CASE WHEN t1.team_key < t2.team_key THEN t2.source_team_id ELSE t1.source_team_id END
                        AS team2_source_team_id,
                    CASE WHEN t1.team_key < t2.team_key THEN t2.win ELSE t1.win END AS team2_win,
                    CASE
                        WHEN t1.patch_version IS NOT NULL THEN t1.patch_version
                        ELSE t2.patch_version
                    END AS patch_version,
                    COALESCE(t1.game_length_seconds, t2.game_length_seconds) AS game_length_seconds
                FROM gold_fact_game_team t1
                JOIN gold_fact_game_team t2
                    ON t1.game_id = t2.game_id
                   AND t1.team_key < t2.team_key
            ),
            daily_series AS (
                SELECT
                    game_date AS match_date,
                    league_code,
                    season_year,
                    split_name,
                    MAX(playoffs) AS playoffs,
                    team1_key,
                    arg_max(team1_name, game_datetime) AS team1_name,
                    arg_max(team1_source_team_id, game_datetime) AS team1_source_team_id,
                    team2_key,
                    arg_max(team2_name, game_datetime) AS team2_name,
                    arg_max(team2_source_team_id, game_datetime) AS team2_source_team_id,
                    COUNT(*) AS games_played,
                    SUM(team1_win) AS team1_wins,
                    SUM(team2_win) AS team2_wins,
                    arg_max(patch_version, game_datetime) AS patch_version,
                    array_agg(game_id ORDER BY game_number) AS game_ids,
                    array_agg(game_number ORDER BY game_number) AS game_numbers,
                    AVG(game_length_seconds) AS avg_game_length_seconds,
                    MIN(game_datetime) AS start_time,
                    MAX(game_datetime) AS end_time
                FROM game_matchups
                GROUP BY
                    game_date,
                    league_code,
                    season_year,
                    split_name,
                    team1_key,
                    team2_key
            ),
            with_prev_day AS (
                SELECT
                    *,
                    LAG(match_date) OVER (
                        PARTITION BY league_code, season_year, team1_key, team2_key
                        ORDER BY match_date
                    ) AS prev_match_date,
                    LAG(team1_wins) OVER (
                        PARTITION BY league_code, season_year, team1_key, team2_key
                        ORDER BY match_date
                    ) AS prev_team1_wins,
                    LAG(team2_wins) OVER (
                        PARTITION BY league_code, season_year, team1_key, team2_key
                        ORDER BY match_date
                    ) AS prev_team2_wins,
                    LAG(games_played) OVER (
                        PARTITION BY league_code, season_year, team1_key, team2_key
                        ORDER BY match_date
                    ) AS prev_games_played,
                    LAG(game_ids) OVER (
                        PARTITION BY league_code, season_year, team1_key, team2_key
                        ORDER BY match_date
                    ) AS prev_game_ids,
                    LAG(game_numbers) OVER (
                        PARTITION BY league_code, season_year, team1_key, team2_key
                        ORDER BY match_date
                    ) AS prev_game_numbers,
                    LAG(start_time) OVER (
                        PARTITION BY league_code, season_year, team1_key, team2_key
                        ORDER BY match_date
                    ) AS prev_start_time
                FROM daily_series
            ),
            merged_series AS (
                SELECT
                    CASE
                        WHEN prev_match_date IS NOT NULL
                         AND datediff('day', prev_match_date, match_date) = 1
                        THEN prev_match_date
                        ELSE match_date
                    END AS series_date,
                    league_code,
                    season_year,
                    split_name,
                    playoffs,
                    team1_key,
                    team1_name,
                    team1_source_team_id,
                    team2_key,
                    team2_name,
                    team2_source_team_id,
                    CASE
                        WHEN prev_match_date IS NOT NULL
                         AND datediff('day', prev_match_date, match_date) = 1
                        THEN games_played + prev_games_played
                        ELSE games_played
                    END AS games_played,
                    CASE
                        WHEN prev_match_date IS NOT NULL
                         AND datediff('day', prev_match_date, match_date) = 1
                        THEN team1_wins + prev_team1_wins
                        ELSE team1_wins
                    END AS team1_wins,
                    CASE
                        WHEN prev_match_date IS NOT NULL
                         AND datediff('day', prev_match_date, match_date) = 1
                        THEN team2_wins + prev_team2_wins
                        ELSE team2_wins
                    END AS team2_wins,
                    patch_version,
                    CASE
                        WHEN prev_match_date IS NOT NULL
                         AND datediff('day', prev_match_date, match_date) = 1
                        THEN list_concat(prev_game_ids, game_ids)
                        ELSE game_ids
                    END AS game_ids,
                    CASE
                        WHEN prev_match_date IS NOT NULL
                         AND datediff('day', prev_match_date, match_date) = 1
                        THEN list_concat(prev_game_numbers, game_numbers)
                        ELSE game_numbers
                    END AS game_numbers,
                    avg_game_length_seconds,
                    CASE
                        WHEN prev_match_date IS NOT NULL
                         AND datediff('day', prev_match_date, match_date) = 1
                        THEN prev_start_time
                        ELSE start_time
                    END AS start_time,
                    end_time,
                    CASE
                        WHEN prev_match_date IS NOT NULL
                         AND datediff('day', prev_match_date, match_date) = 1
                        THEN TRUE
                        ELSE FALSE
                    END AS is_multi_day_merge,
                    CASE
                        WHEN lead(match_date) OVER (
                            PARTITION BY league_code, season_year, team1_key, team2_key
                            ORDER BY match_date
                        ) IS NOT NULL
                         AND datediff(
                            'day',
                            match_date,
                            lead(match_date) OVER (
                                PARTITION BY league_code, season_year, team1_key, team2_key
                                ORDER BY match_date
                            )
                        ) = 1
                        THEN TRUE
                        ELSE FALSE
                    END AS is_first_day_of_multi
                FROM with_prev_day
            ),
            finalized AS (
                SELECT
                    'series:' || league_code || ':' || season_year || ':' || CAST(series_date AS VARCHAR)
                        || ':' || team1_key || ':' || team2_key AS series_key,
                    series_date,
                    league_code,
                    season_year,
                    split_name,
                    playoffs,
                    patch_version,
                    team1_key,
                    team1_name,
                    team1_source_team_id,
                    team2_key,
                    team2_name,
                    team2_source_team_id,
                    games_played,
                    team1_wins,
                    team2_wins,
                    CASE
                        WHEN team1_wins > team2_wins THEN team1_key
                        WHEN team2_wins > team1_wins THEN team2_key
                        ELSE NULL
                    END AS series_winner_key,
                    CASE
                        WHEN team1_wins > team2_wins THEN team1_name
                        WHEN team2_wins > team1_wins THEN team2_name
                        ELSE NULL
                    END AS series_winner_name,
                    CASE
                        WHEN games_played = 1 THEN 1
                        WHEN games_played <= 3 THEN 3
                        WHEN games_played <= 5 THEN 5
                        ELSE NULL
                    END AS best_of_inferred,
                    CASE
                        WHEN games_played = 1 THEN 'Bo1'
                        WHEN games_played <= 3 THEN 'Bo3'
                        WHEN games_played <= 5 THEN 'Bo5'
                        ELSE 'Unknown'
                    END AS series_format_inferred,
                    CAST(greatest(team1_wins, team2_wins) AS VARCHAR) ||
                        '-' || CAST(least(team1_wins, team2_wins) AS VARCHAR) AS score,
                    avg_game_length_seconds,
                    game_ids,
                    game_numbers,
                    start_time,
                    end_time,
                    is_multi_day_merge,
                    {series_team1_norm} AS team1_name_norm,
                    {series_team2_norm} AS team2_name_norm
                FROM merged_series
                WHERE NOT is_first_day_of_multi
            ),
            official_matches AS (
                SELECT
                    official_match_id,
                    official_start_time,
                    official_match_date,
                    CASE
                        WHEN official_league_name = 'WORLDS' THEN 'WLDs'
                        WHEN official_league_name = 'CBLOL' THEN 'CBLOL'
                        ELSE official_league_name
                    END AS league_code,
                    official_best_of,
                    official_block_name,
                    official_strategy_type,
                    official_team1_name,
                    official_team2_name,
                    CASE
                        WHEN regexp_replace(lower(coalesce(official_team1_name, '')), '[^a-z0-9]+', '', 'g')
                           < regexp_replace(lower(coalesce(official_team2_name, '')), '[^a-z0-9]+', '', 'g')
                        THEN regexp_replace(lower(coalesce(official_team1_name, '')), '[^a-z0-9]+', '', 'g')
                        ELSE regexp_replace(lower(coalesce(official_team2_name, '')), '[^a-z0-9]+', '', 'g')
                    END AS team_lo_norm,
                    CASE
                        WHEN regexp_replace(lower(coalesce(official_team1_name, '')), '[^a-z0-9]+', '', 'g')
                           < regexp_replace(lower(coalesce(official_team2_name, '')), '[^a-z0-9]+', '', 'g')
                        THEN regexp_replace(lower(coalesce(official_team2_name, '')), '[^a-z0-9]+', '', 'g')
                        ELSE regexp_replace(lower(coalesce(official_team1_name, '')), '[^a-z0-9]+', '', 'g')
                    END AS team_hi_norm,
                    official_team1_game_wins,
                    official_team2_game_wins
                FROM bronze_official_matches
            ),
            official_coverage AS (
                SELECT
                    league_code,
                    MIN(official_match_date) AS first_official_match_date,
                    MAX(official_match_date) AS last_official_match_date
                FROM official_matches
                GROUP BY league_code
            )
            SELECT
                '{snapshot_id}' AS snapshot_id,
                f.series_key,
                f.series_date,
                f.league_code,
                f.season_year,
                COALESCE(NULLIF(f.split_name, ''), f.league_code) AS split_name,
                f.playoffs,
                f.patch_version,
                f.team1_key,
                f.team1_name,
                f.team1_source_team_id,
                f.team2_key,
                f.team2_name,
                f.team2_source_team_id,
                f.games_played,
                f.team1_wins,
                f.team2_wins,
                f.series_winner_key,
                f.series_winner_name,
                f.best_of_inferred,
                f.series_format_inferred,
                f.score,
                f.avg_game_length_seconds,
                f.game_ids,
                f.game_numbers,
                f.start_time,
                f.end_time,
                f.is_multi_day_merge,
                f.team1_name_norm,
                f.team2_name_norm,
                om.official_match_id,
                om.official_start_time,
                om.official_block_name,
                om.official_strategy_type,
                om.official_best_of,
                om.official_team1_name,
                om.official_team2_name,
                om.official_team1_game_wins,
                om.official_team2_game_wins,
                oc.first_official_match_date,
                oc.last_official_match_date,
                CASE
                    WHEN om.official_match_id IS NOT NULL THEN 'matched_official'
                    WHEN oc.league_code IS NOT NULL
                     AND f.series_date BETWEEN oc.first_official_match_date AND oc.last_official_match_date
                    THEN 'unmatched_in_official_coverage'
                    ELSE 'no_official_reference_available'
                END AS official_match_status,
                'oracle_series_derivation_v1' AS derivation_rule
            FROM finalized f
            LEFT JOIN official_matches om
                ON f.series_date = om.official_match_date
               AND f.league_code = om.league_code
               AND (
                    CASE
                        WHEN f.team1_name_norm < f.team2_name_norm THEN f.team1_name_norm
                        ELSE f.team2_name_norm
                    END
               ) = om.team_lo_norm
               AND (
                    CASE
                        WHEN f.team1_name_norm < f.team2_name_norm THEN f.team2_name_norm
                        ELSE f.team1_name_norm
                    END
               ) = om.team_hi_norm
               AND (f.best_of_inferred = om.official_best_of OR om.official_best_of IS NULL)
            LEFT JOIN official_coverage oc
                ON f.league_code = oc.league_code
            QUALIFY row_number() OVER (
                PARTITION BY f.series_key
                ORDER BY om.official_start_time NULLS LAST, om.official_match_id
            ) = 1
            """.format(
                snapshot_id=self.snapshot_id,
                series_team1_norm=self._team_norm_sql("team1_name"),
                series_team2_norm=self._team_norm_sql("team2_name"),
            )
        )

        self.con.execute(
            """
            CREATE OR REPLACE TABLE gold_external_reconciliation AS
            WITH series_base AS (
                SELECT
                    snapshot_id,
                    series_key,
                    series_date,
                    league_code,
                    season_year,
                    split_name,
                    playoffs,
                    patch_version,
                    team1_key,
                    team1_name,
                    team1_source_team_id,
                    team2_key,
                    team2_name,
                    team2_source_team_id,
                    games_played,
                    team1_wins,
                    team2_wins,
                    best_of_inferred,
                    score,
                    start_time,
                    end_time,
                    official_match_id,
                    official_start_time,
                    official_best_of,
                    official_team1_name,
                    official_team2_name,
                    official_team1_game_wins,
                    official_team2_game_wins,
                    official_match_status,
                    team1_name_norm,
                    team2_name_norm,
                    CASE
                        WHEN team1_name_norm < team2_name_norm THEN team1_name_norm
                        ELSE team2_name_norm
                    END AS series_team_lo_norm,
                    CASE
                        WHEN team1_name_norm < team2_name_norm THEN team2_name_norm
                        ELSE team1_name_norm
                    END AS series_team_hi_norm,
                    CASE
                        WHEN team1_name_norm < team2_name_norm THEN team1_wins
                        ELSE team2_wins
                    END AS series_team_lo_wins,
                    CASE
                        WHEN team1_name_norm < team2_name_norm THEN team2_wins
                        ELSE team1_wins
                    END AS series_team_hi_wins
                FROM gold_fact_series
            ),
            leaguepedia_candidates AS (
                SELECT
                    s.series_key,
                    l.leaguepedia_match_id,
                    l.leaguepedia_match_date,
                    l.leaguepedia_start_time,
                    l.leaguepedia_best_of,
                    l.leaguepedia_overview_page,
                    l.leaguepedia_shown_name,
                    l.leaguepedia_patch,
                    l.leaguepedia_round,
                    l.leaguepedia_phase,
                    l.leaguepedia_tab,
                    l.leaguepedia_match_day,
                    l.leaguepedia_team1_name,
                    l.leaguepedia_team2_name,
                    l.leaguepedia_team1_score,
                    l.leaguepedia_team2_score,
                    l.team_lo_score,
                    l.team_hi_score,
                    ABS(datediff('day', s.series_date, l.leaguepedia_match_date)) AS day_distance,
                    ABS(datediff('minute', s.start_time, l.leaguepedia_start_time)) AS minute_distance,
                    ROW_NUMBER() OVER (
                        PARTITION BY s.series_key
                        ORDER BY
                            ABS(datediff('day', s.series_date, l.leaguepedia_match_date)),
                            ABS(datediff('minute', s.start_time, l.leaguepedia_start_time)),
                            l.leaguepedia_match_id
                    ) AS candidate_rank,
                    COUNT(*) OVER (PARTITION BY s.series_key) AS candidate_count
                FROM series_base s
                JOIN bronze_leaguepedia_matches l
                    ON s.series_team_lo_norm = l.team_lo_norm
                   AND s.series_team_hi_norm = l.team_hi_norm
                   AND (
                        s.best_of_inferred = l.leaguepedia_best_of
                        OR l.leaguepedia_best_of IS NULL
                        OR s.best_of_inferred IS NULL
                   )
                   AND ABS(datediff('day', s.series_date, l.leaguepedia_match_date)) <= 1
            ),
            best_leaguepedia_match AS (
                SELECT *
                FROM leaguepedia_candidates
                WHERE candidate_rank = 1
            ),
            leaguepedia_coverage AS (
                SELECT
                    s.league_code,
                    s.season_year,
                    MIN(lm.leaguepedia_match_date) AS first_leaguepedia_match_date,
                    MAX(lm.leaguepedia_match_date) AS last_leaguepedia_match_date,
                    COUNT(*) AS leaguepedia_match_count
                FROM series_base s
                JOIN best_leaguepedia_match lm
                    ON s.series_key = lm.series_key
                GROUP BY s.league_code, s.season_year
            )
            SELECT
                s.snapshot_id,
                s.series_key,
                s.series_date,
                s.league_code,
                s.season_year,
                s.split_name,
                s.playoffs,
                s.patch_version,
                s.team1_key,
                s.team1_name,
                s.team1_source_team_id,
                s.team2_key,
                s.team2_name,
                s.team2_source_team_id,
                s.games_played,
                s.team1_wins,
                s.team2_wins,
                s.best_of_inferred,
                s.score,
                s.official_match_id,
                s.official_start_time,
                s.official_best_of,
                s.official_team1_name,
                s.official_team2_name,
                s.official_team1_game_wins,
                s.official_team2_game_wins,
                s.official_match_status,
                lm.leaguepedia_match_id,
                lm.leaguepedia_match_date,
                lm.leaguepedia_start_time,
                lm.leaguepedia_best_of,
                lm.leaguepedia_overview_page,
                lm.leaguepedia_shown_name,
                lm.leaguepedia_patch,
                lm.leaguepedia_round,
                lm.leaguepedia_phase,
                lm.leaguepedia_tab,
                lm.leaguepedia_match_day,
                lm.leaguepedia_team1_name,
                lm.leaguepedia_team2_name,
                lm.leaguepedia_team1_score,
                lm.leaguepedia_team2_score,
                lm.candidate_count AS leaguepedia_candidate_count,
                CASE
                    WHEN lm.leaguepedia_match_id IS NOT NULL THEN 'matched_leaguepedia'
                    WHEN lc.leaguepedia_match_count > 0
                     AND s.series_date BETWEEN lc.first_leaguepedia_match_date AND lc.last_leaguepedia_match_date
                    THEN 'unmatched_in_leaguepedia_coverage'
                    ELSE 'no_leaguepedia_reference_available'
                END AS leaguepedia_match_status,
                CASE
                    WHEN lm.leaguepedia_match_id IS NOT NULL
                     AND lm.team_lo_score = s.series_team_lo_wins
                     AND lm.team_hi_score = s.series_team_hi_wins
                    THEN TRUE
                    WHEN lm.leaguepedia_match_id IS NOT NULL
                    THEN FALSE
                    ELSE NULL
                END AS leaguepedia_score_matches_gold,
                CASE
                    WHEN s.official_match_status = 'matched_official'
                     AND lm.leaguepedia_match_id IS NOT NULL
                    THEN 'riot_and_leaguepedia_matched'
                    WHEN s.official_match_status = 'matched_official'
                    THEN 'riot_only'
                    WHEN lm.leaguepedia_match_id IS NOT NULL
                    THEN 'leaguepedia_only'
                    ELSE 'no_external_match'
                END AS triangulation_status,
                CASE
                    WHEN s.official_match_status = 'matched_official'
                     AND lm.leaguepedia_match_id IS NOT NULL
                     AND (
                        lm.team_lo_score <> s.series_team_lo_wins
                        OR lm.team_hi_score <> s.series_team_hi_wins
                     )
                    THEN TRUE
                    WHEN s.official_match_status = 'matched_official'
                     AND lm.leaguepedia_match_id IS NOT NULL
                    THEN FALSE
                    ELSE NULL
                END AS multi_source_score_conflict
            FROM series_base s
            LEFT JOIN best_leaguepedia_match lm
                ON s.series_key = lm.series_key
            LEFT JOIN leaguepedia_coverage lc
                ON s.league_code = lc.league_code
               AND s.season_year = lc.season_year
            """
        )

        self.con.execute(
            """
            CREATE OR REPLACE TABLE gold_source_coverage AS
            WITH coverage_by_league AS (
                SELECT
                    snapshot_id,
                    league_code,
                    COUNT(*) AS total_series,
                    COUNT(*) FILTER (
                        WHERE official_match_status IN (
                            'matched_official',
                            'unmatched_in_official_coverage'
                        )
                    ) AS official_covered_series,
                    COUNT(*) FILTER (
                        WHERE official_match_status = 'matched_official'
                    ) AS official_matched_series,
                    COUNT(*) FILTER (
                        WHERE leaguepedia_match_status IN (
                            'matched_leaguepedia',
                            'unmatched_in_leaguepedia_coverage'
                        )
                    ) AS leaguepedia_covered_series,
                    COUNT(*) FILTER (
                        WHERE leaguepedia_match_status = 'matched_leaguepedia'
                    ) AS leaguepedia_matched_series,
                    COUNT(*) FILTER (
                        WHERE triangulation_status = 'riot_and_leaguepedia_matched'
                    ) AS triangulated_series,
                    COUNT(*) FILTER (
                        WHERE multi_source_score_conflict = TRUE
                    ) AS multi_source_score_conflicts
                FROM gold_external_reconciliation
                GROUP BY snapshot_id, league_code
            )
            SELECT
                snapshot_id,
                league_code,
                total_series,
                official_covered_series,
                official_matched_series,
                ROUND(
                    CASE
                        WHEN official_covered_series = 0 THEN 0
                        ELSE official_matched_series * 100.0 / official_covered_series
                    END,
                    4
                ) AS official_match_rate_pct,
                leaguepedia_covered_series,
                leaguepedia_matched_series,
                ROUND(
                    CASE
                        WHEN leaguepedia_covered_series = 0 THEN 0
                        ELSE leaguepedia_matched_series * 100.0 / leaguepedia_covered_series
                    END,
                    4
                ) AS leaguepedia_match_rate_pct,
                triangulated_series,
                multi_source_score_conflicts
            FROM coverage_by_league

            UNION ALL

            SELECT
                snapshot_id,
                'ALL' AS league_code,
                SUM(total_series) AS total_series,
                SUM(official_covered_series) AS official_covered_series,
                SUM(official_matched_series) AS official_matched_series,
                ROUND(
                    CASE
                        WHEN SUM(official_covered_series) = 0 THEN 0
                        ELSE SUM(official_matched_series) * 100.0 / SUM(official_covered_series)
                    END,
                    4
                ) AS official_match_rate_pct,
                SUM(leaguepedia_covered_series) AS leaguepedia_covered_series,
                SUM(leaguepedia_matched_series) AS leaguepedia_matched_series,
                ROUND(
                    CASE
                        WHEN SUM(leaguepedia_covered_series) = 0 THEN 0
                        ELSE SUM(leaguepedia_matched_series) * 100.0 / SUM(leaguepedia_covered_series)
                    END,
                    4
                ) AS leaguepedia_match_rate_pct,
                SUM(triangulated_series) AS triangulated_series,
                SUM(multi_source_score_conflicts) AS multi_source_score_conflicts
            FROM coverage_by_league
            GROUP BY snapshot_id
            """
        )

    def _build_model_ready_tables(self) -> None:
        """Create filtered Gold tables suitable for initial model training."""
        logger.info("Building model-ready Gold tables")

        core_leagues = ", ".join(f"'{league}'" for league in self.MODEL_CORE_LEAGUES)
        self.con.execute(
            f"""
            CREATE OR REPLACE TABLE gold_model_core_series AS
            SELECT
                f.snapshot_id,
                f.series_key,
                f.series_date,
                f.league_code,
                f.season_year,
                f.split_name,
                f.playoffs,
                f.patch_version,
                f.team1_key,
                f.team1_name,
                f.team1_source_team_id,
                f.team2_key,
                f.team2_name,
                f.team2_source_team_id,
                f.games_played,
                f.team1_wins,
                f.team2_wins,
                f.series_winner_key,
                f.series_winner_name,
                CASE
                    WHEN f.series_winner_key = f.team1_key THEN 1
                    WHEN f.series_winner_key = f.team2_key THEN 0
                    ELSE NULL
                END AS label_team1_win,
                f.best_of_inferred,
                f.series_format_inferred,
                f.score,
                f.avg_game_length_seconds,
                f.start_time,
                f.end_time,
                r.official_match_status,
                r.leaguepedia_match_status,
                r.leaguepedia_score_matches_gold,
                r.triangulation_status,
                r.multi_source_score_conflict
            FROM gold_fact_series f
            JOIN gold_external_reconciliation r
                ON f.series_key = r.series_key
            WHERE f.league_code IN ({core_leagues})
              AND r.leaguepedia_match_status = 'matched_leaguepedia'
              AND coalesce(r.leaguepedia_score_matches_gold, FALSE) = TRUE
              AND r.official_match_status <> 'unmatched_in_official_coverage'
              AND coalesce(r.multi_source_score_conflict, FALSE) = FALSE
              AND f.team1_source_team_id IS NOT NULL
              AND f.team2_source_team_id IS NOT NULL
              AND coalesce(f.patch_version, '') <> ''
              AND f.series_winner_key IS NOT NULL
            """
        )

        self.con.execute(
            """
            CREATE OR REPLACE TABLE gold_fact_game_player AS
            SELECT
                '{snapshot_id}' AS snapshot_id,
                p.game_id AS game_key,
                CASE
                    WHEN p.participant_id IS NOT NULL THEN p.game_id || ':' || CAST(p.participant_id AS VARCHAR)
                    ELSE p.game_id || ':' || p.side || ':' || coalesce(p.role, 'unknown')
                END AS game_player_key,
                p.game_id,
                p.game_date,
                p.game_datetime,
                p.season_year,
                p.league_code,
                p.split_name,
                p.playoffs,
                p.patch_version,
                p.game_number,
                p.data_completeness,
                p.player_key,
                p.source_player_id,
                p.player_name,
                p.player_name_norm,
                p.participant_id,
                COALESCE(
                    t.team_key,
                    CASE
                        WHEN p.source_team_id IS NOT NULL THEN 'team:' || p.source_team_id
                        ELSE 'teamname:' ||
                            regexp_replace(lower(coalesce(p.team_name, 'unknown')), '[^a-z0-9]+', '', 'g')
                    END
                ) AS team_key,
                p.source_team_id,
                p.team_name,
                t.opponent_team_key,
                t.opponent_team_name,
                p.side,
                p.role,
                p.champion_name,
                p.win,
                p.game_length_seconds,
                p.kills,
                p.deaths,
                p.assists,
                CASE
                    WHEN p.deaths IS NULL OR p.deaths = 0 THEN CAST(p.kills + p.assists AS DOUBLE)
                    ELSE (p.kills + p.assists) * 1.0 / p.deaths
                END AS kda,
                p.damage_to_champions,
                p.dpm,
                p.damage_share,
                p.damage_taken_per_minute,
                p.damage_mitigated_per_minute,
                p.damage_to_towers,
                p.total_gold,
                p.earned_gold,
                p.earned_gpm,
                p.earned_gold_share,
                p.gold_spent,
                p.minion_kills,
                p.monster_kills,
                p.total_cs,
                p.cs_per_minute,
                p.enemy_jungle_cs,
                p.own_jungle_cs,
                p.vision_score,
                p.vision_score_per_minute,
                p.wards_placed,
                p.wards_killed,
                p.control_wards_bought,
                p.gold_at_10,
                p.xp_at_10,
                p.cs_at_10,
                p.gold_diff_at_10,
                p.xp_diff_at_10,
                p.cs_diff_at_10,
                p.kills_at_10,
                p.assists_at_10,
                p.deaths_at_10,
                p.gold_at_15,
                p.xp_at_15,
                p.cs_at_15,
                p.gold_diff_at_15,
                p.xp_diff_at_15,
                p.cs_diff_at_15,
                p.kills_at_15,
                p.assists_at_15,
                p.deaths_at_15,
                p.double_kills,
                p.triple_kills,
                p.quadra_kills,
                p.penta_kills,
                p.first_blood_kill,
                p.first_blood_assist,
                p.first_blood_victim,
                CASE WHEN p.source_player_id IS NULL THEN TRUE ELSE FALSE END AS has_missing_source_player_id,
                'oracle_elixir' AS source_system,
                p.source_file
            FROM bronze_oracle_player_rows p
            LEFT JOIN gold_team_game_base t
                ON p.game_id = t.game_id
               AND p.side = t.side
            """.format(snapshot_id=self.snapshot_id)
        )

        self.con.execute(
            """
            CREATE OR REPLACE TABLE gold_fact_draft AS
            SELECT
                '{snapshot_id}' AS snapshot_id,
                game_id AS game_key,
                game_id,
                game_date,
                game_datetime,
                season_year,
                league_code,
                split_name,
                playoffs,
                patch_version,
                game_number,
                team_key,
                team_name,
                opponent_team_key,
                opponent_team_name,
                side,
                first_pick,
                pick_1,
                pick_2,
                pick_3,
                pick_4,
                pick_5,
                ban_1,
                ban_2,
                ban_3,
                ban_4,
                ban_5,
                CASE
                    WHEN pick_1 IS NOT NULL AND pick_2 IS NOT NULL AND pick_3 IS NOT NULL
                      AND pick_4 IS NOT NULL AND pick_5 IS NOT NULL
                    THEN TRUE
                    ELSE FALSE
                END AS draft_complete,
                'oracle_elixir' AS source_system,
                source_file
            FROM gold_team_game_base
            """.format(snapshot_id=self.snapshot_id)
        )

        self.con.execute(
            """
            CREATE OR REPLACE TABLE gold_fact_series AS
            WITH game_matchups AS (
                SELECT
                    t1.game_id,
                    t1.game_date,
                    t1.game_datetime,
                    t1.season_year,
                    t1.league_code,
                    t1.split_name,
                    t1.playoffs,
                    t1.patch_version,
                    t1.game_number,
                    CASE WHEN t1.team_key < t2.team_key THEN t1.team_key ELSE t2.team_key END AS team1_key,
                    CASE WHEN t1.team_key < t2.team_key THEN t1.team_name ELSE t2.team_name END AS team1_name,
                    CASE WHEN t1.team_key < t2.team_key THEN t1.source_team_id ELSE t2.source_team_id END
                        AS team1_source_team_id,
                    CASE WHEN t1.team_key < t2.team_key THEN t1.win ELSE t2.win END AS team1_win,
                    CASE WHEN t1.team_key < t2.team_key THEN t2.team_key ELSE t1.team_key END AS team2_key,
                    CASE WHEN t1.team_key < t2.team_key THEN t2.team_name ELSE t1.team_name END AS team2_name,
                    CASE WHEN t1.team_key < t2.team_key THEN t2.source_team_id ELSE t1.source_team_id END
                        AS team2_source_team_id,
                    CASE WHEN t1.team_key < t2.team_key THEN t2.win ELSE t1.win END AS team2_win,
                    CASE
                        WHEN t1.patch_version IS NOT NULL THEN t1.patch_version
                        ELSE t2.patch_version
                    END AS patch_version,
                    COALESCE(t1.game_length_seconds, t2.game_length_seconds) AS game_length_seconds
                FROM gold_fact_game_team t1
                JOIN gold_fact_game_team t2
                    ON t1.game_id = t2.game_id
                   AND t1.team_key < t2.team_key
            ),
            daily_series AS (
                SELECT
                    game_date AS match_date,
                    league_code,
                    season_year,
                    split_name,
                    MAX(playoffs) AS playoffs,
                    team1_key,
                    arg_max(team1_name, game_datetime) AS team1_name,
                    arg_max(team1_source_team_id, game_datetime) AS team1_source_team_id,
                    team2_key,
                    arg_max(team2_name, game_datetime) AS team2_name,
                    arg_max(team2_source_team_id, game_datetime) AS team2_source_team_id,
                    COUNT(*) AS games_played,
                    SUM(team1_win) AS team1_wins,
                    SUM(team2_win) AS team2_wins,
                    arg_max(patch_version, game_datetime) AS patch_version,
                    array_agg(game_id ORDER BY game_number) AS game_ids,
                    array_agg(game_number ORDER BY game_number) AS game_numbers,
                    AVG(game_length_seconds) AS avg_game_length_seconds,
                    MIN(game_datetime) AS start_time,
                    MAX(game_datetime) AS end_time
                FROM game_matchups
                GROUP BY
                    game_date,
                    league_code,
                    season_year,
                    split_name,
                    team1_key,
                    team2_key
            ),
            with_prev_day AS (
                SELECT
                    *,
                    LAG(match_date) OVER (
                        PARTITION BY league_code, season_year, team1_key, team2_key
                        ORDER BY match_date
                    ) AS prev_match_date,
                    LAG(team1_wins) OVER (
                        PARTITION BY league_code, season_year, team1_key, team2_key
                        ORDER BY match_date
                    ) AS prev_team1_wins,
                    LAG(team2_wins) OVER (
                        PARTITION BY league_code, season_year, team1_key, team2_key
                        ORDER BY match_date
                    ) AS prev_team2_wins,
                    LAG(games_played) OVER (
                        PARTITION BY league_code, season_year, team1_key, team2_key
                        ORDER BY match_date
                    ) AS prev_games_played,
                    LAG(game_ids) OVER (
                        PARTITION BY league_code, season_year, team1_key, team2_key
                        ORDER BY match_date
                    ) AS prev_game_ids,
                    LAG(game_numbers) OVER (
                        PARTITION BY league_code, season_year, team1_key, team2_key
                        ORDER BY match_date
                    ) AS prev_game_numbers,
                    LAG(start_time) OVER (
                        PARTITION BY league_code, season_year, team1_key, team2_key
                        ORDER BY match_date
                    ) AS prev_start_time
                FROM daily_series
            ),
            merged_series AS (
                SELECT
                    CASE
                        WHEN prev_match_date IS NOT NULL
                         AND datediff('day', prev_match_date, match_date) = 1
                        THEN prev_match_date
                        ELSE match_date
                    END AS series_date,
                    league_code,
                    season_year,
                    split_name,
                    playoffs,
                    team1_key,
                    team1_name,
                    team1_source_team_id,
                    team2_key,
                    team2_name,
                    team2_source_team_id,
                    CASE
                        WHEN prev_match_date IS NOT NULL
                         AND datediff('day', prev_match_date, match_date) = 1
                        THEN games_played + prev_games_played
                        ELSE games_played
                    END AS games_played,
                    CASE
                        WHEN prev_match_date IS NOT NULL
                         AND datediff('day', prev_match_date, match_date) = 1
                        THEN team1_wins + prev_team1_wins
                        ELSE team1_wins
                    END AS team1_wins,
                    CASE
                        WHEN prev_match_date IS NOT NULL
                         AND datediff('day', prev_match_date, match_date) = 1
                        THEN team2_wins + prev_team2_wins
                        ELSE team2_wins
                    END AS team2_wins,
                    patch_version,
                    CASE
                        WHEN prev_match_date IS NOT NULL
                         AND datediff('day', prev_match_date, match_date) = 1
                        THEN list_concat(prev_game_ids, game_ids)
                        ELSE game_ids
                    END AS game_ids,
                    CASE
                        WHEN prev_match_date IS NOT NULL
                         AND datediff('day', prev_match_date, match_date) = 1
                        THEN list_concat(prev_game_numbers, game_numbers)
                        ELSE game_numbers
                    END AS game_numbers,
                    avg_game_length_seconds,
                    CASE
                        WHEN prev_match_date IS NOT NULL
                         AND datediff('day', prev_match_date, match_date) = 1
                        THEN prev_start_time
                        ELSE start_time
                    END AS start_time,
                    end_time,
                    CASE
                        WHEN prev_match_date IS NOT NULL
                         AND datediff('day', prev_match_date, match_date) = 1
                        THEN TRUE
                        ELSE FALSE
                    END AS is_multi_day_merge,
                    CASE
                        WHEN lead(match_date) OVER (
                            PARTITION BY league_code, season_year, team1_key, team2_key
                            ORDER BY match_date
                        ) IS NOT NULL
                         AND datediff(
                            'day',
                            match_date,
                            lead(match_date) OVER (
                                PARTITION BY league_code, season_year, team1_key, team2_key
                                ORDER BY match_date
                            )
                        ) = 1
                        THEN TRUE
                        ELSE FALSE
                    END AS is_first_day_of_multi
                FROM with_prev_day
            ),
            finalized AS (
                SELECT
                    'series:' || league_code || ':' || season_year || ':' || CAST(series_date AS VARCHAR)
                        || ':' || team1_key || ':' || team2_key AS series_key,
                    series_date,
                    league_code,
                    season_year,
                    split_name,
                    playoffs,
                    patch_version,
                    team1_key,
                    team1_name,
                    team1_source_team_id,
                    team2_key,
                    team2_name,
                    team2_source_team_id,
                    games_played,
                    team1_wins,
                    team2_wins,
                    CASE
                        WHEN team1_wins > team2_wins THEN team1_key
                        WHEN team2_wins > team1_wins THEN team2_key
                        ELSE NULL
                    END AS series_winner_key,
                    CASE
                        WHEN team1_wins > team2_wins THEN team1_name
                        WHEN team2_wins > team1_wins THEN team2_name
                        ELSE NULL
                    END AS series_winner_name,
                    CASE
                        WHEN games_played = 1 THEN 1
                        WHEN games_played <= 3 THEN 3
                        WHEN games_played <= 5 THEN 5
                        ELSE NULL
                    END AS best_of_inferred,
                    CASE
                        WHEN games_played = 1 THEN 'Bo1'
                        WHEN games_played <= 3 THEN 'Bo3'
                        WHEN games_played <= 5 THEN 'Bo5'
                        ELSE 'Unknown'
                    END AS series_format_inferred,
                    CAST(greatest(team1_wins, team2_wins) AS VARCHAR) ||
                        '-' || CAST(least(team1_wins, team2_wins) AS VARCHAR) AS score,
                    avg_game_length_seconds,
                    game_ids,
                    game_numbers,
                    start_time,
                    end_time,
                    is_multi_day_merge,
                    {series_team1_norm} AS team1_name_norm,
                    {series_team2_norm} AS team2_name_norm
                FROM merged_series
                WHERE NOT is_first_day_of_multi
            ),
            official_matches AS (
                SELECT
                    official_match_id,
                    official_start_time,
                    official_match_date,
                    CASE
                        WHEN official_league_name = 'WORLDS' THEN 'WLDs'
                        WHEN official_league_name = 'CBLOL' THEN 'CBLOL'
                        ELSE official_league_name
                    END AS league_code,
                    official_best_of,
                    official_block_name,
                    official_strategy_type,
                    official_team1_name,
                    official_team2_name,
                    CASE
                        WHEN regexp_replace(lower(coalesce(official_team1_name, '')), '[^a-z0-9]+', '', 'g')
                           < regexp_replace(lower(coalesce(official_team2_name, '')), '[^a-z0-9]+', '', 'g')
                        THEN regexp_replace(lower(coalesce(official_team1_name, '')), '[^a-z0-9]+', '', 'g')
                        ELSE regexp_replace(lower(coalesce(official_team2_name, '')), '[^a-z0-9]+', '', 'g')
                    END AS team_lo_norm,
                    CASE
                        WHEN regexp_replace(lower(coalesce(official_team1_name, '')), '[^a-z0-9]+', '', 'g')
                           < regexp_replace(lower(coalesce(official_team2_name, '')), '[^a-z0-9]+', '', 'g')
                        THEN regexp_replace(lower(coalesce(official_team2_name, '')), '[^a-z0-9]+', '', 'g')
                        ELSE regexp_replace(lower(coalesce(official_team1_name, '')), '[^a-z0-9]+', '', 'g')
                    END AS team_hi_norm,
                    official_team1_game_wins,
                    official_team2_game_wins
                FROM bronze_official_matches
            ),
            official_coverage AS (
                SELECT
                    league_code,
                    MIN(official_match_date) AS first_official_match_date,
                    MAX(official_match_date) AS last_official_match_date
                FROM official_matches
                GROUP BY league_code
            )
            SELECT
                '{snapshot_id}' AS snapshot_id,
                f.series_key,
                f.series_date,
                f.league_code,
                f.season_year,
                f.split_name,
                f.playoffs,
                f.patch_version,
                f.team1_key,
                f.team1_name,
                f.team1_source_team_id,
                f.team2_key,
                f.team2_name,
                f.team2_source_team_id,
                f.games_played,
                f.team1_wins,
                f.team2_wins,
                f.series_winner_key,
                f.series_winner_name,
                f.best_of_inferred,
                f.series_format_inferred,
                f.score,
                f.avg_game_length_seconds,
                f.game_ids,
                f.game_numbers,
                f.start_time,
                f.end_time,
                f.is_multi_day_merge,
                f.team1_name_norm,
                f.team2_name_norm,
                om.official_match_id,
                om.official_start_time,
                om.official_block_name,
                om.official_strategy_type,
                om.official_best_of,
                om.official_team1_name,
                om.official_team2_name,
                om.official_team1_game_wins,
                om.official_team2_game_wins,
                oc.first_official_match_date,
                oc.last_official_match_date,
                CASE
                    WHEN om.official_match_id IS NOT NULL THEN 'matched_official'
                    WHEN oc.league_code IS NOT NULL
                     AND f.series_date BETWEEN oc.first_official_match_date AND oc.last_official_match_date
                    THEN 'unmatched_in_official_coverage'
                    ELSE 'no_official_reference_available'
                END AS official_match_status,
                'oracle_series_derivation_v1' AS derivation_rule
            FROM finalized f
            LEFT JOIN official_matches om
                ON f.series_date = om.official_match_date
               AND f.league_code = om.league_code
               AND (
                    CASE
                        WHEN f.team1_name_norm < f.team2_name_norm THEN f.team1_name_norm
                        ELSE f.team2_name_norm
                    END
               ) = om.team_lo_norm
               AND (
                    CASE
                        WHEN f.team1_name_norm < f.team2_name_norm THEN f.team2_name_norm
                        ELSE f.team1_name_norm
                    END
               ) = om.team_hi_norm
               AND (f.best_of_inferred = om.official_best_of OR om.official_best_of IS NULL)
            LEFT JOIN official_coverage oc
                ON f.league_code = oc.league_code
            QUALIFY row_number() OVER (
                PARTITION BY f.series_key
                ORDER BY om.official_start_time NULLS LAST, om.official_match_id
            ) = 1
            """.format(
                snapshot_id=self.snapshot_id,
                series_team1_norm=self._team_norm_sql("team1_name"),
                series_team2_norm=self._team_norm_sql("team2_name"),
            )
        )
        self._build_match_features_prematch()

    def _build_match_features_prematch(self) -> None:
        """Create leakage-safe pre-match features from the model core slice."""
        logger.info("Building prematch features for model core series")

        series_df = self.con.execute(
            """
            SELECT
                snapshot_id,
                series_key,
                series_date,
                start_time,
                league_code,
                season_year,
                split_name,
                playoffs,
                patch_version,
                team1_key,
                team1_name,
                team2_key,
                team2_name,
                games_played,
                team1_wins,
                team2_wins,
                label_team1_win,
                best_of_inferred,
                avg_game_length_seconds
            FROM gold_model_core_series
            ORDER BY coalesce(start_time, CAST(series_date AS TIMESTAMP)), series_key
            """
        ).df()

        roster_df = self.con.execute(
            """
            WITH model_series AS (
                SELECT series_key
                FROM gold_model_core_series
            ),
            series_games AS (
                SELECT
                    s.series_key,
                    unnest(s.game_ids) AS game_id,
                    s.team1_key,
                    s.team2_key
                FROM gold_fact_series s
                JOIN model_series m
                    ON s.series_key = m.series_key
            )
            SELECT
                sg.series_key,
                p.team_key,
                p.player_key,
                COUNT(DISTINCT p.game_id) AS games_played
            FROM series_games sg
            JOIN gold_fact_game_player p
                ON sg.game_id = p.game_id
               AND (p.team_key = sg.team1_key OR p.team_key = sg.team2_key)
            GROUP BY
                sg.series_key,
                p.team_key,
                p.player_key
            ORDER BY
                sg.series_key,
                p.team_key,
                games_played DESC,
                p.player_key
            """
        ).df()

        draft_history_df = self.con.execute(
            """
            WITH model_series AS (
                SELECT series_key
                FROM gold_model_core_series
            ),
            series_games AS (
                SELECT
                    s.series_key,
                    unnest(s.game_ids) AS game_id,
                    s.team1_key,
                    s.team2_key
                FROM gold_fact_series s
                JOIN model_series m
                    ON s.series_key = m.series_key
            )
            SELECT
                sg.series_key,
                p.team_key,
                p.game_id,
                COALESCE(MAX(CASE WHEN d.first_pick THEN 1 ELSE 0 END), 0) AS first_pick,
                p.champion_name
            FROM series_games sg
            JOIN gold_fact_game_player p
                ON sg.game_id = p.game_id
               AND (p.team_key = sg.team1_key OR p.team_key = sg.team2_key)
            LEFT JOIN gold_fact_draft d
                ON d.game_id = p.game_id
               AND d.team_key = p.team_key
            WHERE p.champion_name IS NOT NULL
            GROUP BY
                sg.series_key,
                p.team_key,
                p.game_id,
                p.champion_name
            ORDER BY
                sg.series_key,
                p.team_key,
                p.game_id,
                p.champion_name
            """
        ).df()

        feature_columns = [
            "snapshot_id",
            "feature_version",
            "series_key",
            "series_date",
            "start_time",
            "league_code",
            "season_year",
            "split_name",
            "playoffs",
            "patch_version",
            "team1_key",
            "team1_name",
            "team2_key",
            "team2_name",
            "best_of_inferred",
            "label_team1_win",
            "team1_pre_elo",
            "team2_pre_elo",
            "elo_diff",
            "team1_elo_win_prob",
            "team1_prior_series_count",
            "team2_prior_series_count",
            "series_count_diff",
            "team1_prior_series_win_rate",
            "team2_prior_series_win_rate",
            "prior_series_win_rate_diff",
            "team1_recent5_series_count",
            "team2_recent5_series_count",
            "team1_recent5_series_win_rate",
            "team2_recent5_series_win_rate",
            "recent5_series_win_rate_diff",
            "team1_prior_game_count",
            "team2_prior_game_count",
            "prior_game_count_diff",
            "team1_prior_game_win_rate",
            "team2_prior_game_win_rate",
            "prior_game_win_rate_diff",
            "team1_prior_avg_games_played_per_series",
            "team2_prior_avg_games_played_per_series",
            "prior_avg_games_played_per_series_diff",
            "team1_prior_avg_games_won_per_series",
            "team2_prior_avg_games_won_per_series",
            "prior_avg_games_won_per_series_diff",
            "team1_prior_avg_game_length_seconds",
            "team2_prior_avg_game_length_seconds",
            "prior_avg_game_length_seconds_diff",
            "team1_series_streak",
            "team2_series_streak",
            "series_streak_diff",
            "team1_days_since_last_series",
            "team2_days_since_last_series",
            "days_since_last_series_diff",
            "team1_patch_prior_series_count",
            "team2_patch_prior_series_count",
            "patch_prior_series_count_diff",
            "team1_patch_prior_win_rate",
            "team2_patch_prior_win_rate",
            "patch_prior_win_rate_diff",
            "team1_split_prior_series_count",
            "team2_split_prior_series_count",
            "split_prior_series_count_diff",
            "team1_split_prior_win_rate",
            "team2_split_prior_win_rate",
            "split_prior_win_rate_diff",
            "h2h_prior_series_count",
            "h2h_team1_series_wins",
            "h2h_team2_series_wins",
            "h2h_team1_series_win_rate",
            "h2h_prior_game_count",
            "h2h_team1_game_wins",
            "h2h_team2_game_wins",
            "h2h_team1_game_win_rate",
            "team1_prev_series_roster_overlap",
            "team2_prev_series_roster_overlap",
            "prev_series_roster_overlap_diff",
            "team1_recent3_avg_roster_overlap",
            "team2_recent3_avg_roster_overlap",
            "recent3_avg_roster_overlap_diff",
            "team1_roster_avg_player_prior_series_count",
            "team2_roster_avg_player_prior_series_count",
            "roster_avg_player_prior_series_count_diff",
            "team1_roster_avg_player_prior_series_win_rate",
            "team2_roster_avg_player_prior_series_win_rate",
            "roster_avg_player_prior_series_win_rate_diff",
            "team1_roster_new_player_count",
            "team2_roster_new_player_count",
            "roster_new_player_count_diff",
            "team1_recent10_unique_champions",
            "team2_recent10_unique_champions",
            "recent10_unique_champions_diff",
            "team1_recent25_unique_champions",
            "team2_recent25_unique_champions",
            "recent25_unique_champions_diff",
            "team1_recent25_top5_champion_share",
            "team2_recent25_top5_champion_share",
            "recent25_top5_champion_share_diff",
            "team1_recent10_first_pick_rate",
            "team2_recent10_first_pick_rate",
            "recent10_first_pick_rate_diff",
        ]

        if series_df.empty:
            empty_df = pd.DataFrame(columns=feature_columns)
            self.con.register("match_features_prematch_df", empty_df)
            self.con.execute(
                """
                CREATE OR REPLACE TABLE gold_match_features_prematch AS
                SELECT * FROM match_features_prematch_df
                """
            )
            self.con.unregister("match_features_prematch_df")
            return

        def _neutral_rate(wins: float, total: float) -> float:
            return 0.5 if not total else float(wins) / float(total)

        def _safe_average(total: float, count: float) -> float:
            return 0.0 if not count else float(total) / float(count)

        def _days_since(last_seen: pd.Timestamp | None, current_time: pd.Timestamp) -> float:
            if last_seen is None:
                return float("nan")
            return (current_time - last_seen).total_seconds() / 86400.0

        def _string_or_empty(value: Any) -> str:
            if value is None or pd.isna(value):
                return ""
            return str(value)

        def _average(values: list[float], neutral: float = 0.0) -> float:
            return neutral if not values else float(sum(values)) / float(len(values))

        def _roster_overlap(current_roster: tuple[str, ...], prior_roster: tuple[str, ...] | None) -> int:
            if not current_roster or not prior_roster:
                return 0
            return len(set(current_roster).intersection(prior_roster))

        def _recent_roster_overlap(
            current_roster: tuple[str, ...], roster_history: deque[tuple[str, ...]]
        ) -> float:
            if not current_roster or not roster_history:
                return 0.0
            overlaps = [_roster_overlap(current_roster, prior_roster) for prior_roster in roster_history]
            return _average([float(value) for value in overlaps], neutral=0.0)

        def _draft_window_metrics(
            game_history: deque[dict[str, Any]], window: int
        ) -> dict[str, float]:
            recent_games = list(game_history)[-window:]
            champion_counter: Counter[str] = Counter()
            first_picks: list[int] = []
            for item in recent_games:
                champion_counter.update(item["champions"])
                first_picks.append(1 if item["first_pick"] else 0)

            total_picks = sum(champion_counter.values())
            top5_share = (
                sum(sorted(champion_counter.values(), reverse=True)[:5]) / total_picks
                if total_picks
                else 0.0
            )
            return {
                "unique_champions": float(len(champion_counter)),
                "top5_share": float(top5_share),
                "first_pick_rate": _average([float(value) for value in first_picks], neutral=0.5),
            }

        series_df["series_date"] = pd.to_datetime(series_df["series_date"], utc=True, errors="coerce")
        series_df["start_time"] = pd.to_datetime(series_df["start_time"], utc=True, errors="coerce")
        series_df["event_time"] = series_df["start_time"].fillna(series_df["series_date"])
        series_df = series_df.sort_values(["event_time", "series_key"], kind="stable").reset_index(drop=True)

        series_rosters: dict[tuple[str, str], tuple[str, ...]] = {}
        if not roster_df.empty:
            roster_df = roster_df.sort_values(
                ["series_key", "team_key", "games_played", "player_key"],
                ascending=[True, True, False, True],
                kind="stable",
            )
            for (series_key, team_key), team_frame in roster_df.groupby(
                ["series_key", "team_key"], sort=False
            ):
                roster_players = team_frame.head(5)["player_key"].tolist()
                series_rosters[(series_key, team_key)] = tuple(str(player) for player in roster_players)

        series_team_games: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        if not draft_history_df.empty:
            draft_history_df = draft_history_df.sort_values(
                ["series_key", "team_key", "game_id", "champion_name"],
                kind="stable",
            )
            for (series_key, team_key, game_id), game_frame in draft_history_df.groupby(
                ["series_key", "team_key", "game_id"], sort=False
            ):
                champions = tuple(
                    sorted(
                        {
                            champion
                            for champion in game_frame["champion_name"].tolist()
                            if isinstance(champion, str) and champion
                        }
                    )
                )
                series_team_games[(series_key, team_key)].append(
                    {
                        "game_id": str(game_id),
                        "champions": champions,
                        "first_pick": bool(game_frame["first_pick"].max()),
                    }
                )

        team_state: defaultdict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "elo": 1500.0,
                "series_count": 0,
                "series_wins": 0,
                "game_count": 0,
                "game_wins": 0,
                "games_played_total": 0,
                "games_won_total": 0,
                "game_length_seconds_total": 0.0,
                "series_streak": 0,
                "last_series_at": None,
            }
        )
        recent_series_results: defaultdict[str, deque[int]] = defaultdict(lambda: deque(maxlen=5))
        patch_state: defaultdict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"series_count": 0, "series_wins": 0}
        )
        split_state: defaultdict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"series_count": 0, "series_wins": 0}
        )
        h2h_state: defaultdict[tuple[str, str], dict[str, Any]] = defaultdict(
            lambda: {
                "series_count": 0,
                "game_count": 0,
                "series_wins": defaultdict(int),
                "game_wins": defaultdict(int),
            }
        )
        player_state: defaultdict[str, dict[str, int]] = defaultdict(
            lambda: {"series_count": 0, "series_wins": 0}
        )
        team_roster_history: defaultdict[str, deque[tuple[str, ...]]] = defaultdict(
            lambda: deque(maxlen=3)
        )
        team_draft_history: defaultdict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=25)
        )

        feature_rows: list[dict[str, Any]] = []

        for _, batch in series_df.groupby("event_time", sort=False):
            pending_updates: list[dict[str, Any]] = []

            for row in batch.itertuples(index=False):
                event_time = row.event_time
                patch_version = _string_or_empty(row.patch_version)
                split_name = _string_or_empty(row.split_name)
                games_played = int(row.games_played) if pd.notna(row.games_played) else 0
                team1_wins = int(row.team1_wins) if pd.notna(row.team1_wins) else 0
                team2_wins = int(row.team2_wins) if pd.notna(row.team2_wins) else 0
                best_of_inferred = (
                    int(row.best_of_inferred)
                    if pd.notna(row.best_of_inferred)
                    else max(games_played, 1)
                )
                avg_game_length_seconds = (
                    float(row.avg_game_length_seconds)
                    if pd.notna(row.avg_game_length_seconds)
                    else 0.0
                )
                label_team1_win = int(row.label_team1_win) if pd.notna(row.label_team1_win) else None
                team1_state = team_state[row.team1_key]
                team2_state = team_state[row.team2_key]
                team1_recent = recent_series_results[row.team1_key]
                team2_recent = recent_series_results[row.team2_key]
                patch_key1 = (row.team1_key, patch_version)
                patch_key2 = (row.team2_key, patch_version)
                split_key1 = (row.team1_key, split_name)
                split_key2 = (row.team2_key, split_name)
                team1_patch_state = patch_state[patch_key1]
                team2_patch_state = patch_state[patch_key2]
                team1_split_state = split_state[split_key1]
                team2_split_state = split_state[split_key2]
                matchup_key = tuple(sorted((row.team1_key, row.team2_key)))
                matchup_state = h2h_state[matchup_key]
                team1_roster = series_rosters.get((row.series_key, row.team1_key), tuple())
                team2_roster = series_rosters.get((row.series_key, row.team2_key), tuple())
                team1_roster_history = team_roster_history[row.team1_key]
                team2_roster_history = team_roster_history[row.team2_key]
                team1_prev_series_roster_overlap = _roster_overlap(
                    team1_roster,
                    team1_roster_history[-1] if team1_roster_history else None,
                )
                team2_prev_series_roster_overlap = _roster_overlap(
                    team2_roster,
                    team2_roster_history[-1] if team2_roster_history else None,
                )
                team1_recent3_avg_roster_overlap = _recent_roster_overlap(
                    team1_roster,
                    team1_roster_history,
                )
                team2_recent3_avg_roster_overlap = _recent_roster_overlap(
                    team2_roster,
                    team2_roster_history,
                )

                team1_player_states = [player_state[player_key] for player_key in team1_roster]
                team2_player_states = [player_state[player_key] for player_key in team2_roster]
                team1_roster_avg_player_prior_series_count = _average(
                    [float(state["series_count"]) for state in team1_player_states],
                    neutral=0.0,
                )
                team2_roster_avg_player_prior_series_count = _average(
                    [float(state["series_count"]) for state in team2_player_states],
                    neutral=0.0,
                )
                team1_roster_avg_player_prior_series_win_rate = _average(
                    [
                        _neutral_rate(state["series_wins"], state["series_count"])
                        for state in team1_player_states
                    ],
                    neutral=0.5,
                )
                team2_roster_avg_player_prior_series_win_rate = _average(
                    [
                        _neutral_rate(state["series_wins"], state["series_count"])
                        for state in team2_player_states
                    ],
                    neutral=0.5,
                )
                team1_roster_new_player_count = sum(
                    1 for state in team1_player_states if state["series_count"] == 0
                )
                team2_roster_new_player_count = sum(
                    1 for state in team2_player_states if state["series_count"] == 0
                )

                team1_draft_recent10 = _draft_window_metrics(
                    team_draft_history[row.team1_key],
                    window=10,
                )
                team2_draft_recent10 = _draft_window_metrics(
                    team_draft_history[row.team2_key],
                    window=10,
                )
                team1_draft_recent25 = _draft_window_metrics(
                    team_draft_history[row.team1_key],
                    window=25,
                )
                team2_draft_recent25 = _draft_window_metrics(
                    team_draft_history[row.team2_key],
                    window=25,
                )

                team1_pre_elo = float(team1_state["elo"])
                team2_pre_elo = float(team2_state["elo"])
                team1_elo_win_prob = 1.0 / (1.0 + 10.0 ** ((team2_pre_elo - team1_pre_elo) / 400.0))

                team1_prior_series_count = int(team1_state["series_count"])
                team2_prior_series_count = int(team2_state["series_count"])
                team1_prior_game_count = int(team1_state["game_count"])
                team2_prior_game_count = int(team2_state["game_count"])

                team1_prior_series_win_rate = _neutral_rate(
                    team1_state["series_wins"], team1_prior_series_count
                )
                team2_prior_series_win_rate = _neutral_rate(
                    team2_state["series_wins"], team2_prior_series_count
                )
                team1_prior_game_win_rate = _neutral_rate(
                    team1_state["game_wins"], team1_prior_game_count
                )
                team2_prior_game_win_rate = _neutral_rate(
                    team2_state["game_wins"], team2_prior_game_count
                )

                team1_recent5_series_count = len(team1_recent)
                team2_recent5_series_count = len(team2_recent)
                team1_recent5_series_win_rate = _neutral_rate(sum(team1_recent), team1_recent5_series_count)
                team2_recent5_series_win_rate = _neutral_rate(sum(team2_recent), team2_recent5_series_count)

                team1_patch_prior_series_count = int(team1_patch_state["series_count"])
                team2_patch_prior_series_count = int(team2_patch_state["series_count"])
                team1_patch_prior_win_rate = _neutral_rate(
                    team1_patch_state["series_wins"], team1_patch_prior_series_count
                )
                team2_patch_prior_win_rate = _neutral_rate(
                    team2_patch_state["series_wins"], team2_patch_prior_series_count
                )

                team1_split_prior_series_count = int(team1_split_state["series_count"])
                team2_split_prior_series_count = int(team2_split_state["series_count"])
                team1_split_prior_win_rate = _neutral_rate(
                    team1_split_state["series_wins"], team1_split_prior_series_count
                )
                team2_split_prior_win_rate = _neutral_rate(
                    team2_split_state["series_wins"], team2_split_prior_series_count
                )

                h2h_prior_series_count = int(matchup_state["series_count"])
                h2h_prior_game_count = int(matchup_state["game_count"])
                h2h_team1_series_wins = int(matchup_state["series_wins"][row.team1_key])
                h2h_team2_series_wins = int(matchup_state["series_wins"][row.team2_key])
                h2h_team1_game_wins = int(matchup_state["game_wins"][row.team1_key])
                h2h_team2_game_wins = int(matchup_state["game_wins"][row.team2_key])

                feature_rows.append(
                    {
                        "snapshot_id": row.snapshot_id,
                        "feature_version": "prematch_v2",
                        "series_key": row.series_key,
                        "series_date": row.series_date,
                        "start_time": row.start_time,
                        "league_code": row.league_code,
                        "season_year": row.season_year,
                        "split_name": split_name,
                        "playoffs": row.playoffs,
                        "patch_version": patch_version,
                        "team1_key": row.team1_key,
                        "team1_name": row.team1_name,
                        "team2_key": row.team2_key,
                        "team2_name": row.team2_name,
                        "best_of_inferred": best_of_inferred,
                        "label_team1_win": label_team1_win,
                        "team1_pre_elo": team1_pre_elo,
                        "team2_pre_elo": team2_pre_elo,
                        "elo_diff": team1_pre_elo - team2_pre_elo,
                        "team1_elo_win_prob": team1_elo_win_prob,
                        "team1_prior_series_count": team1_prior_series_count,
                        "team2_prior_series_count": team2_prior_series_count,
                        "series_count_diff": team1_prior_series_count - team2_prior_series_count,
                        "team1_prior_series_win_rate": team1_prior_series_win_rate,
                        "team2_prior_series_win_rate": team2_prior_series_win_rate,
                        "prior_series_win_rate_diff": (
                            team1_prior_series_win_rate - team2_prior_series_win_rate
                        ),
                        "team1_recent5_series_count": team1_recent5_series_count,
                        "team2_recent5_series_count": team2_recent5_series_count,
                        "team1_recent5_series_win_rate": team1_recent5_series_win_rate,
                        "team2_recent5_series_win_rate": team2_recent5_series_win_rate,
                        "recent5_series_win_rate_diff": (
                            team1_recent5_series_win_rate - team2_recent5_series_win_rate
                        ),
                        "team1_prior_game_count": team1_prior_game_count,
                        "team2_prior_game_count": team2_prior_game_count,
                        "prior_game_count_diff": team1_prior_game_count - team2_prior_game_count,
                        "team1_prior_game_win_rate": team1_prior_game_win_rate,
                        "team2_prior_game_win_rate": team2_prior_game_win_rate,
                        "prior_game_win_rate_diff": team1_prior_game_win_rate - team2_prior_game_win_rate,
                        "team1_prior_avg_games_played_per_series": _safe_average(
                            team1_state["games_played_total"], team1_prior_series_count
                        ),
                        "team2_prior_avg_games_played_per_series": _safe_average(
                            team2_state["games_played_total"], team2_prior_series_count
                        ),
                        "prior_avg_games_played_per_series_diff": _safe_average(
                            team1_state["games_played_total"], team1_prior_series_count
                        )
                        - _safe_average(team2_state["games_played_total"], team2_prior_series_count),
                        "team1_prior_avg_games_won_per_series": _safe_average(
                            team1_state["games_won_total"], team1_prior_series_count
                        ),
                        "team2_prior_avg_games_won_per_series": _safe_average(
                            team2_state["games_won_total"], team2_prior_series_count
                        ),
                        "prior_avg_games_won_per_series_diff": _safe_average(
                            team1_state["games_won_total"], team1_prior_series_count
                        )
                        - _safe_average(team2_state["games_won_total"], team2_prior_series_count),
                        "team1_prior_avg_game_length_seconds": _safe_average(
                            team1_state["game_length_seconds_total"], team1_prior_game_count
                        ),
                        "team2_prior_avg_game_length_seconds": _safe_average(
                            team2_state["game_length_seconds_total"], team2_prior_game_count
                        ),
                        "prior_avg_game_length_seconds_diff": _safe_average(
                            team1_state["game_length_seconds_total"], team1_prior_game_count
                        )
                        - _safe_average(team2_state["game_length_seconds_total"], team2_prior_game_count),
                        "team1_series_streak": int(team1_state["series_streak"]),
                        "team2_series_streak": int(team2_state["series_streak"]),
                        "series_streak_diff": int(team1_state["series_streak"]) - int(team2_state["series_streak"]),
                        "team1_days_since_last_series": _days_since(
                            team1_state["last_series_at"], event_time
                        ),
                        "team2_days_since_last_series": _days_since(
                            team2_state["last_series_at"], event_time
                        ),
                        "days_since_last_series_diff": _days_since(
                            team1_state["last_series_at"], event_time
                        )
                        - _days_since(team2_state["last_series_at"], event_time),
                        "team1_patch_prior_series_count": team1_patch_prior_series_count,
                        "team2_patch_prior_series_count": team2_patch_prior_series_count,
                        "patch_prior_series_count_diff": (
                            team1_patch_prior_series_count - team2_patch_prior_series_count
                        ),
                        "team1_patch_prior_win_rate": team1_patch_prior_win_rate,
                        "team2_patch_prior_win_rate": team2_patch_prior_win_rate,
                        "patch_prior_win_rate_diff": team1_patch_prior_win_rate - team2_patch_prior_win_rate,
                        "team1_split_prior_series_count": team1_split_prior_series_count,
                        "team2_split_prior_series_count": team2_split_prior_series_count,
                        "split_prior_series_count_diff": (
                            team1_split_prior_series_count - team2_split_prior_series_count
                        ),
                        "team1_split_prior_win_rate": team1_split_prior_win_rate,
                        "team2_split_prior_win_rate": team2_split_prior_win_rate,
                        "split_prior_win_rate_diff": team1_split_prior_win_rate - team2_split_prior_win_rate,
                        "h2h_prior_series_count": h2h_prior_series_count,
                        "h2h_team1_series_wins": h2h_team1_series_wins,
                        "h2h_team2_series_wins": h2h_team2_series_wins,
                        "h2h_team1_series_win_rate": _neutral_rate(
                            h2h_team1_series_wins, h2h_prior_series_count
                        ),
                        "h2h_prior_game_count": h2h_prior_game_count,
                        "h2h_team1_game_wins": h2h_team1_game_wins,
                        "h2h_team2_game_wins": h2h_team2_game_wins,
                        "h2h_team1_game_win_rate": _neutral_rate(
                            h2h_team1_game_wins, h2h_prior_game_count
                        ),
                        "team1_prev_series_roster_overlap": team1_prev_series_roster_overlap,
                        "team2_prev_series_roster_overlap": team2_prev_series_roster_overlap,
                        "prev_series_roster_overlap_diff": (
                            team1_prev_series_roster_overlap - team2_prev_series_roster_overlap
                        ),
                        "team1_recent3_avg_roster_overlap": team1_recent3_avg_roster_overlap,
                        "team2_recent3_avg_roster_overlap": team2_recent3_avg_roster_overlap,
                        "recent3_avg_roster_overlap_diff": (
                            team1_recent3_avg_roster_overlap - team2_recent3_avg_roster_overlap
                        ),
                        "team1_roster_avg_player_prior_series_count": (
                            team1_roster_avg_player_prior_series_count
                        ),
                        "team2_roster_avg_player_prior_series_count": (
                            team2_roster_avg_player_prior_series_count
                        ),
                        "roster_avg_player_prior_series_count_diff": (
                            team1_roster_avg_player_prior_series_count
                            - team2_roster_avg_player_prior_series_count
                        ),
                        "team1_roster_avg_player_prior_series_win_rate": (
                            team1_roster_avg_player_prior_series_win_rate
                        ),
                        "team2_roster_avg_player_prior_series_win_rate": (
                            team2_roster_avg_player_prior_series_win_rate
                        ),
                        "roster_avg_player_prior_series_win_rate_diff": (
                            team1_roster_avg_player_prior_series_win_rate
                            - team2_roster_avg_player_prior_series_win_rate
                        ),
                        "team1_roster_new_player_count": team1_roster_new_player_count,
                        "team2_roster_new_player_count": team2_roster_new_player_count,
                        "roster_new_player_count_diff": (
                            team1_roster_new_player_count - team2_roster_new_player_count
                        ),
                        "team1_recent10_unique_champions": team1_draft_recent10["unique_champions"],
                        "team2_recent10_unique_champions": team2_draft_recent10["unique_champions"],
                        "recent10_unique_champions_diff": (
                            team1_draft_recent10["unique_champions"]
                            - team2_draft_recent10["unique_champions"]
                        ),
                        "team1_recent25_unique_champions": team1_draft_recent25["unique_champions"],
                        "team2_recent25_unique_champions": team2_draft_recent25["unique_champions"],
                        "recent25_unique_champions_diff": (
                            team1_draft_recent25["unique_champions"]
                            - team2_draft_recent25["unique_champions"]
                        ),
                        "team1_recent25_top5_champion_share": team1_draft_recent25["top5_share"],
                        "team2_recent25_top5_champion_share": team2_draft_recent25["top5_share"],
                        "recent25_top5_champion_share_diff": (
                            team1_draft_recent25["top5_share"]
                            - team2_draft_recent25["top5_share"]
                        ),
                        "team1_recent10_first_pick_rate": team1_draft_recent10["first_pick_rate"],
                        "team2_recent10_first_pick_rate": team2_draft_recent10["first_pick_rate"],
                        "recent10_first_pick_rate_diff": (
                            team1_draft_recent10["first_pick_rate"]
                            - team2_draft_recent10["first_pick_rate"]
                        ),
                    }
                )

                pending_updates.append(
                    {
                        "event_time": event_time,
                        "patch_version": patch_version,
                        "split_name": split_name,
                        "team1_key": row.team1_key,
                        "team2_key": row.team2_key,
                        "team1_series_win": int(label_team1_win == 1),
                        "team2_series_win": int(label_team1_win == 0),
                        "team1_game_wins": team1_wins,
                        "team2_game_wins": team2_wins,
                        "games_played": games_played,
                        "best_of_inferred": best_of_inferred,
                        "avg_game_length_seconds": avg_game_length_seconds,
                        "team1_pre_elo": team1_pre_elo,
                        "team2_pre_elo": team2_pre_elo,
                        "matchup_key": matchup_key,
                        "team1_roster": team1_roster,
                        "team2_roster": team2_roster,
                        "team1_series_games": series_team_games.get((row.series_key, row.team1_key), []),
                        "team2_series_games": series_team_games.get((row.series_key, row.team2_key), []),
                    }
                )

            for update in pending_updates:
                team1_key = update["team1_key"]
                team2_key = update["team2_key"]
                team1_state = team_state[team1_key]
                team2_state = team_state[team2_key]
                event_time = update["event_time"]

                team1_series_win = update["team1_series_win"]
                team2_series_win = update["team2_series_win"]
                games_played = update["games_played"]
                team1_game_wins = update["team1_game_wins"]
                team2_game_wins = update["team2_game_wins"]
                best_of = max(update["best_of_inferred"], 1)
                margin_multiplier = 1.0 + abs(team1_game_wins - team2_game_wins) / best_of
                k_factor = 32.0 * margin_multiplier
                expected_team1 = 1.0 / (
                    1.0 + 10.0 ** ((update["team2_pre_elo"] - update["team1_pre_elo"]) / 400.0)
                )
                expected_team2 = 1.0 - expected_team1

                team1_state["elo"] = float(team1_state["elo"]) + k_factor * (
                    team1_series_win - expected_team1
                )
                team2_state["elo"] = float(team2_state["elo"]) + k_factor * (
                    team2_series_win - expected_team2
                )

                for state, team_series_win, team_game_wins in (
                    (team1_state, team1_series_win, team1_game_wins),
                    (team2_state, team2_series_win, team2_game_wins),
                ):
                    state["series_count"] += 1
                    state["series_wins"] += team_series_win
                    state["game_count"] += games_played
                    state["game_wins"] += team_game_wins
                    state["games_played_total"] += games_played
                    state["games_won_total"] += team_game_wins
                    state["game_length_seconds_total"] += (
                        update["avg_game_length_seconds"] * games_played
                    )
                    previous_streak = int(state["series_streak"])
                    if team_series_win:
                        state["series_streak"] = previous_streak + 1 if previous_streak > 0 else 1
                    else:
                        state["series_streak"] = previous_streak - 1 if previous_streak < 0 else -1
                    state["last_series_at"] = event_time

                recent_series_results[team1_key].append(team1_series_win)
                recent_series_results[team2_key].append(team2_series_win)

                patch_state[(team1_key, update["patch_version"])]["series_count"] += 1
                patch_state[(team1_key, update["patch_version"])]["series_wins"] += team1_series_win
                patch_state[(team2_key, update["patch_version"])]["series_count"] += 1
                patch_state[(team2_key, update["patch_version"])]["series_wins"] += team2_series_win

                split_state[(team1_key, update["split_name"])]["series_count"] += 1
                split_state[(team1_key, update["split_name"])]["series_wins"] += team1_series_win
                split_state[(team2_key, update["split_name"])]["series_count"] += 1
                split_state[(team2_key, update["split_name"])]["series_wins"] += team2_series_win

                matchup_state = h2h_state[update["matchup_key"]]
                matchup_state["series_count"] += 1
                matchup_state["game_count"] += games_played
                matchup_state["series_wins"][team1_key] += team1_series_win
                matchup_state["series_wins"][team2_key] += team2_series_win
                matchup_state["game_wins"][team1_key] += team1_game_wins
                matchup_state["game_wins"][team2_key] += team2_game_wins

                for player_key in update["team1_roster"]:
                    player_state[player_key]["series_count"] += 1
                    player_state[player_key]["series_wins"] += team1_series_win
                for player_key in update["team2_roster"]:
                    player_state[player_key]["series_count"] += 1
                    player_state[player_key]["series_wins"] += team2_series_win

                if update["team1_roster"]:
                    team_roster_history[team1_key].append(update["team1_roster"])
                if update["team2_roster"]:
                    team_roster_history[team2_key].append(update["team2_roster"])

                for game_info in update["team1_series_games"]:
                    team_draft_history[team1_key].append(game_info)
                for game_info in update["team2_series_games"]:
                    team_draft_history[team2_key].append(game_info)

        features_df = pd.DataFrame(feature_rows, columns=feature_columns)
        self.con.register("match_features_prematch_df", features_df)
        self.con.execute(
            """
            CREATE OR REPLACE TABLE gold_match_features_prematch AS
            SELECT * FROM match_features_prematch_df
            """
        )
        self.con.unregister("match_features_prematch_df")

    def _build_validation_summary(self) -> list[dict[str, Any]]:
        logger.info("Running Gold validations")

        checks: list[dict[str, Any]] = []

        def add_check(
            check_name: str,
            severity: str,
            failing_rows: int,
            total_rows: int,
            threshold_pct: float | None,
            description: str,
        ) -> None:
            failure_pct = round((failing_rows / total_rows * 100.0), 4) if total_rows else 0.0
            passed = threshold_pct is None or failure_pct <= threshold_pct
            checks.append(
                {
                    "snapshot_id": self.snapshot_id,
                    "check_name": check_name,
                    "severity": severity,
                    "passed": passed,
                    "failing_rows": int(failing_rows),
                    "total_rows": int(total_rows),
                    "failure_pct": failure_pct,
                    "threshold_pct": threshold_pct,
                    "description": description,
                }
            )

        game_totals = self.con.execute(
            """
            SELECT
                COUNT(DISTINCT game_id) AS total_games,
                SUM(CASE WHEN teams_per_game != 2 THEN 1 ELSE 0 END) AS bad_team_counts,
                SUM(CASE WHEN players_per_game != 10 THEN 1 ELSE 0 END) AS bad_player_counts
            FROM (
                SELECT
                    g.game_id,
                    COUNT(*) AS teams_per_game,
                    COALESCE(p.players_per_game, 0) AS players_per_game
                FROM gold_fact_game_team g
                LEFT JOIN (
                    SELECT game_id, COUNT(*) AS players_per_game
                    FROM gold_fact_game_player
                    GROUP BY game_id
                ) p USING (game_id)
                GROUP BY g.game_id, p.players_per_game
            )
            """
        ).fetchone()
        add_check(
            "games_have_exactly_two_teams",
            "error",
            int(game_totals[1] or 0),
            int(game_totals[0] or 0),
            0.0,
            "Every game should have exactly two team rows in Gold fact_game_team.",
        )
        add_check(
            "games_have_exactly_ten_players",
            "error",
            int(game_totals[2] or 0),
            int(game_totals[0] or 0),
            0.0,
            "Every game should have exactly ten player rows in Gold fact_game_player.",
        )

        duplicates = self.con.execute(
            """
            SELECT
                SUM(CASE WHEN team_dup_count > 1 THEN 1 ELSE 0 END) AS duplicate_game_team_slots,
                SUM(CASE WHEN player_dup_count > 1 THEN 1 ELSE 0 END) AS duplicate_game_player_slots,
                SUM(CASE WHEN series_dup_count > 1 THEN 1 ELSE 0 END) AS duplicate_series_keys
            FROM (
                SELECT
                    0 AS player_dup_count,
                    0 AS series_dup_count,
                    COUNT(*) AS team_dup_count
                FROM gold_fact_game_team
                GROUP BY game_id, side
                UNION ALL
                SELECT
                    COUNT(*) AS player_dup_count,
                    0 AS series_dup_count,
                    0 AS team_dup_count
                FROM gold_fact_game_player
                GROUP BY game_id, game_player_key
                UNION ALL
                SELECT
                    0 AS player_dup_count,
                    COUNT(*) AS series_dup_count,
                    0 AS team_dup_count
                FROM gold_fact_series
                GROUP BY series_key
            )
            """
        ).fetchone()
        add_check(
            "no_duplicate_game_team_slots",
            "error",
            int(duplicates[0] or 0),
            int(self.con.execute("SELECT COUNT(*) FROM gold_fact_game_team").fetchone()[0]),
            0.0,
            "Game/team slot grain must be unique.",
        )
        add_check(
            "no_duplicate_game_player_slots",
            "error",
            int(duplicates[1] or 0),
            int(self.con.execute("SELECT COUNT(*) FROM gold_fact_game_player").fetchone()[0]),
            0.0,
            "Game/player slot grain must be unique.",
        )
        add_check(
            "no_duplicate_series_keys",
            "error",
            int(duplicates[2] or 0),
            int(self.con.execute("SELECT COUNT(*) FROM gold_fact_series").fetchone()[0]),
            0.0,
            "Series grain must be unique.",
        )

        series_total = int(self.con.execute("SELECT COUNT(*) FROM gold_fact_series").fetchone()[0])
        add_check(
            "series_missing_team_ids",
            "warning",
            int(
                self.con.execute(
                    """
                    SELECT COUNT(*)
                    FROM gold_fact_series
                    WHERE team1_source_team_id IS NULL OR team2_source_team_id IS NULL
                    """
                ).fetchone()[0]
            ),
            series_total,
            5.0,
            "Series should have source team IDs on both sides whenever Oracle provides them.",
        )
        add_check(
            "series_missing_split",
            "warning",
            int(
                self.con.execute(
                    """
                    SELECT COUNT(*)
                    FROM gold_fact_series
                    WHERE split_name IS NULL OR split_name = ''
                    """
                ).fetchone()[0]
            ),
            series_total,
            25.0,
            "Split coverage should stay high enough for temporal segmentation.",
        )
        add_check(
            "series_missing_patch",
            "warning",
            int(
                self.con.execute(
                    """
                    SELECT COUNT(*)
                    FROM gold_fact_series
                    WHERE patch_version IS NULL OR patch_version = ''
                    """
                ).fetchone()[0]
            ),
            series_total,
            5.0,
            "Patch coverage should stay high enough for meta-aware modeling.",
        )

        game_team_total = int(
            self.con.execute("SELECT COUNT(*) FROM gold_fact_game_team").fetchone()[0]
        )
        player_total = int(
            self.con.execute("SELECT COUNT(*) FROM gold_fact_game_player").fetchone()[0]
        )
        add_check(
            "game_rows_missing_team_id",
            "warning",
            int(
                self.con.execute(
                    "SELECT COUNT(*) FROM gold_fact_game_team WHERE source_team_id IS NULL"
                ).fetchone()[0]
            ),
            game_team_total,
            5.0,
            "Team IDs should exist for most Gold game/team rows.",
        )
        add_check(
            "player_rows_missing_player_id",
            "warning",
            int(
                self.con.execute(
                    "SELECT COUNT(*) FROM gold_fact_game_player WHERE source_player_id IS NULL"
                ).fetchone()[0]
            ),
            player_total,
            5.0,
            "Player IDs should exist for most Gold game/player rows.",
        )
        add_check(
            "games_with_same_team_key_both_sides",
            "warning",
            int(
                self.con.execute(
                    """
                    SELECT COUNT(*)
                    FROM (
                        SELECT game_id, COUNT(DISTINCT team_key) AS distinct_team_keys
                        FROM gold_fact_game_team
                        GROUP BY game_id
                    )
                    WHERE distinct_team_keys < 2
                    """
                ).fetchone()[0]
            ),
            int(game_totals[0] or 0),
            0.1,
            "Two sides in the same game should normally resolve to different canonical team keys.",
        )
        add_check(
            "games_with_unknown_player_key_collisions",
            "warning",
            int(
                self.con.execute(
                    """
                    SELECT COUNT(*)
                    FROM (
                        SELECT game_id, COUNT(*) AS unknown_rows
                        FROM gold_fact_game_player
                        WHERE player_key = 'playername:unknown'
                        GROUP BY game_id
                    )
                    WHERE unknown_rows > 1
                    """
                ).fetchone()[0]
            ),
            int(game_totals[0] or 0),
            0.1,
            "Unknown player names should stay rare; repeated unknowns in one game need manual review.",
        )

        official_coverage = self.con.execute(
            """
            SELECT
                COUNT(*) FILTER (
                    WHERE official_match_status IN (
                        'matched_official',
                        'unmatched_in_official_coverage'
                    )
                ) AS covered_series,
                COUNT(*) FILTER (
                    WHERE official_match_status = 'matched_official'
                ) AS matched_series
            FROM gold_fact_series
            """
        ).fetchone()
        add_check(
            "official_series_reconciliation_coverage",
            "warning",
            int((official_coverage[0] or 0) - (official_coverage[1] or 0)),
            int(official_coverage[0] or 0),
            60.0,
            "Series covered by official match ingestion should reconcile against the official schedule.",
        )

        leaguepedia_coverage = self.con.execute(
            """
            SELECT
                COUNT(*) FILTER (
                    WHERE leaguepedia_match_status IN (
                        'matched_leaguepedia',
                        'unmatched_in_leaguepedia_coverage'
                    )
                ) AS covered_series,
                COUNT(*) FILTER (
                    WHERE leaguepedia_match_status = 'matched_leaguepedia'
                ) AS matched_series
            FROM gold_external_reconciliation
            """
        ).fetchone()
        add_check(
            "leaguepedia_series_reconciliation_coverage",
            "warning",
            int((leaguepedia_coverage[0] or 0) - (leaguepedia_coverage[1] or 0)),
            int(leaguepedia_coverage[0] or 0),
            55.0,
            "Series inside Leaguepedia schedule coverage should mostly reconcile to MatchSchedule rows.",
        )
        add_check(
            "multi_source_score_conflicts",
            "warning",
            int(
                self.con.execute(
                    """
                    SELECT COUNT(*)
                    FROM gold_external_reconciliation
                    WHERE multi_source_score_conflict = TRUE
                    """
                ).fetchone()[0]
            ),
            int(
                self.con.execute(
                    """
                    SELECT COUNT(*)
                    FROM gold_external_reconciliation
                    WHERE triangulation_status = 'riot_and_leaguepedia_matched'
                    """
                ).fetchone()[0]
            ),
            2.0,
            "When Riot and Leaguepedia both match the same series, the series score should agree.",
        )

        self._build_quality_issues_table()

        validation_df = pd.DataFrame(checks)
        self.con.register("validation_summary_df", validation_df)
        self.con.execute(
            """
            CREATE OR REPLACE TABLE gold_validation_summary AS
            SELECT * FROM validation_summary_df
            """
        )
        return checks

    def _build_quality_issues_table(self) -> None:
        self.con.execute(
            """
            CREATE OR REPLACE TABLE gold_quality_issues AS
            WITH game_team_issues AS (
                SELECT
                    '{snapshot_id}' AS snapshot_id,
                    'game_team_count' AS issue_type,
                    'error' AS severity,
                    game_id AS entity_key,
                    game_id,
                    NULL::VARCHAR AS series_key,
                    CONCAT('Expected 2 team rows, found ', COUNT(*)) AS issue_detail
                FROM gold_fact_game_team
                GROUP BY game_id
                HAVING COUNT(*) <> 2
            ),
            game_player_issues AS (
                SELECT
                    '{snapshot_id}' AS snapshot_id,
                    'game_player_count' AS issue_type,
                    'error' AS severity,
                    game_id AS entity_key,
                    game_id,
                    NULL::VARCHAR AS series_key,
                    CONCAT('Expected 10 player rows, found ', COUNT(*)) AS issue_detail
                FROM gold_fact_game_player
                GROUP BY game_id
                HAVING COUNT(*) <> 10
            ),
            missing_team_ids AS (
                SELECT
                    '{snapshot_id}' AS snapshot_id,
                    'series_missing_team_id' AS issue_type,
                    'warning' AS severity,
                    series_key AS entity_key,
                    NULL::VARCHAR AS game_id,
                    series_key,
                    'At least one side is missing source_team_id' AS issue_detail
                FROM gold_fact_series
                WHERE team1_source_team_id IS NULL OR team2_source_team_id IS NULL
            ),
            same_team_key_both_sides AS (
                SELECT
                    '{snapshot_id}' AS snapshot_id,
                    'game_same_team_key_both_sides' AS issue_type,
                    'warning' AS severity,
                    game_id AS entity_key,
                    game_id,
                    NULL::VARCHAR AS series_key,
                    'Both sides resolved to the same canonical team_key.' AS issue_detail
                FROM (
                    SELECT game_id, COUNT(DISTINCT team_key) AS distinct_team_keys
                    FROM gold_fact_game_team
                    GROUP BY game_id
                )
                WHERE distinct_team_keys < 2
            ),
            unknown_player_collisions AS (
                SELECT
                    '{snapshot_id}' AS snapshot_id,
                    'game_unknown_player_collisions' AS issue_type,
                    'warning' AS severity,
                    game_id AS entity_key,
                    game_id,
                    NULL::VARCHAR AS series_key,
                    'Multiple rows in the same game fell back to playername:unknown.' AS issue_detail
                FROM (
                    SELECT game_id, COUNT(*) AS unknown_rows
                    FROM gold_fact_game_player
                    WHERE player_key = 'playername:unknown'
                    GROUP BY game_id
                )
                WHERE unknown_rows > 1
            ),
            official_unmatched AS (
                SELECT
                    '{snapshot_id}' AS snapshot_id,
                    'official_series_unmatched' AS issue_type,
                    'warning' AS severity,
                    series_key AS entity_key,
                    NULL::VARCHAR AS game_id,
                    series_key,
                    'Series is inside official coverage but did not reconcile to an official match.' AS issue_detail
                FROM gold_fact_series
                WHERE official_match_status = 'unmatched_in_official_coverage'
            ),
            leaguepedia_unmatched AS (
                SELECT
                    '{snapshot_id}' AS snapshot_id,
                    'leaguepedia_series_unmatched' AS issue_type,
                    'warning' AS severity,
                    series_key AS entity_key,
                    NULL::VARCHAR AS game_id,
                    series_key,
                    'Series is inside Leaguepedia coverage but did not reconcile to a MatchSchedule row.'
                        AS issue_detail
                FROM gold_external_reconciliation
                WHERE leaguepedia_match_status = 'unmatched_in_leaguepedia_coverage'
            ),
            multi_source_conflicts AS (
                SELECT
                    '{snapshot_id}' AS snapshot_id,
                    'multi_source_score_conflict' AS issue_type,
                    'warning' AS severity,
                    series_key AS entity_key,
                    NULL::VARCHAR AS game_id,
                    series_key,
                    'Riot/Oracle series score disagrees with the matched Leaguepedia schedule row.'
                        AS issue_detail
                FROM gold_external_reconciliation
                WHERE multi_source_score_conflict = TRUE
            )
            SELECT * FROM game_team_issues
            UNION ALL
            SELECT * FROM game_player_issues
            UNION ALL
            SELECT * FROM missing_team_ids
            UNION ALL
            SELECT * FROM same_team_key_both_sides
            UNION ALL
            SELECT * FROM unknown_player_collisions
            UNION ALL
            SELECT * FROM official_unmatched
            UNION ALL
            SELECT * FROM leaguepedia_unmatched
            UNION ALL
            SELECT * FROM multi_source_conflicts
            """.format(snapshot_id=self.snapshot_id)
        )

    def _collect_table_counts(self) -> dict[str, int]:
        counts = {}
        for table_name in (
            "gold_dim_league",
            "gold_dim_team",
            "gold_dim_player",
            "gold_external_reconciliation",
            "gold_fact_game_team",
            "gold_fact_game_player",
            "gold_fact_draft",
            "gold_fact_series",
            "gold_match_features_prematch",
            "gold_model_core_series",
            "gold_quality_issues",
            "gold_source_coverage",
        ):
            counts[table_name] = int(
                self.con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            )
        return counts

    def _build_manifest(
        self,
        table_counts: dict[str, int],
        validation_summary: list[dict[str, Any]],
        quality_issue_count: int,
    ) -> dict[str, Any]:
        build_finished_at = _utc_now()
        failed_checks = [check for check in validation_summary if not check["passed"]]
        source_files = {
            "oracle_csv_pattern": self.oracle_pattern,
            "official_matches_pattern": self.official_matches_pattern,
            "official_leagues_pattern": self.official_leagues_pattern,
            "official_tournaments_pattern": self.official_tournaments_pattern,
            "leaguepedia_match_results_path": self.leaguepedia_match_results_path,
        }

        return {
            "snapshot_id": self.snapshot_id,
            "build_started_at": self.build_started_at,
            "build_finished_at": build_finished_at,
            "snapshot_dir": self.snapshot_dir,
            "table_counts": table_counts,
            "quality_issue_count": quality_issue_count,
            "validation": {
                "total_checks": len(validation_summary),
                "failed_checks": len(failed_checks),
                "failed_check_names": [check["check_name"] for check in failed_checks],
            },
            "source_files": source_files,
            "truth_registry": {
                "schedule_and_match_identity": "official_matches (Riot LoL Esports API bronze cache)",
                "game_player_stats": "bronze_oracle_raw (Oracle's Elixir CSVs)",
                "drafts_and_box_scores": "bronze_oracle_raw (Oracle's Elixir CSVs)",
                "league_metadata": "bronze_official_leagues with manual league-code mapping",
                "series_shape": "oracle_series_derivation_v1 reconciled to official_matches when available",
                "historical_schedule_fallback": "bronze_leaguepedia_matches (Leaguepedia MatchSchedule bronze cache)",
            },
        }

    def _write_manifest_tables(
        self, manifest: dict[str, Any], validation_summary: list[dict[str, Any]]
    ) -> None:
        manifest_df = pd.DataFrame(
            [
                {
                    "snapshot_id": manifest["snapshot_id"],
                    "build_started_at": manifest["build_started_at"].isoformat(),
                    "build_finished_at": manifest["build_finished_at"].isoformat(),
                    "snapshot_dir": str(manifest["snapshot_dir"]),
                    "quality_issue_count": manifest["quality_issue_count"],
                    "total_checks": manifest["validation"]["total_checks"],
                    "failed_checks": manifest["validation"]["failed_checks"],
                    "table_counts_json": json.dumps(manifest["table_counts"], sort_keys=True),
                    "truth_registry_json": json.dumps(manifest["truth_registry"], sort_keys=True),
                    "source_files_json": json.dumps(
                        manifest["source_files"],
                        sort_keys=True,
                        default=_json_default,
                    ),
                }
            ]
        )
        self.con.register("dataset_manifest_df", manifest_df)
        self.con.execute(
            """
            CREATE OR REPLACE TABLE gold_dataset_manifest AS
            SELECT * FROM dataset_manifest_df
            """
        )

        validation_df = pd.DataFrame(validation_summary)
        self.con.register("validation_summary_export_df", validation_df)
        self.con.execute(
            """
            CREATE OR REPLACE TABLE gold_validation_summary_export AS
            SELECT * FROM validation_summary_export_df
            """
        )

    def _export_snapshot_tables(self) -> dict[str, Path]:
        logger.info("Exporting Gold snapshot tables to %s", self.snapshot_dir)

        table_sources = {
            "dim_league": "gold_dim_league",
            "dim_team": "gold_dim_team",
            "dim_player": "gold_dim_player",
            "external_reconciliation": "gold_external_reconciliation",
            "fact_game_team": "gold_fact_game_team",
            "fact_game_player": "gold_fact_game_player",
            "fact_draft": "gold_fact_draft",
            "fact_series": "gold_fact_series",
            "match_features_prematch": "gold_match_features_prematch",
            "model_core_series": "gold_model_core_series",
            "quality_issues": "gold_quality_issues",
            "source_coverage": "gold_source_coverage",
            "validation_summary": "gold_validation_summary_export",
            "dataset_manifest": "gold_dataset_manifest",
        }
        exported: dict[str, Path] = {}

        for export_name, filename in self.TABLE_EXPORTS.items():
            source_table = table_sources[export_name]
            output_path = self.snapshot_dir / filename
            self.con.execute(
                f"COPY {source_table} TO '{_normalize_path(output_path)}' (FORMAT PARQUET)"
            )
            exported[export_name] = output_path

        return exported

    def _write_json_reports(
        self,
        manifest: dict[str, Any],
        validation_summary: list[dict[str, Any]],
    ) -> tuple[Path, Path]:
        manifest_path = self.snapshot_dir / "manifest.json"
        validation_path = self.snapshot_dir / "validation_report.json"

        manifest_path.write_text(
            json.dumps(manifest, indent=2, default=_json_default),
            encoding="utf-8",
        )
        validation_path.write_text(
            json.dumps(validation_summary, indent=2, default=_json_default),
            encoding="utf-8",
        )
        return manifest_path, validation_path

    def _write_latest_pointer(
        self,
        manifest: dict[str, Any],
        validation_report_path: Path,
    ) -> None:
        latest_pointer = {
            "snapshot_id": self.snapshot_id,
            "snapshot_dir": str(self.snapshot_dir),
            "manifest_path": str(self.snapshot_dir / "manifest.json"),
            "validation_report_path": str(validation_report_path),
            "built_at": manifest["build_finished_at"].isoformat(),
        }
        latest_path = self.gold_path / "latest_snapshot.json"
        latest_path.write_text(json.dumps(latest_pointer, indent=2), encoding="utf-8")
