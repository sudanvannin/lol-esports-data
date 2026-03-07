"""
Silver layer transformation for match data.

Transforms raw Oracle's Elixir data into structured match/series data
with inferred tournament phases.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

# Known tournament structures for phase inference
INTERNATIONAL_TOURNAMENTS = ["WLDs", "MSI"]
MAJOR_LEAGUES = ["LCK", "LPL", "LEC", "LCS", "NA LCS", "EU LCS"]


@dataclass
class TournamentPhaseRules:
    """Rules for inferring tournament phases."""
    
    # Bo5 with 2 teams at the end = Final
    # Bo5 with 2 teams before final = Semi/Quarter
    # Bo1/Bo3 with many teams = Groups/Swiss
    
    @staticmethod
    def infer_series_format(games_count: int) -> str:
        """Infer series format from game count."""
        if games_count == 1:
            return "Bo1"
        elif games_count <= 2:
            return "Bo3"  # Could be 2-0
        elif games_count <= 3:
            return "Bo3"
        elif games_count <= 5:
            return "Bo5"
        else:
            return "Unknown"
    
    @staticmethod
    def is_elimination_format(series_format: str) -> bool:
        """Check if format suggests elimination stage."""
        return series_format in ["Bo5", "Bo3"]


class SilverMatchesTransformer:
    """Transforms Bronze data into Silver matches/series."""
    
    def __init__(
        self,
        bronze_path: str = "data/bronze/oracle_elixir",
        silver_path: str = "data/silver",
    ):
        self.bronze_path = Path(bronze_path)
        self.silver_path = Path(silver_path)
        self.silver_path.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(":memory:")
        
    def _get_csv_pattern(self) -> str:
        """Get CSV glob pattern for Oracle's Elixir files."""
        return str(self.bronze_path / "*_LoL_esports_match_data_from_OraclesElixir.csv")
    
    def transform_matches_to_series(self) -> None:
        """
        Transform individual games into series/matches.
        
        Groups games by:
        - Same day
        - Same two teams
        - Sequential game numbers
        """
        csv = self._get_csv_pattern()
        
        logger.info("Creating silver_series table...")
        
        query = f"""
        CREATE TABLE silver_series AS
        WITH team_games AS (
            -- Get team-level data (one row per team per game)
            SELECT 
                gameid,
                date,
                CAST(date AS DATE) as match_date,
                league,
                year,
                split,
                playoffs,
                patch,
                game,
                teamname,
                teamid,
                side,
                result,
                gamelength,
                -- Picks and bans
                pick1, pick2, pick3, pick4, pick5,
                ban1, ban2, ban3, ban4, ban5,
                -- Team stats
                teamkills,
                teamdeaths,
                firstblood,
                firsttower,
                firstdragon,
                firstbaron,
                firstherald,
                towers,
                dragons,
                barons,
                heralds,
                elders,
                void_grubs
            FROM read_csv_auto('{csv}', ignore_errors=true)
            WHERE position = 'team'
        ),
        game_matchups AS (
            -- Pair teams in each game
            SELECT 
                t1.gameid,
                t1.date,
                t1.match_date,
                t1.league,
                t1.year,
                t1.split,
                t1.playoffs,
                t1.patch,
                t1.game as game_number,
                t1.gamelength,
                -- Team 1 (alphabetically first)
                CASE WHEN t1.teamname < t2.teamname THEN t1.teamname ELSE t2.teamname END as team1,
                CASE WHEN t1.teamname < t2.teamname THEN t1.teamid ELSE t2.teamid END as team1_id,
                CASE WHEN t1.teamname < t2.teamname THEN t1.result ELSE t2.result END as team1_win,
                CASE WHEN t1.teamname < t2.teamname THEN t1.side ELSE t2.side END as team1_side,
                -- Team 2
                CASE WHEN t1.teamname < t2.teamname THEN t2.teamname ELSE t1.teamname END as team2,
                CASE WHEN t1.teamname < t2.teamname THEN t2.teamid ELSE t1.teamid END as team2_id,
                CASE WHEN t1.teamname < t2.teamname THEN t2.result ELSE t1.result END as team2_win,
                CASE WHEN t1.teamname < t2.teamname THEN t2.side ELSE t1.side END as team2_side,
                -- Team 1 draft
                CASE WHEN t1.teamname < t2.teamname THEN t1.pick1 ELSE t2.pick1 END as team1_pick1,
                CASE WHEN t1.teamname < t2.teamname THEN t1.pick2 ELSE t2.pick2 END as team1_pick2,
                CASE WHEN t1.teamname < t2.teamname THEN t1.pick3 ELSE t2.pick3 END as team1_pick3,
                CASE WHEN t1.teamname < t2.teamname THEN t1.pick4 ELSE t2.pick4 END as team1_pick4,
                CASE WHEN t1.teamname < t2.teamname THEN t1.pick5 ELSE t2.pick5 END as team1_pick5,
                CASE WHEN t1.teamname < t2.teamname THEN t1.ban1 ELSE t2.ban1 END as team1_ban1,
                CASE WHEN t1.teamname < t2.teamname THEN t1.ban2 ELSE t2.ban2 END as team1_ban2,
                CASE WHEN t1.teamname < t2.teamname THEN t1.ban3 ELSE t2.ban3 END as team1_ban3,
                CASE WHEN t1.teamname < t2.teamname THEN t1.ban4 ELSE t2.ban4 END as team1_ban4,
                CASE WHEN t1.teamname < t2.teamname THEN t1.ban5 ELSE t2.ban5 END as team1_ban5,
                -- Team 2 draft
                CASE WHEN t1.teamname < t2.teamname THEN t2.pick1 ELSE t1.pick1 END as team2_pick1,
                CASE WHEN t1.teamname < t2.teamname THEN t2.pick2 ELSE t1.pick2 END as team2_pick2,
                CASE WHEN t1.teamname < t2.teamname THEN t2.pick3 ELSE t1.pick3 END as team2_pick3,
                CASE WHEN t1.teamname < t2.teamname THEN t2.pick4 ELSE t1.pick4 END as team2_pick4,
                CASE WHEN t1.teamname < t2.teamname THEN t2.pick5 ELSE t1.pick5 END as team2_pick5,
                CASE WHEN t1.teamname < t2.teamname THEN t2.ban1 ELSE t1.ban1 END as team2_ban1,
                CASE WHEN t1.teamname < t2.teamname THEN t2.ban2 ELSE t1.ban2 END as team2_ban2,
                CASE WHEN t1.teamname < t2.teamname THEN t2.ban3 ELSE t1.ban3 END as team2_ban3,
                CASE WHEN t1.teamname < t2.teamname THEN t2.ban4 ELSE t1.ban4 END as team2_ban4,
                CASE WHEN t1.teamname < t2.teamname THEN t2.ban5 ELSE t1.ban5 END as team2_ban5,
                -- Stats
                CASE WHEN t1.teamname < t2.teamname THEN t1.teamkills ELSE t2.teamkills END as team1_kills,
                CASE WHEN t1.teamname < t2.teamname THEN t2.teamkills ELSE t1.teamkills END as team2_kills
            FROM team_games t1
            JOIN team_games t2 ON t1.gameid = t2.gameid AND t1.teamname < t2.teamname
        ),
        daily_series AS (
            -- First aggregate by day
            SELECT 
                match_date,
                league,
                year,
                split,
                MAX(playoffs) as playoffs,
                team1,
                team1_id,
                team2,
                team2_id,
                COUNT(*) as games_played,
                SUM(team1_win) as team1_wins,
                SUM(team2_win) as team2_wins,
                MIN(patch) as patch,
                ARRAY_AGG(gameid ORDER BY game_number) as game_ids,
                ARRAY_AGG(game_number ORDER BY game_number) as game_numbers,
                AVG(gamelength) as avg_game_length,
                MIN(date) as start_time,
                MAX(date) as end_time
            FROM game_matchups
            GROUP BY match_date, league, year, split, team1, team1_id, team2, team2_id
        ),
        -- Detect multi-day series (same teams on consecutive days)
        with_prev_day AS (
            SELECT 
                *,
                LAG(match_date) OVER (
                    PARTITION BY league, year, team1, team2 
                    ORDER BY match_date
                ) as prev_match_date,
                LAG(team1_wins) OVER (
                    PARTITION BY league, year, team1, team2 
                    ORDER BY match_date
                ) as prev_team1_wins,
                LAG(team2_wins) OVER (
                    PARTITION BY league, year, team1, team2 
                    ORDER BY match_date
                ) as prev_team2_wins,
                LAG(games_played) OVER (
                    PARTITION BY league, year, team1, team2 
                    ORDER BY match_date
                ) as prev_games_played
            FROM daily_series
        ),
        series_agg AS (
            -- Merge consecutive day series
            SELECT 
                -- Use first day's date for multi-day series
                CASE 
                    WHEN prev_match_date IS NOT NULL 
                         AND DATEDIFF('day', prev_match_date, match_date) = 1
                    THEN prev_match_date
                    ELSE match_date
                END as match_date,
                league,
                year,
                split,
                playoffs,
                team1,
                team1_id,
                team2,
                team2_id,
                -- Sum wins from both days if consecutive
                CASE 
                    WHEN prev_match_date IS NOT NULL 
                         AND DATEDIFF('day', prev_match_date, match_date) = 1
                    THEN games_played + prev_games_played
                    ELSE games_played
                END as games_played,
                CASE 
                    WHEN prev_match_date IS NOT NULL 
                         AND DATEDIFF('day', prev_match_date, match_date) = 1
                    THEN team1_wins + prev_team1_wins
                    ELSE team1_wins
                END as team1_wins,
                CASE 
                    WHEN prev_match_date IS NOT NULL 
                         AND DATEDIFF('day', prev_match_date, match_date) = 1
                    THEN team2_wins + prev_team2_wins
                    ELSE team2_wins
                END as team2_wins,
                patch,
                game_ids,
                game_numbers,
                avg_game_length,
                start_time,
                end_time,
                -- Flag to filter out the "first day" when merged
                CASE 
                    WHEN LEAD(match_date) OVER (
                        PARTITION BY league, year, team1, team2 
                        ORDER BY match_date
                    ) IS NOT NULL
                    AND DATEDIFF('day', match_date, LEAD(match_date) OVER (
                        PARTITION BY league, year, team1, team2 
                        ORDER BY match_date
                    )) = 1
                    THEN TRUE
                    ELSE FALSE
                END as is_first_day_of_multi
            FROM with_prev_day
        )
        SELECT 
            -- Generate series ID
            league || '_' || year || '_' || match_date || '_' || team1_id || '_' || team2_id as series_id,
            match_date,
            league,
            year,
            split,
            playoffs,
            patch,
            team1,
            team1_id,
            team2,
            team2_id,
            games_played,
            team1_wins,
            team2_wins,
            -- Determine winner
            CASE 
                WHEN team1_wins > team2_wins THEN team1
                WHEN team2_wins > team1_wins THEN team2
                ELSE 'Draw'
            END as series_winner,
            -- Infer format
            CASE 
                WHEN games_played = 1 THEN 'Bo1'
                WHEN games_played <= 2 THEN 'Bo3'
                WHEN games_played <= 3 THEN 'Bo3'
                WHEN games_played <= 5 THEN 'Bo5'
                ELSE 'Unknown'
            END as series_format,
            -- Score string (winner's score first)
            CASE 
                WHEN team1_wins >= team2_wins THEN team1_wins || '-' || team2_wins
                ELSE team2_wins || '-' || team1_wins
            END as score,
            avg_game_length,
            game_ids,
            game_numbers,
            start_time,
            end_time
        FROM series_agg
        WHERE NOT is_first_day_of_multi  -- Filter out first day of multi-day series
        ORDER BY match_date, league
        """
        
        self.con.execute(query)
        
        count = self.con.execute("SELECT COUNT(*) FROM silver_series").fetchone()[0]
        logger.info(f"Created silver_series with {count} series")
    
    def infer_tournament_phases(self) -> None:
        """
        Infer tournament phases based on patterns.
        
        Strategy:
        - Final: Last series between 2 unique teams in the tournament
        - Semifinal: 2nd and 3rd to last elimination series
        - Uses games_played >= 3 as proxy for elimination (Bo5/Bo3)
        """
        logger.info("Inferring tournament phases...")
        
        query = """
        CREATE TABLE silver_series_with_phase AS
        WITH tournament_stats AS (
            SELECT 
                league,
                year,
                MAX(match_date) as last_date,
                MIN(match_date) as first_date
            FROM silver_series
            GROUP BY league, year
        ),
        -- Merge series that span multiple days (same teams within 2 days)
        merged_series AS (
            SELECT 
                s.*,
                t.last_date,
                t.first_date,
                DATEDIFF('day', s.match_date, t.last_date) as days_from_end
            FROM silver_series s
            JOIN tournament_stats t ON s.league = t.league AND s.year = t.year
        ),
        -- For international tournaments, rank series by date (last = final)
        international_ranked AS (
            SELECT 
                *,
                -- Rank elimination-style series (3+ games) by date descending
                ROW_NUMBER() OVER (
                    PARTITION BY league, year 
                    ORDER BY match_date DESC, games_played DESC
                ) as series_rank,
                -- Count how many teams play on each date
                COUNT(*) OVER (PARTITION BY league, year, match_date) as series_on_date
            FROM merged_series
            WHERE league IN ('WLDs', 'MSI')
        ),
        international_phases AS (
            SELECT 
                *,
                CASE 
                    -- Final: Last series with 3+ games (Bo5 or Bo3 sweep)
                    WHEN series_rank = 1 AND games_played >= 3 THEN 'Final'
                    -- Could also be final if it's the last day with only 1 series
                    WHEN days_from_end = 0 AND series_on_date = 1 THEN 'Final'
                    -- Semifinal: 2nd or 3rd series with 3+ games, within 7 days of end
                    WHEN series_rank IN (2, 3) AND games_played >= 3 AND days_from_end <= 10 THEN 'Semifinal'
                    -- Quarterfinal: 4-7th series with 3+ games
                    WHEN series_rank BETWEEN 4 AND 8 AND games_played >= 3 AND days_from_end <= 20 THEN 'Quarterfinal'
                    -- Swiss/Groups: Bo1 or many teams on same day
                    WHEN games_played <= 2 AND series_on_date >= 2 THEN 'Swiss/Groups'
                    -- Play-in: Early in tournament
                    WHEN days_from_end > 25 THEN 'Play-in'
                    -- Default
                    ELSE 'Group Stage'
                END as tournament_phase
            FROM international_ranked
        ),
        -- Regular leagues
        regular_leagues AS (
            SELECT 
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY league, year, split
                    ORDER BY match_date DESC, games_played DESC
                ) as series_rank
            FROM merged_series
            WHERE league NOT IN ('WLDs', 'MSI')
        ),
        regular_phases AS (
            SELECT 
                *,
                CASE 
                    -- Playoff final
                    WHEN playoffs = 1 AND series_rank = 1 AND games_played >= 3 THEN 'Final'
                    -- Playoff semifinal  
                    WHEN playoffs = 1 AND series_rank IN (2, 3) AND games_played >= 3 THEN 'Semifinal'
                    -- Other playoffs
                    WHEN playoffs = 1 THEN 'Playoffs'
                    -- Regular season
                    ELSE 'Regular Season'
                END as tournament_phase
            FROM regular_leagues
        ),
        -- Combine all
        all_phases AS (
            SELECT series_id, match_date, league, year, split, playoffs, patch,
                   team1, team1_id, team2, team2_id, games_played, team1_wins, team2_wins,
                   series_winner, series_format, score, tournament_phase, avg_game_length,
                   game_ids, game_numbers, start_time, end_time, days_from_end
            FROM international_phases
            UNION ALL
            SELECT series_id, match_date, league, year, split, playoffs, patch,
                   team1, team1_id, team2, team2_id, games_played, team1_wins, team2_wins,
                   series_winner, series_format, score, tournament_phase, avg_game_length,
                   game_ids, game_numbers, start_time, end_time, days_from_end
            FROM regular_phases
        )
        SELECT * FROM all_phases
        ORDER BY league, year, match_date
        """
        
        self.con.execute(query)
        
        # Verify phases for international
        verify = self.con.execute("""
            SELECT league, year, tournament_phase, COUNT(*) as count
            FROM silver_series_with_phase
            WHERE league IN ('WLDs', 'MSI')
            GROUP BY league, year, tournament_phase
            ORDER BY league, year, tournament_phase
        """).fetchdf()
        
        logger.info(f"Phase distribution:\n{verify.to_string()}")
    
    def extract_champions(self) -> None:
        """Extract tournament champions (final winners)."""
        logger.info("Extracting tournament champions...")
        
        query = """
        CREATE TABLE silver_champions AS
        SELECT 
            league,
            year,
            split,
            match_date as final_date,
            series_winner as champion,
            CASE 
                WHEN team1 = series_winner THEN team2
                ELSE team1
            END as runner_up,
            -- Ensure winner's score is first
            CASE 
                WHEN team1 = series_winner THEN team1_wins || '-' || team2_wins
                ELSE team2_wins || '-' || team1_wins
            END as final_score,
            series_format,
            games_played
        FROM silver_series_with_phase
        WHERE tournament_phase = 'Final'
        ORDER BY league, year, match_date
        """
        
        self.con.execute(query)
        
        count = self.con.execute("SELECT COUNT(*) FROM silver_champions").fetchone()[0]
        logger.info(f"Extracted {count} tournament champions")
    
    def save_to_parquet(self) -> dict[str, Path]:
        """Save all Silver tables to Parquet files."""
        logger.info("Saving Silver tables to Parquet...")
        
        files = {}
        
        # Save series with phases
        series_path = self.silver_path / "series.parquet"
        self.con.execute(f"COPY silver_series_with_phase TO '{series_path}' (FORMAT PARQUET)")
        files["series"] = series_path
        
        # Save champions
        champions_path = self.silver_path / "champions.parquet"
        self.con.execute(f"COPY silver_champions TO '{champions_path}' (FORMAT PARQUET)")
        files["champions"] = champions_path
        
        logger.info(f"Saved: {list(files.keys())}")
        return files
    
    def run(self) -> dict[str, Path]:
        """Run full Silver transformation pipeline."""
        logger.info("Starting Silver transformation...")
        
        self.transform_matches_to_series()
        self.infer_tournament_phases()
        self.extract_champions()
        files = self.save_to_parquet()
        
        logger.info("Silver transformation complete!")
        return files
