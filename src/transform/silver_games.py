"""
Silver layer transformation for games and players.

Creates structured Parquet tables from Oracle's Elixir CSVs:
- silver.games: Game-level data (one row per team per game)
- silver.players: Player-level data (one row per player per game)
"""

import logging
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)


class SilverGamesTransformer:
    """Transforms Bronze CSV data into Silver Parquet tables."""
    
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
        return str(self.bronze_path / "*_LoL_esports_match_data_from_OraclesElixir.csv")
    
    def create_games_table(self) -> Path:
        """
        Create silver.games table - one row per team per game.
        
        Includes team-level stats, draft, objectives.
        """
        logger.info("Creating silver.games table...")
        csv = self._get_csv_pattern()
        
        output_path = self.silver_path / "games"
        output_path.mkdir(exist_ok=True)
        
        query = f"""
        COPY (
            SELECT 
                -- Identifiers
                gameid,
                CAST(date AS TIMESTAMP) as game_datetime,
                CAST(date AS DATE) as game_date,
                EXTRACT(YEAR FROM CAST(date AS DATE)) as year,
                EXTRACT(MONTH FROM CAST(date AS DATE)) as month,
                league,
                split,
                playoffs,
                game,
                patch,
                
                -- Team info
                teamid,
                teamname,
                side,
                result,
                
                -- Game stats
                gamelength,
                teamkills,
                teamdeaths,
                
                -- Draft
                pick1, pick2, pick3, pick4, pick5,
                ban1, ban2, ban3, ban4, ban5,
                
                -- Objectives
                firstblood,
                firsttower,
                firstdragon,
                firstherald,
                firstbaron,
                towers,
                dragons,
                heralds,
                barons,
                elders,
                void_grubs,
                inhibitors,
                
                -- Opponent objectives
                opp_towers,
                opp_dragons,
                opp_heralds,
                opp_barons,
                opp_elders,
                opp_void_grubs,
                opp_inhibitors
                
            FROM read_csv_auto('{csv}', ignore_errors=true)
            WHERE position = 'team'
            ORDER BY game_date, league, gameid
        ) TO '{output_path}' (
            FORMAT PARQUET,
            PARTITION_BY (year, league),
            OVERWRITE_OR_IGNORE
        )
        """
        
        self.con.execute(query)
        
        # Count rows
        count_query = f"""
        SELECT COUNT(*) FROM read_parquet('{output_path}/**/*.parquet')
        """
        count = self.con.execute(count_query).fetchone()[0]
        logger.info(f"Created silver.games with {count:,} rows")
        
        return output_path
    
    def create_players_table(self) -> Path:
        """
        Create silver.players table - one row per player per game.
        
        Includes player stats, champion, performance metrics.
        """
        logger.info("Creating silver.players table...")
        csv = self._get_csv_pattern()
        
        output_path = self.silver_path / "players"
        output_path.mkdir(exist_ok=True)
        
        query = f"""
        COPY (
            SELECT 
                -- Identifiers
                gameid,
                CAST(date AS TIMESTAMP) as game_datetime,
                CAST(date AS DATE) as game_date,
                EXTRACT(YEAR FROM CAST(date AS DATE)) as year,
                EXTRACT(MONTH FROM CAST(date AS DATE)) as month,
                league,
                split,
                playoffs,
                game,
                patch,
                
                -- Player info
                playerid,
                playername,
                teamid,
                teamname,
                position,
                side,
                champion,
                result,
                
                -- Game length for rate calculations
                gamelength,
                
                -- Basic stats
                kills,
                deaths,
                assists,
                
                -- Damage stats
                damagetochampions,
                dpm,
                damageshare,
                damagetakenperminute,
                damagemitigatedperminute,
                damagetotowers,
                
                -- Gold stats
                totalgold,
                earnedgold,
                earnedgoldshare,
                "earned gpm" as earned_gpm,
                goldspent,
                
                -- CS stats
                minionkills,
                monsterkills,
                "total cs" as total_cs,
                cspm,
                monsterkillsenemyjungle,
                monsterkillsownjungle,
                
                -- Vision
                visionscore,
                vspm,
                wardsplaced,
                wardskilled,
                controlwardsbought,
                
                -- Early game (at 10 min)
                goldat10,
                xpat10,
                csat10,
                golddiffat10,
                xpdiffat10,
                csdiffat10,
                killsat10,
                assistsat10,
                deathsat10,
                
                -- Early game (at 15 min)
                goldat15,
                xpat15,
                csat15,
                golddiffat15,
                xpdiffat15,
                csdiffat15,
                killsat15,
                assistsat15,
                deathsat15,
                
                -- Opponent stats at 10
                opp_goldat10,
                opp_xpat10,
                opp_csat10,
                opp_killsat10,
                opp_assistsat10,
                opp_deathsat10,
                
                -- Opponent stats at 15
                opp_goldat15,
                opp_xpat15,
                opp_csat15,
                opp_killsat15,
                opp_assistsat15,
                opp_deathsat15,
                
                -- Multi-kills
                doublekills,
                triplekills,
                quadrakills,
                pentakills,
                
                -- First blood
                firstbloodkill,
                firstbloodassist,
                firstbloodvictim
                
            FROM read_csv_auto('{csv}', ignore_errors=true)
            WHERE position != 'team'
            ORDER BY game_date, league, gameid, teamname, position
        ) TO '{output_path}' (
            FORMAT PARQUET,
            PARTITION_BY (year, league),
            OVERWRITE_OR_IGNORE
        )
        """
        
        self.con.execute(query)
        
        # Count rows
        count_query = f"""
        SELECT COUNT(*) FROM read_parquet('{output_path}/**/*.parquet')
        """
        count = self.con.execute(count_query).fetchone()[0]
        logger.info(f"Created silver.players with {count:,} rows")
        
        return output_path
    
    def verify_tables(self) -> dict:
        """Verify created tables with sample queries."""
        logger.info("Verifying Silver tables...")
        
        results = {}
        
        # Games table stats
        games_query = """
        SELECT 
            COUNT(*) as total_games,
            COUNT(DISTINCT league) as leagues,
            MIN(game_date) as min_date,
            MAX(game_date) as max_date
        FROM read_parquet('data/silver/games/**/*.parquet')
        """
        results['games'] = self.con.execute(games_query).fetchdf().to_dict('records')[0]
        
        # Players table stats
        players_query = """
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT playername) as unique_players,
            COUNT(DISTINCT champion) as unique_champions,
            MIN(game_date) as min_date,
            MAX(game_date) as max_date
        FROM read_parquet('data/silver/players/**/*.parquet')
        """
        results['players'] = self.con.execute(players_query).fetchdf().to_dict('records')[0]
        
        # Sample query: Chovy's most played champions
        chovy_query = """
        SELECT 
            champion,
            COUNT(*) as games,
            SUM(result) as wins,
            ROUND(100.0 * SUM(result) / COUNT(*), 1) as winrate
        FROM read_parquet('data/silver/players/**/*.parquet')
        WHERE LOWER(playername) = 'chovy'
        GROUP BY champion
        ORDER BY games DESC
        LIMIT 5
        """
        results['chovy_champions'] = self.con.execute(chovy_query).fetchdf().to_dict('records')
        
        return results
    
    def run(self) -> dict[str, Path]:
        """Run full Silver transformation pipeline."""
        logger.info("=" * 60)
        logger.info("SILVER LAYER - GAMES & PLAYERS")
        logger.info("=" * 60)
        
        files = {}
        
        files['games'] = self.create_games_table()
        files['players'] = self.create_players_table()
        
        # Verify
        stats = self.verify_tables()
        
        logger.info("=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Games: {stats['games']['total_games']:,} rows")
        logger.info(f"Players: {stats['players']['total_rows']:,} rows")
        logger.info(f"Unique players: {stats['players']['unique_players']:,}")
        logger.info(f"Unique champions: {stats['players']['unique_champions']}")
        logger.info(f"Date range: {stats['games']['min_date']} to {stats['games']['max_date']}")
        
        return files
