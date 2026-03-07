"""Calculate win rate for teams/players from collected match data."""

import duckdb

con = duckdb.connect(":memory:")

print("=" * 70)
print("WIN RATE ANALYSIS - LCK Teams")
print("=" * 70)

# Flatten matches and calculate win rates
query = """
WITH matches_flat AS (
    SELECT 
        match_id,
        start_time,
        league.slug as league_slug,
        tournament.slug as tournament_slug,
        teams[1].name as team1_name,
        teams[1].code as team1_code,
        teams[1].result.outcome as team1_outcome,
        teams[1].result.gameWins as team1_wins,
        teams[2].name as team2_name,
        teams[2].code as team2_code,
        teams[2].result.outcome as team2_outcome,
        teams[2].result.gameWins as team2_wins
    FROM read_json_auto('data/bronze/matches/**/*.json')
),
team_results AS (
    SELECT team1_name as team, team1_outcome as outcome, start_time, league_slug
    FROM matches_flat
    WHERE team1_name IS NOT NULL
    UNION ALL
    SELECT team2_name as team, team2_outcome as outcome, start_time, league_slug
    FROM matches_flat
    WHERE team2_name IS NOT NULL
)
SELECT 
    team,
    COUNT(*) as total_matches,
    SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
    ROUND(100.0 * SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate
FROM team_results
WHERE league_slug = 'lck'
GROUP BY team
HAVING COUNT(*) >= 10
ORDER BY win_rate DESC
"""

result = con.execute(query).fetchdf()
print("\nLCK TEAM WIN RATES (all-time, min 10 matches):")
print("-" * 70)
print(result.to_string(index=False))

# Specific analysis for Gen.G (Chovy's team)
print("\n" + "=" * 70)
print("GEN.G ESPORTS (CHOVY'S TEAM) - DETAILED")
print("=" * 70)

geng_query = """
WITH matches_flat AS (
    SELECT 
        match_id,
        start_time::DATE as match_date,
        YEAR(start_time::DATE) as year,
        league.slug as league_slug,
        tournament.slug as tournament_slug,
        teams[1].name as team1_name,
        teams[1].result.outcome as team1_outcome,
        teams[2].name as team2_name,
        teams[2].result.outcome as team2_outcome
    FROM read_json_auto('data/bronze/matches/**/*.json')
)
SELECT 
    year,
    COUNT(*) as matches,
    SUM(CASE 
        WHEN (team1_name LIKE '%Gen.G%' AND team1_outcome = 'win')
          OR (team2_name LIKE '%Gen.G%' AND team2_outcome = 'win')
        THEN 1 ELSE 0 
    END) as wins,
    ROUND(100.0 * SUM(CASE 
        WHEN (team1_name LIKE '%Gen.G%' AND team1_outcome = 'win')
          OR (team2_name LIKE '%Gen.G%' AND team2_outcome = 'win')
        THEN 1 ELSE 0 
    END) / COUNT(*), 1) as win_rate
FROM matches_flat
WHERE team1_name LIKE '%Gen.G%' OR team2_name LIKE '%Gen.G%'
GROUP BY year
ORDER BY year
"""

result = con.execute(geng_query).fetchdf()
print("\nGEN.G WIN RATE BY YEAR:")
print("-" * 70)
print(result.to_string(index=False))

# Overall
overall_query = """
WITH matches_flat AS (
    SELECT 
        match_id,
        teams[1].name as team1_name,
        teams[1].result.outcome as team1_outcome,
        teams[2].name as team2_name,
        teams[2].result.outcome as team2_outcome
    FROM read_json_auto('data/bronze/matches/**/*.json')
)
SELECT 
    COUNT(*) as total_matches,
    SUM(CASE 
        WHEN (team1_name LIKE '%Gen.G%' AND team1_outcome = 'win')
          OR (team2_name LIKE '%Gen.G%' AND team2_outcome = 'win')
        THEN 1 ELSE 0 
    END) as total_wins
FROM matches_flat
WHERE team1_name LIKE '%Gen.G%' OR team2_name LIKE '%Gen.G%'
"""

result = con.execute(overall_query).fetchdf()
total_matches = result.iloc[0, 0]
total_wins = result.iloc[0, 1]
win_rate = (total_wins / total_matches * 100) if total_matches > 0 else 0

print(f"\nGEN.G OVERALL:")
print(f"  Total Matches: {total_matches}")
print(f"  Wins: {total_wins}")
print(f"  Losses: {total_matches - total_wins}")
print(f"  Win Rate: {win_rate:.1f}%")

print("\n" + "=" * 70)
print("NOTA: Este win rate e aproximado (baseado no time Gen.G)")
print("Para win rate individual do Chovy, seria necessario dados de")
print("participacao por jogador em cada partida.")
print("=" * 70)
