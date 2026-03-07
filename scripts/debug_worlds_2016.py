"""Debug Worlds 2016 final issue."""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import duckdb

con = duckdb.connect(":memory:")
csv = "data/bronze/oracle_elixir/*_LoL_esports_match_data_from_OraclesElixir.csv"

print("=" * 70)
print("WORLDS 2016 - SKT vs Samsung Galaxy")
print("=" * 70)

query = f"""
SELECT 
    gameid,
    CAST(date AS DATE) as match_date,
    game,
    teamname,
    side,
    result
FROM read_csv_auto('{csv}', ignore_errors=true)
WHERE league = 'WLDs' 
    AND year = 2016 
    AND position = 'team'
    AND (teamname LIKE '%Samsung%' OR teamname LIKE '%SK Telecom%')
    AND CAST(date AS DATE) >= '2016-10-28'
ORDER BY date, game, teamname
"""
result = con.execute(query).fetchdf()
print(result.to_string())

print("\n\nResumo por dia:")
query = f"""
SELECT 
    CAST(date AS DATE) as match_date,
    SUM(CASE WHEN teamname LIKE '%SK Telecom%' AND result = 1 THEN 1 ELSE 0 END) as skt_wins,
    SUM(CASE WHEN teamname LIKE '%Samsung%' AND result = 1 THEN 1 ELSE 0 END) as ssg_wins
FROM read_csv_auto('{csv}', ignore_errors=true)
WHERE league = 'WLDs' 
    AND year = 2016 
    AND position = 'team'
    AND (teamname LIKE '%Samsung%' OR teamname LIKE '%SK Telecom%')
    AND CAST(date AS DATE) >= '2016-10-28'
GROUP BY CAST(date AS DATE)
ORDER BY match_date
"""
result = con.execute(query).fetchdf()
print(result.to_string())
