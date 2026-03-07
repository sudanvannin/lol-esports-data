"""Debug MSI finals issues."""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import duckdb

con = duckdb.connect(":memory:")
csv = "data/bronze/oracle_elixir/*_LoL_esports_match_data_from_OraclesElixir.csv"

print("=" * 70)
print("MSI 2015 - Ultimas series")
print("=" * 70)

query = f"""
SELECT 
    CAST(date AS DATE) as match_date,
    teamname,
    SUM(result) as wins,
    COUNT(*) as games
FROM read_csv_auto('{csv}', ignore_errors=true)
WHERE league = 'MSI' AND year = 2015 AND position = 'team'
GROUP BY CAST(date AS DATE), teamname
ORDER BY match_date DESC, wins DESC
LIMIT 20
"""
result = con.execute(query).fetchdf()
print(result.to_string())

print("\n\n" + "=" * 70)
print("MSI 2016 - Ultimas series")
print("=" * 70)

query = f"""
SELECT 
    CAST(date AS DATE) as match_date,
    teamname,
    SUM(result) as wins,
    COUNT(*) as games
FROM read_csv_auto('{csv}', ignore_errors=true)
WHERE league = 'MSI' AND year = 2016 AND position = 'team'
GROUP BY CAST(date AS DATE), teamname
ORDER BY match_date DESC, wins DESC
LIMIT 20
"""
result = con.execute(query).fetchdf()
print(result.to_string())

print("\n\n" + "=" * 70)
print("MSI 2016 - Final (SKT vs CLG)")
print("=" * 70)

query = f"""
SELECT 
    gameid,
    CAST(date AS DATE) as match_date,
    game,
    teamname,
    result
FROM read_csv_auto('{csv}', ignore_errors=true)
WHERE league = 'MSI' AND year = 2016 AND position = 'team'
    AND (teamname LIKE '%SK Telecom%' OR teamname LIKE '%Counter Logic%')
ORDER BY date DESC, game
LIMIT 20
"""
result = con.execute(query).fetchdf()
print(result.to_string())
