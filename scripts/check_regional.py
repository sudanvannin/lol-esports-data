"""Check regional champions data."""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import duckdb

con = duckdb.connect(":memory:")

print("=" * 70)
print("CAMPEOES POR LIGA")
print("=" * 70)

query = """
SELECT league, COUNT(*) as finals
FROM read_parquet('data/silver/champions.parquet')
GROUP BY league
ORDER BY finals DESC
"""
result = con.execute(query).fetchdf()
print(result.to_string(index=False))

print("\n" + "=" * 70)
print("EXEMPLO: LCK 2024")
print("=" * 70)

query = """
SELECT year, split, champion, runner_up, final_score
FROM read_parquet('data/silver/champions.parquet')
WHERE league = 'LCK' AND year = 2024
ORDER BY final_date
"""
result = con.execute(query).fetchdf()
print(result.to_string(index=False))

print("\n" + "=" * 70)
print("EXEMPLO: CBLOL 2024")
print("=" * 70)

query = """
SELECT year, split, champion, runner_up, final_score
FROM read_parquet('data/silver/champions.parquet')
WHERE league = 'CBLOL' AND year = 2024
ORDER BY final_date
"""
result = con.execute(query).fetchdf()
print(result.to_string(index=False))
