"""Explain champions data."""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import duckdb

con = duckdb.connect(":memory:")

print("=" * 70)
print("608 FINAIS = CAMPEOES POR TORNEIO/SPLIT/ANO")
print("=" * 70)

# Total unique champions (teams that won)
query = """
SELECT COUNT(DISTINCT champion) as unique_teams_that_won
FROM read_parquet('data/silver/champions.parquet')
"""
result = con.execute(query).fetchone()[0]
print(f"\nTimes diferentes que ganharam pelo menos 1 titulo: {result}")

# Most titles
print("\n" + "=" * 70)
print("TIMES COM MAIS TITULOS (todas as ligas)")
print("=" * 70)

query = """
SELECT champion, COUNT(*) as titles
FROM read_parquet('data/silver/champions.parquet')
GROUP BY champion
ORDER BY titles DESC
LIMIT 15
"""
result = con.execute(query).fetchdf()
print(result.to_string(index=False))

# Breakdown by type
print("\n" + "=" * 70)
print("DISTRIBUICAO DAS 608 FINAIS")
print("=" * 70)

query = """
SELECT 
    CASE 
        WHEN league = 'WLDs' THEN 'Worlds'
        WHEN league = 'MSI' THEN 'MSI'
        WHEN league IN ('LCK', 'LPL', 'LEC', 'LCS', 'NA LCS', 'EU LCS') THEN 'Major Regions'
        WHEN league IN ('CBLOL', 'LLA', 'LJL', 'PCS', 'VCS', 'TCL', 'LCL') THEN 'Minor Regions'
        ELSE 'Other (Academy, Regional)'
    END as category,
    COUNT(*) as finals
FROM read_parquet('data/silver/champions.parquet')
GROUP BY category
ORDER BY finals DESC
"""
result = con.execute(query).fetchdf()
print(result.to_string(index=False))
