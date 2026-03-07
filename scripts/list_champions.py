"""List all champions in data."""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import duckdb

con = duckdb.connect(":memory:")

query = """
SELECT DISTINCT champion
FROM read_parquet('data/silver/players/**/*.parquet')
ORDER BY champion
"""

result = con.execute(query).fetchdf()
print(f"Total: {len(result)} campeoes\n")

for i, row in result.iterrows():
    print(f"  {row['champion']}")
