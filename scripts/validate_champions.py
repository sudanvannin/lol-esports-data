"""Validate champions against known results."""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import duckdb

con = duckdb.connect(":memory:")

print("=" * 70)
print("VALIDACAO DE CAMPEOES")
print("=" * 70)

# Known Worlds champions (from user)
worlds_known = {
    2011: "Fnatic",
    2012: "Taipei Assassins",
    2013: "SK Telecom T1",
    2014: "Samsung White",
    2015: "SK Telecom T1",
    2016: "SK Telecom T1",
    2017: "Samsung Galaxy",
    2018: "Invictus Gaming",
    2019: "FunPlus Phoenix",
    2020: "Damwon Gaming",  # Dplus Kia
    2021: "EDward Gaming",
    2022: "DRX",
    2023: "T1",
    2024: "T1",
    2025: "T1",
}

# Known MSI champions
msi_known = {
    2015: "EDward Gaming",
    2016: "SK Telecom T1",
    2017: "SK Telecom T1",
    2018: "Royal Never Give Up",
    2019: "G2 Esports",
    # 2020: Cancelled (COVID)
    2021: "Royal Never Give Up",
    2022: "Royal Never Give Up",
    2023: "JD Gaming",
    2024: "Gen.G",
    2025: "Gen.G",
}

# Team name mappings (rebrands)
team_aliases = {
    "Dplus Kia": "Damwon Gaming",
    "DWG KIA": "Damwon Gaming",
    "Dplus KIA": "Damwon Gaming",
}

# Query our data
query = """
SELECT league, year, champion, runner_up, final_score
FROM read_parquet('data/silver/champions.parquet')
WHERE league IN ('WLDs', 'MSI')
ORDER BY league, year
"""
result = con.execute(query).fetchdf()

print("\n=== WORLDS ===")
print(f"{'Year':<6} {'Data':<20} {'Known':<20} {'Match':<8}")
print("-" * 60)

for year in sorted(worlds_known.keys()):
    known = worlds_known[year]
    row = result[(result['league'] == 'WLDs') & (result['year'] == year)]
    
    if len(row) == 0:
        print(f"{year:<6} {'NO DATA':<20} {known:<20} {'N/A':<8}")
    else:
        data_champ = row.iloc[0]['champion']
        # Check with aliases
        data_normalized = team_aliases.get(data_champ, data_champ)
        match = "OK" if known in data_normalized or data_normalized in known else "WRONG"
        print(f"{year:<6} {data_champ:<20} {known:<20} {match:<8}")

print("\n=== MSI ===")
print(f"{'Year':<6} {'Data':<20} {'Known':<20} {'Match':<8}")
print("-" * 60)

for year in sorted(msi_known.keys()):
    known = msi_known[year]
    row = result[(result['league'] == 'MSI') & (result['year'] == year)]
    
    if len(row) == 0:
        print(f"{year:<6} {'NO DATA':<20} {known:<20} {'N/A':<8}")
    else:
        data_champ = row.iloc[0]['champion']
        # Check with aliases
        data_normalized = team_aliases.get(data_champ, data_champ)
        match = "OK" if known in data_normalized or data_normalized in known else "WRONG"
        print(f"{year:<6} {data_champ:<20} {known:<20} {match:<8}")

print("\n=== TITULOS DO FAKER (T1/SKT) ===")
faker_titles = []
for _, row in result.iterrows():
    champ = row['champion']
    if 'SK Telecom' in champ or champ == 'T1':
        faker_titles.append(f"{row['league']} {int(row['year'])}")

print(f"Total: {len(faker_titles)} titulos internacionais")
for t in faker_titles:
    print(f"  - {t}")
