"""Check available columns in Oracle's Elixir data."""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import duckdb

con = duckdb.connect(":memory:")
csv = "data/bronze/oracle_elixir/*_LoL_esports_match_data_from_OraclesElixir.csv"

# Get column names
query = f"""
SELECT * FROM read_csv_auto('{csv}', ignore_errors=true) LIMIT 1
"""
result = con.execute(query).fetchdf()

print("=" * 70)
print("COLUNAS DISPONIVEIS NO ORACLE'S ELIXIR")
print("=" * 70)

columns = list(result.columns)
print(f"\nTotal: {len(columns)} colunas\n")

# Categorizar
categories = {
    "Identificacao": [],
    "Composicao/Draft": [],
    "Estatisticas Basicas": [],
    "Early Game (@10, @15)": [],
    "Objetivos": [],
    "Dano/Gold": [],
    "Visao": [],
    "Outros": []
}

for col in columns:
    col_lower = col.lower()
    if any(x in col_lower for x in ['gameid', 'playerid', 'teamid', 'name', 'league', 'year', 'date', 'game', 'split', 'playoffs', 'patch', 'position', 'side']):
        categories["Identificacao"].append(col)
    elif any(x in col_lower for x in ['champion', 'ban', 'pick']):
        categories["Composicao/Draft"].append(col)
    elif any(x in col_lower for x in ['kill', 'death', 'assist', 'result']):
        categories["Estatisticas Basicas"].append(col)
    elif any(x in col_lower for x in ['at10', 'at15', 'at20', 'at25', 'first']):
        categories["Early Game (@10, @15)"].append(col)
    elif any(x in col_lower for x in ['dragon', 'baron', 'herald', 'tower', 'inhib', 'elder', 'void']):
        categories["Objetivos"].append(col)
    elif any(x in col_lower for x in ['gold', 'damage', 'dpm', 'cs', 'minion', 'earn']):
        categories["Dano/Gold"].append(col)
    elif any(x in col_lower for x in ['ward', 'vision', 'control']):
        categories["Visao"].append(col)
    else:
        categories["Outros"].append(col)

for cat, cols in categories.items():
    if cols:
        print(f"\n{cat} ({len(cols)}):")
        for c in sorted(cols):
            print(f"  - {c}")
