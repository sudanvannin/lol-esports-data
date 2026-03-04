"""
Query Bronze layer data using DuckDB.
Usage: python scripts/query_bronze.py
"""

import duckdb

BRONZE_PATH = "data/bronze"


def main():
    con = duckdb.connect(":memory:")

    print("=" * 70)
    print("BRONZE LAYER - DATA EXPLORER")
    print("=" * 70)

    # Query 1: Ligas
    print("\n[1] LIGAS DISPONIVEIS")
    print("-" * 70)
    query = f"""
    SELECT 
        json_extract_string(content, '$.name') as name,
        json_extract_string(content, '$.slug') as slug,
        json_extract_string(content, '$.region') as region
    FROM read_json_auto('{BRONZE_PATH}/leagues/**/*.json') as content
    ORDER BY name
    """
    result = con.execute(query).fetchdf()
    print(result.to_string(index=False))
    print(f"\nTotal: {len(result)} ligas")

    # Query 2: Torneios por liga
    print("\n\n[2] TORNEIOS POR LIGA (Top 10)")
    print("-" * 70)
    query = f"""
    SELECT 
        json_extract_string(content, '$.league_id') as league_id,
        json_extract_string(content, '$.name') as tournament_name,
        json_extract_string(content, '$.start_date') as start_date,
        json_extract_string(content, '$.end_date') as end_date
    FROM read_json_auto('{BRONZE_PATH}/tournaments/**/*.json') as content
    ORDER BY start_date DESC
    LIMIT 10
    """
    result = con.execute(query).fetchdf()
    print(result.to_string(index=False))

    # Query 3: Contagem de torneios por liga
    print("\n\n[3] CONTAGEM DE TORNEIOS POR LIGA")
    print("-" * 70)
    query = f"""
    WITH leagues AS (
        SELECT 
            json_extract_string(content, '$.id') as id,
            json_extract_string(content, '$.name') as name
        FROM read_json_auto('{BRONZE_PATH}/leagues/**/*.json') as content
    ),
    tournaments AS (
        SELECT 
            json_extract_string(content, '$.league_id') as league_id
        FROM read_json_auto('{BRONZE_PATH}/tournaments/**/*.json') as content
    )
    SELECT 
        l.name as liga,
        COUNT(*) as total_torneios
    FROM tournaments t
    JOIN leagues l ON t.league_id = l.id
    GROUP BY l.name
    ORDER BY total_torneios DESC
    LIMIT 15
    """
    result = con.execute(query).fetchdf()
    print(result.to_string(index=False))

    # Query 4: Torneios ativos (com datas futuras)
    print("\n\n[4] TORNEIOS ATIVOS/FUTUROS (2026)")
    print("-" * 70)
    query = f"""
    SELECT 
        json_extract_string(content, '$.name') as tournament,
        json_extract_string(content, '$.start_date')::DATE as inicio,
        json_extract_string(content, '$.end_date')::DATE as fim
    FROM read_json_auto('{BRONZE_PATH}/tournaments/**/*.json') as content
    WHERE json_extract_string(content, '$.end_date')::DATE >= CURRENT_DATE
    ORDER BY inicio
    LIMIT 15
    """
    result = con.execute(query).fetchdf()
    print(result.to_string(index=False))

    # Query 5: Estatisticas gerais
    print("\n\n[5] ESTATISTICAS GERAIS")
    print("-" * 70)
    
    leagues_count = con.execute(f"""
        SELECT COUNT(*) FROM read_json_auto('{BRONZE_PATH}/leagues/**/*.json')
    """).fetchone()[0]
    
    tournaments_count = con.execute(f"""
        SELECT COUNT(*) FROM read_json_auto('{BRONZE_PATH}/tournaments/**/*.json')
    """).fetchone()[0]

    print(f"  Total de Ligas:     {leagues_count}")
    print(f"  Total de Torneios:  {tournaments_count}")

    print("\n" + "=" * 70)
    print("Para queries customizadas, use: duckdb.connect(':memory:')")
    print("=" * 70)


if __name__ == "__main__":
    main()
