import os
import duckdb

token = os.environ.get("MOTHERDUCK_TOKEN")
if token:
    print("Testing DB Connection SQL...\n")
    try:
        con = duckdb.connect(f"md:lolesports?motherduck_token={token}")
        
        # Test 1: Recent series (Home Page)
        print("--- Test 1: Home Page Series ---")
        res = con.execute("""
            SELECT match_date, league, team1, team2, score, series_winner,
                   series_format, tournament_phase
            FROM lolesports.series
            ORDER BY match_date DESC
            LIMIT 5
        """).fetchdf()
        print(res.head())
        
        # Test 2: Active Leagues
        print("\n--- Test 2: Home Page Active Leagues ---")
        res = con.execute("""
            SELECT league, MAX(match_date) as last_match, COUNT(*) as total_series
            FROM series
            WHERE match_date >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY league
            ORDER BY last_match DESC
        """).fetchdf()
        print(res.head())
        
        
    except Exception as e:
        print(f"SQL Error: {e}")
else:
    print("Token not found.")
