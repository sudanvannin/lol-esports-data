"""Test series games API."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import requests

sys.path.insert(0, ".")
from web.db import get_team_head_to_head

h2h = get_team_head_to_head("T1", "Gen.G")
if len(h2h) > 0:
    row = h2h.iloc[0]
    date = row["match_date"].strftime("%Y-%m-%d")
    t1, t2 = row["team1"], row["team2"]
    print(f"Trying: {t1} vs {t2} on {date}")
    r = requests.get(f"http://localhost:8000/api/series_games?team1={t1}&team2={t2}&date={date}", timeout=30)
    print(f"Status: {r.status_code}")
    print(f"Content type: {r.headers.get('content-type')}")
    text = r.text[:500]
    print(f"Body: {text}")
