"""FastAPI web application for LoL Esports Stats."""

from pathlib import Path

from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import db

app = FastAPI(title="LoL Esports Stats")

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")



# ----- AUTOCOMPLETE API -----

@app.get("/api/autocomplete")
async def autocomplete(q: str = Query(""), mode: str = Query("players")):
    if len(q) < 2:
        return JSONResponse([])
    if mode == "teams":
        results = db.search_teams(q)
        return JSONResponse([
            {"label": r["teamname"], "sub": r["league"]}
            for r in results.to_dict("records")
        ])
    else:
        results = db.search_players(q)
        return JSONResponse([
            {"label": r["playername"], "sub": f"{r['teamname']} ({r['position']})"}
            for r in results.to_dict("records")
        ])


@app.get("/api/series_games")
async def series_games(
    team1: str = Query(""),
    team2: str = Query(""),
    date: str = Query(""),
):
    """Return all games in a series between two teams on a given date."""
    if not team1 or not team2 or not date:
        return JSONResponse([])
    games = db.get_series_games(team1, team2, date)
    return JSONResponse(games)


# ----- HOME -----

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    recent = db.get_recent_series(limit=25)
    leagues = db.get_active_leagues()
    upcoming = db.get_upcoming_matches(limit=8)
    upcoming_meta = db.get_upcoming_matches_meta()
    return templates.TemplateResponse("home.html", {
        "request": request,
        "recent_series": recent.to_dict("records"),
        "active_leagues": leagues.to_dict("records"),
        "upcoming_matches": upcoming.to_dict("records"),
        "upcoming_meta": upcoming_meta,
    })


@app.get("/upcoming", response_class=HTMLResponse)
async def upcoming(
    request: Request,
    league: str = Query(None),
):
    upcoming_matches = db.get_upcoming_matches(limit=100, league=league)
    upcoming_leagues = db.get_upcoming_match_leagues()
    upcoming_meta = db.get_upcoming_matches_meta()
    return templates.TemplateResponse("upcoming.html", {
        "request": request,
        "rows": upcoming_matches.to_dict("records"),
        "league": league,
        "available_leagues": upcoming_leagues.to_dict("records"),
        "upcoming_meta": upcoming_meta,
    })


@app.get("/search", response_class=HTMLResponse)
async def search(request: Request, q: str = Query("")):
    if not q.strip():
        return RedirectResponse("/")
    players = db.search_players(q)
    teams = db.search_teams(q)
    return templates.TemplateResponse("search.html", {
        "request": request,
        "query": q,
        "players": players.to_dict("records"),
        "teams": teams.to_dict("records"),
    })


# ----- PLAYER -----

@app.get("/player/{name}", response_class=HTMLResponse)
async def player(
    request: Request,
    name: str,
    year: int = Query(None),
    split: str = Query(None),
):
    info = db.get_player_info(name)
    if not info:
        return templates.TemplateResponse("404.html", {
            "request": request, "message": f"Player '{name}' not found"
        }, status_code=404)
    career = db.get_player_career_stats(name, year=year, split=split)
    by_year = db.get_player_by_year(name)
    champions = db.get_player_champions(name, year=year, split=split)
    recent = db.get_player_recent_games(name, year=year, split=split)
    splits = db.get_player_splits(name)
    return templates.TemplateResponse("player.html", {
        "request": request,
        "info": info,
        "career": career,
        "by_year": by_year.to_dict("records"),
        "champions": champions.to_dict("records"),
        "recent_games": recent.to_dict("records"),
        "available_splits": splits.to_dict("records"),
        "filter_year": year,
        "filter_split": split,
    })


# ----- TEAM -----

@app.get("/team/{name}", response_class=HTMLResponse)
async def team(request: Request, name: str):
    info = db.get_team_info(name)
    if not info:
        return templates.TemplateResponse("404.html", {
            "request": request, "message": f"Team '{name}' not found"
        }, status_code=404)
    actual_name = info["teamname"]
    roster = db.get_team_roster(actual_name)
    stats = db.get_team_stats_by_split(actual_name)
    titles = db.get_team_titles(actual_name)
    recent = db.get_team_recent_series(actual_name)
    betting = db.get_team_betting_stats(actual_name)
    wr_by_split = db.get_team_winrate_by_split(actual_name)
    form = db.get_team_form(actual_name, limit=10)
    return templates.TemplateResponse("team.html", {
        "request": request,
        "info": info,
        "roster": roster.to_dict("records"),
        "stats_by_split": stats.to_dict("records"),
        "titles": titles.to_dict("records"),
        "recent_series": recent.to_dict("records"),
        "betting": betting,
        "wr_by_split": wr_by_split.to_dict("records"),
        "form": form.to_dict("records"),
    })


# ----- TOURNAMENT -----

@app.get("/tournaments", response_class=HTMLResponse)
async def tournaments_list(request: Request):
    leagues = db.get_tournament_leagues()
    return templates.TemplateResponse("tournaments.html", {
        "request": request,
        "leagues": leagues.to_dict("records"),
    })


@app.get("/tournament/{league}/{year}", response_class=HTMLResponse)
async def tournament(request: Request, league: str, year: int):
    results = db.get_tournament_results(league, year)
    champion = db.get_tournament_champion(league, year)
    years = db.get_tournament_years(league)
    return templates.TemplateResponse("tournament.html", {
        "request": request,
        "league": league,
        "year": year,
        "results": results.to_dict("records"),
        "champions": champion.to_dict("records"),
        "available_years": years["year"].tolist(),
    })


# ----- GAME -----

@app.get("/game/{gameid:path}", response_class=HTMLResponse)
async def game(request: Request, gameid: str):
    players = db.get_game_players(gameid)
    teams = db.get_game_teams(gameid)
    if len(teams) == 0:
        return templates.TemplateResponse("404.html", {
            "request": request, "message": f"Game '{gameid}' not found"
        }, status_code=404)
    return templates.TemplateResponse("game.html", {
        "request": request,
        "gameid": gameid,
        "players": players.to_dict("records"),
        "teams": teams.to_dict("records"),
    })


# ----- COMPARE -----

@app.get("/compare", response_class=HTMLResponse)
async def compare(
    request: Request,
    p1: str = Query(""),
    p2: str = Query(""),
    mode: str = Query("players"),
    split: str = Query(None),
):
    if mode == "teams":
        if not p1 or not p2:
            return templates.TemplateResponse("compare.html", {
                "request": request, "p1": p1, "p2": p2,
                "mode": "teams", "comparison": None, "h2h": None,
                "h2h_series": None, "h2h_summary": None,
                "bet1": None, "bet2": None, "form1": [], "form2": [],
                "target_split": split,
            })
        
        # Split is formatted as "YEAR-LEAGUE-SPLIT-PLAYOFFS", e.g. "2024-LCK-Spring-0" or "all"
        target_year = None
        target_league = None
        target_split_name = None
        target_playoffs = None
        
        if split and split != "all":
            parts = split.split("-")
            if len(parts) >= 4:
                target_year = int(parts[0])
                target_league = parts[1]
                target_split_name = parts[2]
                target_playoffs = int(parts[3])
                
        comparison = db.get_team_comparison(
            p1, p2, year=target_year, split=target_split_name, 
            playoffs=target_playoffs, league=target_league
        )
        h2h_series = db.get_team_head_to_head(
            p1, p2, year=target_year, split=target_split_name,
            playoffs=target_playoffs, league=target_league
        )
        h2h_summary = db.get_team_h2h_summary(
            p1, p2, year=target_year, split=target_split_name,
            playoffs=target_playoffs, league=target_league
        )
        rows = comparison.to_dict("records")
        # resolve exact names from comparison rows
        name1 = rows[0]["teamname"] if rows else p1
        name2 = rows[1]["teamname"] if len(rows) > 1 else p2
        # General stats (overall, no H2H filter)
        gen1 = db.get_team_betting_stats(name1)
        gen2 = db.get_team_betting_stats(name2)
        # H2H stats (filtered to games between these two teams)
        bet1 = db.get_team_betting_stats(
            name1, year=target_year, split=target_split_name, 
            playoffs=target_playoffs, league=target_league, h2h_opponent=name2
        )
        bet2 = db.get_team_betting_stats(
            name2, year=target_year, split=target_split_name, 
            playoffs=target_playoffs, league=target_league, h2h_opponent=name1
        )
        form1 = db.get_team_form(name1, limit=10).to_dict("records")
        form2 = db.get_team_form(name2, limit=10).to_dict("records")
        h2h_by_split = db.get_team_h2h_by_split(name1, name2)
        return templates.TemplateResponse("compare.html", {
            "request": request,
            "p1": p1, "p2": p2,
            "mode": "teams",
            "comparison": rows,
            "h2h": None,
            "h2h_series": h2h_series.to_dict("records"),
            "h2h_summary": h2h_summary,
            "gen1": gen1, "gen2": gen2,
            "bet1": bet1, "bet2": bet2,
            "form1": form1, "form2": form2,
            "name1": name1, "name2": name2,
            "h2h_by_split": h2h_by_split,
            "target_split": split or "all",
        })
    else:
        if not p1 or not p2:
            return templates.TemplateResponse("compare.html", {
                "request": request, "p1": p1, "p2": p2,
                "mode": "players", "comparison": None, "h2h": None,
                "h2h_series": None, "h2h_summary": None,
            })
        comparison = db.get_player_comparison(p1, p2)
        h2h = db.get_head_to_head(p1, p2)
        return templates.TemplateResponse("compare.html", {
            "request": request,
            "p1": p1,
            "p2": p2,
            "mode": "players",
            "comparison": comparison.to_dict("records"),
            "h2h": h2h.to_dict("records")[0] if len(h2h) > 0 else None,
            "h2h_series": None,
            "h2h_summary": None,
        })


# ----- RANKINGS -----

@app.get("/rankings", response_class=HTMLResponse)
async def rankings(
    request: Request,
    stat: str = Query("kda"),
    position: str = Query(None),
    league: str = Query(None),
    year: int = Query(None),
    split: str = Query(None),
    min_games: int = Query(30),
):
    data = db.get_player_rankings(
        stat=stat, position=position, league=league,
        year=year, split=split, min_games=min_games,
    )
    years = db.get_available_years()
    leagues = db.get_available_leagues()
    splits = db.get_available_splits()
    return templates.TemplateResponse("rankings.html", {
        "request": request,
        "rankings": data.to_dict("records"),
        "stat": stat,
        "position": position,
        "league": league,
        "year": year,
        "split": split,
        "min_games": min_games,
        "available_years": years,
        "available_leagues": leagues,
        "available_splits": splits,
    })


# ----- BETTING -----

@app.get("/betting", response_class=HTMLResponse)
async def betting(
    request: Request,
    league: str = Query(None),
    year: str = Query(None),
    split: str = Query(None),
    playoffs: str = Query(None),
    team: str = Query(""),
):
    # Convert empty strings to None, then to proper types
    year_int = int(year) if year else None
    playoffs_int = int(playoffs) if playoffs and playoffs.strip() != "" else None
    league = league or None
    split = split or None

    data = db.get_betting_stats(
        team=team or None,
        league=league,
        year=year_int,
        split=split,
        playoffs=playoffs_int,
    )
    available_leagues, available_years, available_splits = db.get_betting_filters()
    return templates.TemplateResponse("betting.html", {
        "request": request,
        "rows": data.to_dict("records"),
        "league": league,
        "year": year_int,
        "split": split,
        "playoffs": playoffs_int,
        "team": team,
        "available_leagues": available_leagues,
        "available_years": available_years,
        "available_splits": available_splits,
    })
