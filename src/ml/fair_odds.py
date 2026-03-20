"""Manual fair-odds scoring on top of the trained prematch baseline."""

from __future__ import annotations

import json
import math
import pickle
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .prematch_baseline import LABEL_COLUMN, META_COLUMNS, load_feature_frame

DEFAULT_LATEST_RUN_POINTER = Path("data/models/prematch_baseline/latest_run.json")
TEAM_HISTORY_STALE_DAYS = 180.0


@dataclass(slots=True)
class MarketOddsComparison:
    """Comparison between model probabilities and a two-way market."""

    team1_odds: float
    team2_odds: float
    team1_implied_prob_raw: float
    team2_implied_prob_raw: float
    overround: float
    team1_implied_prob_devig: float
    team2_implied_prob_devig: float
    team1_edge_vs_devig: float
    team2_edge_vs_devig: float
    team1_ev_per_unit: float
    team2_ev_per_unit: float


@dataclass(slots=True)
class FairOddsQuote:
    """Serializable fair-odds output for a single manual matchup."""

    run_id: str
    snapshot_id: str
    model_name: str
    model_path: str
    feature_version: str
    history_source: str
    series_key: str
    match_time: str
    league_code: str
    split_name: str
    patch_version: str
    best_of: int
    playoffs: bool
    season_year: int
    team1_key: str
    team1_name: str
    team2_key: str
    team2_name: str
    team1_win_prob: float
    team2_win_prob: float
    team1_fair_odds: float
    team2_fair_odds: float
    team1_core_series_count: int
    team2_core_series_count: int
    team1_latest_core_series_at: str | None
    team2_latest_core_series_at: str | None
    team1_latest_fact_series_at: str | None
    team2_latest_fact_series_at: str | None
    warnings: list[str]
    market_comparison: MarketOddsComparison | None = None


def _normalize_team_name(value: Any) -> str:
    text = "" if value is None else str(value)
    return "".join(character.lower() for character in text if character.isalnum())


def _string_or_empty(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _neutral_rate(wins: float, total: float) -> float:
    return 0.5 if not total else float(wins) / float(total)


def _safe_average(total: float, count: float) -> float:
    return 0.0 if not count else float(total) / float(count)


def _average(values: list[float], neutral: float = 0.0) -> float:
    return neutral if not values else float(sum(values)) / float(len(values))


def _roster_overlap(current_roster: tuple[str, ...], prior_roster: tuple[str, ...] | None) -> int:
    if not current_roster or not prior_roster:
        return 0
    return len(set(current_roster).intersection(prior_roster))


def _recent_roster_overlap(
    current_roster: tuple[str, ...],
    roster_history: list[tuple[str, ...]],
) -> float:
    if not current_roster or not roster_history:
        return 0.0
    overlaps = [_roster_overlap(current_roster, prior_roster) for prior_roster in roster_history]
    return _average([float(value) for value in overlaps], neutral=0.0)


def _draft_window_metrics(game_history: list[dict[str, Any]], window: int) -> dict[str, float]:
    recent_games = game_history[-window:]
    champion_counter: Counter[str] = Counter()
    first_picks: list[int] = []

    for item in recent_games:
        champion_counter.update(item["champions"])
        first_picks.append(1 if item["first_pick"] else 0)

    total_picks = sum(champion_counter.values())
    top5_share = (
        sum(sorted(champion_counter.values(), reverse=True)[:5]) / total_picks if total_picks else 0.0
    )
    return {
        "unique_champions": float(len(champion_counter)),
        "top5_share": float(top5_share),
        "first_pick_rate": _average([float(value) for value in first_picks], neutral=0.5),
    }


def _days_since(last_seen: pd.Timestamp | None, current_time: pd.Timestamp) -> float:
    if last_seen is None or pd.isna(last_seen):
        return float("nan")
    return (current_time - last_seen).total_seconds() / 86400.0


def _probability_to_fair_odds(probability: float) -> float:
    probability = float(probability)
    if probability <= 0.0:
        return math.inf
    return float(1.0 / probability)


def compare_two_way_market(
    team1_model_prob: float,
    team2_model_prob: float,
    team1_odds: float,
    team2_odds: float,
) -> MarketOddsComparison:
    """Remove the vig from a two-way market and compare it to model probabilities."""
    if team1_odds <= 1.0 or team2_odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0")

    team1_raw = 1.0 / float(team1_odds)
    team2_raw = 1.0 / float(team2_odds)
    total = team1_raw + team2_raw
    team1_devig = team1_raw / total
    team2_devig = team2_raw / total

    return MarketOddsComparison(
        team1_odds=float(team1_odds),
        team2_odds=float(team2_odds),
        team1_implied_prob_raw=float(team1_raw),
        team2_implied_prob_raw=float(team2_raw),
        overround=float(total - 1.0),
        team1_implied_prob_devig=float(team1_devig),
        team2_implied_prob_devig=float(team2_devig),
        team1_edge_vs_devig=float(team1_model_prob - team1_devig),
        team2_edge_vs_devig=float(team2_model_prob - team2_devig),
        team1_ev_per_unit=float(team1_model_prob * float(team1_odds) - 1.0),
        team2_ev_per_unit=float(team2_model_prob * float(team2_odds) - 1.0),
    )


def _serialize_timestamp(value: pd.Timestamp | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).tz_convert("UTC").isoformat()


def _iter_game_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None and not pd.isna(item)]
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(item) for item in converted if item is not None and not pd.isna(item)]
    if pd.isna(value):
        return []
    return [str(value)]


class PrematchFairOddsScorer:
    """Score a future series and return model fair odds."""

    def __init__(self, run_pointer: Path = DEFAULT_LATEST_RUN_POINTER) -> None:
        pointer_payload = json.loads(run_pointer.read_text(encoding="utf-8"))
        self.run_id = str(pointer_payload["run_id"])
        self.output_dir = Path(pointer_payload["output_dir"])

        metrics_path = Path(pointer_payload["metrics_path"])
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.snapshot_id = str(metrics_payload["snapshot_id"])
        self.model_name = str(metrics_payload["best_model_name"])
        self.model_path = self.output_dir / f"{self.model_name}.pkl"
        self.features_path = Path(metrics_payload["features_path"])
        self.snapshot_dir = self.features_path.parent

        with self.model_path.open("rb") as file_obj:
            self.model = pickle.load(file_obj)

        self.features_df = load_feature_frame(self.features_path)
        self.feature_columns = [
            column
            for column in self.features_df.columns
            if column not in META_COLUMNS + [LABEL_COLUMN, "event_time", "event_date"]
        ]
        feature_versions = self.features_df["feature_version"].dropna()
        self.feature_version = (
            str(feature_versions.iloc[-1]) if not feature_versions.empty else "unknown_feature_version"
        )

        self.core_series_df = self._load_core_series(self.snapshot_dir / "model_core_series.parquet")
        self.fact_series_df = self._load_fact_series(self.snapshot_dir / "fact_series.parquet")
        self.fact_game_player_df = self._load_fact_game_player(self.snapshot_dir / "fact_game_player.parquet")
        self.fact_draft_df = self._load_fact_draft(self.snapshot_dir / "fact_draft.parquet")
        self.game_index_df = self._build_game_index(self.fact_series_df)

        self.game_to_series_key = dict(
            zip(self.game_index_df["game_id"], self.game_index_df["series_key"], strict=False)
        )
        self.game_to_event_time = dict(
            zip(self.game_index_df["game_id"], self.game_index_df["event_time"], strict=False)
        )
        self.game_to_series_winner = dict(
            zip(self.game_index_df["game_id"], self.game_index_df["series_winner_key"], strict=False)
        )

        self.seen_team_keys = set(self.features_df["team1_key"].dropna()).union(
            set(self.features_df["team2_key"].dropna())
        )
        self.seen_leagues = set(self.features_df["league_code"].dropna())
        self.seen_splits = set(self.features_df["split_name"].dropna())
        self.seen_patches = set(self.features_df["patch_version"].dropna())
        self.team_key_to_name, self.team_alias_lookup = self._build_team_lookup()

    @staticmethod
    def _load_core_series(path: Path) -> pd.DataFrame:
        columns = [
            "snapshot_id",
            "series_key",
            "series_date",
            "start_time",
            "league_code",
            "season_year",
            "split_name",
            "playoffs",
            "patch_version",
            "team1_key",
            "team1_name",
            "team2_key",
            "team2_name",
            "games_played",
            "team1_wins",
            "team2_wins",
            "label_team1_win",
            "best_of_inferred",
            "avg_game_length_seconds",
        ]
        df = pd.read_parquet(path, columns=columns)
        df["series_date"] = pd.to_datetime(df["series_date"], utc=True, errors="coerce")
        df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
        df["event_time"] = df["start_time"].fillna(df["series_date"])
        return df.sort_values(["event_time", "series_key"], kind="stable").reset_index(drop=True)

    @staticmethod
    def _load_fact_series(path: Path) -> pd.DataFrame:
        columns = [
            "series_key",
            "series_date",
            "start_time",
            "team1_key",
            "team1_name",
            "team2_key",
            "team2_name",
            "team1_wins",
            "team2_wins",
            "series_winner_key",
            "game_ids",
        ]
        df = pd.read_parquet(path, columns=columns)
        df["series_date"] = pd.to_datetime(df["series_date"], utc=True, errors="coerce")
        df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
        df["event_time"] = df["start_time"].fillna(df["series_date"])
        return df.sort_values(["event_time", "series_key"], kind="stable").reset_index(drop=True)

    @staticmethod
    def _load_fact_game_player(path: Path) -> pd.DataFrame:
        columns = ["game_id", "team_key", "player_key", "champion_name"]
        df = pd.read_parquet(path, columns=columns)
        df["game_id"] = df["game_id"].map(str)
        df["team_key"] = df["team_key"].map(str)
        df["player_key"] = df["player_key"].map(str)
        return df

    @staticmethod
    def _load_fact_draft(path: Path) -> pd.DataFrame:
        columns = ["game_id", "team_key", "first_pick"]
        df = pd.read_parquet(path, columns=columns)
        df["game_id"] = df["game_id"].map(str)
        df["team_key"] = df["team_key"].map(str)
        return df

    @staticmethod
    def _build_game_index(fact_series_df: pd.DataFrame) -> pd.DataFrame:
        game_index_df = fact_series_df[["series_key", "event_time", "series_winner_key", "game_ids"]].copy()
        game_index_df["game_ids"] = game_index_df["game_ids"].apply(_iter_game_ids)
        game_index_df = game_index_df.explode("game_ids", ignore_index=True)
        game_index_df = game_index_df.rename(columns={"game_ids": "game_id"})
        game_index_df = game_index_df.dropna(subset=["game_id"])
        game_index_df["game_id"] = game_index_df["game_id"].map(str)
        return game_index_df

    def _build_team_lookup(self) -> tuple[dict[str, str], defaultdict[str, set[str]]]:
        frames: list[pd.DataFrame] = []
        for source_df in (self.features_df, self.fact_series_df):
            for key_column, name_column in (("team1_key", "team1_name"), ("team2_key", "team2_name")):
                part = source_df[[key_column, name_column, "event_time"]].rename(
                    columns={key_column: "team_key", name_column: "team_name"}
                )
                frames.append(part)

        team_df = pd.concat(frames, ignore_index=True)
        team_df = team_df.dropna(subset=["team_key"]).copy()
        team_df["team_key"] = team_df["team_key"].map(str)
        team_df["team_name"] = team_df["team_name"].fillna(team_df["team_key"]).map(str)
        team_df = team_df.sort_values(["event_time", "team_key", "team_name"], kind="stable")

        key_to_name: dict[str, str] = {}
        alias_lookup: defaultdict[str, set[str]] = defaultdict(set)
        for row in team_df.itertuples(index=False):
            key_to_name[row.team_key] = row.team_name
            for alias in {row.team_key, row.team_name}:
                normalized = _normalize_team_name(alias)
                if normalized:
                    alias_lookup[normalized].add(row.team_key)
        return key_to_name, alias_lookup

    def resolve_team(self, team_input: str) -> tuple[str, str]:
        """Resolve a user-supplied team string into the canonical team key and name."""
        if team_input in self.team_key_to_name:
            return team_input, self.team_key_to_name[team_input]

        normalized = _normalize_team_name(team_input)
        direct_matches = self.team_alias_lookup.get(normalized, set())
        if len(direct_matches) == 1:
            team_key = next(iter(direct_matches))
            return team_key, self.team_key_to_name[team_key]

        fuzzy_matches = {
            team_key
            for alias, keys in self.team_alias_lookup.items()
            if normalized and (normalized in alias or alias in normalized)
            for team_key in keys
        }
        core_fuzzy_matches = sorted(team_key for team_key in fuzzy_matches if team_key in self.seen_team_keys)
        if len(core_fuzzy_matches) == 1:
            team_key = core_fuzzy_matches[0]
            return team_key, self.team_key_to_name[team_key]
        if len(fuzzy_matches) == 1:
            team_key = next(iter(fuzzy_matches))
            return team_key, self.team_key_to_name[team_key]

        if fuzzy_matches:
            candidates = ", ".join(
                sorted(f"{team_key} ({self.team_key_to_name[team_key]})" for team_key in fuzzy_matches)
            )
            raise ValueError(f"Ambiguous team '{team_input}'. Candidates: {candidates}")

        raise ValueError(f"Unknown team '{team_input}'")

    def _build_global_core_state(
        self,
        event_time: pd.Timestamp,
    ) -> tuple[
        defaultdict[str, dict[str, Any]],
        defaultdict[str, deque[int]],
        defaultdict[tuple[str, str], dict[str, int]],
        defaultdict[tuple[str, str], dict[str, int]],
        defaultdict[tuple[str, str], dict[str, Any]],
    ]:
        team_state: defaultdict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "elo": 1500.0,
                "series_count": 0,
                "series_wins": 0,
                "game_count": 0,
                "game_wins": 0,
                "games_played_total": 0,
                "games_won_total": 0,
                "game_length_seconds_total": 0.0,
                "series_streak": 0,
                "last_series_at": None,
            }
        )
        recent_series_results: defaultdict[str, deque[int]] = defaultdict(lambda: deque(maxlen=5))
        patch_state: defaultdict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"series_count": 0, "series_wins": 0}
        )
        split_state: defaultdict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"series_count": 0, "series_wins": 0}
        )
        h2h_state: defaultdict[tuple[str, str], dict[str, Any]] = defaultdict(
            lambda: {
                "series_count": 0,
                "game_count": 0,
                "series_wins": defaultdict(int),
                "game_wins": defaultdict(int),
            }
        )

        for row in self.core_series_df.itertuples(index=False):
            if pd.isna(row.event_time) or row.event_time >= event_time:
                break

            team1_key = str(row.team1_key)
            team2_key = str(row.team2_key)
            patch_version = _string_or_empty(row.patch_version)
            split_name = _string_or_empty(row.split_name)
            games_played = int(row.games_played) if pd.notna(row.games_played) else 0
            team1_wins = int(row.team1_wins) if pd.notna(row.team1_wins) else 0
            team2_wins = int(row.team2_wins) if pd.notna(row.team2_wins) else 0
            label_team1_win = int(row.label_team1_win) if pd.notna(row.label_team1_win) else int(team1_wins > team2_wins)
            best_of_inferred = (
                int(row.best_of_inferred) if pd.notna(row.best_of_inferred) else max(games_played, 1)
            )
            avg_game_length_seconds = (
                float(row.avg_game_length_seconds) if pd.notna(row.avg_game_length_seconds) else 0.0
            )

            team1_state = team_state[team1_key]
            team2_state = team_state[team2_key]
            team1_series_win = int(label_team1_win == 1)
            team2_series_win = int(label_team1_win == 0)
            best_of = max(best_of_inferred, 1)
            margin_multiplier = 1.0 + abs(team1_wins - team2_wins) / best_of
            k_factor = 32.0 * margin_multiplier
            expected_team1 = 1.0 / (1.0 + 10.0 ** ((team2_state["elo"] - team1_state["elo"]) / 400.0))
            expected_team2 = 1.0 - expected_team1

            team1_state["elo"] = float(team1_state["elo"]) + k_factor * (team1_series_win - expected_team1)
            team2_state["elo"] = float(team2_state["elo"]) + k_factor * (team2_series_win - expected_team2)

            for state, team_series_win, team_game_wins in (
                (team1_state, team1_series_win, team1_wins),
                (team2_state, team2_series_win, team2_wins),
            ):
                state["series_count"] += 1
                state["series_wins"] += team_series_win
                state["game_count"] += games_played
                state["game_wins"] += team_game_wins
                state["games_played_total"] += games_played
                state["games_won_total"] += team_game_wins
                state["game_length_seconds_total"] += avg_game_length_seconds * games_played
                previous_streak = int(state["series_streak"])
                if team_series_win:
                    state["series_streak"] = previous_streak + 1 if previous_streak > 0 else 1
                else:
                    state["series_streak"] = previous_streak - 1 if previous_streak < 0 else -1
                state["last_series_at"] = row.event_time

            recent_series_results[team1_key].append(team1_series_win)
            recent_series_results[team2_key].append(team2_series_win)

            patch_state[(team1_key, patch_version)]["series_count"] += 1
            patch_state[(team1_key, patch_version)]["series_wins"] += team1_series_win
            patch_state[(team2_key, patch_version)]["series_count"] += 1
            patch_state[(team2_key, patch_version)]["series_wins"] += team2_series_win

            split_state[(team1_key, split_name)]["series_count"] += 1
            split_state[(team1_key, split_name)]["series_wins"] += team1_series_win
            split_state[(team2_key, split_name)]["series_count"] += 1
            split_state[(team2_key, split_name)]["series_wins"] += team2_series_win

            matchup_key = tuple(sorted((team1_key, team2_key)))
            matchup_state = h2h_state[matchup_key]
            matchup_state["series_count"] += 1
            matchup_state["game_count"] += games_played
            matchup_state["series_wins"][team1_key] += team1_series_win
            matchup_state["series_wins"][team2_key] += team2_series_win
            matchup_state["game_wins"][team1_key] += team1_wins
            matchup_state["game_wins"][team2_key] += team2_wins

        return team_state, recent_series_results, patch_state, split_state, h2h_state

    def _build_local_team_context(
        self,
        event_time: pd.Timestamp,
        team_keys: set[str],
    ) -> dict[str, dict[str, Any]]:
        team_context: dict[str, dict[str, Any]] = {
            team_key: {
                "current_roster": tuple(),
                "roster_history": [],
                "draft_history": [],
                "latest_fact_series_at": None,
                "player_summary": {},
            }
            for team_key in team_keys
        }

        prior_series_df = self.fact_series_df.loc[
            (self.fact_series_df["event_time"] < event_time)
            & (
                self.fact_series_df["team1_key"].isin(team_keys)
                | self.fact_series_df["team2_key"].isin(team_keys)
            )
        ].copy()
        if prior_series_df.empty:
            return team_context

        prior_series_df = prior_series_df.sort_values(["event_time", "series_key"], kind="stable")
        relevant_game_ids = {
            game_id for raw_ids in prior_series_df["game_ids"].tolist() for game_id in _iter_game_ids(raw_ids)
        }

        player_rows = self.fact_game_player_df.loc[
            self.fact_game_player_df["game_id"].isin(relevant_game_ids)
            & self.fact_game_player_df["team_key"].isin(team_keys)
        ].copy()
        player_rows["series_key"] = player_rows["game_id"].map(self.game_to_series_key)

        draft_rows = self.fact_draft_df.loc[
            self.fact_draft_df["game_id"].isin(relevant_game_ids)
            & self.fact_draft_df["team_key"].isin(team_keys)
        ].copy()
        draft_rows["series_key"] = draft_rows["game_id"].map(self.game_to_series_key)

        series_rosters: dict[tuple[str, str], tuple[str, ...]] = {}
        if not player_rows.empty:
            roster_counts = (
                player_rows.groupby(["series_key", "team_key", "player_key"], as_index=False)
                .agg(games_played=("game_id", "nunique"))
                .sort_values(
                    ["series_key", "team_key", "games_played", "player_key"],
                    ascending=[True, True, False, True],
                    kind="stable",
                )
            )
            for (series_key, team_key), frame in roster_counts.groupby(["series_key", "team_key"], sort=False):
                series_rosters[(str(series_key), str(team_key))] = tuple(frame.head(5)["player_key"].map(str).tolist())

        first_pick_lookup: dict[tuple[str, str, str], bool] = {}
        if not draft_rows.empty:
            draft_summary = (
                draft_rows.groupby(["series_key", "team_key", "game_id"], as_index=False)
                .agg(first_pick=("first_pick", "max"))
            )
            first_pick_lookup = {
                (str(row.series_key), str(row.team_key), str(row.game_id)): bool(row.first_pick)
                for row in draft_summary.itertuples(index=False)
            }

        series_team_games: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        if not player_rows.empty:
            sorted_players = player_rows.sort_values(
                ["series_key", "team_key", "game_id", "champion_name"],
                kind="stable",
            )
            for (series_key, team_key, game_id), frame in sorted_players.groupby(
                ["series_key", "team_key", "game_id"],
                sort=False,
            ):
                champions = tuple(
                    sorted(
                        {
                            champion
                            for champion in frame["champion_name"].tolist()
                            if isinstance(champion, str) and champion
                        }
                    )
                )
                series_team_games[(str(series_key), str(team_key))].append(
                    {
                        "game_id": str(game_id),
                        "champions": champions,
                        "first_pick": first_pick_lookup.get(
                            (str(series_key), str(team_key), str(game_id)),
                            False,
                        ),
                    }
                )

        team_roster_events: defaultdict[str, list[tuple[pd.Timestamp, tuple[str, ...]]]] = defaultdict(list)
        team_draft_history: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

        for row in prior_series_df.itertuples(index=False):
            present_keys = {str(row.team1_key), str(row.team2_key)}
            for team_key in team_keys:
                if team_key not in present_keys:
                    continue
                team_context[team_key]["latest_fact_series_at"] = row.event_time
                roster = series_rosters.get((str(row.series_key), team_key), tuple())
                if roster:
                    team_roster_events[team_key].append((row.event_time, roster))
                team_draft_history[team_key].extend(series_team_games.get((str(row.series_key), team_key), []))

        for team_key, context in team_context.items():
            ordered_rosters = [roster for _, roster in sorted(team_roster_events[team_key], key=lambda item: item[0])]
            if ordered_rosters:
                context["current_roster"] = ordered_rosters[-1]
                context["roster_history"] = ordered_rosters[-3:]
            context["draft_history"] = team_draft_history[team_key]

            current_roster = context["current_roster"]
            if not current_roster:
                continue

            current_players = set(current_roster)
            player_history_rows = self.fact_game_player_df.loc[
                self.fact_game_player_df["player_key"].isin(current_players)
            ].copy()
            player_history_rows["event_time"] = player_history_rows["game_id"].map(self.game_to_event_time)
            player_history_rows["series_key"] = player_history_rows["game_id"].map(self.game_to_series_key)
            player_history_rows["series_winner_key"] = player_history_rows["game_id"].map(
                self.game_to_series_winner
            )
            player_history_rows = player_history_rows.loc[
                player_history_rows["event_time"].notna() & (player_history_rows["event_time"] < event_time)
            ]

            if player_history_rows.empty:
                continue

            player_series = (
                player_history_rows.groupby(["player_key", "series_key"], as_index=False)
                .agg(
                    team_key=("team_key", "first"),
                    series_winner_key=("series_winner_key", "first"),
                )
                .sort_values(["player_key", "series_key"], kind="stable")
            )
            context["player_summary"] = {
                str(player_key): {
                    "series_count": int(len(frame)),
                    "series_wins": int((frame["team_key"] == frame["series_winner_key"]).sum()),
                }
                for player_key, frame in player_series.groupby("player_key", sort=False)
            }

        return team_context

    def _build_feature_row(
        self,
        event_time: pd.Timestamp,
        team1_key: str,
        team1_name: str,
        team2_key: str,
        team2_name: str,
        league_code: str,
        split_name: str,
        patch_version: str,
        best_of: int,
        playoffs: bool,
    ) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
        team_state, recent_series_results, patch_state, split_state, h2h_state = self._build_global_core_state(
            event_time
        )
        local_context = self._build_local_team_context(event_time, {team1_key, team2_key})

        warnings: list[str] = []
        model_split_name = split_name
        if model_split_name not in self.seen_splits and league_code in self.seen_splits:
            warnings.append(f"Split '{split_name}' nao apareceu no treino core; usando fallback '{league_code}'.")
            model_split_name = league_code

        if team1_key not in self.seen_team_keys:
            warnings.append(f"{team1_name} nao apareceu no treino core; team1_key sera tratado como desconhecido.")
        if team2_key not in self.seen_team_keys:
            warnings.append(f"{team2_name} nao apareceu no treino core; team2_key sera tratado como desconhecido.")
        if league_code not in self.seen_leagues:
            warnings.append(f"Liga {league_code} nao apareceu no treino core; essa categoria sera tratada como desconhecida.")
        if model_split_name not in self.seen_splits:
            warnings.append(f"Split '{split_name}' nao apareceu no treino core.")
        if patch_version not in self.seen_patches:
            warnings.append(f"Patch '{patch_version}' nao apareceu no treino core.")

        team1_state = team_state[team1_key]
        team2_state = team_state[team2_key]
        team1_recent = recent_series_results[team1_key]
        team2_recent = recent_series_results[team2_key]
        team1_patch_state = patch_state[(team1_key, patch_version)]
        team2_patch_state = patch_state[(team2_key, patch_version)]
        team1_split_state = split_state[(team1_key, model_split_name)]
        team2_split_state = split_state[(team2_key, model_split_name)]
        matchup_key = tuple(sorted((team1_key, team2_key)))
        matchup_state = h2h_state[matchup_key]

        team1_local = local_context[team1_key]
        team2_local = local_context[team2_key]
        team1_roster = team1_local["current_roster"]
        team2_roster = team2_local["current_roster"]
        team1_roster_history = team1_local["roster_history"]
        team2_roster_history = team2_local["roster_history"]

        team1_prev_series_roster_overlap = _roster_overlap(
            team1_roster,
            team1_roster_history[-1] if team1_roster_history else None,
        )
        team2_prev_series_roster_overlap = _roster_overlap(
            team2_roster,
            team2_roster_history[-1] if team2_roster_history else None,
        )
        team1_recent3_avg_roster_overlap = _recent_roster_overlap(team1_roster, team1_roster_history)
        team2_recent3_avg_roster_overlap = _recent_roster_overlap(team2_roster, team2_roster_history)

        team1_player_states = [
            team1_local["player_summary"].get(player_key, {"series_count": 0, "series_wins": 0})
            for player_key in team1_roster
        ]
        team2_player_states = [
            team2_local["player_summary"].get(player_key, {"series_count": 0, "series_wins": 0})
            for player_key in team2_roster
        ]
        team1_roster_avg_player_prior_series_count = _average(
            [float(state["series_count"]) for state in team1_player_states],
            neutral=0.0,
        )
        team2_roster_avg_player_prior_series_count = _average(
            [float(state["series_count"]) for state in team2_player_states],
            neutral=0.0,
        )
        team1_roster_avg_player_prior_series_win_rate = _average(
            [_neutral_rate(state["series_wins"], state["series_count"]) for state in team1_player_states],
            neutral=0.5,
        )
        team2_roster_avg_player_prior_series_win_rate = _average(
            [_neutral_rate(state["series_wins"], state["series_count"]) for state in team2_player_states],
            neutral=0.5,
        )
        team1_roster_new_player_count = sum(1 for state in team1_player_states if state["series_count"] == 0)
        team2_roster_new_player_count = sum(1 for state in team2_player_states if state["series_count"] == 0)

        team1_draft_recent10 = _draft_window_metrics(team1_local["draft_history"], window=10)
        team2_draft_recent10 = _draft_window_metrics(team2_local["draft_history"], window=10)
        team1_draft_recent25 = _draft_window_metrics(team1_local["draft_history"], window=25)
        team2_draft_recent25 = _draft_window_metrics(team2_local["draft_history"], window=25)

        team1_pre_elo = float(team1_state["elo"])
        team2_pre_elo = float(team2_state["elo"])
        team1_elo_win_prob = 1.0 / (1.0 + 10.0 ** ((team2_pre_elo - team1_pre_elo) / 400.0))

        team1_prior_series_count = int(team1_state["series_count"])
        team2_prior_series_count = int(team2_state["series_count"])
        team1_prior_game_count = int(team1_state["game_count"])
        team2_prior_game_count = int(team2_state["game_count"])

        team1_prior_series_win_rate = _neutral_rate(team1_state["series_wins"], team1_prior_series_count)
        team2_prior_series_win_rate = _neutral_rate(team2_state["series_wins"], team2_prior_series_count)
        team1_prior_game_win_rate = _neutral_rate(team1_state["game_wins"], team1_prior_game_count)
        team2_prior_game_win_rate = _neutral_rate(team2_state["game_wins"], team2_prior_game_count)

        team1_recent5_series_count = len(team1_recent)
        team2_recent5_series_count = len(team2_recent)
        team1_recent5_series_win_rate = _neutral_rate(sum(team1_recent), team1_recent5_series_count)
        team2_recent5_series_win_rate = _neutral_rate(sum(team2_recent), team2_recent5_series_count)

        team1_patch_prior_series_count = int(team1_patch_state["series_count"])
        team2_patch_prior_series_count = int(team2_patch_state["series_count"])
        team1_patch_prior_win_rate = _neutral_rate(team1_patch_state["series_wins"], team1_patch_prior_series_count)
        team2_patch_prior_win_rate = _neutral_rate(team2_patch_state["series_wins"], team2_patch_prior_series_count)

        team1_split_prior_series_count = int(team1_split_state["series_count"])
        team2_split_prior_series_count = int(team2_split_state["series_count"])
        team1_split_prior_win_rate = _neutral_rate(team1_split_state["series_wins"], team1_split_prior_series_count)
        team2_split_prior_win_rate = _neutral_rate(team2_split_state["series_wins"], team2_split_prior_series_count)

        h2h_prior_series_count = int(matchup_state["series_count"])
        h2h_prior_game_count = int(matchup_state["game_count"])
        h2h_team1_series_wins = int(matchup_state["series_wins"][team1_key])
        h2h_team2_series_wins = int(matchup_state["series_wins"][team2_key])
        h2h_team1_game_wins = int(matchup_state["game_wins"][team1_key])
        h2h_team2_game_wins = int(matchup_state["game_wins"][team2_key])

        team1_last_core = team1_state["last_series_at"]
        team2_last_core = team2_state["last_series_at"]
        team1_last_fact = team1_local["latest_fact_series_at"]
        team2_last_fact = team2_local["latest_fact_series_at"]

        for team_name, last_core in ((team1_name, team1_last_core), (team2_name, team2_last_core)):
            if last_core is None:
                warnings.append(f"Sem historico core para {team_name}; features globais ficaram neutras.")
                continue
            history_gap = _days_since(last_core, event_time)
            if history_gap > TEAM_HISTORY_STALE_DAYS:
                warnings.append(
                    f"Historico core de {team_name} esta {history_gap:.0f} dias atras do jogo alvo."
                )

        for team_name, last_fact in ((team1_name, team1_last_fact), (team2_name, team2_last_fact)):
            if last_fact is None:
                warnings.append(f"Sem historico local de roster/draft para {team_name}.")

        feature_row = {
            "snapshot_id": self.snapshot_id,
            "feature_version": self.feature_version,
            "series_key": f"manual::{team1_key}::{team2_key}::{event_time.strftime('%Y%m%dT%H%M%SZ')}",
            "series_date": event_time.floor("D"),
            "start_time": event_time,
            "league_code": league_code,
            "season_year": int(event_time.year),
            "split_name": model_split_name,
            "playoffs": bool(playoffs),
            "patch_version": patch_version,
            "team1_key": team1_key,
            "team1_name": team1_name,
            "team2_key": team2_key,
            "team2_name": team2_name,
            "best_of_inferred": int(best_of),
            "label_team1_win": None,
            "team1_pre_elo": team1_pre_elo,
            "team2_pre_elo": team2_pre_elo,
            "elo_diff": team1_pre_elo - team2_pre_elo,
            "team1_elo_win_prob": team1_elo_win_prob,
            "team1_prior_series_count": team1_prior_series_count,
            "team2_prior_series_count": team2_prior_series_count,
            "series_count_diff": team1_prior_series_count - team2_prior_series_count,
            "team1_prior_series_win_rate": team1_prior_series_win_rate,
            "team2_prior_series_win_rate": team2_prior_series_win_rate,
            "prior_series_win_rate_diff": team1_prior_series_win_rate - team2_prior_series_win_rate,
            "team1_recent5_series_count": team1_recent5_series_count,
            "team2_recent5_series_count": team2_recent5_series_count,
            "team1_recent5_series_win_rate": team1_recent5_series_win_rate,
            "team2_recent5_series_win_rate": team2_recent5_series_win_rate,
            "recent5_series_win_rate_diff": team1_recent5_series_win_rate - team2_recent5_series_win_rate,
            "team1_prior_game_count": team1_prior_game_count,
            "team2_prior_game_count": team2_prior_game_count,
            "prior_game_count_diff": team1_prior_game_count - team2_prior_game_count,
            "team1_prior_game_win_rate": team1_prior_game_win_rate,
            "team2_prior_game_win_rate": team2_prior_game_win_rate,
            "prior_game_win_rate_diff": team1_prior_game_win_rate - team2_prior_game_win_rate,
            "team1_prior_avg_games_played_per_series": _safe_average(
                team1_state["games_played_total"], team1_prior_series_count
            ),
            "team2_prior_avg_games_played_per_series": _safe_average(
                team2_state["games_played_total"], team2_prior_series_count
            ),
            "prior_avg_games_played_per_series_diff": _safe_average(
                team1_state["games_played_total"], team1_prior_series_count
            )
            - _safe_average(team2_state["games_played_total"], team2_prior_series_count),
            "team1_prior_avg_games_won_per_series": _safe_average(
                team1_state["games_won_total"], team1_prior_series_count
            ),
            "team2_prior_avg_games_won_per_series": _safe_average(
                team2_state["games_won_total"], team2_prior_series_count
            ),
            "prior_avg_games_won_per_series_diff": _safe_average(
                team1_state["games_won_total"], team1_prior_series_count
            )
            - _safe_average(team2_state["games_won_total"], team2_prior_series_count),
            "team1_prior_avg_game_length_seconds": _safe_average(
                team1_state["game_length_seconds_total"], team1_prior_game_count
            ),
            "team2_prior_avg_game_length_seconds": _safe_average(
                team2_state["game_length_seconds_total"], team2_prior_game_count
            ),
            "prior_avg_game_length_seconds_diff": _safe_average(
                team1_state["game_length_seconds_total"], team1_prior_game_count
            )
            - _safe_average(team2_state["game_length_seconds_total"], team2_prior_game_count),
            "team1_series_streak": int(team1_state["series_streak"]),
            "team2_series_streak": int(team2_state["series_streak"]),
            "series_streak_diff": int(team1_state["series_streak"]) - int(team2_state["series_streak"]),
            "team1_days_since_last_series": _days_since(team1_last_core, event_time),
            "team2_days_since_last_series": _days_since(team2_last_core, event_time),
            "days_since_last_series_diff": _days_since(team1_last_core, event_time)
            - _days_since(team2_last_core, event_time),
            "team1_patch_prior_series_count": team1_patch_prior_series_count,
            "team2_patch_prior_series_count": team2_patch_prior_series_count,
            "patch_prior_series_count_diff": team1_patch_prior_series_count - team2_patch_prior_series_count,
            "team1_patch_prior_win_rate": team1_patch_prior_win_rate,
            "team2_patch_prior_win_rate": team2_patch_prior_win_rate,
            "patch_prior_win_rate_diff": team1_patch_prior_win_rate - team2_patch_prior_win_rate,
            "team1_split_prior_series_count": team1_split_prior_series_count,
            "team2_split_prior_series_count": team2_split_prior_series_count,
            "split_prior_series_count_diff": team1_split_prior_series_count - team2_split_prior_series_count,
            "team1_split_prior_win_rate": team1_split_prior_win_rate,
            "team2_split_prior_win_rate": team2_split_prior_win_rate,
            "split_prior_win_rate_diff": team1_split_prior_win_rate - team2_split_prior_win_rate,
            "h2h_prior_series_count": h2h_prior_series_count,
            "h2h_team1_series_wins": h2h_team1_series_wins,
            "h2h_team2_series_wins": h2h_team2_series_wins,
            "h2h_team1_series_win_rate": _neutral_rate(h2h_team1_series_wins, h2h_prior_series_count),
            "h2h_prior_game_count": h2h_prior_game_count,
            "h2h_team1_game_wins": h2h_team1_game_wins,
            "h2h_team2_game_wins": h2h_team2_game_wins,
            "h2h_team1_game_win_rate": _neutral_rate(h2h_team1_game_wins, h2h_prior_game_count),
            "team1_prev_series_roster_overlap": team1_prev_series_roster_overlap,
            "team2_prev_series_roster_overlap": team2_prev_series_roster_overlap,
            "prev_series_roster_overlap_diff": team1_prev_series_roster_overlap - team2_prev_series_roster_overlap,
            "team1_recent3_avg_roster_overlap": team1_recent3_avg_roster_overlap,
            "team2_recent3_avg_roster_overlap": team2_recent3_avg_roster_overlap,
            "recent3_avg_roster_overlap_diff": team1_recent3_avg_roster_overlap - team2_recent3_avg_roster_overlap,
            "team1_roster_avg_player_prior_series_count": team1_roster_avg_player_prior_series_count,
            "team2_roster_avg_player_prior_series_count": team2_roster_avg_player_prior_series_count,
            "roster_avg_player_prior_series_count_diff": (
                team1_roster_avg_player_prior_series_count - team2_roster_avg_player_prior_series_count
            ),
            "team1_roster_avg_player_prior_series_win_rate": team1_roster_avg_player_prior_series_win_rate,
            "team2_roster_avg_player_prior_series_win_rate": team2_roster_avg_player_prior_series_win_rate,
            "roster_avg_player_prior_series_win_rate_diff": (
                team1_roster_avg_player_prior_series_win_rate
                - team2_roster_avg_player_prior_series_win_rate
            ),
            "team1_roster_new_player_count": team1_roster_new_player_count,
            "team2_roster_new_player_count": team2_roster_new_player_count,
            "roster_new_player_count_diff": team1_roster_new_player_count - team2_roster_new_player_count,
            "team1_recent10_unique_champions": team1_draft_recent10["unique_champions"],
            "team2_recent10_unique_champions": team2_draft_recent10["unique_champions"],
            "recent10_unique_champions_diff": (
                team1_draft_recent10["unique_champions"] - team2_draft_recent10["unique_champions"]
            ),
            "team1_recent25_unique_champions": team1_draft_recent25["unique_champions"],
            "team2_recent25_unique_champions": team2_draft_recent25["unique_champions"],
            "recent25_unique_champions_diff": (
                team1_draft_recent25["unique_champions"] - team2_draft_recent25["unique_champions"]
            ),
            "team1_recent25_top5_champion_share": team1_draft_recent25["top5_share"],
            "team2_recent25_top5_champion_share": team2_draft_recent25["top5_share"],
            "recent25_top5_champion_share_diff": (
                team1_draft_recent25["top5_share"] - team2_draft_recent25["top5_share"]
            ),
            "team1_recent10_first_pick_rate": team1_draft_recent10["first_pick_rate"],
            "team2_recent10_first_pick_rate": team2_draft_recent10["first_pick_rate"],
            "recent10_first_pick_rate_diff": (
                team1_draft_recent10["first_pick_rate"] - team2_draft_recent10["first_pick_rate"]
            ),
        }

        metadata = {
            "team1_core_series_count": team1_prior_series_count,
            "team2_core_series_count": team2_prior_series_count,
            "team1_latest_core_series_at": team1_last_core,
            "team2_latest_core_series_at": team2_last_core,
            "team1_latest_fact_series_at": team1_last_fact,
            "team2_latest_fact_series_at": team2_last_fact,
        }
        return feature_row, warnings, metadata

    def score_match(
        self,
        team1: str,
        team2: str,
        match_time: str,
        league_code: str,
        split_name: str,
        patch_version: str,
        best_of: int = 5,
        playoffs: bool = False,
        team1_odds: float | None = None,
        team2_odds: float | None = None,
    ) -> FairOddsQuote:
        """Return model fair odds for a future matchup."""
        event_time = pd.Timestamp(match_time)
        if event_time.tzinfo is None:
            event_time = event_time.tz_localize("UTC")
        else:
            event_time = event_time.tz_convert("UTC")

        team1_key, team1_name = self.resolve_team(team1)
        team2_key, team2_name = self.resolve_team(team2)

        feature_row, warnings, metadata = self._build_feature_row(
            event_time=event_time,
            team1_key=team1_key,
            team1_name=team1_name,
            team2_key=team2_key,
            team2_name=team2_name,
            league_code=league_code,
            split_name=split_name,
            patch_version=patch_version,
            best_of=best_of,
            playoffs=playoffs,
        )

        scoring_frame = pd.DataFrame([{column: feature_row.get(column) for column in self.feature_columns}])
        team1_win_prob = float(self.model.predict_proba(scoring_frame)[0, 1])
        team2_win_prob = float(1.0 - team1_win_prob)

        market_comparison = None
        if team1_odds is not None or team2_odds is not None:
            if team1_odds is None or team2_odds is None:
                raise ValueError("team1_odds and team2_odds must be provided together")
            market_comparison = compare_two_way_market(
                team1_model_prob=team1_win_prob,
                team2_model_prob=team2_win_prob,
                team1_odds=float(team1_odds),
                team2_odds=float(team2_odds),
            )

        return FairOddsQuote(
            run_id=self.run_id,
            snapshot_id=self.snapshot_id,
            model_name=self.model_name,
            model_path=str(self.model_path),
            feature_version=self.feature_version,
            history_source="model_core_global + fact_series_local_roster_draft",
            series_key=str(feature_row["series_key"]),
            match_time=event_time.isoformat(),
            league_code=league_code,
            split_name=split_name,
            patch_version=patch_version,
            best_of=int(best_of),
            playoffs=bool(playoffs),
            season_year=int(event_time.year),
            team1_key=team1_key,
            team1_name=team1_name,
            team2_key=team2_key,
            team2_name=team2_name,
            team1_win_prob=team1_win_prob,
            team2_win_prob=team2_win_prob,
            team1_fair_odds=_probability_to_fair_odds(team1_win_prob),
            team2_fair_odds=_probability_to_fair_odds(team2_win_prob),
            team1_core_series_count=int(metadata["team1_core_series_count"]),
            team2_core_series_count=int(metadata["team2_core_series_count"]),
            team1_latest_core_series_at=_serialize_timestamp(metadata["team1_latest_core_series_at"]),
            team2_latest_core_series_at=_serialize_timestamp(metadata["team2_latest_core_series_at"]),
            team1_latest_fact_series_at=_serialize_timestamp(metadata["team1_latest_fact_series_at"]),
            team2_latest_fact_series_at=_serialize_timestamp(metadata["team2_latest_fact_series_at"]),
            warnings=warnings,
            market_comparison=market_comparison,
        )
