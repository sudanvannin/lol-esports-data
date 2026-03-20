"""Tests for fair-odds helpers."""

from __future__ import annotations

import math

from src.ml.fair_odds import _probability_to_fair_odds, compare_two_way_market


def test_probability_to_fair_odds_converts_probability():
    assert math.isclose(_probability_to_fair_odds(0.5), 2.0)
    assert math.isclose(_probability_to_fair_odds(0.25), 4.0)
    assert math.isinf(_probability_to_fair_odds(0.0))


def test_compare_two_way_market_removes_vig_and_computes_edge():
    comparison = compare_two_way_market(
        team1_model_prob=0.60,
        team2_model_prob=0.40,
        team1_odds=1.80,
        team2_odds=2.20,
    )

    assert math.isclose(comparison.team1_implied_prob_raw, 1.0 / 1.80)
    assert math.isclose(comparison.team2_implied_prob_raw, 1.0 / 2.20)
    assert math.isclose(
        comparison.team1_implied_prob_devig + comparison.team2_implied_prob_devig,
        1.0,
    )
    assert comparison.team1_edge_vs_devig > 0.0
    assert comparison.team2_edge_vs_devig < 0.0
