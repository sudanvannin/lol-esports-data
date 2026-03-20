"""Tests for totals fair-line helpers."""

from __future__ import annotations

from src.ml.game_totals_fair_lines import estimate_discrete_market


def test_estimate_discrete_market_returns_probabilities_for_poisson_case():
    distribution, over_prob, under_prob = estimate_discrete_market(
        mean_value=1.2,
        variance_value=1.2,
        line=1.5,
    )

    assert distribution == "poisson"
    assert 0.0 < over_prob < 1.0
    assert 0.0 < under_prob < 1.0
    assert round(over_prob + under_prob, 6) == 1.0


def test_estimate_discrete_market_uses_negative_binomial_for_overdispersion():
    distribution, over_prob, under_prob = estimate_discrete_market(
        mean_value=30.0,
        variance_value=45.0,
        line=29.5,
    )

    assert distribution == "negative_binomial"
    assert 0.0 < over_prob < 1.0
    assert 0.0 < under_prob < 1.0
    assert round(over_prob + under_prob, 6) == 1.0
