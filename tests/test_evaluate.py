"""Unit tests for evaluate.py decision-support functions.

cost_curve / optimal_threshold drive the production t=0.13 threshold; psi
drives the drift monitor; gains_table is the headline credit-scoring report.
All are pure numeric functions and get pinned here.
"""
from __future__ import annotations

import numpy as np

import evaluate

# ---------- cost_curve / optimal_threshold ----------

def test_cost_formula_is_weighted_sum():
    y = [1, 1, 0, 0]
    scores = [0.9, 0.6, 0.4, 0.1]
    curve = evaluate.cost_curve(y, scores, c_fn=5.0, c_fp=1.0)
    # cost column must equal c_fn*fn + c_fp*fp everywhere.
    np.testing.assert_allclose(
        curve['cost'].to_numpy(),
        5.0 * curve['fn'].to_numpy() + 1.0 * curve['fp'].to_numpy(),
    )


def test_perfect_ranking_reaches_zero_cost():
    # With perfectly separable scores there exists a threshold (0.5) with no
    # FN and no FP -> optimal cost is exactly 0, recall and precision 1.
    y = [1, 1, 0, 0]
    scores = [0.9, 0.6, 0.4, 0.1]
    curve = evaluate.cost_curve(y, scores)
    opt = evaluate.optimal_threshold(curve)
    assert opt['cost'] == 0.0
    assert opt['recall'] == 1.0
    assert opt['precision'] == 1.0


def test_weights_scale_cost_linearly():
    y = [1, 0, 1, 0]
    scores = [0.8, 0.7, 0.2, 0.1]
    base = evaluate.cost_curve(y, scores)
    weighted = evaluate.cost_curve(y, scores, weights=[2.0, 2.0, 2.0, 2.0])
    np.testing.assert_allclose(weighted['cost'].to_numpy(),
                               2.0 * base['cost'].to_numpy())


def test_higher_fn_cost_does_not_raise_optimal_threshold():
    # Asymmetry sanity: making false-negatives more expensive should push the
    # optimal threshold no higher (reject at least as aggressively).
    rng = np.random.RandomState(3)
    y = rng.randint(0, 2, size=400)
    scores = np.clip(0.5 * y + 0.3 * rng.rand(400), 0, 1)
    t_cheap = evaluate.optimal_threshold(evaluate.cost_curve(y, scores, c_fn=2, c_fp=1))['threshold']
    t_pricey = evaluate.optimal_threshold(evaluate.cost_curve(y, scores, c_fn=20, c_fp=1))['threshold']
    assert t_pricey <= t_cheap


# ---------- psi ----------

def test_psi_identical_distributions_is_zero():
    x = np.linspace(0, 1, 1000)
    assert evaluate.psi(x, x) == 0.0


def test_psi_grows_with_shift():
    expected = np.linspace(0, 1, 2000)
    small = np.linspace(0, 1, 2000) + 0.05
    large = np.linspace(0, 1, 2000) + 0.50
    psi_none = evaluate.psi(expected, expected)
    psi_small = evaluate.psi(expected, small)
    psi_large = evaluate.psi(expected, large)
    assert psi_none < psi_small < psi_large


def test_psi_is_nonnegative():
    rng = np.random.RandomState(5)
    a = rng.rand(1000)
    b = rng.rand(1000)
    assert evaluate.psi(a, b) >= 0.0


# ---------- gains_table ----------

def test_gains_table_cumulative_capture_ends_at_one():
    rng = np.random.RandomState(7)
    n = 1000
    scores = rng.rand(n)
    # Make defaults more likely at high scores so deciles are meaningful.
    y = (rng.rand(n) < scores).astype(int)
    g = evaluate.gains_table(y, scores, n_bins=10)
    assert len(g) == 10
    # Last decile's cumulative capture must be 100% of all defaults.
    np.testing.assert_allclose(g['cum_capture_rate'].iloc[-1], 1.0, rtol=1e-9)
    # Lift in the top (highest-risk) decile should exceed 1.0 for a real signal.
    assert g['lift'].iloc[0] > 1.0
