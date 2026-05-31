"""Unit tests for features.py — the Phase 8 interaction features.

Covers _safe_div (the zero-division guard) and add_interactions (the seven
underwriting ratios). The arithmetic and NaN-propagation behavior here is
relied on identically by training, serving, and batch scoring, so it must be
pinned down.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import features


def _base_frame() -> pd.DataFrame:
    """Minimal frame with every column add_interactions() reads."""
    return pd.DataFrame({
        'loan_amnt': [10000.0, 20000.0],
        'annual_inc': [50000.0, 0.0],          # row 1 has zero income -> NaN ratios
        'installment': [300.0, 600.0],
        'int_rate': [10.0, 20.0],
        'term': [36, 60],
        'fico_mean': [750.0, 650.0],
        'dti': [20.0, 40.0],
        'revol_util': [50.0, 80.0],
        'revol_bal': [5000.0, 10000.0],
        'credit_history_years': [10, 0],       # row 1 has 0 years -> clip to 1
        'total_acc': [20.0, 5.0],
        'delinq_2yrs': [1.0, 2.0],
    })


def test_safe_div_normal():
    out = features._safe_div(pd.Series([10.0, 9.0]), pd.Series([2.0, 3.0]))
    assert list(out) == [5.0, 3.0]
    assert str(out.dtype) == 'float32'


def test_safe_div_zero_and_negative_denominator_is_nan():
    out = features._safe_div(pd.Series([10.0, 10.0, 10.0]),
                             pd.Series([2.0, 0.0, -5.0]))
    assert out.iloc[0] == 5.0
    assert pd.isna(out.iloc[1])   # divide by zero -> NaN, not inf
    assert pd.isna(out.iloc[2])   # negative denominator -> NaN


def test_add_interactions_adds_all_columns():
    out = features.add_interactions(_base_frame())
    for col in features.INTERACTION_COLS:
        assert col in out.columns


def test_add_interactions_does_not_mutate_input():
    df = _base_frame()
    before = df.copy()
    features.add_interactions(df)
    pd.testing.assert_frame_equal(df, before)


def test_loan_to_income_value_and_zero_income_nan():
    out = features.add_interactions(_base_frame())
    # row 0: 10000 / 50000 = 0.2
    assert out['loan_to_income'].iloc[0] == 0.2
    # row 1: annual_inc == 0 -> NaN
    assert pd.isna(out['loan_to_income'].iloc[1])


def test_int_rate_x_term_is_simple_product():
    out = features.add_interactions(_base_frame())
    assert out['int_rate_x_term'].iloc[0] == 10.0 * 36
    assert out['int_rate_x_term'].iloc[1] == 20.0 * 60


def test_fico_dti_risk_formula():
    out = features.add_interactions(_base_frame())
    # (850 - fico_mean) * (dti / 100)
    expected0 = (850 - 750.0) * (20.0 / 100)
    np.testing.assert_allclose(out['fico_dti_risk'].iloc[0], expected0, rtol=1e-5)


def test_credit_history_clip_prevents_explosion():
    out = features.add_interactions(_base_frame())
    # row 1: credit_history_years = 0 -> clipped to 1, so accounts_per_year = total_acc / 1
    assert out['accounts_per_year'].iloc[1] == 5.0
    assert out['delinq_per_year'].iloc[1] == 2.0
    # row 0: 20 accounts / 10 years = 2.0
    assert out['accounts_per_year'].iloc[0] == 2.0
