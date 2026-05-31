"""Unit tests for models.py metric + split helpers.

ks_statistic and evaluate are the project's evaluation surface; split_xy and
numeric_cols decide what the model is allowed to see. None of these fit a
model, so they're fast and pure. (build_lr/build_xgb are exercised by the
phase smoke tests, not here.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import models

# ---------- ks_statistic ----------

def test_ks_perfect_separation_is_one():
    y = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert models.ks_statistic(y, scores) == 1.0


def test_ks_single_class_is_nan():
    # No positives (or no negatives) -> KS undefined.
    assert np.isnan(models.ks_statistic([0, 0, 0], [0.1, 0.2, 0.3]))
    assert np.isnan(models.ks_statistic([1, 1, 1], [0.1, 0.2, 0.3]))


def test_ks_is_in_unit_interval():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, size=500)
    scores = rng.rand(500)
    ks = models.ks_statistic(y, scores)
    assert 0.0 <= ks <= 1.0


def test_ks_invariant_under_monotonic_rescale():
    # KS is rank-based, so multiplying scores by a positive constant can't change it.
    rng = np.random.RandomState(1)
    y = rng.randint(0, 2, size=300)
    scores = rng.rand(300)
    ks_a = models.ks_statistic(y, scores)
    ks_b = models.ks_statistic(y, scores * 7.5)
    np.testing.assert_allclose(ks_a, ks_b, rtol=1e-9)


# ---------- evaluate ----------

def test_evaluate_perfect_classifier():
    y = pd.Series([0, 0, 1, 1])
    scores = np.array([0.05, 0.1, 0.9, 0.95])
    m = models.evaluate(y, scores)
    assert m.roc_auc == 1.0
    assert m.ks == 1.0
    assert m.brier < 0.05          # near-perfect probabilities -> tiny Brier
    assert set(m.as_row()) == {'roc_auc', 'pr_auc', 'brier', 'log_loss', 'ks'}


# ---------- split_xy / numeric_cols ----------

def test_split_xy_drops_target_and_metadata():
    df = pd.DataFrame({
        'loan_amnt': [1000, 2000],
        'issue_year': [2015, 2016],
        'default': [0, 1],
    })
    X, y = models.split_xy(df)
    assert 'default' not in X.columns
    assert 'issue_year' not in X.columns
    assert list(X.columns) == ['loan_amnt']
    assert list(y) == [0, 1]
    assert str(y.dtype) == 'int8'


def test_numeric_cols_excludes_categoricals():
    from prepare import CATEGORICAL_COLS
    X = pd.DataFrame(columns=['loan_amnt', 'dti'] + CATEGORICAL_COLS)
    nums = models.numeric_cols(X)
    assert 'loan_amnt' in nums and 'dti' in nums
    for c in CATEGORICAL_COLS:
        assert c not in nums
