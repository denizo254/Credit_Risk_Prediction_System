"""Tests for explain.py — TreeSHAP reason codes.

Trains a tiny synthetic XGBoost model in-test (no dataset, no artifact) so it
runs in CI. Verifies the SHAP contributions are correct (they sum to the model
margin) and that reason codes surface the true risk driver.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from xgboost import DMatrix, XGBClassifier

import explain


@pytest.fixture(scope='module')
def fitted():
    """A small model whose risk is driven almost entirely by feature x0."""
    rng = np.random.RandomState(0)
    n = 400
    X = pd.DataFrame({
        'x0': rng.randn(n),
        'x1': rng.randn(n),
        'x2': rng.randn(n),
        'x3': rng.randn(n),
    })
    y = (X['x0'] > 0).astype(int)  # x0 is the sole signal
    clf = XGBClassifier(n_estimators=30, max_depth=3, learning_rate=0.3,
                        enable_categorical=True, eval_metric='logloss',
                        random_state=0)
    clf.fit(X, y)
    return clf, X


def test_contributions_shape(fitted):
    clf, X = fitted
    contribs = explain.shap_contributions(clf, X)
    # n_features + 1 (bias) columns.
    assert contribs.shape == (len(X), X.shape[1] + 1)


def test_contributions_sum_to_margin(fitted):
    # The defining property of (Tree)SHAP: per-row contributions + bias == the
    # raw model margin. This is the correctness check.
    clf, X = fitted
    contribs = explain.shap_contributions(clf, X)
    margin = clf.get_booster().predict(DMatrix(X, enable_categorical=True),
                                       output_margin=True)
    np.testing.assert_allclose(contribs.sum(axis=1), margin, rtol=1e-4, atol=1e-4)


def test_reason_codes_identify_driver(fitted):
    clf, X = fitted
    # A clearly high-risk row (large positive x0).
    high = pd.DataFrame([{'x0': 3.0, 'x1': 0.0, 'x2': 0.0, 'x3': 0.0}])
    reasons = explain.reason_codes(clf, high, row=0, top_n=5)
    assert reasons, 'expected at least one risk-increasing reason'
    assert reasons[0].feature == 'x0'
    assert reasons[0].contribution > 0


def test_positive_only_filters_negatives(fitted):
    clf, X = fitted
    reasons = explain.reason_codes(clf, X, row=0, top_n=10, positive_only=True)
    assert all(c.contribution > 0 for c in reasons)


def test_top_n_is_respected(fitted):
    clf, X = fitted
    reasons = explain.reason_codes(clf, X, row=0, top_n=2, positive_only=False)
    assert len(reasons) == 2


def test_calibrated_wrapper_uses_base_booster(fitted):
    # A CalibratedXGB wraps the booster in `.base`; explanations must match the
    # bare model since isotonic doesn't change feature attributions.
    from sklearn.isotonic import IsotonicRegression

    from evaluate import CalibratedXGB

    clf, X = fitted
    raw = clf.predict_proba(X)[:, 1]
    iso = IsotonicRegression(out_of_bounds='clip').fit(raw, (X['x0'] > 0).astype(int))
    cal = CalibratedXGB(base=clf, iso=iso)

    bare = explain.reason_codes(clf, X, row=5, top_n=4, positive_only=False)
    wrapped = explain.reason_codes(cal, X, row=5, top_n=4, positive_only=False)
    assert [c.feature for c in bare] == [c.feature for c in wrapped]


def test_scalar_handles_types():
    assert explain._scalar(np.nan) is None
    assert explain._scalar(3) == 3.0
    assert isinstance(explain._scalar(np.float32(1.5)), float)
    assert explain._scalar('MORTGAGE') == 'MORTGAGE'


def test_label_falls_back_to_feature_name(fitted):
    clf, X = fitted
    # 'x0' has no entry in FEATURE_LABELS, so label should equal the name.
    reasons = explain.reason_codes(clf, X, row=0, top_n=1, positive_only=False)
    assert reasons[0].label == reasons[0].feature
