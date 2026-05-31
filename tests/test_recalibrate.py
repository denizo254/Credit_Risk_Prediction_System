"""Tests for recalibrate.py — rolling isotonic recalibration.

Window selection and calibration math are pure; the recalibrate() path is
exercised with a tiny in-test XGBoost model (no dataset, no artifact).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

import recalibrate
from evaluate import CalibratedXGB


@pytest.fixture(scope='module')
def base_and_frame():
    """A fitted bare XGB plus a featv2-shaped labeled frame for it."""
    rng = np.random.RandomState(0)
    n = 400
    X = pd.DataFrame({'x0': rng.randn(n), 'x1': rng.randn(n)})
    y = (X['x0'] > 0).astype(int)
    clf = XGBClassifier(n_estimators=25, max_depth=3, enable_categorical=True,
                        eval_metric='logloss', random_state=0).fit(X, y)
    frame = X.assign(default=y, issue_year=2017)
    return clf, frame


# ---------- recent_years ----------

def test_recent_years_default_is_latest_year():
    df = pd.DataFrame({'issue_year': [2014, 2015, 2016, 2017, 2018], 'v': range(5)})
    out = recalibrate.recent_years(df, years=1)
    assert set(out['issue_year']) == {2018}


def test_recent_years_window_and_max_year():
    df = pd.DataFrame({'issue_year': [2014, 2015, 2016, 2017, 2018], 'v': range(5)})
    out = recalibrate.recent_years(df, years=2, max_year=2017)
    assert set(out['issue_year']) == {2016, 2017}


def test_recent_years_requires_issue_year():
    with pytest.raises(ValueError, match='issue_year'):
        recalibrate.recent_years(pd.DataFrame({'a': [1]}), years=1)


# ---------- base_of ----------

def test_base_of_unwraps_calibrated_and_passes_bare(base_and_frame):
    clf, _ = base_and_frame
    cal = CalibratedXGB(base=clf, iso=None)
    assert recalibrate.base_of(cal) is clf
    assert recalibrate.base_of(clf) is clf


# ---------- recalibrate ----------

def test_recalibrate_reuses_base_and_returns_calibrated(base_and_frame):
    clf, frame = base_and_frame
    new = recalibrate.recalibrate(clf, frame)
    assert isinstance(new, CalibratedXGB)
    assert new.base is clf                    # base is frozen / reused, not retrained
    X = frame.drop(columns=['default', 'issue_year'])
    p = new.predict_proba(X)[:, 1]
    assert ((0.0 <= p) & (p <= 1.0)).all()


def test_recalibrate_accepts_calibrated_input(base_and_frame):
    clf, frame = base_and_frame
    wrapped = CalibratedXGB(base=clf, iso=None)
    new = recalibrate.recalibrate(wrapped, frame)
    assert new.base is clf


# ---------- calibration_summary ----------

def test_calibration_summary_gap_is_meanpred_minus_observed(base_and_frame):
    clf, frame = base_and_frame
    new = recalibrate.recalibrate(clf, frame)
    s = recalibrate.calibration_summary(new, frame)
    assert s['n'] == len(frame)
    assert s['gap'] == round(s['mean_pred'] - s['observed'], 4)
    assert 0.0 <= s['brier'] <= 1.0
