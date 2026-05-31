"""Tests for monitor.py's ground-truth join + realized-performance helpers.

All synthetic frames / temp files — no model, no dataset — so they run in CI.
Covers the column logic, the both-class vs single-class metric handling, the
monthly rollup, the file reader, and the report() integration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import monitor


def _predictions():
    ts = pd.to_datetime(['2026-01-15', '2026-01-20', '2026-02-10', '2026-02-11'], utc=True)
    return pd.DataFrame({
        'ts': ts,
        'application_id': ['a', 'b', 'c', 'd'],
        'p_default': [0.2, 0.8, 0.6, 0.3],
        'decision': ['approve', 'reject', 'reject', 'approve'],
    })


# ---------- join_outcomes ----------

def test_join_matches_on_id():
    preds = _predictions()
    truth = pd.DataFrame({'application_id': ['a', 'b', 'c', 'd'], 'actual': [0, 1, 1, 0]})
    joined = monitor.join_outcomes(preds, truth)
    assert len(joined) == 4
    assert 'actual' in joined.columns


def test_join_drops_unmatched_and_missing_id():
    preds = _predictions()
    preds.loc[3, 'application_id'] = None       # one logged pred without an id
    truth = pd.DataFrame({'application_id': ['a', 'b'], 'actual': [0, 1]})  # only 2 labels
    joined = monitor.join_outcomes(preds, truth)
    assert sorted(joined['application_id']) == ['a', 'b']


def test_join_without_id_column_returns_empty():
    preds = _predictions().drop(columns=['application_id'])
    truth = pd.DataFrame({'application_id': ['a'], 'actual': [1]})
    joined = monitor.join_outcomes(preds, truth)
    assert joined.empty
    assert 'actual' in joined.columns


# ---------- realized_metrics ----------

def test_realized_metrics_both_classes():
    m = monitor.realized_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert m['n'] == 4
    assert m['actual_default_rate'] == 0.5
    assert 0.0 <= m['brier'] <= 1.0
    assert m['roc_auc'] == 1.0      # perfectly separated
    assert m['ks'] == 1.0
    assert m['log_loss'] is not None


def test_realized_metrics_single_class_nulls_ranking():
    # All non-defaults: Brier still defined, but ROC-AUC/KS/log-loss undefined.
    m = monitor.realized_metrics([0, 0, 0], [0.1, 0.2, 0.3])
    assert m['brier'] is not None
    assert m['roc_auc'] is None
    assert m['ks'] is None
    assert m['log_loss'] is None


# ---------- realized_monthly ----------

def test_realized_monthly_groups_by_month():
    preds = _predictions()
    truth = pd.DataFrame({'application_id': ['a', 'b', 'c', 'd'], 'actual': [0, 1, 1, 0]})
    joined = monitor.join_outcomes(preds, truth)
    monthly = monitor.realized_monthly(joined)
    assert list(monthly.index) == ['2026-01', '2026-02']
    assert monthly.loc['2026-01', 'n'] == 2


# ---------- load_ground_truth ----------

def test_load_ground_truth_csv(tmp_path):
    p = tmp_path / 'truth.csv'
    pd.DataFrame({'application_id': [1, 2], 'default': [0, 1]}).to_csv(p, index=False)
    gt = monitor.load_ground_truth(p)
    assert list(gt.columns) == ['application_id', 'actual']
    assert gt['application_id'].tolist() == ['1', '2']   # coerced to str for joining
    assert gt['actual'].tolist() == [0, 1]


def test_load_ground_truth_custom_columns(tmp_path):
    p = tmp_path / 'truth.csv'
    pd.DataFrame({'loan_id': ['x'], 'charged_off': [1]}).to_csv(p, index=False)
    gt = monitor.load_ground_truth(p, id_col='loan_id', label_col='charged_off')
    assert gt['application_id'].tolist() == ['x']
    assert gt['actual'].tolist() == [1]


def test_load_ground_truth_missing_column_raises(tmp_path):
    p = tmp_path / 'truth.csv'
    pd.DataFrame({'application_id': [1]}).to_csv(p, index=False)  # no label column
    with pytest.raises(ValueError, match='missing column'):
        monitor.load_ground_truth(p)


# ---------- report() integration ----------

def test_report_includes_realized_when_truth_given():
    preds = _predictions()
    truth = pd.DataFrame({'application_id': ['a', 'b', 'c', 'd'], 'actual': [0, 1, 1, 0]})
    ref = np.linspace(0, 1, 50)  # small ref -> PSI suppressed, irrelevant here
    r = monitor.report(preds, ref, truth=truth)
    assert 'realized' in r
    assert r['realized']['n'] == 4
    assert r['realized']['n_unlabeled'] == 0
    assert 'realized_monthly' in r


def test_report_realized_note_when_no_match():
    preds = _predictions()
    truth = pd.DataFrame({'application_id': ['zzz'], 'actual': [1]})
    ref = np.linspace(0, 1, 50)
    r = monitor.report(preds, ref, truth=truth)
    assert 'realized' not in r
    assert 'realized_note' in r
