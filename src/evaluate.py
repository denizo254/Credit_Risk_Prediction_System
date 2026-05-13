"""Phase 5 — evaluation utilities.

What lives here:
  - `CalibratedXGB`        : transparent (base, isotonic) wrapper. The base
                             model never sees the calibration slice — that's
                             the whole point of holding 2016 out.
  - `fit_calibrated`       : end-to-end train-then-calibrate on the
                             time-aware base/calib split.
  - `cost_curve`           : expected cost across thresholds for a given
                             cost matrix (FN, FP).
  - `optimal_threshold`    : argmin of `cost_curve`.
  - `gains_table`          : per-decile cumulative captures + lift.
  - `psi`                  : Population Stability Index for score drift.
  - `metrics_by_group`     : ROC-AUC, KS, default rate per subgroup.

Design choices the notebook is going to reference:
  - Isotonic calibration is fit on out-of-sample probabilities (the 2016
    slice the base XGB never saw). `cv='prefit'` would force us into
    sklearn-version-specific API churn — wrapping IsotonicRegression
    directly keeps this stable across sklearn 1.4–1.8+.
  - Cost matrix is per-row by default. Loan-amount weighting is a
    one-line override (`weights=X['loan_amnt']`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from models import build_xgb, ks_statistic, split_xy

BASE_MAX_YEAR = 2015   # base XGB trains on issue_year <= this
CALIB_YEAR = 2016      # isotonic fits on this year (held out from base)


@dataclass
class CalibratedXGB:
    """XGB + isotonic regression on its raw P(default)."""
    base: object
    iso: IsotonicRegression

    def predict_proba(self, X) -> np.ndarray:
        raw = self.base.predict_proba(X)[:, 1]
        cal = self.iso.transform(raw)
        return np.column_stack([1 - cal, cal])

    def predict_proba_pos(self, X) -> np.ndarray:
        return self.predict_proba(X)[:, 1]


def time_calibration_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the training set into a base-training slice and a calibration slice.

    base = issue_year <= BASE_MAX_YEAR; calib = issue_year == CALIB_YEAR.
    The calib slice is the most recent year in train — keeps the calibrator
    aligned with the *near future* it's about to score.
    """
    base = train[train['issue_year'] <= BASE_MAX_YEAR].copy()
    calib = train[train['issue_year'] == CALIB_YEAR].copy()
    return base, calib


def fit_calibrated(train: pd.DataFrame) -> CalibratedXGB:
    """Fit XGB on base, isotonic on calib. Returns a CalibratedXGB."""
    base_df, calib_df = time_calibration_split(train)
    X_base, y_base = split_xy(base_df)
    X_calib, y_calib = split_xy(calib_df)

    base = build_xgb(y_base)
    base.fit(X_base, y_base)

    raw_calib = base.predict_proba(X_calib)[:, 1]
    iso = IsotonicRegression(out_of_bounds='clip').fit(raw_calib, y_calib)
    return CalibratedXGB(base=base, iso=iso)


# -------- Threshold selection ------------------------------------------------

def cost_curve(
    y_true: Iterable[int],
    y_score: Iterable[float],
    c_fn: float = 5.0,
    c_fp: float = 1.0,
    weights: Iterable[float] | None = None,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Expected cost across a grid of thresholds.

    Decision rule: predict default (i.e. reject loan) iff score >= threshold.
    Costs are asymmetric — c_fn >> c_fp in lending. The default 5:1 ratio
    is a reasonable starting point (FN = lose ~50% principal,
    FP = lose ~10% interest).

    Returns columns: threshold, fn, fp, cost, precision, recall.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    w = np.ones_like(y, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    rows = []
    for t in thresholds:
        pred = (s >= t).astype(int)
        fn = w[(pred == 0) & (y == 1)].sum()
        fp = w[(pred == 1) & (y == 0)].sum()
        tp = w[(pred == 1) & (y == 1)].sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / w[y == 1].sum() if w[y == 1].sum() > 0 else 0.0
        rows.append({
            'threshold': t,
            'fn': float(fn),
            'fp': float(fp),
            'cost': float(c_fn * fn + c_fp * fp),
            'precision': float(precision),
            'recall': float(recall),
        })
    return pd.DataFrame(rows)


def optimal_threshold(curve: pd.DataFrame) -> dict:
    """Pick the row of `cost_curve()` with minimum cost."""
    i = curve['cost'].idxmin()
    return curve.loc[i].to_dict()


# -------- Gains / lift / PSI -------------------------------------------------

def gains_table(y_true: Iterable[int], y_score: Iterable[float], n_bins: int = 10) -> pd.DataFrame:
    """Per-decile cumulative captures + lift.

    Industry-standard credit-scoring presentation. Decile 1 = top n% by score
    (i.e. *highest* predicted default risk).
    """
    df = pd.DataFrame({'y': np.asarray(y_true).astype(int),
                       'score': np.asarray(y_score, dtype=float)})
    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    df['decile'] = pd.qcut(df.index, n_bins, labels=range(1, n_bins + 1))

    overall_rate = df['y'].mean()
    g = df.groupby('decile', observed=True).agg(
        n=('y', 'size'),
        n_defaults=('y', 'sum'),
        default_rate=('y', 'mean'),
        score_min=('score', 'min'),
        score_max=('score', 'max'),
    )
    g['lift'] = g['default_rate'] / overall_rate
    g['cum_defaults'] = g['n_defaults'].cumsum()
    g['cum_capture_rate'] = g['cum_defaults'] / df['y'].sum()
    return g.round(4)


def psi(expected: Iterable[float], actual: Iterable[float], n_bins: int = 10) -> float:
    """Population Stability Index between two score distributions.

    <0.10 stable / 0.10-0.25 moderate drift / >0.25 significant drift.
    Quantile bins are derived from `expected` (= train scores) so the
    comparison is apples-to-apples.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.quantile(expected, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e_pct, _ = np.histogram(expected, bins=edges)
    a_pct, _ = np.histogram(actual, bins=edges)
    e_pct = np.maximum(e_pct / e_pct.sum(), 1e-6)
    a_pct = np.maximum(a_pct / a_pct.sum(), 1e-6)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def metrics_by_group(
    df_with_score: pd.DataFrame,
    score_col: str = 'score',
    y_col: str = 'default',
    group_col: str = 'grade',
) -> pd.DataFrame:
    """Per-subgroup discrimination — confirms the model adds signal *within* each grade."""
    rows = []
    for g, sub in df_with_score.groupby(group_col, observed=True):
        if sub[y_col].nunique() < 2 or len(sub) < 100:
            continue
        rows.append({
            group_col: g,
            'n': len(sub),
            'default_rate': float(sub[y_col].mean()),
            'roc_auc': float(roc_auc_score(sub[y_col], sub[score_col])),
            'ks': float(ks_statistic(sub[y_col], sub[score_col])),
        })
    return pd.DataFrame(rows).sort_values(group_col)
