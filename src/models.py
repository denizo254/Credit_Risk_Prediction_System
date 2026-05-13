"""Phase 4 modeling — model builders + metric helpers.

Two models, one shared evaluation surface:

  - `build_lr()`   : ColumnTransformer (median-impute + scale numerics, OHE
                     categoricals) -> LogisticRegression(class_weight='balanced').
                     Interpretable baseline.
  - `build_xgb()`  : XGBClassifier with enable_categorical=True so pandas
                     `category` dtype columns are consumed natively (no OHE
                     blow-up on addr_state's 50 levels).

  - `evaluate()`   : returns ROC-AUC, PR-AUC, Brier, log-loss, and KS.
                     KS (Kolmogorov-Smirnov) is the standard credit-risk
                     discrimination metric — max separation between the score
                     CDFs of defaults vs non-defaults.

Neither builder fits any data. Caller controls `fit()` so the same builders
work in CV, in the notebook, and in the smoke test.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prepare import CATEGORICAL_COLS

PROJECT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT / 'outputs' / 'models'

TARGET = 'default'
# `issue_year` is metadata (used for the time split) — not a predictive feature.
NON_FEATURE_COLS = (TARGET, 'issue_year')


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop target + metadata, return (X, y)."""
    X = df.drop(columns=list(NON_FEATURE_COLS))
    y = df[TARGET].astype('int8')
    return X, y


def numeric_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in CATEGORICAL_COLS]


def build_lr(X_sample: pd.DataFrame) -> Pipeline:
    """Logistic regression baseline. `X_sample` is used only to pick column lists."""
    num = numeric_cols(X_sample)
    pre = ColumnTransformer([
        ('num', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
        ]), num),
        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=True)),
        ]), CATEGORICAL_COLS),
    ])
    clf = LogisticRegression(
        max_iter=200,
        class_weight='balanced',
        solver='lbfgs',
    )
    return Pipeline([('pre', pre), ('clf', clf)])


def build_xgb(y_train: pd.Series, **overrides):
    """XGBoost with `enable_categorical=True` so pandas categoricals pass through.

    `scale_pos_weight` is set from the training class balance — XGB's
    documented way of handling imbalance without losing the natural prior.
    """
    # Local import: xgboost adds ~2s of import cost; only pay it if used.
    from xgboost import XGBClassifier

    n_neg, n_pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    params = dict(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        scale_pos_weight=n_neg / max(n_pos, 1),
        tree_method='hist',
        enable_categorical=True,
        eval_metric='logloss',
        n_jobs=-1,
        random_state=42,
    )
    params.update(overrides)
    return XGBClassifier(**params)


@dataclass
class Metrics:
    roc_auc: float
    pr_auc: float
    brier: float
    log_loss: float
    ks: float

    def as_row(self) -> dict[str, float]:
        return {
            'roc_auc': self.roc_auc, 'pr_auc': self.pr_auc,
            'brier': self.brier, 'log_loss': self.log_loss, 'ks': self.ks,
        }


def ks_statistic(y_true: Iterable[int], y_score: Iterable[float]) -> float:
    """KS = max |CDF_default(score) - CDF_repay(score)|. Higher = better separation.

    Industry rule of thumb: KS < 0.20 is weak, 0.30–0.50 is operationally useful,
    >0.50 is excellent (and worth re-checking for leakage).
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(y_score)
    y_sorted = y_true[order]
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    cum_pos = np.cumsum(y_sorted) / n_pos
    cum_neg = np.cumsum(1 - y_sorted) / n_neg
    return float(np.max(np.abs(cum_pos - cum_neg)))


def evaluate(y_true: pd.Series, y_score: np.ndarray) -> Metrics:
    """All five metrics for a binary classifier's probability output."""
    return Metrics(
        roc_auc=float(roc_auc_score(y_true, y_score)),
        pr_auc=float(average_precision_score(y_true, y_score)),
        brier=float(brier_score_loss(y_true, y_score)),
        log_loss=float(log_loss(y_true, y_score)),
        ks=ks_statistic(y_true, y_score),
    )


def model_path(name: str) -> Path:
    """`outputs/models/<name>.joblib`."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR / f'{name}.joblib'
