"""Phase 7 — hyperparameter sweep for XGBoost.

Random search with **time-aware** cross-validation. Same constraint as the
rest of the project: never split randomly across time when the production
use is "score next year's loans."

Why random search instead of full grid:
  - Grid size for the space below is 576; with 2 folds that's 1,152 fits.
    Even at 15s/fit that's 4.8 hours. Random search at 15 trials samples
    the space well enough to identify a *direction* of improvement.

Why 25% subsample of base training:
  - Each fit on the full 831K rows takes ~80s. Subsampling preserves the
    feature-to-feature relationships that hyperparameters tune for, while
    cutting per-fit time to ~10-15s. Best params transfer back to full data
    because we're optimizing model *capacity* settings, not data quirks.

Why ROC-AUC instead of PR-AUC inside CV:
  - PR-AUC scales with base rate. Time-series folds have different base
    rates (default rate rises over time), so PR-AUC values aren't directly
    comparable across folds. ROC-AUC is invariant to base rate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit

from models import build_xgb

PROJECT = Path(__file__).resolve().parent.parent
TUNE_RESULTS = PROJECT / 'outputs' / 'tune_results.csv'

# Carefully sized space. Each axis spans values that XGBoost docs flag as
# the typical operating band; together they cover under-/over-fitting and
# regularization vs. signal trade-offs without exploding the grid.
SEARCH_SPACE = {
    'max_depth':        [4, 6, 8],
    'learning_rate':    [0.05, 0.10],
    'n_estimators':     [200, 400],
    'min_child_weight': [1, 5, 20],
    'subsample':        [0.7, 0.9],
    'colsample_bytree': [0.6, 0.9],
    'reg_lambda':       [1.0, 5.0],
    'gamma':            [0.0, 0.5],
}

# Phase 4's hand-picked defaults — included as trial 0 so we know the
# baseline AUC under the same CV protocol.
DEFAULT_PARAMS = {
    'max_depth':        6,
    'learning_rate':    0.10,
    'n_estimators':     400,
    'min_child_weight': 5,
    'subsample':        0.9,
    'colsample_bytree': 0.8,
    'reg_lambda':       1.0,
    'gamma':            0.0,
}

N_TRIALS = 15
CV_FOLDS = 2
SUBSAMPLE_FRAC = 0.25
RANDOM_STATE = 42


def random_search(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = N_TRIALS,
    cv_folds: int = CV_FOLDS,
    subsample_frac: float = SUBSAMPLE_FRAC,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return a DataFrame of trial params + per-fold + mean ROC-AUC,
    sorted descending by mean_auc.

    Subsample is time-ordered: we pick `subsample_frac` of the rows then
    sort by `issue_year` so TimeSeriesSplit sees an honest chronology.
    """
    rng = np.random.RandomState(random_state)
    n = len(X)
    n_sub = int(n * subsample_frac)
    sub_idx = rng.choice(n, n_sub, replace=False)
    sub_idx = np.sort(sub_idx)

    # Sort the subsample by issue_year (if present) so TimeSeriesSplit
    # sees real chronology, then drop issue_year — it's metadata, not a
    # feature, and the production model never sees it.
    X_sub = X.iloc[sub_idx].reset_index(drop=True)
    y_sub = y.iloc[sub_idx].reset_index(drop=True)
    if 'issue_year' in X_sub.columns:
        order = X_sub['issue_year'].argsort(kind='stable')
        X_sub = X_sub.iloc[order].reset_index(drop=True)
        y_sub = y_sub.iloc[order].reset_index(drop=True)
        X_sub = X_sub.drop(columns=['issue_year'])

    # Trial 0 = the Phase-4 defaults, then n_trials-1 random samples.
    trials: list[dict] = [DEFAULT_PARAMS.copy()]
    trials += list(ParameterSampler(SEARCH_SPACE, n_iter=n_trials - 1,
                                    random_state=random_state))

    cv = TimeSeriesSplit(n_splits=cv_folds)
    results = []
    for trial_idx, params in enumerate(trials):
        fold_scores = []
        for fold_idx, (tr, te) in enumerate(cv.split(X_sub)):
            X_tr, y_tr = X_sub.iloc[tr], y_sub.iloc[tr]
            X_te, y_te = X_sub.iloc[te], y_sub.iloc[te]
            model = build_xgb(y_tr, **params)
            model.fit(X_tr, y_tr)
            p = model.predict_proba(X_te)[:, 1]
            fold_scores.append(float(roc_auc_score(y_te, p)))

        mean = float(np.mean(fold_scores))
        std = float(np.std(fold_scores))
        row = {'trial': trial_idx, **params,
               'fold_aucs': fold_scores, 'mean_auc': mean, 'std_auc': std,
               'is_default': trial_idx == 0}
        results.append(row)

        if verbose:
            tag = '[default]' if trial_idx == 0 else ''
            short = f'd={params["max_depth"]} lr={params["learning_rate"]} ne={params["n_estimators"]} '
            short += f'mcw={params["min_child_weight"]} ss={params["subsample"]} '
            short += f'cs={params["colsample_bytree"]} l2={params["reg_lambda"]} g={params["gamma"]}'
            print(f'  trial {trial_idx:>2d}: AUC={mean:.4f} +/- {std:.4f}  {tag}  {short}')

    df = pd.DataFrame(results).sort_values('mean_auc', ascending=False).reset_index(drop=True)
    TUNE_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=['fold_aucs']).to_csv(TUNE_RESULTS, index=False)
    return df


def best_params_from(df: pd.DataFrame) -> dict:
    """Pull the param dict out of the top row, stripping bookkeeping columns."""
    drop = {'trial', 'fold_aucs', 'mean_auc', 'std_auc', 'is_default'}
    return {k: v for k, v in df.iloc[0].to_dict().items() if k not in drop}
