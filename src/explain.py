"""Phase 9 — per-prediction explainability (reason codes).

Why this exists: the MODEL_CARD flags that adverse-action notices with reason
codes are required by US lending law (Reg B) and were missing. This module
produces them.

How it works: XGBoost ships exact TreeSHAP via
`booster.predict(dmatrix, pred_contribs=True)` — so we get true SHAP feature
attributions without adding the (heavy) `shap` dependency, and with native
support for the model's `enable_categorical=True` columns.

We explain the **base** XGBoost booster of the calibrated model. Isotonic
calibration is a monotone transform of the score, so it changes the *level*
of the probability but not the *direction* or *ranking* of feature
contributions — the attribution that explains "why this applicant scores
riskier than average" is unchanged.

Contributions are in **margin (log-odds) units**: they sum to the raw model
margin plus a bias term. A positive contribution pushes toward default
(higher risk); negative pushes toward repayment. For adverse-action-style
reason codes we surface the top positive contributors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Human-readable labels for the model's 32 features (25 base + 7 interactions).
# Falls back to the raw feature name if a label is missing.
FEATURE_LABELS: dict[str, str] = {
    'loan_amnt': 'Loan amount',
    'term': 'Loan term (months)',
    'int_rate': 'Interest rate',
    'installment': 'Monthly installment',
    'grade': 'LendingClub grade',
    'sub_grade': 'LendingClub sub-grade',
    'emp_length': 'Employment length',
    'emp_length_missing': 'Employment length missing',
    'home_ownership': 'Home ownership',
    'annual_inc': 'Annual income',
    'verification_status': 'Income verification status',
    'purpose': 'Loan purpose',
    'addr_state': 'State of residence',
    'application_type': 'Application type',
    'dti': 'Debt-to-income ratio',
    'revol_util': 'Revolving utilization',
    'revol_bal': 'Revolving balance',
    'fico_mean': 'FICO score',
    'delinq_2yrs': 'Delinquencies (2yr)',
    'pub_rec': 'Public records',
    'pub_rec_bankruptcies': 'Bankruptcies',
    'mort_acc': 'Mortgage accounts',
    'open_acc': 'Open credit lines',
    'total_acc': 'Total credit lines',
    'credit_history_years': 'Length of credit history',
    # Interaction features (features.py)
    'loan_to_income': 'Loan-to-income ratio',
    'installment_to_income': 'Installment-to-income ratio',
    'int_rate_x_term': 'Total interest exposure (rate × term)',
    'fico_dti_risk': 'FICO × DTI risk product',
    'revol_util_x_logbal': 'Utilization × log(balance)',
    'accounts_per_year': 'Account-opening velocity',
    'delinq_per_year': 'Delinquency rate (per year of history)',
}


@dataclass
class Contribution:
    """One feature's SHAP attribution for a single prediction."""
    feature: str
    label: str
    value: float | str | None   # the applicant's value for this feature
    contribution: float         # SHAP value in log-odds; + = toward default


def _booster(model):
    """Return the XGBoost Booster from a CalibratedXGB wrapper or a bare model."""
    base = getattr(model, 'base', model)  # CalibratedXGB.base, else the model itself
    return base.get_booster()


def _scalar(v):
    """Make a feature value JSON-safe: NaN -> None, numeric -> float, else str."""
    if pd.isna(v):
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    return str(v)


def shap_contributions(model, X: pd.DataFrame) -> np.ndarray:
    """Exact TreeSHAP contributions for every row in `X`.

    Returns an array of shape (n_rows, n_features + 1); the final column is the
    bias (expected value) term. Column order matches the booster's features.
    """
    from xgboost import DMatrix  # local import — only pay xgboost cost when used

    booster = _booster(model)
    dmat = DMatrix(X, enable_categorical=True)
    return booster.predict(dmat, pred_contribs=True)


def _row_contributions(contribs_row: np.ndarray, names: list[str],
                       x_row: pd.Series) -> list[Contribution]:
    """Build sorted Contribution objects for one row (most risk-increasing first)."""
    items = [
        Contribution(
            feature=name,
            label=FEATURE_LABELS.get(name, name),
            value=_scalar(x_row[name]),
            contribution=float(s),
        )
        # contribs_row has a trailing bias column we drop with zip's shorter arg.
        for name, s in zip(names, contribs_row, strict=False)
    ]
    items.sort(key=lambda c: c.contribution, reverse=True)
    return items


def _select(items: list[Contribution], top_n: int, positive_only: bool) -> list[Contribution]:
    if positive_only:
        items = [c for c in items if c.contribution > 0]
    return items[:top_n]


def reason_codes(model, X: pd.DataFrame, row: int = 0, top_n: int = 5,
                 positive_only: bool = True) -> list[Contribution]:
    """Top contributors driving one prediction's risk.

    `positive_only=True` (the default) returns only features that push toward
    default — the adverse-action use case ("why was this flagged risky?").
    Set it False to get the top contributors regardless of sign.
    """
    contribs = shap_contributions(model, X)
    names = list(_booster(model).feature_names)
    items = _row_contributions(contribs[row], names, X.iloc[row])
    return _select(items, top_n, positive_only)


def reason_codes_batch(model, X: pd.DataFrame, top_n: int = 5,
                       positive_only: bool = True) -> list[list[Contribution]]:
    """Reason codes for every row, computing SHAP once for the whole frame.

    Use this instead of calling `reason_codes` in a loop — the TreeSHAP pass
    over all rows is a single booster call, so per-row looping would be O(n²).
    """
    contribs = shap_contributions(model, X)
    names = list(_booster(model).feature_names)
    return [
        _select(_row_contributions(contribs[i], names, X.iloc[i]), top_n, positive_only)
        for i in range(len(X))
    ]
