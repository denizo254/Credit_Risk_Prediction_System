"""Phase 8 — interaction features.

These are the underwriting-style combinations Phase 7 flagged as the most
likely next gain. A depth-4 tree can split on any single feature but only
catches one interaction per branch — handing the model pre-computed
interactions saves it depth.

Each feature is a *known* credit-risk signal, not a random feature-cross:

  loan_to_income            loan_amnt / annual_inc
                            Standard underwriting ratio. Higher = thinner cushion.
  installment_to_income     (installment * 12) / annual_inc
                            Approximates Debt-Service Coverage Ratio.
  int_rate_x_term           int_rate * term
                            Total interest exposure over loan life
                            (better captures "expensive long loan" than either alone).
  fico_dti_risk             (850 - fico_mean) * (dti / 100)
                            Monotone-risky in both args. Subtraction so the
                            product peaks at low FICO + high DTI.
  revol_util_x_logbal       (revol_util / 100) * log1p(revol_bal)
                            "Maxed out AND a lot of debt" — bigger signal than either alone.
  accounts_per_year         total_acc / max(credit_history_years, 1)
                            Account-opening velocity. New-credit-seeking behavior.
  delinq_per_year           delinq_2yrs / max(credit_history_years, 1)
                            Delinquency rate normalized by history length.

NaN handling: if any operand is NaN, the interaction is NaN. XGBoost handles
that natively; the Phase 4 LR pipeline imputes via median (so the pipeline
needs to see these in numeric_cols, which it already does — `features.py`'s
output columns are auto-discovered as numerics in models.numeric_cols).

Zero-division: annual_inc <= 0 happens in ~0.001% of rows. We replace inf
with NaN so XGBoost can decide rather than fitting on an extreme value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

INTERACTION_COLS = [
    'loan_to_income',
    'installment_to_income',
    'int_rate_x_term',
    'fico_dti_risk',
    'revol_util_x_logbal',
    'accounts_per_year',
    'delinq_per_year',
]


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Division with zero/negative denominators replaced by NaN."""
    den_safe = den.where(den > 0, np.nan)
    return (num / den_safe).astype('float32')


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Append the seven INTERACTION_COLS to `df`. Returns a new DataFrame."""
    out = df.copy()

    out['loan_to_income'] = _safe_div(out['loan_amnt'], out['annual_inc'])
    out['installment_to_income'] = _safe_div(out['installment'] * 12, out['annual_inc'])
    out['int_rate_x_term'] = (out['int_rate'] * out['term']).astype('float32')
    out['fico_dti_risk'] = ((850 - out['fico_mean']) * (out['dti'] / 100)).astype('float32')
    out['revol_util_x_logbal'] = (
        (out['revol_util'] / 100) * np.log1p(out['revol_bal'])
    ).astype('float32')

    # accounts_per_year / delinq_per_year — clip history at 1 year minimum so
    # very-thin-file borrowers don't get an exploded rate.
    history_clip = out['credit_history_years'].astype('float32').clip(lower=1)
    out['accounts_per_year'] = (out['total_acc'] / history_clip).astype('float32')
    out['delinq_per_year'] = (out['delinq_2yrs'] / history_clip).astype('float32')

    return out
