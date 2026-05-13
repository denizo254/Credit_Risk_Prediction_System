"""Shared data-loading helpers for the credit risk project.

Phases 3+ should import from here instead of re-implementing the load logic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT / 'club loan data' / 'accepted_2007_to_2018q4.csv' / 'accepted_2007_to_2018Q4.csv'
INTERIM_PARQUET = PROJECT / 'data' / 'interim' / 'accepted_curated.parquet'

FEATURE_COLS = [
    'loan_amnt', 'term', 'int_rate', 'installment',
    'grade', 'sub_grade',
    'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
    'purpose', 'addr_state', 'application_type',
    'dti', 'revol_util', 'revol_bal',
    'fico_range_low', 'fico_range_high',
    'delinq_2yrs', 'pub_rec', 'pub_rec_bankruptcies',
    'mort_acc', 'open_acc', 'total_acc',
    'earliest_cr_line', 'issue_d',
]
TARGET_COL = 'loan_status'

DEFAULT_STATUSES = frozenset({
    'Charged Off', 'Default',
    'Late (31-120 days)', 'Late (16-30 days)',
    'Does not meet the credit policy. Status:Charged Off',
})
REPAY_STATUSES = frozenset({
    'Fully Paid',
    'Does not meet the credit policy. Status:Fully Paid',
})


def derive_default_flag(loan_status: pd.Series) -> pd.Series:
    """Map raw loan_status strings to {0, 1, NaN}. NaN = censored, caller should drop."""
    return loan_status.map(
        lambda s: 1 if s in DEFAULT_STATUSES else (0 if s in REPAY_STATUSES else np.nan)
    )


def load_raw(usecols: list[str] | None = None) -> pd.DataFrame:
    """Load curated columns from the raw LendingClub CSV. ~1 minute on a laptop."""
    cols = usecols if usecols is not None else (FEATURE_COLS + [TARGET_COL])
    return pd.read_csv(RAW_CSV, usecols=cols, low_memory=False)


def load_curated() -> pd.DataFrame:
    """Load the cleaned parquet written by notebook 02. Fast (<2s)."""
    if not INTERIM_PARQUET.exists():
        raise FileNotFoundError(
            f'{INTERIM_PARQUET} not found. Run notebooks/02_data_understanding.ipynb first.'
        )
    return pd.read_parquet(INTERIM_PARQUET)
