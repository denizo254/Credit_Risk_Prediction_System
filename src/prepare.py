"""Phase 3 data preparation — pure cleaning + feature engineering.

Reads `data/interim/accepted_curated.parquet` (produced by notebook 02) and
returns a model-ready frame plus a time-based train/test split.

Design choices, locked in here so notebook 03 and Phase 4 stay aligned:
  - Type coercion only; no scaling, no encoding, no statistical imputation.
    Those are model-dependent and belong in the Phase 4 sklearn Pipeline so
    they stay inside CV folds.
  - Two semantic imputations are done here because they are NOT statistical:
    `mort_acc` and `pub_rec_bankruptcies` NaN -> 0 (the column literally
    counts events; absence = zero, not "unknown").
  - `emp_length` missingness (6.5%) is preserved as a separate indicator
    column; the value itself stays NaN for the pipeline to impute.
  - Time-based split (train <= 2016, test >= 2017) — see notebook for why.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from load import INTERIM_PARQUET

PROJECT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT / 'data' / 'processed'
TRAIN_PARQUET = PROCESSED_DIR / 'train.parquet'
TEST_PARQUET = PROCESSED_DIR / 'test.parquet'
# Phase 8 variant — same rows, plus interaction features from features.py.
# Kept separate so xgb_v1/v2/v3 stay loadable on the v1 parquets.
TRAIN_PARQUET_FEATV2 = PROCESSED_DIR / 'train_featv2.parquet'
TEST_PARQUET_FEATV2 = PROCESSED_DIR / 'test_featv2.parquet'

TRAIN_MAX_YEAR = 2016
TEST_MIN_YEAR = 2017

CATEGORICAL_COLS = [
    'grade', 'sub_grade', 'home_ownership', 'verification_status',
    'purpose', 'addr_state', 'application_type',
]

_EMP_LENGTH_MAP = {
    '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
    '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
    '10+ years': 10,
}


def _parse_term(s: pd.Series) -> pd.Series:
    """' 36 months' -> 36 (int8). LendingClub only uses 36 or 60."""
    return s.str.strip().str.extract(r'(\d+)', expand=False).astype('int8')


def _parse_emp_length(s: pd.Series) -> pd.Series:
    """'< 1 year'->0, '1 year'->1, ..., '10+ years'->10. NaN preserved."""
    return s.map(_EMP_LENGTH_MAP).astype('Float32')


def _parse_earliest_cr_year(s: pd.Series) -> pd.Series:
    """'Aug-2003' -> 2003. Pandas parses 'Aug-2003' directly with format='%b-%Y'."""
    return pd.to_datetime(s, format='%b-%Y', errors='coerce').dt.year.astype('Int16')


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all Phase 3 cleaning + feature engineering. Pure: no I/O."""
    out = df.copy()

    out['term'] = _parse_term(out['term'])
    out['emp_length_missing'] = out['emp_length'].isna().astype('int8')
    out['emp_length'] = _parse_emp_length(out['emp_length'])

    earliest_year = _parse_earliest_cr_year(out['earliest_cr_line'])
    out['credit_history_years'] = (out['issue_year'] - earliest_year).astype('Int16')

    out['fico_mean'] = ((out['fico_range_low'] + out['fico_range_high']) / 2).astype('float32')

    # Semantic imputations only — see module docstring.
    out['mort_acc'] = out['mort_acc'].fillna(0).astype('int16')
    out['pub_rec_bankruptcies'] = out['pub_rec_bankruptcies'].fillna(0).astype('int16')

    for c in CATEGORICAL_COLS:
        out[c] = out[c].astype('category')

    out = out.drop(columns=[
        'loan_status', 'earliest_cr_line', 'issue_d',
        'fico_range_low', 'fico_range_high',
    ])

    return out


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-time split. Train: issue_year <= 2016. Test: issue_year >= 2017."""
    train = df[df['issue_year'] <= TRAIN_MAX_YEAR].copy()
    test = df[df['issue_year'] >= TEST_MIN_YEAR].copy()
    return train, test


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the train/test parquets written by `prepare_and_save()`."""
    if not TRAIN_PARQUET.exists() or not TEST_PARQUET.exists():
        raise FileNotFoundError(
            f'{TRAIN_PARQUET} or {TEST_PARQUET} missing. '
            'Run src/prepare.py (or notebook 03) first.'
        )
    return pd.read_parquet(TRAIN_PARQUET), pd.read_parquet(TEST_PARQUET)


def prepare_and_save() -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end: read interim, clean, split, write parquets, return (train, test)."""
    curated = pd.read_parquet(INTERIM_PARQUET)
    cleaned = clean(curated)
    train, test = time_split(cleaned)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(TRAIN_PARQUET, index=False)
    test.to_parquet(TEST_PARQUET, index=False)
    return train, test


def load_processed_featv2() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Phase-8 feature-v2 parquets — clean() plus features.add_interactions()."""
    if not TRAIN_PARQUET_FEATV2.exists() or not TEST_PARQUET_FEATV2.exists():
        raise FileNotFoundError(
            f'{TRAIN_PARQUET_FEATV2} or {TEST_PARQUET_FEATV2} missing. '
            'Run prepare_and_save_featv2() (or src/_smoke_phase8.py) first.'
        )
    return pd.read_parquet(TRAIN_PARQUET_FEATV2), pd.read_parquet(TEST_PARQUET_FEATV2)


def prepare_and_save_featv2() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Phase 8: clean() + add_interactions(), then time-split + write featv2 parquets."""
    from features import add_interactions  # local import — features.py optional

    curated = pd.read_parquet(INTERIM_PARQUET)
    cleaned = clean(curated)
    enriched = add_interactions(cleaned)
    train, test = time_split(enriched)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(TRAIN_PARQUET_FEATV2, index=False)
    test.to_parquet(TEST_PARQUET_FEATV2, index=False)
    return train, test


if __name__ == '__main__':
    tr, te = prepare_and_save()
    print(f'train: {tr.shape}  default rate: {tr["default"].mean()*100:.2f}%')
    print(f'test:  {te.shape}  default rate: {te["default"].mean()*100:.2f}%')
