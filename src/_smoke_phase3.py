"""Phase 3 smoke — run prepare.clean() + time_split() end-to-end.

Not a deliverable. Confirms the curated parquet survives cleaning, the split
sizes match expectations, and no NaNs slipped through where we didn't expect any.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load import INTERIM_PARQUET
from prepare import (
    CATEGORICAL_COLS, TRAIN_PARQUET, TEST_PARQUET,
    TRAIN_MAX_YEAR, TEST_MIN_YEAR,
    clean, time_split,
)


t0 = time.time()
print(f'Loading {INTERIM_PARQUET.name} ...')
curated = pd.read_parquet(INTERIM_PARQUET)
print(f'  Loaded in {time.time() - t0:.1f}s  shape={curated.shape}')

print('\n=== Cleaning ===')
t1 = time.time()
df = clean(curated)
print(f'  clean() done in {time.time() - t1:.1f}s  shape={df.shape}')
print(f'  Memory: {df.memory_usage(deep=True).sum() / 1e6:.0f} MB')

print('\n=== Dtypes after clean ===')
print(df.dtypes.to_string())

print('\n=== Missingness after clean (>0%) ===')
miss = (df.isna().mean() * 100).sort_values(ascending=False)
miss = miss[miss > 0]
if len(miss):
    print(miss.round(3).to_string())
else:
    print('  (none)')

# Cols where Phase 3 should leave NO NaNs (semantic imputation or no-NaN input).
must_be_clean = ['mort_acc', 'pub_rec_bankruptcies', 'default', 'issue_year',
                 'term', 'fico_mean'] + CATEGORICAL_COLS
for c in must_be_clean:
    n_na = df[c].isna().sum()
    assert n_na == 0, f'{c} still has {n_na} NaN after clean()'
print('  All semantic-imputed / required columns clean: OK')

# Engineered features sanity
print('\n=== Engineered features ===')
print('credit_history_years describe:')
print(df['credit_history_years'].describe().round(2).to_string())
assert df['credit_history_years'].min() >= 0, 'negative credit history — date parse bug?'
print(f'fico_mean range: [{df["fico_mean"].min()}, {df["fico_mean"].max()}]')
print(f'emp_length_missing share: {df["emp_length_missing"].mean()*100:.2f}%')

print('\n=== Time split ===')
train, test = time_split(df)
print(f'train (issue_year <= {TRAIN_MAX_YEAR}): {len(train):,} rows '
      f'({len(train)/len(df)*100:.1f}%)  default rate: {train["default"].mean()*100:.2f}%')
print(f'test  (issue_year >= {TEST_MIN_YEAR}): {len(test):,} rows '
      f'({len(test)/len(df)*100:.1f}%)  default rate: {test["default"].mean()*100:.2f}%')
assert len(train) + len(test) == len(df), 'train+test != total — year gap?'
assert train['issue_year'].max() < test['issue_year'].min(), 'time-split leakage!'

print('\n=== Saving processed parquets ===')
TRAIN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
train.to_parquet(TRAIN_PARQUET, index=False)
test.to_parquet(TEST_PARQUET, index=False)
print(f'  Wrote {TRAIN_PARQUET}  ({TRAIN_PARQUET.stat().st_size / 1e6:.1f} MB)')
print(f'  Wrote {TEST_PARQUET}  ({TEST_PARQUET.stat().st_size / 1e6:.1f} MB)')

print(f'\nTotal elapsed: {time.time() - t0:.1f}s')
