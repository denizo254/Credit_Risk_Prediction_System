"""Phase 2 smoke test — runs the headline cells from notebook 02 end-to-end.

Not part of the deliverable; just confirms the curated load, target derivation,
and class-imbalance numbers before the user opens Jupyter.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load import FEATURE_COLS, INTERIM_PARQUET, RAW_CSV, TARGET_COL, derive_default_flag

t0 = time.time()
print(f'Reading {RAW_CSV.name} ({RAW_CSV.stat().st_size / 1e9:.2f} GB)...')
df = pd.read_csv(RAW_CSV, usecols=FEATURE_COLS + [TARGET_COL], low_memory=False)
print(f'  Loaded in {time.time() - t0:.1f}s  shape={df.shape}  mem={df.memory_usage(deep=True).sum()/1e6:.0f} MB')

print('\n=== Raw loan_status distribution ===')
print(df[TARGET_COL].value_counts(dropna=False).to_string())

print('\n=== Missingness (>0%) ===')
miss = (df.isna().mean() * 100).sort_values(ascending=False)
print(miss[miss > 0].round(2).to_string())

print('\n=== Target derivation ===')
df['default'] = derive_default_flag(df[TARGET_COL])
n_before = len(df)
df = df.dropna(subset=['default']).copy()
df['default'] = df['default'].astype('int8')
print(f'Dropped censored: {n_before - len(df):,} rows ({(n_before - len(df)) / n_before * 100:.1f}%)')
print(f'Remaining:        {len(df):,} rows')

print('\n=== Class imbalance ===')
counts = df['default'].value_counts()
print(counts.to_string())
print(f'Default rate:      {df["default"].mean() * 100:.2f}%')
print(f'Imbalance ratio:   {counts[0] / counts[1]:.1f}:1  (repay:default)')

print('\n=== Default rate by grade ===')
by_grade = df.groupby('grade')['default'].agg(['mean', 'count'])
by_grade['mean'] = (by_grade['mean'] * 100).round(2)
print(by_grade.to_string())

print('\n=== Default rate by year ===')
df['issue_year'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce').dt.year
by_year = df.groupby('issue_year')['default'].agg(['mean', 'count'])
by_year['mean'] = (by_year['mean'] * 100).round(2)
print(by_year.to_string())

print('\n=== Saving interim parquet ===')
INTERIM_PARQUET.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(INTERIM_PARQUET, index=False)
print(f'  Wrote {INTERIM_PARQUET}  ({INTERIM_PARQUET.stat().st_size / 1e6:.1f} MB)')
print(f'\nTotal elapsed: {time.time() - t0:.1f}s')
