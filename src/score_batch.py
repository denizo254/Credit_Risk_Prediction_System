"""Phase 6 — batch scoring CLI.

Usage:
    python score_batch.py --input data/processed/test.parquet --output outputs/scores.parquet
    python score_batch.py --input new_apps.csv --output scored.csv --threshold 0.15

The input must already be in the Phase-3 prepared shape (use
`prepare.clean()` on raw LendingClub data first). The CLI writes one extra
column `p_default` plus a `decision` column.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import CATEGORICAL_COLS, load_processed_featv2
from features import INTERACTION_COLS, add_interactions
from models import model_path

DEFAULT_THRESHOLD = 0.13
DEFAULT_MODEL = 'xgb_v4_interactions'


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    if path.suffix in ('.csv', '.txt'):
        return pd.read_csv(path)
    raise ValueError(f'Unsupported input format: {path.suffix}')


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.parquet':
        df.to_parquet(path, index=False)
    elif path.suffix in ('.csv', '.txt'):
        df.to_csv(path, index=False)
    else:
        raise ValueError(f'Unsupported output format: {path.suffix}')


def _align_categoricals(df: pd.DataFrame, categories: dict[str, list]) -> pd.DataFrame:
    """Force categoricals to share training-set levels — same trick serve.py uses."""
    for c in CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = pd.Categorical(df[c], categories=categories[c])
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Batch score loan applications.')
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args(argv)

    t0 = time.time()
    print(f'Loading model {args.model}...')
    model = joblib.load(model_path(args.model))

    train, _ = load_processed_featv2()
    categories = {c: list(train[c].cat.categories) for c in CATEGORICAL_COLS}
    feature_order = [c for c in train.columns if c not in ('default', 'issue_year')]

    print(f'Reading {args.input}...')
    df = _read(args.input)
    print(f'  {len(df):,} rows')

    # If the input is a v1-feature parquet (no interactions), compute them.
    if not all(c in df.columns for c in INTERACTION_COLS):
        df = add_interactions(df)

    X = df.reindex(columns=feature_order)
    X = _align_categoricals(X, categories)

    print(f'Scoring (threshold={args.threshold})...')
    probs = model.predict_proba(X)[:, 1]
    out = df.copy()
    out['p_default'] = probs
    out['decision'] = ['reject' if p >= args.threshold else 'approve' for p in probs]

    _write(out, args.output)
    n_reject = int((out['decision'] == 'reject').sum())
    print(f'Wrote {args.output}  ({len(out):,} rows, {n_reject:,} rejects '
          f'= {n_reject/len(out)*100:.1f}%)')
    print(f'Elapsed: {time.time() - t0:.1f}s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
