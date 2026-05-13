"""Phase 6 — monitoring helpers + CLI.

Reads `outputs/logs/predictions.jsonl` (written by serve.py) and reports the
production-health signals a risk team would actually look at:

  - PSI(train -> production scores)  — Phase 5's `psi()` reused, with
                                       train scores as the reference.
  - Predicted default-rate trend     — daily mean p_default; if it climbs,
                                       either the population is shifting
                                       or the model is drifting.
  - Brier score (when labels arrive) — needs a separate ground-truth file
                                       keyed by some application_id.

Alert thresholds match Phase 5's recommendations:
  - PSI >= 0.25            => SIGNIFICANT drift, trigger retrain
  - PSI in [0.10, 0.25)    => MODERATE drift, investigate
  - PSI <  0.10            => stable

Usage:
    python monitor.py                       # full report on all logged predictions
    python monitor.py --since 2025-01-01    # window
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import load_processed_featv2, CATEGORICAL_COLS
from models import split_xy, model_path
from evaluate import psi

PROJECT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT / 'outputs' / 'logs' / 'predictions.jsonl'

PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25
# PSI on small samples is unstable — empty quantile bins blow up the log ratio.
# Industry rule of thumb: need ~100 per decile bin, so 1000 total.
PSI_MIN_SAMPLES = 1000


def load_predictions(since: datetime | None = None) -> pd.DataFrame:
    """Read predictions.jsonl into a tidy DataFrame, optionally filtered by date."""
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=['ts', 'p_default', 'decision', 'threshold', 'model'])
    rows = []
    with LOG_PATH.open('r', encoding='utf-8') as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df['ts'] = pd.to_datetime(df['ts'], utc=True)
    if since is not None:
        df = df[df['ts'] >= since]
    return df


def reference_train_scores(model_name: str = 'xgb_v4_interactions') -> np.ndarray:
    """Score the train set with the current production model — this is the
    reference distribution PSI is measured against. v4 needs the featv2
    parquets (with interactions); older models would need load_processed()
    if you point this back at them via the `model_name` arg."""
    model = joblib.load(model_path(model_name))
    train, _ = load_processed_featv2()
    X_train, _ = split_xy(train)
    return model.predict_proba(X_train)[:, 1]


def psi_band(value: float) -> str:
    if value >= PSI_SIGNIFICANT:
        return 'SIGNIFICANT'
    if value >= PSI_MODERATE:
        return 'MODERATE'
    return 'STABLE'


def report(predictions: pd.DataFrame, ref_scores: np.ndarray) -> dict:
    """Compute the headline monitoring numbers."""
    if predictions.empty:
        return {'n_predictions': 0, 'note': 'No predictions logged yet.'}

    prod_scores = predictions['p_default'].astype(float).to_numpy()
    if len(prod_scores) < PSI_MIN_SAMPLES:
        psi_val = float('nan')
        psi_note = f'PSI suppressed (n={len(prod_scores)} < {PSI_MIN_SAMPLES})'
    else:
        psi_val = psi(ref_scores, prod_scores)
        psi_note = None

    daily = (
        predictions.assign(date=predictions['ts'].dt.date)
        .groupby('date')
        .agg(n=('p_default', 'size'),
             mean_p_default=('p_default', 'mean'),
             reject_rate=('decision', lambda s: (s == 'reject').mean()))
        .round(4)
    )

    return {
        'n_predictions': int(len(predictions)),
        'window': (predictions['ts'].min().isoformat(),
                   predictions['ts'].max().isoformat()),
        'psi': round(float(psi_val), 4) if psi_val == psi_val else None,
        'psi_band': psi_band(psi_val) if psi_val == psi_val else 'INSUFFICIENT_DATA',
        'psi_note': psi_note,
        'mean_p_default_prod': round(float(prod_scores.mean()), 4),
        'mean_p_default_train_ref': round(float(ref_scores.mean()), 4),
        'reject_rate': round(float((predictions['decision'] == 'reject').mean()), 4),
        'daily_table': daily,
    }


def print_report(r: dict) -> None:
    print(f'\n=== Monitoring report ===')
    if r.get('n_predictions', 0) == 0:
        print(f'  {r["note"]}')
        return
    print(f'  n predictions:        {r["n_predictions"]:,}')
    print(f'  window:               {r["window"][0]}  ->  {r["window"][1]}')
    if r['psi'] is None:
        print(f'  PSI(train -> prod):   n/a  [{r["psi_band"]}]  ({r["psi_note"]})')
    else:
        print(f'  PSI(train -> prod):   {r["psi"]:.4f}  [{r["psi_band"]}]')
    print(f'  mean p_default:       {r["mean_p_default_prod"]:.4f}  '
          f'(train ref {r["mean_p_default_train_ref"]:.4f})')
    print(f'  reject rate:          {r["reject_rate"]*100:.1f}%')
    print()
    print('  daily:')
    print(r['daily_table'].to_string())

    if r['psi_band'] == 'SIGNIFICANT':
        print('\n  ALERT: PSI in significant-drift band - schedule a model refresh.')
    elif r['psi_band'] == 'MODERATE':
        print('\n  Watch: PSI in moderate-drift band - investigate population changes.')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Production-monitoring report.')
    parser.add_argument('--since', default=None, help='ISO date (YYYY-MM-DD).')
    parser.add_argument('--model', default='xgb_v4_interactions')
    args = parser.parse_args(argv)

    since = None
    if args.since:
        since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)

    preds = load_predictions(since)
    ref = reference_train_scores(args.model)
    r = report(preds, ref)
    print_report(r)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
