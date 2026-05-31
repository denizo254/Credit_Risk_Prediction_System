"""Phase 6 — monitoring helpers + CLI.

Reads `outputs/logs/predictions.jsonl` (written by serve.py) and reports the
production-health signals a risk team would actually look at:

  - PSI(train -> production scores)  — Phase 5's `psi()` reused, with
                                       train scores as the reference.
  - Predicted default-rate trend     — daily mean p_default; if it climbs,
                                       either the population is shifting
                                       or the model is drifting.
  - Realized performance (when labels  — Brier / ROC-AUC / KS, overall and by
    arrive)                              month, on the subset of predictions
                                         whose outcomes are supplied via a
                                         ground-truth file and joined on
                                         application_id.

Alert thresholds match Phase 5's recommendations:
  - PSI >= 0.25            => SIGNIFICANT drift, trigger retrain
  - PSI in [0.10, 0.25)    => MODERATE drift, investigate
  - PSI <  0.10            => stable

Usage:
    python monitor.py                       # full report on all logged predictions
    python monitor.py --since 2025-01-01    # window
    python monitor.py --truth outcomes.csv  # + realized Brier/AUC/KS over time
                                            #   (outcomes.csv: application_id,default)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import psi
from models import ks_statistic, model_path, split_xy
from prepare import load_processed_featv2

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


# -------- Realized performance (ground-truth join) ---------------------------

def load_ground_truth(path: Path, id_col: str = 'application_id',
                      label_col: str = 'default') -> pd.DataFrame:
    """Load realized outcomes keyed by application id.

    Accepts CSV or parquet with at least `id_col` and `label_col`. Returns a
    frame with columns ['application_id', 'actual'] (actual = 0/1).
    """
    path = Path(path)
    if path.suffix == '.parquet':
        df = pd.read_parquet(path)
    elif path.suffix in ('.csv', '.txt'):
        df = pd.read_csv(path)
    else:
        raise ValueError(f'Unsupported ground-truth format: {path.suffix}')
    missing = [c for c in (id_col, label_col) if c not in df.columns]
    if missing:
        raise ValueError(f'Ground-truth file missing column(s): {", ".join(missing)}')
    out = df[[id_col, label_col]].rename(
        columns={id_col: 'application_id', label_col: 'actual'})
    out['application_id'] = out['application_id'].astype(str)
    out['actual'] = out['actual'].astype(int)
    return out


def join_outcomes(predictions: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Inner-join logged predictions to realized outcomes on application_id.

    Predictions with no application_id, or with no matching label yet, are
    dropped — the result is exactly the set we can actually score.
    """
    if predictions.empty or 'application_id' not in predictions.columns:
        return predictions.iloc[:0].assign(actual=pd.Series(dtype='int64'))
    preds = predictions.dropna(subset=['application_id']).copy()
    preds['application_id'] = preds['application_id'].astype(str)
    return preds.merge(truth, on='application_id', how='inner')


def realized_metrics(y_true, y_score) -> dict:
    """Performance on a labeled slice. Brier is always defined; ROC-AUC / KS /
    log-loss need both classes present, else they're reported as None."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_score, dtype=float)
    m = {
        'n': int(len(y)),
        'actual_default_rate': round(float(y.mean()), 4),
        'mean_p_default': round(float(p.mean()), 4),
        'brier': round(float(brier_score_loss(y, p)), 4),
    }
    if len(np.unique(y)) == 2:
        m['roc_auc'] = round(float(roc_auc_score(y, p)), 4)
        m['ks'] = round(float(ks_statistic(y, p)), 4)
        m['log_loss'] = round(float(log_loss(y, p)), 4)
    else:
        m['roc_auc'] = m['ks'] = m['log_loss'] = None
    return m


def realized_monthly(joined: pd.DataFrame) -> pd.DataFrame:
    """Per-calendar-month realized metrics on the joined slice."""
    # strftime (not to_period) so tz-aware timestamps don't warn about dropping tz.
    j = joined.assign(month=joined['ts'].dt.strftime('%Y-%m'))
    rows = [{'month': month, **realized_metrics(g['actual'], g['p_default'])}
            for month, g in j.groupby('month')]
    return pd.DataFrame(rows).set_index('month')


def report(predictions: pd.DataFrame, ref_scores: np.ndarray,
           truth: pd.DataFrame | None = None) -> dict:
    """Compute the headline monitoring numbers.

    If `truth` (a ground-truth frame from `load_ground_truth`) is supplied,
    realized performance is computed on the subset of predictions whose
    outcomes are known.
    """
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

    result = {
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

    if truth is not None:
        joined = join_outcomes(predictions, truth)
        if joined.empty:
            result['realized_note'] = (
                'No logged predictions matched a ground-truth label '
                '(missing application_id, or outcomes not in yet).')
        else:
            realized = realized_metrics(joined['actual'], joined['p_default'])
            realized['n_unlabeled'] = int(len(predictions) - len(joined))
            result['realized'] = realized
            result['realized_monthly'] = realized_monthly(joined)

    return result


def print_report(r: dict) -> None:
    print('\n=== Monitoring report ===')
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

    if 'realized' in r:
        m = r['realized']
        auc = f'{m["roc_auc"]:.4f}' if m['roc_auc'] is not None else 'n/a (single class)'
        ks = f'{m["ks"]:.4f}' if m['ks'] is not None else 'n/a'
        print('\n  realized performance (labeled subset):')
        print(f'    labeled / unlabeled:  {m["n"]:,} / {m["n_unlabeled"]:,}')
        print(f'    actual default rate:  {m["actual_default_rate"]:.4f}  '
              f'(mean predicted {m["mean_p_default"]:.4f})')
        print(f'    Brier: {m["brier"]:.4f}   ROC-AUC: {auc}   KS: {ks}')
        print('\n  realized by month:')
        print(r['realized_monthly'].to_string())
    elif 'realized_note' in r:
        print(f'\n  realized performance:  {r["realized_note"]}')

    if r['psi_band'] == 'SIGNIFICANT':
        print('\n  ALERT: PSI in significant-drift band - schedule a model refresh.')
    elif r['psi_band'] == 'MODERATE':
        print('\n  Watch: PSI in moderate-drift band - investigate population changes.')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Production-monitoring report.')
    parser.add_argument('--since', default=None, help='ISO date (YYYY-MM-DD).')
    parser.add_argument('--model', default='xgb_v4_interactions')
    parser.add_argument('--truth', default=None,
                        help='CSV/parquet of realized outcomes for the labeled subset.')
    parser.add_argument('--id-col', default='application_id',
                        help='ID column in the ground-truth file (default: application_id).')
    parser.add_argument('--label-col', default='default',
                        help='0/1 outcome column in the ground-truth file (default: default).')
    args = parser.parse_args(argv)

    since = None
    if args.since:
        since = datetime.fromisoformat(args.since).replace(tzinfo=UTC)

    truth = None
    if args.truth:
        truth = load_ground_truth(Path(args.truth),
                                  id_col=args.id_col, label_col=args.label_col)

    preds = load_predictions(since)
    ref = reference_train_scores(args.model)
    r = report(preds, ref, truth=truth)
    print_report(r)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
