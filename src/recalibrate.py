"""Phase 9 — rolling recalibration.

The cheapest, most responsive answer to the calibration drift the monitor
(`monitor.py --truth`) detects. The base XGBoost is expensive to retrain and
its *ranking* (ROC-AUC / KS) holds up out-of-time; what drifts is the
*level* of the probabilities — the 2017-2018 default rate runs above the 2016
slice the isotonic layer was fit on, so the model under-predicts.

This module freezes the existing base booster and refits **only** the isotonic
calibration layer on a recent window of labeled outcomes. Seconds to run, no
GPU, and it tracks the population as it moves. Retrain the base only when the
*ranking* degrades (a separate, rarer event).

Usage:
    python recalibrate.py --calib recent_labeled.parquet --out xgb_v5_recalibrated
    # window + before/after report on a held-out eval set:
    python recalibrate.py --calib test_featv2.parquet --years 1 --max-year 2017 \
        --eval test_2018.parquet --out xgb_v5_recalibrated

`--calib` / `--eval` are featv2-shaped parquets (model features + `default` +
`issue_year`) — the same shape `prepare.load_processed_featv2()` produces.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import CalibratedXGB
from models import model_path, split_xy

DEFAULT_BASE = 'xgb_v4_interactions'


def base_of(model):
    """The underlying XGB booster-bearing estimator, whether `model` is a
    CalibratedXGB wrapper or a bare classifier."""
    return getattr(model, 'base', model)


def recent_years(df: pd.DataFrame, years: int = 1, max_year: int | None = None) -> pd.DataFrame:
    """The most recent `years` of rows by `issue_year` (default: ending at the
    latest year present). `years=1` approximates a rolling-12-month window for
    this year-granularity data."""
    if 'issue_year' not in df.columns:
        raise ValueError("calibration frame needs an 'issue_year' column to window on")
    top = int(max_year) if max_year is not None else int(df['issue_year'].max())
    lo = top - years + 1
    return df[(df['issue_year'] >= lo) & (df['issue_year'] <= top)]


def recalibrate(model, calib_df: pd.DataFrame) -> CalibratedXGB:
    """Refit the isotonic layer on `calib_df`, reusing `model`'s frozen base.

    Returns a new CalibratedXGB sharing the same base booster — the base is not
    retrained, so this is fast and the ranking is unchanged by construction.
    """
    base = base_of(model)
    X, y = split_xy(calib_df)
    raw = base.predict_proba(X)[:, 1]
    iso = IsotonicRegression(out_of_bounds='clip').fit(raw, y)
    return CalibratedXGB(base=base, iso=iso)


def calibration_summary(model, df: pd.DataFrame) -> dict:
    """Level-calibration snapshot on a labeled frame: mean predicted vs observed
    default rate, their gap, and Brier score."""
    X, y = split_xy(df)
    p = model.predict_proba(X)[:, 1]
    mean_pred = float(p.mean())
    observed = float(y.mean())
    return {
        'n': int(len(y)),
        'mean_pred': round(mean_pred, 4),
        'observed': round(observed, 4),
        'gap': round(mean_pred - observed, 4),
        'brier': round(float(brier_score_loss(y, p)), 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Rolling recalibration: refit only the isotonic layer on recent labeled data.')
    parser.add_argument('--calib', required=True, type=Path,
                        help='Labeled featv2 parquet (features + default + issue_year).')
    parser.add_argument('--base', default=DEFAULT_BASE,
                        help='Model whose frozen XGB base is reused (default: %(default)s).')
    parser.add_argument('--out', required=True,
                        help='Output model name -> outputs/models/<name>.joblib.')
    parser.add_argument('--years', type=int, default=1,
                        help='Rolling window length in years (default: 1 ~ 12 months).')
    parser.add_argument('--max-year', type=int, default=None,
                        help='Most recent year of the window (default: max in --calib).')
    parser.add_argument('--eval', type=Path, default=None,
                        help='Optional labeled featv2 parquet for a before/after report.')
    args = parser.parse_args(argv)

    model = joblib.load(model_path(args.base))
    calib = pd.read_parquet(args.calib)
    window = recent_years(calib, years=args.years, max_year=args.max_year)
    print(f'Recalibrating {args.base} base on {len(window):,} loans '
          f'(years {int(window["issue_year"].min())}-{int(window["issue_year"].max())})...')
    new_model = recalibrate(model, window)

    if args.eval is not None:
        ev = pd.read_parquet(args.eval)
        before = calibration_summary(model, ev)
        after = calibration_summary(new_model, ev)
        print(f'  eval set: {before["n"]:,} loans, observed default {before["observed"]:.4f}')
        print(f'  before (current calibrator):    mean_pred {before["mean_pred"]:.4f}  '
              f'gap {before["gap"]:+.4f}  Brier {before["brier"]:.4f}')
        print(f'  after  (rolling recalibration):  mean_pred {after["mean_pred"]:.4f}  '
              f'gap {after["gap"]:+.4f}  Brier {after["brier"]:.4f}')

    out_path = model_path(args.out)
    joblib.dump(new_model, out_path)
    print(f'Saved -> {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
