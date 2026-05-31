"""Phase 5 smoke — calibrate, pick threshold, stability check.

Not a deliverable. Verifies:
  - retraining XGB on 2007-2015 + isotonic on 2016 actually improves Brier
    on the 2017-2018 test set.
  - cost_curve / optimal_threshold / gains_table / psi / metrics_by_group
    all run cleanly on the real data.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (
    BASE_MAX_YEAR,
    CALIB_YEAR,
    cost_curve,
    fit_calibrated,
    gains_table,
    metrics_by_group,
    optimal_threshold,
    psi,
    time_calibration_split,
)
from models import evaluate, model_path, split_xy
from prepare import load_processed

warnings.filterwarnings('ignore', category=UserWarning)


def fmt(m) -> str:
    return (f'ROC-AUC={m.roc_auc:.4f}  PR-AUC={m.pr_auc:.4f}  '
            f'Brier={m.brier:.4f}  log_loss={m.log_loss:.4f}  KS={m.ks:.4f}')


t0 = time.time()
print('Loading processed splits...')
train, test = load_processed()
X_test, y_test = split_xy(test)

base_df, calib_df = time_calibration_split(train)
print(f'  base  (<= {BASE_MAX_YEAR}): {len(base_df):>9,} rows  rate={base_df["default"].mean()*100:.2f}%')
print(f'  calib (== {CALIB_YEAR}):   {len(calib_df):>9,} rows  rate={calib_df["default"].mean()*100:.2f}%')
print(f'  test:                {len(test):>9,} rows  rate={y_test.mean()*100:.2f}%')

# -------- Fit calibrated model --------
print('\n=== Fitting calibrated XGB (base on 2007-2015, isotonic on 2016) ===')
t1 = time.time()
cal_xgb = fit_calibrated(train)
print(f'  done in {time.time() - t1:.1f}s')

# -------- Test metrics, calibrated vs uncalibrated --------
p_uncal = cal_xgb.base.predict_proba(X_test)[:, 1]   # the base alone
p_cal   = cal_xgb.predict_proba(X_test)[:, 1]        # after isotonic
m_uncal = evaluate(y_test, p_uncal)
m_cal   = evaluate(y_test, p_cal)
print('\n=== Test metrics: uncalibrated base vs isotonic-calibrated ===')
print(f'  uncal:  {fmt(m_uncal)}')
print(f'  cal:    {fmt(m_cal)}')
print(f'  Brier improvement: {m_uncal.brier - m_cal.brier:+.4f}  '
      f'(positive means calibration helped)')

joblib.dump(cal_xgb, model_path('xgb_v2_calibrated'))
print(f'  Saved -> {model_path("xgb_v2_calibrated")}')

# -------- Threshold selection --------
print('\n=== Cost-based threshold selection (c_fn=5, c_fp=1) ===')
curve = cost_curve(y_test, p_cal, c_fn=5.0, c_fp=1.0)
opt = optimal_threshold(curve)
print(f'  optimal threshold: {opt["threshold"]:.3f}')
print(f'  expected cost:     {opt["cost"]:,.0f} (unit costs)')
print(f'  precision @ opt:   {opt["precision"]:.3f}')
print(f'  recall    @ opt:   {opt["recall"]:.3f}')
print(f'  vs threshold=0.5 baseline cost: {curve.loc[curve["threshold"].sub(0.5).abs().idxmin(), "cost"]:,.0f}')

# Loan-amount-weighted version
print('\n=== Cost-based threshold, weighted by loan_amnt ===')
curve_w = cost_curve(y_test, p_cal, c_fn=0.5, c_fp=0.1, weights=test['loan_amnt'])
opt_w = optimal_threshold(curve_w)
print(f'  optimal threshold: {opt_w["threshold"]:.3f}')
print(f'  expected $ cost:   ${opt_w["cost"]:,.0f}')

# -------- Per-year stability --------
print('\n=== Per-year stability ===')
test_scored = test.assign(score=p_cal)
year_metrics = metrics_by_group(test_scored, score_col='score', group_col='issue_year')
print(year_metrics.round(4).to_string(index=False))

# -------- Per-grade stability --------
print('\n=== Per-grade stability ===')
grade_metrics = metrics_by_group(test_scored, score_col='score', group_col='grade')
print(grade_metrics.round(4).to_string(index=False))

# -------- Gains / lift --------
print('\n=== Gains table (deciles, decile 1 = highest predicted risk) ===')
gt = gains_table(y_test, p_cal, n_bins=10)
print(gt.round(4).to_string())

# -------- PSI: train vs test score distributions --------
print('\n=== Population Stability Index (PSI) ===')
X_train, _ = split_xy(train)
p_train = cal_xgb.predict_proba(X_train)[:, 1]
psi_val = psi(p_train, p_cal)
print(f'  PSI(train_scores -> test_scores) = {psi_val:.4f}')
band = 'STABLE' if psi_val < 0.10 else ('MODERATE drift' if psi_val < 0.25 else 'SIGNIFICANT drift')
print(f'  Interpretation: {band}')

print(f'\nTotal elapsed: {time.time() - t0:.1f}s')
