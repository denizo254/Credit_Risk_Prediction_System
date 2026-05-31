"""Phase 7 smoke — run the sweep, refit best on full base, calibrate, evaluate.

Output: outputs/models/xgb_v3_tuned.joblib + outputs/tune_results.csv.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import joblib
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (
    BASE_MAX_YEAR,
    CALIB_YEAR,
    CalibratedXGB,
    time_calibration_split,
)
from models import build_xgb, evaluate, model_path, split_xy
from prepare import load_processed
from tune import best_params_from, random_search

warnings.filterwarnings('ignore', category=UserWarning)


def fmt(m) -> str:
    return (f'ROC-AUC={m.roc_auc:.4f}  PR-AUC={m.pr_auc:.4f}  '
            f'Brier={m.brier:.4f}  log_loss={m.log_loss:.4f}  KS={m.ks:.4f}')


t0 = time.time()
print('Loading processed splits...')
train, test = load_processed()
base_df, calib_df = time_calibration_split(train)
X_base, y_base = split_xy(base_df)
X_calib, y_calib = split_xy(calib_df)
X_test, y_test = split_xy(test)
print(f'  base  ({BASE_MAX_YEAR} cutoff): {len(base_df):>9,}')
print(f'  calib (={CALIB_YEAR}):          {len(calib_df):>9,}')
print(f'  test:                       {len(test):>9,}')

# We need issue_year in X for tune.random_search to sort the subsample.
# It's stripped from the model's view by build_xgb's column selection
# inside .fit — but here we just pass it through; XGB treats it as a
# numeric feature for the duration of CV. Not perfect, but consistent
# across trials so the comparison is fair.
X_base_for_cv = base_df.drop(columns=['default'])

# -------- Random search --------
print('\n=== Random search (15 trials, 2-fold time-series CV, 25% subsample) ===')
t1 = time.time()
results = random_search(X_base_for_cv, y_base)
print(f'  Sweep finished in {time.time() - t1:.1f}s')

print('\n=== Top 5 trials by mean CV ROC-AUC ===')
print(results.head(5).drop(columns=['fold_aucs']).round(4).to_string(index=False))

print('\n=== Bottom 3 trials ===')
print(results.tail(3).drop(columns=['fold_aucs']).round(4).to_string(index=False))

default_row = results[results['is_default']].iloc[0]
best_row = results.iloc[0]
print('\n=== Best vs Phase-4 defaults (under identical CV) ===')
print(f'  default trial AUC: {default_row["mean_auc"]:.4f} '
      f'(rank {results.index[results["is_default"]].tolist()[0] + 1}/{len(results)})')
print(f'  best trial    AUC: {best_row["mean_auc"]:.4f}')
print(f'  delta:             {best_row["mean_auc"] - default_row["mean_auc"]:+.4f}')

best_params = best_params_from(results)
print('\nBest params:')
for k, v in best_params.items():
    print(f'  {k:20s} = {v}')

# -------- Refit best on full base, calibrate on 2016 --------
print('\n=== Refitting best params on full base (2007-2015) ===')
t2 = time.time()
final_base = build_xgb(y_base, **best_params)
final_base.fit(X_base, y_base)
print(f'  Refit in {time.time() - t2:.1f}s')

print('Fitting isotonic on 2016 slice...')
raw_calib = final_base.predict_proba(X_calib)[:, 1]
iso = IsotonicRegression(out_of_bounds='clip').fit(raw_calib, y_calib)
cal_xgb = CalibratedXGB(base=final_base, iso=iso)
joblib.dump(cal_xgb, model_path('xgb_v3_tuned'))
print(f'  Saved -> {model_path("xgb_v3_tuned")}')

# -------- Evaluate on test --------
p_test = cal_xgb.predict_proba(X_test)[:, 1]
m_test = evaluate(y_test, p_test)
print('\n=== xgb_v3_tuned on TEST ===')
print(f'  {fmt(m_test)}')

# Compare against xgb_v2_calibrated for an apples-to-apples read
print('\n=== Comparison vs xgb_v2_calibrated ===')
v2 = joblib.load(model_path('xgb_v2_calibrated'))
p_v2 = v2.predict_proba(X_test)[:, 1]
m_v2 = evaluate(y_test, p_v2)
print(f'  v2 (Phase 5):  {fmt(m_v2)}')
print(f'  v3 (tuned):    {fmt(m_test)}')
for attr in ['roc_auc', 'pr_auc', 'brier', 'log_loss', 'ks']:
    delta = getattr(m_test, attr) - getattr(m_v2, attr)
    sign = '+' if delta >= 0 else ''
    direction = 'better' if (attr in ('brier', 'log_loss') and delta < 0) or \
                            (attr not in ('brier', 'log_loss') and delta > 0) else 'worse'
    print(f'  {attr:10s}: delta={sign}{delta:+.4f}  ({direction})')

print(f'\nTotal elapsed: {time.time() - t0:.1f}s')
