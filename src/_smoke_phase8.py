"""Phase 8 smoke — regenerate featv2 parquets, train v4 with v3's hyperparams.

Steps:
  1. Generate `train_featv2.parquet` and `test_featv2.parquet` (clean + add_interactions).
  2. Read the v3 hyperparams from outputs/tune_results.csv (top row) so the
     comparison v3 vs v4 is *features-only*, same model capacity.
  3. Refit XGB(base) on 2007-2015, isotonic on 2016, save xgb_v4_interactions.
  4. Evaluate v4 on test_featv2; compare against v3 (already evaluated on
     test.parquet — same rows, just fewer columns).
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (
    BASE_MAX_YEAR,
    CALIB_YEAR,
    CalibratedXGB,
)
from features import INTERACTION_COLS
from models import build_xgb, evaluate, model_path, split_xy
from prepare import (
    load_processed,
    prepare_and_save_featv2,
)
from tune import TUNE_RESULTS

warnings.filterwarnings('ignore', category=UserWarning)


def fmt(m) -> str:
    return (f'ROC-AUC={m.roc_auc:.4f}  PR-AUC={m.pr_auc:.4f}  '
            f'Brier={m.brier:.4f}  log_loss={m.log_loss:.4f}  KS={m.ks:.4f}')


t0 = time.time()

# ---------- 1. Regenerate parquets with interactions ----------
print('Regenerating featv2 parquets (clean + add_interactions)...')
t1 = time.time()
train, test = prepare_and_save_featv2()
print(f'  done in {time.time() - t1:.1f}s')
print(f'  train: {train.shape}  test: {test.shape}')
print(f'  new columns: {INTERACTION_COLS}')

# ---------- Sanity: how much did interactions move? ----------
print('\n=== Interaction feature sanity (train) ===')
for c in INTERACTION_COLS:
    s = train[c]
    print(f'  {c:25s}  mean={s.mean():+.4f}  std={s.std():.4f}  '
          f'NaN={(s.isna().mean()*100):.2f}%  '
          f'corr_with_default={s.corr(train["default"].astype(float)):+.4f}')

# ---------- 2. Read v3 hyperparams from the Phase 7 tune results ----------
print('\nReading best hyperparams from Phase 7 tune results...')
tune_df = pd.read_csv(TUNE_RESULTS).sort_values('mean_auc', ascending=False)
best_row = tune_df.iloc[0]
v3_params = {k: best_row[k] for k in [
    'max_depth', 'learning_rate', 'n_estimators', 'min_child_weight',
    'subsample', 'colsample_bytree', 'reg_lambda', 'gamma',
]}
# Cast int-y params
for k in ('max_depth', 'n_estimators', 'min_child_weight'):
    v3_params[k] = int(v3_params[k])
for k in ('learning_rate', 'subsample', 'colsample_bytree', 'reg_lambda', 'gamma'):
    v3_params[k] = float(v3_params[k])
print(f'  v3 hyperparams: {v3_params}')

# ---------- 3. Train v4 base + isotonic ----------
base_df = train[train['issue_year'] <= BASE_MAX_YEAR]
calib_df = train[train['issue_year'] == CALIB_YEAR]
X_base, y_base = split_xy(base_df)
X_calib, y_calib = split_xy(calib_df)
X_test, y_test = split_xy(test)
print(f'\nSplits — base: {len(base_df):,}  calib: {len(calib_df):,}  test: {len(test):,}')
print(f'Feature count: {X_base.shape[1]} (was 25 in v3; expect 25 + {len(INTERACTION_COLS)} = 32)')

print('\n=== Fitting v4 (v3 hyperparams + interactions) ===')
t2 = time.time()
base = build_xgb(y_base, **v3_params)
base.fit(X_base, y_base)
print(f'  base fit in {time.time() - t2:.1f}s')

raw_calib = base.predict_proba(X_calib)[:, 1]
iso = IsotonicRegression(out_of_bounds='clip').fit(raw_calib, y_calib)
v4 = CalibratedXGB(base=base, iso=iso)
joblib.dump(v4, model_path('xgb_v4_interactions'))
print(f'  Saved -> {model_path("xgb_v4_interactions")}')

# ---------- 4. Evaluate v4 on test, compare to v3 ----------
p_v4 = v4.predict_proba(X_test)[:, 1]
m_v4 = evaluate(y_test, p_v4)

# v3 evaluation: load on the v1 test parquet (which lacks interactions).
v3 = joblib.load(model_path('xgb_v3_tuned'))
_, test_v1 = load_processed()
X_test_v1, y_test_v1 = split_xy(test_v1)
p_v3 = v3.predict_proba(X_test_v1)[:, 1]
m_v3 = evaluate(y_test_v1, p_v3)

# Sanity: test rows must match between the two parquets, else metric comparison is invalid.
assert len(test) == len(test_v1), 'test parquets have different row counts!'

print('\n=== Test-set comparison: v3 (Phase 7) vs v4 (v3 hyperparams + 7 interactions) ===')
print(f'  v3 (Phase 7):  {fmt(m_v3)}')
print(f'  v4 (Phase 8):  {fmt(m_v4)}')

for attr in ['roc_auc', 'pr_auc', 'brier', 'log_loss', 'ks']:
    d = getattr(m_v4, attr) - getattr(m_v3, attr)
    better = (d < 0) if attr in ('brier', 'log_loss') else (d > 0)
    tag = 'better' if better else 'worse'
    print(f'  {attr:10s} delta={d:+.4f}  ({tag})')

# ---------- Feature importance: did the interactions earn their keep? ----------
print('\n=== v4 top-15 features by gain (look for INTERACTION_COLS in here) ===')
gain = pd.Series(v4.base.get_booster().get_score(importance_type='gain'))
gain = gain.sort_values(ascending=False).head(15)
for name, val in gain.items():
    tag = ' <-- INTERACTION' if name in INTERACTION_COLS else ''
    print(f'  {name:25s} {val:>9.2f}{tag}')

n_interactions_in_top15 = sum(1 for n in gain.index if n in INTERACTION_COLS)
print(f'\n  {n_interactions_in_top15}/7 interactions made the top 15 — '
      f'{"signal earned" if n_interactions_in_top15 >= 3 else "weak signal"}')

print(f'\nTotal elapsed: {time.time() - t0:.1f}s')
