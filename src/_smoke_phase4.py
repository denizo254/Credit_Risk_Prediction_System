"""Phase 4 smoke — fit LR + XGB on train, evaluate on the time-held-out test set.

Not a deliverable. Confirms both pipelines run end-to-end and prints the
headline metrics so we know what numbers the notebook will show.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import build_lr, build_xgb, evaluate, model_path, split_xy
from prepare import load_processed

# LogisticRegression with OHE'd 100+ columns sometimes warns on convergence
# at max_iter=200. The smoke is happy with a near-converged model — full
# convergence is for the notebook to tune if you care.
warnings.filterwarnings('ignore', category=UserWarning)


def fmt(m) -> str:
    return (f'ROC-AUC={m.roc_auc:.4f}  PR-AUC={m.pr_auc:.4f}  '
            f'Brier={m.brier:.4f}  log_loss={m.log_loss:.4f}  KS={m.ks:.4f}')


t0 = time.time()
print('Loading processed splits...')
train, test = load_processed()
X_train, y_train = split_xy(train)
X_test,  y_test  = split_xy(test)
print(f'  train: X={X_train.shape}  y mean={y_train.mean()*100:.2f}%')
print(f'  test:  X={X_test.shape}   y mean={y_test.mean()*100:.2f}%')

# -------- Logistic regression baseline --------
print('\n=== Logistic Regression (class_weight=balanced) ===')
t1 = time.time()
lr = build_lr(X_train)
lr.fit(X_train, y_train)
print(f'  fit in {time.time() - t1:.1f}s')

p_train = lr.predict_proba(X_train)[:, 1]
p_test  = lr.predict_proba(X_test)[:, 1]
m_train = evaluate(y_train, p_train)
m_test  = evaluate(y_test,  p_test)
print(f'  TRAIN  {fmt(m_train)}')
print(f'  TEST   {fmt(m_test)}')

joblib.dump(lr, model_path('lr_baseline_v1'))
print(f'  Saved -> {model_path("lr_baseline_v1")}')

# -------- XGBoost --------
print('\n=== XGBoost ===')
t2 = time.time()
xgb = build_xgb(y_train)
xgb.fit(X_train, y_train)
print(f'  fit in {time.time() - t2:.1f}s')

p_train_x = xgb.predict_proba(X_train)[:, 1]
p_test_x  = xgb.predict_proba(X_test)[:, 1]
m_train_x = evaluate(y_train, p_train_x)
m_test_x  = evaluate(y_test,  p_test_x)
print(f'  TRAIN  {fmt(m_train_x)}')
print(f'  TEST   {fmt(m_test_x)}')

joblib.dump(xgb, model_path('xgb_v1'))
print(f'  Saved -> {model_path("xgb_v1")}')

# -------- Side-by-side --------
print('\n=== TEST set comparison ===')
rows = [
    {'model': 'lr_baseline_v1', **m_test.as_row()},
    {'model': 'xgb_v1',         **m_test_x.as_row()},
]
print(pd.DataFrame(rows).set_index('model').round(4).to_string())

print(f'\nTotal elapsed: {time.time() - t0:.1f}s')
