"""Phase 6 smoke — exercise the deployment stack end-to-end without uvicorn.

Steps:
  1. Build a sample request from a real test row + a hand-crafted edge case.
  2. Hit serve.app via fastapi.testclient (no subprocess, in-process call).
  3. Run score_batch.py on a small slice of test.parquet.
  4. Run monitor.py against the predictions log produced in step 2.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
import monitor
import serve as serve_module
from prepare import CATEGORICAL_COLS, load_processed

PROJECT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT / 'outputs' / 'logs' / 'predictions.jsonl'
SAMPLE_PARQUET = PROJECT / 'outputs' / 'sample_apps.parquet'
SCORED_PARQUET = PROJECT / 'outputs' / 'sample_scored.parquet'


def row_to_payload(row: pd.Series) -> dict:
    """Convert a prepared DataFrame row to the LoanApplication schema."""
    payload = {}
    for key, val in row.items():
        if key in ('default', 'issue_year'):
            continue
        if pd.isna(val):
            payload[key] = None
        elif key in CATEGORICAL_COLS:
            payload[key] = str(val)
        elif key in ('term', 'emp_length_missing', 'pub_rec_bankruptcies',
                     'mort_acc', 'credit_history_years'):
            payload[key] = int(val)
        else:
            payload[key] = float(val)
    return payload


t0 = time.time()

# Wipe previous log so this smoke is reproducible.
if LOG_PATH.exists():
    LOG_PATH.unlink()

# ---------- 1. Build sample payloads from real test rows ----------
print('Loading test set & building sample payloads...')
_, test = load_processed()
sample = test.sample(n=5, random_state=42).reset_index(drop=True)
payloads = [row_to_payload(sample.iloc[i]) for i in range(len(sample))]

# ---------- 2. Online inference via TestClient ----------
print('\n=== /health ===')
with TestClient(serve_module.app) as client:
    r = client.get('/health')
    print(f'  status={r.status_code}  body={r.json()}')
    assert r.status_code == 200 and r.json()['status'] == 'ok'

    print('\n=== /predict (single) ===')
    for i, p in enumerate(payloads[:2]):
        r = client.post('/predict', json=p)
        assert r.status_code == 200, r.text
        body = r.json()
        print(f'  app {i}: actual_default={sample.iloc[i]["default"]}  '
              f'p_default={body["p_default"]:.4f}  decision={body["decision"]}')

    print('\n=== /predict/batch ===')
    r = client.post('/predict/batch', json={'applications': payloads})
    assert r.status_code == 200, r.text
    body = r.json()
    preds = body['predictions']
    print(f'  scored {len(preds)} apps in batch')
    for i, p in enumerate(preds):
        print(f'    app {i}: actual={sample.iloc[i]["default"]}  '
              f'p={p["p_default"]:.4f}  -> {p["decision"]}')

    # Validation rejects malformed payload
    print('\n=== /predict (invalid payload) ===')
    bad = dict(payloads[0])
    del bad['loan_amnt']
    r = client.post('/predict', json=bad)
    print(f'  status={r.status_code} (expected 422)')
    assert r.status_code == 422

# ---------- 3. Batch CLI ----------
print('\n=== score_batch CLI on a 1000-row slice ===')
slice_df = test.sample(n=1000, random_state=7).drop(columns=['default'])
SAMPLE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
slice_df.to_parquet(SAMPLE_PARQUET, index=False)

result = subprocess.run(
    [sys.executable, str(Path(__file__).resolve().parent / 'score_batch.py'),
     '--input', str(SAMPLE_PARQUET),
     '--output', str(SCORED_PARQUET)],
    capture_output=True, text=True, check=False,
)
print(result.stdout)
if result.returncode != 0:
    print('STDERR:', result.stderr)
    raise SystemExit(f'score_batch failed: rc={result.returncode}')

scored = pd.read_parquet(SCORED_PARQUET)
print(f'  scored shape: {scored.shape}')
print(f'  mean p_default: {scored["p_default"].mean():.4f}')
print(f'  reject rate:    {(scored["decision"] == "reject").mean()*100:.1f}%')

# ---------- 4. Monitoring on the JSONL log ----------
print('\n=== monitor.report on the predictions log ===')
preds = monitor.load_predictions()
print(f'  loaded {len(preds)} logged predictions')
ref = monitor.reference_train_scores()
r = monitor.report(preds, ref)
monitor.print_report(r)

print(f'\nTotal elapsed: {time.time() - t0:.1f}s')
