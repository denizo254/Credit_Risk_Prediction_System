"""Phase 6 — online inference service.

FastAPI app that loads `xgb_v2_calibrated` once at startup and serves:
  GET  /health           — readiness probe
  POST /predict          — single application
  POST /predict/batch    — list of applications

What this service handles (and what it doesn't):
  - Pydantic validation rejects malformed payloads before they touch the model.
  - Categorical levels are pinned to the training distribution at startup;
    unknown grades/states pass through as NaN (XGBoost handles them natively).
  - Every successful prediction is appended to `outputs/logs/predictions.jsonl`
    for offline monitoring (Phase 6's monitor.py reads from there).

What it deliberately does NOT do — leave for the platform team:
  - AuthN/AuthZ — wrap in an API gateway.
  - Rate limiting — same.
  - Distributed tracing / structured logging — replace the JSONL append with
    your tracing library's hook.
  - Hot-reload on model update — restart the process; this is the cheapest
    correct behavior for a model that retrains quarterly.
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import CATEGORICAL_COLS, load_processed_featv2
from features import add_interactions
from models import model_path

# Operating threshold from Phase 5's cost-curve analysis (c_fn:c_fp = 5:1).
# Override via env var without code change.
DECISION_THRESHOLD = float(os.environ.get('DECISION_THRESHOLD', '0.13'))
MODEL_NAME = os.environ.get('MODEL_NAME', 'xgb_v4_interactions')

PROJECT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT / 'outputs' / 'logs'
LOG_PATH = LOG_DIR / 'predictions.jsonl'

logger = logging.getLogger('serve')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s | %(message)s')


# ---------- Pydantic schema ----------

class LoanApplication(BaseModel):
    """Prepared (post-Phase-3) features. Numeric NaN allowed where the pipeline
    expects to impute (`emp_length`, `dti`, `revol_util`, etc.)."""
    # Loan
    loan_amnt: float
    term: int = Field(..., description='Loan term in months (36 or 60)')
    int_rate: float
    installment: float
    # LC rating
    grade: str
    sub_grade: str
    # Borrower
    emp_length: Optional[float] = None
    emp_length_missing: int = 0
    home_ownership: str
    annual_inc: float
    verification_status: str
    # Loan context
    purpose: str
    addr_state: str
    application_type: str
    # Debt load
    dti: Optional[float] = None
    revol_util: Optional[float] = None
    revol_bal: float
    # Credit bureau
    fico_mean: float
    delinq_2yrs: Optional[float] = None
    pub_rec: Optional[float] = None
    pub_rec_bankruptcies: int = 0
    mort_acc: int = 0
    open_acc: Optional[float] = None
    total_acc: Optional[float] = None
    credit_history_years: Optional[int] = None


class PredictResponse(BaseModel):
    p_default: float
    decision: str   # 'approve' | 'reject'
    threshold: float
    model: str


class BatchRequest(BaseModel):
    applications: list[LoanApplication]


class BatchResponse(BaseModel):
    predictions: list[PredictResponse]


# ---------- Startup / state ----------

class ServiceState:
    model = None
    categories: dict[str, list] = {}
    feature_order: list[str] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = model_path(MODEL_NAME)
    logger.info(f'Loading model from {path}')
    ServiceState.model = joblib.load(path)

    # Pin categorical levels to training distribution so predict() doesn't blow
    # up on a new grade/state showing up in production. v4 was trained on the
    # featv2 parquets, so we read those (same rows + 7 interaction columns).
    train, _ = load_processed_featv2()
    ServiceState.categories = {c: list(train[c].cat.categories) for c in CATEGORICAL_COLS}
    ServiceState.feature_order = [c for c in train.columns if c not in ('default', 'issue_year')]
    logger.info(f'Ready. threshold={DECISION_THRESHOLD}  features={len(ServiceState.feature_order)}')

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    yield
    logger.info('Shutting down')


app = FastAPI(title='Credit Risk Prediction Service', lifespan=lifespan)


# ---------- Helpers ----------

def _to_dataframe(records: list[dict]) -> pd.DataFrame:
    """Build a DataFrame from validated records, with the same dtype shape the
    model saw at fit time (especially categorical levels). Computes the 7
    interaction features the API doesn't expose to callers — they're derived
    from the 25 base fields, so the request schema stays narrow."""
    df = pd.DataFrame(records)
    df = add_interactions(df)
    df = df.reindex(columns=ServiceState.feature_order)
    for c in CATEGORICAL_COLS:
        df[c] = pd.Categorical(df[c], categories=ServiceState.categories[c])
    return df


def _decision(p: float) -> str:
    return 'reject' if p >= DECISION_THRESHOLD else 'approve'


def _log_prediction(record: dict, p: float, decision: str) -> None:
    """Append-only JSONL log for offline monitoring."""
    entry = {
        'ts': datetime.now(timezone.utc).isoformat(),
        'p_default': p,
        'decision': decision,
        'threshold': DECISION_THRESHOLD,
        'model': MODEL_NAME,
        'features': record,
    }
    with LOG_PATH.open('a', encoding='utf-8') as f:
        f.write(json.dumps(entry, default=str) + '\n')


# ---------- Endpoints ----------

@app.get('/health')
def health():
    return {
        'status': 'ok' if ServiceState.model is not None else 'loading',
        'model': MODEL_NAME,
        'threshold': DECISION_THRESHOLD,
    }


@app.post('/predict', response_model=PredictResponse)
def predict(app_in: LoanApplication) -> PredictResponse:
    if ServiceState.model is None:
        raise HTTPException(503, 'model not loaded')
    record = app_in.model_dump()
    df = _to_dataframe([record])
    p = float(ServiceState.model.predict_proba(df)[0, 1])
    decision = _decision(p)
    _log_prediction(record, p, decision)
    return PredictResponse(
        p_default=p, decision=decision,
        threshold=DECISION_THRESHOLD, model=MODEL_NAME,
    )


@app.post('/predict/batch', response_model=BatchResponse)
def predict_batch(req: BatchRequest) -> BatchResponse:
    if ServiceState.model is None:
        raise HTTPException(503, 'model not loaded')
    if not req.applications:
        return BatchResponse(predictions=[])

    records = [a.model_dump() for a in req.applications]
    df = _to_dataframe(records)
    probs = ServiceState.model.predict_proba(df)[:, 1].astype(float)
    decisions = [_decision(p) for p in probs]
    for rec, p, d in zip(records, probs, decisions):
        _log_prediction(rec, float(p), d)
    return BatchResponse(predictions=[
        PredictResponse(p_default=float(p), decision=d,
                        threshold=DECISION_THRESHOLD, model=MODEL_NAME)
        for p, d in zip(probs, decisions)
    ])


if __name__ == '__main__':
    # python serve.py  ->  http://127.0.0.1:8000/docs
    import uvicorn
    uvicorn.run('serve:app', host='127.0.0.1', port=8000, reload=False)
