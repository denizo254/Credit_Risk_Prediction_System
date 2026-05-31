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
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explain import reason_codes, reason_codes_batch
from features import add_interactions
from models import model_path
from prepare import CATEGORICAL_COLS, load_processed_featv2

# Operating threshold from Phase 5's cost-curve analysis (c_fn:c_fp = 5:1).
# Override via env var without code change.
DECISION_THRESHOLD = float(os.environ.get('DECISION_THRESHOLD', '0.13'))
MODEL_NAME = os.environ.get('MODEL_NAME', 'xgb_v4_interactions')
# Number of reason codes returned when a request asks for ?explain=true.
TOP_N_REASONS = int(os.environ.get('TOP_N_REASONS', '5'))

PROJECT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT / 'outputs' / 'logs'
LOG_PATH = LOG_DIR / 'predictions.jsonl'

logger = logging.getLogger('serve')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s | %(message)s')


# ---------- Pydantic schema ----------

class LoanApplication(BaseModel):
    """Prepared (post-Phase-3) features for a single application.

    Bounds reject obviously-malformed input (negatives, absurd magnitudes)
    before it reaches the model — without that, a typo like a negative income
    or a 9999% rate scores silently as garbage. Ranges are intentionally
    *permissive* (wider than the training data) so plausible real applications
    are never rejected; the goal is to catch nonsense, not to re-underwrite.

    Optional numeric fields accept None (the pipeline / XGBoost impute the
    missing value); when a value IS supplied it must satisfy the bound.
    Categorical strings are validated against the training distribution at
    score time — unknown levels pass through as NaN (XGBoost handles them).
    """
    # extra='forbid' turns an unexpected/misspelled field into a 422 instead of
    # silently ignoring it (which would score on defaults for the real field).
    model_config = ConfigDict(extra='forbid')

    # Optional caller-supplied id. Not a model feature (dropped before scoring);
    # logged so realized outcomes can later be joined for performance monitoring.
    application_id: str | None = Field(default=None, max_length=64)

    # Loan
    loan_amnt: float = Field(gt=0, le=100_000)
    term: Literal[36, 60] = Field(description='Loan term in months (36 or 60)')
    int_rate: float = Field(gt=0, le=100, description='Annual rate in percent')
    installment: float = Field(gt=0, le=10_000)
    # LC rating
    grade: str = Field(min_length=1, max_length=2)
    sub_grade: str = Field(min_length=1, max_length=3)
    # Borrower
    emp_length: float | None = Field(default=None, ge=0, le=10)
    emp_length_missing: Literal[0, 1] = 0
    home_ownership: str = Field(min_length=1, max_length=20)
    annual_inc: float = Field(ge=0, le=100_000_000)
    verification_status: str = Field(min_length=1, max_length=30)
    # Loan context
    purpose: str = Field(min_length=1, max_length=40)
    addr_state: str = Field(min_length=2, max_length=2, description='US state code')
    application_type: str = Field(min_length=1, max_length=20)
    # Debt load
    # LendingClub uses -1 as a sentinel for dti, so the floor allows it through.
    dti: float | None = Field(default=None, ge=-1, le=1000)
    revol_util: float | None = Field(default=None, ge=0, le=1000)
    revol_bal: float = Field(ge=0, le=100_000_000)
    # Credit bureau
    fico_mean: float = Field(ge=300, le=850, description='FICO score (300-850)')
    delinq_2yrs: float | None = Field(default=None, ge=0, le=100)
    pub_rec: float | None = Field(default=None, ge=0, le=100)
    pub_rec_bankruptcies: int = Field(default=0, ge=0, le=100)
    mort_acc: int = Field(default=0, ge=0, le=100)
    open_acc: float | None = Field(default=None, ge=0, le=200)
    total_acc: float | None = Field(default=None, ge=0, le=200)
    credit_history_years: int | None = Field(default=None, ge=0, le=100)


class ReasonCode(BaseModel):
    """One feature's contribution to this application's risk score (TreeSHAP)."""
    feature: str
    label: str
    value: float | str | None
    contribution: float   # log-odds; positive pushes toward default


class PredictResponse(BaseModel):
    p_default: float
    decision: str   # 'approve' | 'reject'
    threshold: float
    model: str
    # Populated only when the request is made with ?explain=true. Lists the
    # top features pushing this application toward default (Reg B reason codes).
    reasons: list[ReasonCode] | None = None


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


def _to_reason_codes(contributions) -> list[ReasonCode]:
    """Convert explain.Contribution objects into the API response model."""
    return [
        ReasonCode(feature=c.feature, label=c.label, value=c.value,
                   contribution=round(c.contribution, 4))
        for c in contributions
    ]


def _log_prediction(record: dict, p: float, decision: str) -> None:
    """Append-only JSONL log for offline monitoring."""
    entry = {
        'ts': datetime.now(UTC).isoformat(),
        'application_id': record.get('application_id'),
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
def predict(app_in: LoanApplication, explain: bool = False) -> PredictResponse:
    if ServiceState.model is None:
        raise HTTPException(503, 'model not loaded')
    record = app_in.model_dump()
    df = _to_dataframe([record])
    p = float(ServiceState.model.predict_proba(df)[0, 1])
    decision = _decision(p)
    _log_prediction(record, p, decision)
    reasons = None
    if explain:
        reasons = _to_reason_codes(
            reason_codes(ServiceState.model, df, row=0, top_n=TOP_N_REASONS)
        )
    return PredictResponse(
        p_default=p, decision=decision,
        threshold=DECISION_THRESHOLD, model=MODEL_NAME, reasons=reasons,
    )


@app.post('/predict/batch', response_model=BatchResponse)
def predict_batch(req: BatchRequest, explain: bool = False) -> BatchResponse:
    if ServiceState.model is None:
        raise HTTPException(503, 'model not loaded')
    if not req.applications:
        return BatchResponse(predictions=[])

    records = [a.model_dump() for a in req.applications]
    df = _to_dataframe(records)
    probs = ServiceState.model.predict_proba(df)[:, 1].astype(float)
    decisions = [_decision(p) for p in probs]
    for rec, p, d in zip(records, probs, decisions, strict=False):
        _log_prediction(rec, float(p), d)

    # One SHAP pass over the whole batch (not per-row) when explanations asked.
    reasons = (
        [_to_reason_codes(r)
         for r in reason_codes_batch(ServiceState.model, df, top_n=TOP_N_REASONS)]
        if explain else [None] * len(probs)
    )
    return BatchResponse(predictions=[
        PredictResponse(p_default=float(p), decision=d,
                        threshold=DECISION_THRESHOLD, model=MODEL_NAME, reasons=rc)
        for p, d, rc in zip(probs, decisions, reasons, strict=False)
    ])


if __name__ == '__main__':
    # python serve.py  ->  http://127.0.0.1:8000/docs
    import uvicorn
    uvicorn.run('serve:app', host='127.0.0.1', port=8000, reload=False)
