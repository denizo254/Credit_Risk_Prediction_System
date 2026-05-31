# Credit Risk Prediction System

Binary default classifier on the LendingClub accepted-loans dataset (2007-2018). Built end-to-end across the CRISP-DM phases: data understanding, preparation, modeling, evaluation, deployment, hyperparameter tuning, and feature engineering.

Production model is **`xgb_v4_interactions`** — XGBoost with isotonic calibration and 7 underwriting-style interaction features. On the time-held-out 2017-2018 test set (244K loans): **ROC-AUC 0.7047, KS 0.2989, decile-1 lift 1.94×**. Operating decision threshold `t = 0.13` is the argmin of an asymmetric cost curve (FN : FP = 5 : 1).

## Layout

```
src/                   Importable modules + smoke tests (one per phase)
  load.py              Raw CSV loader, target derivation
  prepare.py           clean(), time_split(), featv2 paths
  features.py          add_interactions() — 7 interaction features
  models.py            build_lr(), build_xgb(), evaluate(), Metrics
  evaluate.py          calibration, cost_curve, gains, PSI, per-group metrics
  tune.py              random search with time-series CV
  serve.py             FastAPI app: /health, /predict, /predict/batch
  score_batch.py       CLI batch scorer
  monitor.py           prediction-log analyzer (PSI / reject-rate / Brier)
  _smoke_phase{2..8}.py  end-to-end runners for each phase
notebooks/             One notebook per CRISP-DM phase, with rationale
app/
  streamlit_app.py     Demo UI — single-application scoring with sidebar threshold control
outputs/
  models/              Trained joblib artifacts (committed)
  figures/             Saved plots from notebook runs
  logs/                Runtime prediction logs (gitignored)
data/
  interim/             Phase-2 curated parquet (gitignored, regenerable)
  processed/           Phase-3+ train/test parquets (gitignored, regenerable)
club loan data/        Raw 1.6 GB CSV from LendingClub (gitignored, fetch separately)
```

## Notebooks

| Notebook | Phase | What it covers |
|---|---|---|
| [`02_data_understanding.ipynb`](notebooks/02_data_understanding.ipynb) | 2 | Curated 26-feature load, target derivation, missingness profile, class imbalance, default rate by grade and by year |
| [`03_data_preparation.ipynb`](notebooks/03_data_preparation.ipynb) | 3 | Type coercion, engineered features, semantic vs statistical imputation, time-based train/test split |
| [`04_modeling.ipynb`](notebooks/04_modeling.ipynb) | 4 | LR baseline + XGBoost v1; ROC / PR / calibration / score-distribution / feature-importance plots |
| [`05_evaluation.ipynb`](notebooks/05_evaluation.ipynb) | 5 | Isotonic calibration on held-out 2016, cost-curve threshold selection, per-year & per-grade stability, gains/lift, PSI |
| [`06_deployment.ipynb`](notebooks/06_deployment.ipynb) | 6 | FastAPI service architecture, batch scoring CLI, monitoring scaffolding, operational playbook |
| [`07_hyperparameter_tuning.ipynb`](notebooks/07_hyperparameter_tuning.ipynb) | 7 | Random search with time-series CV; Phase-4 defaults ranked 15/15; CV → test gap |
| [`08_interaction_features.ipynb`](notebooks/08_interaction_features.ipynb) | 8 | 7 underwriting-style interactions; `int_rate_x_term` becomes the #1 feature by gain |

Each notebook calls into the shared `src/` modules — the notebook explains *why*, the module is the *what*. Re-running a notebook end-to-end uses the corresponding `src/_smoke_phase{N}.py` for the heavy lifting.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # PowerShell:  .venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

The raw `accepted_2007_to_2018Q4.csv` (~1.6 GB) is **not** in the repo — obtain it from the LendingClub data source and place it at:

```
club loan data/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv
```

(Yes, the nested-folder shape is intentional — that's how the upstream archive unpacks.)

## Regenerating from scratch

Each phase has a smoke test that produces the artifacts the next phase reads. Run in order:

```bash
python src/_smoke_phase2.py   # raw CSV -> data/interim/accepted_curated.parquet      (~5 min)
python src/_smoke_phase3.py   # cleaning + time split -> data/processed/{train,test}.parquet
python src/_smoke_phase4.py   # LR + XGB v1                                            (~3 min)
python src/_smoke_phase5.py   # xgb_v2_calibrated + Phase 5 metrics                    (~2 min)
python src/_smoke_phase6.py   # exercise FastAPI + CLI + monitor                       (~1 min)
python src/_smoke_phase7.py   # 15-trial random search -> xgb_v3_tuned                 (~7 min)
python src/_smoke_phase8.py   # interactions + xgb_v4_interactions                     (~1 min)
```

You can also just load the pre-trained models from `outputs/models/` and skip retraining.

## Streamlit demo

A one-click demo UI for the production model — useful for showing recruiters or stakeholders what the model does on a single application:

```bash
streamlit run app/streamlit_app.py
# -> http://localhost:8501
```

Fill in 25 fields (sensible medians pre-filled), hit **Score**, see P(default), the threshold-based decision, the risk band, and how the model's score compares to (a) the LendingClub-published default rate for that grade and (b) the test-set base rate of 27.2%. Threshold is adjustable in the sidebar to let you watch decisions shift in real time.

The demo is the *same* model artifact and feature pipeline as the FastAPI service — it's not a separate model, just a different surface.

## Running the service

```bash
python src/serve.py
# -> http://127.0.0.1:8000/docs  (interactive OpenAPI)
```

Single-loan scoring:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{ "loan_amnt": 12000, "term": 36, "int_rate": 11.5, ...25 fields... }'
```

The API accepts the 25 base fields (post-Phase-3 cleaned shape). The 7 interactions are computed internally inside the handler, so the request schema stays narrow.

**Reason codes (explainability).** Add `?explain=true` to either `/predict` or `/predict/batch` to get the top features driving each application's risk, as exact TreeSHAP contributions (log-odds; positive = toward default):

```bash
curl -X POST 'http://127.0.0.1:8000/predict?explain=true' \
  -H 'Content-Type: application/json' -d '{ ...25 fields... }'
# response adds:  "reasons": [ {"feature": "int_rate_x_term", "label": "Total interest exposure (rate × term)", "value": 1847.4, "contribution": 0.5247}, ... ]
```

Reason codes are opt-in so the default response stays lean. Count is configurable via `TOP_N_REASONS` (default 5). These are a technical basis for Reg B adverse-action notices, not a compliant notice on their own — see `MODEL_CARD.md`.

Override defaults at runtime:

```bash
MODEL_NAME=xgb_v3_tuned DECISION_THRESHOLD=0.15 python src/serve.py
```

## Batch scoring

```bash
python src/score_batch.py \
  --input  data/processed/test.parquet \
  --output outputs/scored_test.parquet
```

Works on either v1-feature or v2-feature input parquets (missing interactions are computed automatically).

## Monitoring

`serve.py` appends every prediction to `outputs/logs/predictions.jsonl`. Get a report any time:

```bash
python src/monitor.py
# or, with a window:
python src/monitor.py --since 2026-05-01
```

The PSI computation is suppressed below 1,000 logged predictions to avoid empty-bin blow-up.

## Model lineage

| Model | Features | Calibrated | Test ROC-AUC | Test KS | Notes |
|---|---|---|---|---|---|
| `lr_baseline_v1` | v1 (25) | no | 0.7000 | 0.2907 | Phase-4 baseline; class-balanced LR |
| `xgb_v1` | v1 (25) | no | 0.7091 | 0.3035 | Phase-4 XGBoost defaults |
| `xgb_v2_calibrated` | v1 (25) | yes (2016) | 0.7000 | 0.2907 | Phase-5 isotonic on held-out 2016 |
| `xgb_v3_tuned` | v1 (25) | yes | 0.7040 | 0.2975 | Phase-7 random search; depth-4, 200 trees |
| **`xgb_v4_interactions`** | **featv2 (32)** | **yes** | **0.7047** | **0.2989** | **Production** — v3 hyperparams + 7 interactions |

Full intended-use, limitations, and fair-lending considerations live in [`MODEL_CARD.md`](MODEL_CARD.md). Phase-by-phase commentary lives in `notebooks/02_..` through `notebooks/08_..`.

## Design choices worth knowing

- **Time-aware everything.** Train ≤ 2016, test ≥ 2017. CV inside training uses `TimeSeriesSplit`, never random k-fold. The production use is "score next year's loans" — random splits would smuggle future-cycle data into training.
- **Isotonic calibration on a held-out 2016 slice.** Base XGB trains on 2007-2015; isotonic fits on 2016. The base never sees the calibration data, so there's no probability-leakage.
- **Cost-curve threshold, not 0.5.** The asymmetric cost matrix (FN : FP = 5 : 1) puts the optimum at `t = 0.13` — at this threshold, the model catches 91% of defaults at the price of false-rejecting 67% of those it flags.
- **Statistical imputation lives inside the sklearn `Pipeline`.** `prepare.py` only does *semantic* imputation (`mort_acc.fillna(0)`) where zero is what the column literally measures when missing. Median imputation for `emp_length` / `dti` / `revol_util` is deferred to the model pipeline so medians are computed per CV fold, not globally.

## Honest caveats

1. **Test-set ceiling is real.** Phase 7 tuning improved CV ROC-AUC by +0.023 but only +0.004 on test. Phase 8 interactions added another +0.001. The gap is concept drift between 2007-2016 training and 2017-2018 test — no amount of training-distribution work closes it.
2. **2018 horizon is survivorship-biased.** 2018 loans haven't had 60 months to mature; observed default rate of 27% is a lower bound. Real long-run performance will be worse.
3. **Decile-1 lift of 1.94× is operationally useful, not best-in-class.** A specialist credit-risk team would expect 3×+. The unclosed gap is in *features that scale with the test horizon*, not more tuning.
4. **Calibrated probabilities run low out-of-time.** PSI(train→test) is a stable 0.011 — but that only says the score *shape* holds steady, not that the probabilities are right. The 2017-2018 default rate (27.2%) tops the 2016 calibration year (24.7%), so the isotonic layer under-predicts on test by ~2 pp (mean predicted ≈25% vs observed 27%). Rank-ordering is unaffected; the *levels* need rolling recalibration (see below). Don't read a low PSI as "probabilities are accurate."

## Where the next gain actually lives

1. **Rolling recalibration.** Once production accumulates a year of labeled outcomes, refit just the isotonic on rolling-12-month data. The base XGB stays the same; the calibrator adapts to drift. Cheapest fix; most responsive to the actual problem.
2. **Sub-grade-conditional specialists.** Per-grade KS in Phase 5 varied 0.13 → 0.22. Fit small specialists per grade band (A-B, C-D, E-G), route at score time. Each specialist sees a narrower default-rate range.
3. **External signals.** Macro indicators (unemployment, prime rate at issue date) — these are exactly the variables driving the 2017-2018 drift the model can't internally see.

## License

No license declared. If you want to use this commercially, contact the repository owner first.
