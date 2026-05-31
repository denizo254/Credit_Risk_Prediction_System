# CLAUDE.md

Guidance for working in this repo. Keep it current when modules or conventions change.

## What this is

A binary **credit-default classifier** on the LendingClub accepted-loans dataset (2007–2018),
built end-to-end across the CRISP-DM phases. Production model: **`xgb_v4_interactions`** —
XGBoost + isotonic calibration + 7 underwriting interaction features. Operating threshold
`t = 0.13` (argmin of a 5:1 FN:FP cost curve). See `README.md` and `MODEL_CARD.md` for the
modeling story; this file is the engineering map.

## Environment & commands

Windows + PowerShell. Local venv at `.venv` (Python 3.13). Always call it explicitly:

```powershell
& ".\.venv\Scripts\ruff.exe" check .              # lint (config: ruff.toml)
& ".\.venv\Scripts\python.exe" -m pytest -q       # unit tests (tests/) — fast, data-free
& ".\.venv\Scripts\python.exe" src\serve.py       # FastAPI service -> :8000/docs
streamlit run app\streamlit_app.py                # demo UI
```

- **Exact reproducible env:** `pip install -r requirements.lock`. Loose runtime pins in
  `requirements.txt`; dev tools (pytest, ruff) in `requirements-dev.txt`.
- **CI** (`.github/workflows/ci.yml`) runs ruff + pytest on every push/PR. It is **data-free**
  by design — needs neither the 1.6 GB dataset nor model artifacts. Keep it that way.
- **Smoke tests** `src/_smoke_phase{2..8}.py` are heavy integration runners (retrain on
  ~0.8–1.1M rows, minutes each) and need the dataset/parquets. They are NOT unit tests.

## Module map (`src/`)

| Module | Role |
|---|---|
| `load.py` | Raw CSV paths, `FEATURE_COLS`, `derive_default_flag()` (the label definition) |
| `contracts.py` | `validate_curated()` — schema contract enforced at the top of `clean()` |
| `prepare.py` | `clean()`, `time_split()` (train ≤2016 / test ≥2017), `load_processed[_featv2]()` |
| `features.py` | `add_interactions()` — the 7 `INTERACTION_COLS` |
| `models.py` | `build_lr()`, `build_xgb()` (`enable_categorical`), `evaluate()`, `ks_statistic()`, `split_xy()`, `model_path()` |
| `tune.py` | Random search with `TimeSeriesSplit` |
| `evaluate.py` | `CalibratedXGB`, `fit_calibrated()`, `cost_curve()`, `gains_table()`, `psi()`, `metrics_by_group()` |
| `explain.py` | TreeSHAP reason codes (`reason_codes()`, `reason_codes_batch()`) — native xgboost `pred_contribs` |
| `serve.py` | FastAPI: `/health`, `/predict`, `/predict/batch` (opt-in `?explain=true`) |
| `score_batch.py` | CLI batch scorer; fails loud on missing input columns |
| `monitor.py` | `predictions.jsonl` analyzer: PSI + realized Brier/AUC/KS via `--truth` ground-truth join |
| `recalibrate.py` | Rolling isotonic recalibration on a recent labeled window (base frozen) |

Operational CLIs: `score_batch.py`, `monitor.py [--truth outcomes.csv]`, `recalibrate.py
--calib <featv2.parquet> --out <name>`. Serving overrides via env: `MODEL_NAME`,
`DECISION_THRESHOLD`, `TOP_N_REASONS`.

## Data shapes

- **Curated** (`data/interim/accepted_curated.parquet`): 29 columns, input to `clean()`
  (validated by `contracts.validate_curated`).
- **Processed v1** (`data/processed/{train,test}.parquet`): 25 features + `default` + `issue_year`.
- **featv2** (`*_featv2.parquet`): v1 + 7 interactions = 32 features. v4 trains/serves on these.
- `issue_year` is split metadata (dropped by `split_xy`, never a feature). `application_id` is
  optional request metadata, logged for monitoring, dropped before scoring.

## Conventions / design discipline (don't violate these)

- **Time-aware everything.** Out-of-time split (train ≤2016, test ≥2017); CV uses
  `TimeSeriesSplit`, never random k-fold. The use case is "score next year's loans."
- **Statistical transforms stay inside the sklearn `Pipeline`** (median impute, scale) so they
  fit per CV fold. `prepare.clean()` does only *semantic* imputation (`mort_acc.fillna(0)`).
- **Model builders never `.fit()`** — the caller controls fitting, so the same code runs in CV,
  notebooks, smoke tests, and serving.
- **Flat imports.** Modules do `from prepare import ...` (not package-relative). `src/` is put
  on `sys.path` by entrypoints and by `tests/conftest.py`.
- **Lean dependencies.** Explainability uses xgboost-native TreeSHAP (no `shap`); the data
  contract is hand-rolled (no `pandera`). Prefer this over heavy deps.
- **Categorical pinning.** Serving/scoring pins category levels to the training distribution so
  unknown grades/states become NaN (XGBoost handles them) rather than crashing.

## Testing conventions

- Tests live in `tests/`, mirror modules (`test_<module>.py`), and must be **data-free**: pure
  functions, synthetic frames, or a tiny in-test XGBoost model. Never depend on the dataset or
  committed artifacts (CI has neither). ~92 tests today.
- Cover the math/contract surface (KS, PSI, cost curve, SHAP additivity, schema, join logic).
  Model *builders* are left to the smoke tests.

## Gotchas

- **PowerShell mangles `git commit -m` messages with embedded quotes.** For multi-line/quoted
  messages, write to a temp file and `git commit -F <file>`.
- Committed `.joblib` artifacts are version-sensitive (trained on sklearn 1.8 / xgboost 3.2 — see
  `requirements.lock`). Loading under other versions may warn.
- Git: branch off `main`, fast-forward merge, then push. `.gitattributes` normalizes line endings.
