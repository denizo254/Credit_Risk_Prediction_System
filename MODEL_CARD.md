# Model Card — `xgb_v4_interactions`

Documentation for the production credit-risk classifier shipped in this repository. Follows the structure of [Mitchell et al., "Model Cards for Model Reporting" (FAT* 2019)](https://arxiv.org/abs/1810.03677).

## Model details

- **Model name:** `xgb_v4_interactions`
- **Model version:** v4 (current production)
- **Artifact:** `outputs/models/xgb_v4_interactions.joblib` (441 KB)
- **Model type:** Gradient-boosted decision tree (XGBoost) with post-hoc isotonic regression calibration
- **Architecture:** XGBoost classifier (`max_depth=4`, `n_estimators=200`, `learning_rate=0.1`, `min_child_weight=20`, `reg_lambda=5.0`, `gamma=0.5`) trained with `scale_pos_weight ≈ 3.9` to handle class imbalance, wrapped in a thin `CalibratedXGB` dataclass that applies `IsotonicRegression(out_of_bounds='clip')` to the raw probability output.
- **Inputs:** 25 application-time fields (loan terms, borrower characteristics, credit-bureau history). The service-side feature pipeline adds 7 deterministic interaction features (loan-to-income, FICO×DTI risk product, etc.) before scoring, so the model sees 32 features total.
- **Output:** Calibrated probability of default `P(default) ∈ [0, 1]` and a binary decision `{approve, reject}` against an operating threshold (`t = 0.13` by default).
- **License:** None declared. Contact the repository owner before commercial use.
- **Repository:** [`denizo254/Credit_Risk_Prediction_System`](https://github.com/denizo254/Credit_Risk_Prediction_System)

## Intended use

### Primary intended use
- **Decision support for unsecured personal-loan underwriting** on borrower profiles similar to LendingClub's 2007-2018 issuance.
- **Educational / portfolio demonstration** of an end-to-end CRISP-DM credit-risk pipeline.
- **Reference implementation** of time-aware splitting, isotonic calibration, cost-curve threshold selection, and PSI drift monitoring.

### Primary intended users
- ML practitioners studying credit-risk modeling
- Students of CRISP-DM working through the canonical phases on a real, sizable dataset

### Out-of-scope use
- **Production lending decisions at any regulated financial institution** without further work: this model has not been adversarial-tested for fair lending compliance (ECOA, Reg B, FCRA), has not been audited against Reg B / Disparate Impact analyses, and does not include reason codes / adverse action notices required by US lending law.
- **Borrower populations outside LendingClub's 2007-2018 footprint:** small-business loans, mortgages, auto loans, international markets, post-2018 macroeconomic regimes.
- **Scoring at any threshold below `t = 0.05` or above `t = 0.50`** without re-running the cost-curve analysis in `notebooks/05_evaluation.ipynb` against the actual business cost matrix.

## Factors

### Relevant factors
The training data spans the LendingClub origination years 2007-2018, with US-only retail borrowers. Performance is reported on a strict 2017-2018 out-of-time holdout to mirror the production use case ("score next year's loans").

### Evaluation factors
- **Time:** Per-year breakdowns for 2017 and 2018 (Phase 5)
- **LendingClub grade:** Per-grade KS for A through G (Phase 5)
- **No demographic factors** are evaluated in this card — race, gender, age are not present in the LendingClub dataset, but several available features (`addr_state`, `home_ownership`, `annual_inc`) are well-documented proxies. See [Ethical considerations](#ethical-considerations).

## Metrics

### Model performance measures
Reported on the 2017-2018 out-of-time test set (244,959 loans, base default rate 27.20%):

| Metric | Value | Interpretation |
|---|---|---|
| **ROC-AUC** | 0.7047 | Rank-ordering quality |
| **PR-AUC** | 0.4466 | Precision across recall levels (sensitive to imbalance) |
| **Brier score** | 0.1786 | Calibration + sharpness of probabilities |
| **Log-loss** | 0.5335 | Penalizes confidently wrong predictions |
| **KS statistic** | 0.2989 | Max separation of score CDFs (defaults vs repays) |
| **Decile-1 lift** | 1.94× | Top 10% of risk scores capture 19.4% of defaults |
| **PSI(train → test)** | 0.011 | Stable — score distribution does not drift |

### Decision threshold
- **Operating threshold:** `t = 0.13` (predict default iff `P(default) ≥ 0.13`).
- **Derivation:** argmin of an expected-cost curve over `t ∈ [0.01, 0.99]` with `cost(FN) : cost(FP) = 5 : 1`. The asymmetry reflects unsecured-lending economics — false approvals lose ~50% of principal, false rejections lose only the interest spread (~10% of principal).
- **At `t = 0.13` on the test set:** precision 32.8%, recall 91.0% — the model catches 91% of defaults at the cost of false-rejecting two thirds of those it flags. This trade-off is correct *for the stated 5:1 cost ratio* and must be re-derived if the business cost ratio changes.

### Approaches to uncertainty
Isotonic calibration on a held-out 2016 slice produces probabilities that approximately match observed default rates within their predicted bin (Phase 5 reliability diagrams). The model does *not* expose epistemic uncertainty (e.g., bootstrap variance, prediction intervals) — every score is a point estimate. Phase 7 random search measured CV-fold standard deviation of ~0.01 on ROC-AUC, so absolute test-set numbers should be read with that grain of salt.

## Evaluation data

### Datasets
- **Source:** LendingClub accepted-loans dataset `accepted_2007_to_2018Q4.csv` (~1.6 GB, ~2.26M rows × 151 columns).
- **Holdout:** 243,959 loans issued in 2017-2018, time-held-out (`issue_year ≥ 2017`).
- **Censored loans excluded:** 886,786 loans with `loan_status ∈ {Current, In Grace Period, Issued}` were dropped before evaluation since their outcome was unknown at dataset cutoff.

### Motivation
The 2017-2018 holdout is chosen because (a) it covers a full two-year cohort of distinct macroeconomic conditions versus training, (b) it includes the late-cycle default uptick observed in Phase 2, and (c) it is the realistic out-of-time window the production model would face. **Caveat:** 2018-issued loans have not had the full 36-60 month maturity period observed in the source data — observed default rate of 27% is therefore a lower bound on the true long-run default rate, biased downward by survivorship.

### Preprocessing
Each evaluation row goes through `prepare.clean()` (type coercion, semantic imputation of `mort_acc`/`pub_rec_bankruptcies` to zero, time-based feature engineering) and `features.add_interactions()` (7 underwriting interactions). No statistical imputation is applied — the model relies on XGBoost's native NaN handling for `emp_length`, `dti`, `revol_util`, etc.

## Training data

### Datasets
- **Base model:** 831,403 loans issued in 2007-2015 (`issue_year ≤ 2015`), 18.66% default rate.
- **Calibration slice:** 298,553 loans issued in 2016, 24.69% default rate. The base XGBoost never sees this slice — it is used exclusively to fit the isotonic regression layer.
- **Combined training:** 1,129,956 loans (2007-2016), 20.25% default rate.

### Demographics
- **Geography:** US states (50 + DC). State-level distribution mirrors LendingClub's historical origination footprint and is heavily concentrated in CA, TX, NY, FL.
- **Income:** Annual income median ≈ $65K (training set), with strong right skew.
- **FICO:** Range 612-847 mean. LendingClub pre-screens at ≥ 600, so very-low-FICO borrowers are absent.
- **Age, race, gender:** **Not present in the dataset.** This is a fundamental gap that prevents direct fair-lending testing.

## Quantitative analyses

### Per-year stability (Phase 5)

| Year | Loans | Default rate | ROC-AUC | KS |
|---|---|---|---|---|
| 2017 | 178,817 | 27.21% | 0.7010 | 0.2923 |
| 2018 | 65,142 | 27.17% | 0.6982 | 0.2918 |

Model performance is stable across the two-year test horizon (∆ ROC-AUC = 0.003).

### Per-grade discrimination (Phase 5)

| LC grade | Loans | Default rate | KS within grade |
|---|---|---|---|
| A | 40,985 | 9.24% | 0.220 |
| B | 66,384 | 20.08% | 0.175 |
| C | 76,156 | 31.04% | 0.176 |
| D | 38,355 | 39.54% | 0.157 |
| E | 14,845 | 45.02% | 0.142 |
| F | 4,761 | 50.87% | 0.134 |
| G | 2,473 | 54.19% | 0.139 |

The model adds genuine ranking signal *within every grade* (KS > 0 everywhere). Discrimination is strongest within grade A (KS 0.22) — i.e., the model is best at separating the rare bad apples in nominally-safe pools, which is operationally where lending decisions are most leveraged.

### Decile lift (Phase 7)

| Decile | Default rate | Lift | Cumulative captured |
|---|---|---|---|
| 1 (highest risk) | 52.4% | 1.94× | 19.4% |
| 2 | 43.1% | 1.59× | 35.0% |
| 3 | 37.5% | 1.39× | 48.9% |
| 4-7 (middle) | 20-32% | 0.7-1.2× | 87.3% |
| 8-10 (lowest risk) | 6-17% | 0.2-0.6× | 100.0% |

## Ethical considerations

### Known proxy variables for protected classes
Even though demographic variables are absent from the dataset, several model features are well-documented proxies for protected classes:

- **`addr_state`:** strongly correlated with racial composition (e.g., MS, LA, AL have very different demographic profiles than VT, NH, ID); using it as a feature is a *redlining risk* in the legal sense of disparate impact.
- **`home_ownership`:** racial and intergenerational wealth gaps make this a proxy for race and economic class.
- **`annual_inc`:** correlates strongly with race and gender in US labor-market data.
- **`emp_length`:** career interruptions correlate with gender (childcare) and disability.

The model has **not** been tested for disparate impact under standard fair-lending frameworks (4/5ths rule, AIR, fair-lending statistical tests). A production deployment in any regulated lending context would require this analysis as a precondition.

### Historical bias in training data
The training set reflects who LendingClub *accepted* into its loan pipeline from 2007-2018. Rejected applications are not in `accepted_2007_to_2018Q4.csv`; they live in a separate `rejected_*.csv` not used here. Selection bias from LendingClub's own underwriting model is therefore baked into both the input distribution and the labels.

### Decision automation risks
At `t = 0.13`, the model **rejects 67% of applications it flags as risky**. False rejections deny credit to borrowers who would have repaid. For any production use, this should be paired with:
- A human-in-the-loop review for borderline scores (e.g., `0.10 ≤ P ≤ 0.20`).
- Adverse action notices with reason codes (required by US Reg B for any credit denial).
- A periodic audit of approval/rejection rates broken down by available proxy variables.

## Caveats and recommendations

### Caveats
1. **Concept drift between training (≤2016) and test (2017-2018) is real.** Phase 7 hyperparameter tuning improved cross-validated ROC-AUC by +0.023, but only +0.004 on the time-held-out test set — the gap is drift the in-distribution search cannot close.
2. **Survivorship bias on 2018 loans.** Loans issued in 2018 had not reached the 36-60 month maturity window when the dataset was published. The observed 27% test default rate is a lower bound.
3. **Decile-1 lift of 1.94× is operationally useful, not industry-leading.** A well-tuned commercial credit-risk model would target ≥ 3×. Phase 8 interaction-feature engineering moved this only +0.017×, suggesting the gap is structural (drift, missing macro signals) rather than capacity-bound.
4. **Class weighting de-calibrates the base XGBoost.** The isotonic layer restores calibration on the 2016 slice, but the recovery is only as good as the calibration slice resembling production. A material shift in applicant mix would require recalibration before deployment.
5. **No bureau-attribute updates.** The model assumes the borrower's bureau snapshot at application time is current. In production, the input pipeline must enforce this.

### Recommendations for further work
- **Rolling recalibration in production.** Refit only the isotonic layer on rolling-12-month labeled outcomes. Cheapest fix; most responsive to actual drift.
- **Sub-grade-conditional specialists.** Fit small models per grade band (A-B, C-D, E-G), route by grade at score time. Per-grade KS variability (0.13 → 0.22) suggests different bands respond to different signals.
- **External macro features.** Unemployment rate, prime-rate at issue date, regional economic indicators — these would directly address the concept drift the in-distribution model cannot internally see.
- **Fair-lending audit.** Before any regulated-context deployment, run disparate-impact analyses using available proxy variables and any matched demographic dataset.

### Monitoring requirements
The production deployment via `src/serve.py` logs every prediction to `outputs/logs/predictions.jsonl`. `src/monitor.py` computes PSI between production scores and the training reference. Suggested SLAs:
- **PSI < 0.10:** stable, no action.
- **0.10 ≤ PSI < 0.25:** investigate; check feature-level drift.
- **PSI ≥ 0.25 sustained for 7+ days:** trigger model refresh (recalibrate, then retrain if needed).

PSI computation is suppressed below 1,000 logged predictions to avoid empty-bin blow-up in the log-ratio.
