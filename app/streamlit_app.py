"""Streamlit demo for the production credit-risk model.

Run from the project root:
    streamlit run app/streamlit_app.py

This is a *demo* surface, not the production scoring service. For real
applications the FastAPI app in `src/serve.py` is what you'd deploy behind
an API gateway. Streamlit is here to give a one-click, recruiter-friendly
look at what the model does on a single application.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / 'src'))

from explain import reason_codes  # noqa: E402
from features import add_interactions  # noqa: E402
from models import model_path  # noqa: E402
from prepare import CATEGORICAL_COLS, load_processed_featv2  # noqa: E402

# LendingClub-published default rates by grade (Phase 2 finding) — for the
# "vs LC grade expectation" context display.
LC_DEFAULT_RATE_BY_GRADE = {
    'A': 0.0670, 'B': 0.1465, 'C': 0.2423, 'D': 0.3230,
    'E': 0.4007, 'F': 0.4642, 'G': 0.5111,
}
TEST_BASE_RATE = 0.272


@st.cache_resource
def _load_state(model_name: str = 'xgb_v4_interactions'):
    model = joblib.load(model_path(model_name))
    train, _ = load_processed_featv2()
    categories = {c: list(train[c].cat.categories) for c in CATEGORICAL_COLS}
    feature_order = [c for c in train.columns if c not in ('default', 'issue_year')]
    return model, categories, feature_order


def _build_record(values: dict) -> pd.DataFrame:
    """Construct the model-ready 32-column DataFrame from form values."""
    df = pd.DataFrame([values])
    df = add_interactions(df)
    df = df.reindex(columns=st.session_state.feature_order)
    for c in CATEGORICAL_COLS:
        df[c] = pd.Categorical(df[c], categories=st.session_state.categories[c])
    return df


def _risk_band(p: float) -> str:
    if p < 0.10:
        return 'Low risk'
    if p < 0.25:
        return 'Moderate risk'
    if p < 0.50:
        return 'Elevated risk'
    return 'High risk'


# ---------- Page setup ----------

st.set_page_config(
    page_title='Credit Risk Demo',
    layout='wide',
    initial_sidebar_state='expanded',
)

model, categories, feature_order = _load_state()
st.session_state.categories = categories
st.session_state.feature_order = feature_order

# ---------- Sidebar ----------

with st.sidebar:
    st.header('About this model')
    st.markdown(
        """
        **`xgb_v4_interactions`** — XGBoost (depth-4, 200 trees) with isotonic
        calibration on a held-out 2016 slice and 7 underwriting interaction
        features (loan-to-income, FICO×DTI risk product, etc).

        - Trained on 1.13M LendingClub loans (2007-2016)
        - Tested on 244K loans (2017-2018, out-of-time)
        - Test ROC-AUC **0.7047** · KS **0.2989**
        - PSI(train → test) = 0.011 — stable

        Threshold `t = 0.13` is the argmin of an expected-cost curve with
        cost(FN) : cost(FP) = 5 : 1.
        """
    )
    threshold = st.slider('Decision threshold', 0.05, 0.50, 0.13, 0.01)
    st.caption(
        'Lower threshold → reject more loans, catch more defaults. '
        'Higher threshold → approve more loans, miss more defaults.'
    )

# ---------- Main ----------

st.title('Credit Risk Prediction')
st.markdown(
    'Enter a loan application below. The model returns the probability of '
    'default and a decision against the operating threshold.'
)

with st.form('application'):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('**Loan**')
        loan_amnt = st.number_input('Loan amount ($)', 1000, 40000, 12000, 500)
        term = st.selectbox('Term (months)', [36, 60])
        int_rate = st.slider('Interest rate (%)', 5.0, 30.0, 11.5, 0.1)
        installment = st.number_input('Monthly installment ($)', 50, 1500, 400, 25)
        grade = st.selectbox('LC grade', categories['grade'], index=2)
        sub_grade = st.selectbox('LC sub-grade', categories['sub_grade'], index=12)
        purpose = st.selectbox('Loan purpose', categories['purpose'])

    with col2:
        st.markdown('**Borrower**')
        emp_length = st.slider('Employment length (years)', 0, 10, 5, 1)
        home_ownership = st.selectbox('Home ownership', categories['home_ownership'])
        annual_inc = st.number_input('Annual income ($)', 10_000, 500_000, 65_000, 5000)
        verification_status = st.selectbox('Income verification', categories['verification_status'])
        application_type = st.selectbox('Application type', categories['application_type'])
        addr_state = st.selectbox('State', categories['addr_state'], index=4)
        dti = st.slider('DTI (%)', 0.0, 60.0, 18.0, 0.5)

    with col3:
        st.markdown('**Credit history**')
        fico_mean = st.slider('FICO score', 620, 850, 700, 1)
        credit_history_years = st.slider('Credit history (years)', 1, 50, 15, 1)
        open_acc = st.slider('Open accounts', 1, 50, 10, 1)
        total_acc = st.slider('Total accounts', 1, 100, 25, 1)
        mort_acc = st.slider('Mortgage accounts', 0, 10, 1, 1)
        revol_bal = st.number_input('Revolving balance ($)', 0, 100_000, 8000, 500)
        revol_util = st.slider('Revolving utilization (%)', 0.0, 150.0, 35.0, 0.5)
        delinq_2yrs = st.slider('Delinquencies (2yr)', 0, 10, 0, 1)
        pub_rec = st.slider('Public records', 0, 10, 0, 1)
        pub_rec_bankruptcies = st.slider('Bankruptcies', 0, 5, 0, 1)

    submitted = st.form_submit_button('Score this application', type='primary')

if submitted:
    values = dict(
        loan_amnt=loan_amnt, term=term, int_rate=int_rate, installment=installment,
        grade=grade, sub_grade=sub_grade,
        emp_length=emp_length, emp_length_missing=0,
        home_ownership=home_ownership, annual_inc=annual_inc,
        verification_status=verification_status,
        purpose=purpose, addr_state=addr_state, application_type=application_type,
        dti=dti, revol_util=revol_util, revol_bal=revol_bal,
        fico_mean=fico_mean,
        delinq_2yrs=delinq_2yrs, pub_rec=pub_rec, pub_rec_bankruptcies=pub_rec_bankruptcies,
        mort_acc=mort_acc, open_acc=open_acc, total_acc=total_acc,
        credit_history_years=credit_history_years,
    )

    X = _build_record(values)
    p_default = float(model.predict_proba(X)[0, 1])
    decision = 'REJECT' if p_default >= threshold else 'APPROVE'
    band = _risk_band(p_default)

    st.markdown('---')
    st.subheader('Result')

    r1, r2, r3 = st.columns(3)
    r1.metric('Probability of default', f'{p_default*100:.2f}%')
    r2.metric('Decision', decision, delta=f'threshold t = {threshold:.2f}',
              delta_color='off')
    r3.metric('Risk band', band)

    # Contextual comparison: model vs. LC's grade-implied default rate.
    lc_expected = LC_DEFAULT_RATE_BY_GRADE.get(grade, TEST_BASE_RATE)
    delta_vs_grade = p_default - lc_expected
    delta_vs_base = p_default - TEST_BASE_RATE

    c1, c2 = st.columns(2)
    c1.metric(
        f'vs LC grade {grade} base default rate',
        f'{lc_expected*100:.2f}%',
        delta=f'{delta_vs_grade*100:+.2f} pp',
        delta_color='inverse',
    )
    c2.metric(
        'vs population base rate (test set)',
        f'{TEST_BASE_RATE*100:.2f}%',
        delta=f'{delta_vs_base*100:+.2f} pp',
        delta_color='inverse',
    )

    if abs(delta_vs_grade) < 0.03:
        st.info(
            f'Model agrees with LendingClub\'s own grade-{grade} pricing within '
            f'{abs(delta_vs_grade)*100:.1f} pp. No internal disagreement.'
        )
    elif delta_vs_grade > 0:
        st.warning(
            f'Model is **{delta_vs_grade*100:+.1f} pp** more pessimistic than '
            f'LendingClub\'s grade-{grade} cohort average. The model sees risk '
            f'in this application that grade alone misses.'
        )
    else:
        st.success(
            f'Model is **{delta_vs_grade*100:+.1f} pp** more optimistic than '
            f'LendingClub\'s grade-{grade} cohort average. The model thinks this '
            f'is a better-than-typical grade-{grade} applicant.'
        )

    # ---------- Why this score? (TreeSHAP reason codes) ----------
    st.markdown('---')
    st.subheader('Why this score?')
    st.caption(
        'Top features driving this application, as exact TreeSHAP contributions '
        '(log-odds). Bars to the **right** push toward default; to the **left**, '
        'toward repayment.'
    )

    contribs = reason_codes(model, X, row=0, top_n=99, positive_only=False)
    top = sorted(contribs, key=lambda c: abs(c.contribution), reverse=True)[:8]
    chart_df = (
        pd.DataFrame({
            'feature': [c.label for c in top],
            'contribution (log-odds)': [c.contribution for c in top],
        })
        .set_index('feature')
        .sort_values('contribution (log-odds)')
    )
    st.bar_chart(chart_df, horizontal=True)

    with st.expander('Reason codes (table)'):
        st.dataframe(
            pd.DataFrame([
                # value -> string so the column has one Arrow-friendly dtype
                # (it mixes numbers and category codes like 'C3').
                {'feature': c.label,
                 'value': '—' if c.value is None else str(c.value),
                 'contribution': round(c.contribution, 4)}
                for c in top
            ]),
            hide_index=True,
        )

    with st.expander('Show the 32-feature row sent to the model'):
        st.dataframe(X.T.rename(columns={0: 'value'}))

st.markdown('---')
st.caption(
    'Model artifact: `outputs/models/xgb_v4_interactions.joblib`. '
    'Source: `src/serve.py` is the production scoring service. '
    'This demo uses the same model and feature pipeline.'
)
