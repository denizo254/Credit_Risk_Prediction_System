"""Validation tests for serve.LoanApplication.

These exercise the request schema only — no model artifact, no dataset, no
server — so they run fast in CI. The point is to prove malformed payloads are
rejected at the boundary (422) rather than silently scored as garbage.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from serve import LoanApplication

# A plausible, fully-valid application. Tests start from this and mutate one
# field at a time so a failure points at exactly one rule.
VALID = dict(
    loan_amnt=12000, term=36, int_rate=11.5, installment=400,
    grade='C', sub_grade='C3',
    emp_length=5, emp_length_missing=0,
    home_ownership='MORTGAGE', annual_inc=65000, verification_status='Verified',
    purpose='debt_consolidation', addr_state='CA', application_type='Individual',
    dti=18.0, revol_util=35.0, revol_bal=8000,
    fico_mean=700.0,
    delinq_2yrs=0, pub_rec=0, pub_rec_bankruptcies=0,
    mort_acc=1, open_acc=10, total_acc=25,
    credit_history_years=15,
)


def test_valid_payload_accepted():
    app = LoanApplication(**VALID)
    assert app.loan_amnt == 12000
    assert app.term == 36


def test_optional_fields_accept_none():
    payload = {**VALID, 'dti': None, 'revol_util': None, 'emp_length': None,
               'open_acc': None, 'total_acc': None, 'credit_history_years': None}
    app = LoanApplication(**payload)
    assert app.dti is None


def test_dti_sentinel_minus_one_accepted():
    # LendingClub's -1 sentinel must pass (it's a real value in the data).
    app = LoanApplication(**{**VALID, 'dti': -1})
    assert app.dti == -1


@pytest.mark.parametrize('field,bad_value', [
    ('loan_amnt', -5000),       # negative
    ('loan_amnt', 0),           # gt=0
    ('term', 48),               # not in {36, 60}
    ('int_rate', 9999),         # absurd rate
    ('int_rate', -1),
    ('installment', 0),         # gt=0
    ('annual_inc', -100),       # negative income
    ('fico_mean', 900),         # above FICO ceiling
    ('fico_mean', 250),         # below FICO floor
    ('emp_length', 11),         # > 10
    ('emp_length_missing', 2),  # not in {0, 1}
    ('dti', -2),                # below the -1 sentinel floor
    ('revol_util', -1),         # negative
    ('addr_state', 'CALIF'),    # must be a 2-char code
])
def test_out_of_range_values_rejected(field, bad_value):
    with pytest.raises(ValidationError):
        LoanApplication(**{**VALID, field: bad_value})


def test_unexpected_field_rejected():
    # extra='forbid' — a misspelled/extra field is an error, not silently dropped.
    with pytest.raises(ValidationError):
        LoanApplication(**{**VALID, 'loan_amount': 12000})  # typo: should be loan_amnt


def test_missing_required_field_rejected():
    payload = dict(VALID)
    del payload['loan_amnt']
    with pytest.raises(ValidationError):
        LoanApplication(**payload)
