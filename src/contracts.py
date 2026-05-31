"""Phase 3 data contract — fail fast at the load -> prepare boundary.

`prepare.clean()` assumes the curated frame (written by notebook 02 to
`data/interim/accepted_curated.parquet`) has a specific set of columns with
sane types. Without a check, an upstream change — a renamed source column, a
column that comes through entirely null, a numeric field arriving as text —
surfaces as a cryptic KeyError or a silent NaN cascade deep inside cleaning.
`validate_curated()` turns that into a single, legible error up front.

Dependency-free on purpose (no pandera): the checks are simple, and this keeps
the install light and consistent with the rest of the project. The raw-CSV
side of the boundary already self-validates — `pandas.read_csv(usecols=...)`
raises clearly when a requested column is absent — so the gap worth closing is
this curated -> clean handoff.
"""
from __future__ import annotations

import pandas as pd
from pandas.api import types as pdt

# The curated schema, mirroring data/interim/accepted_curated.parquet.
# Numeric fields the model consumes or that feed engineered features.
NUMERIC_COLUMNS = [
    'loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 'delinq_2yrs',
    'fico_range_low', 'fico_range_high', 'open_acc', 'pub_rec', 'revol_bal',
    'revol_util', 'total_acc', 'mort_acc', 'pub_rec_bankruptcies', 'issue_year',
]
# Text fields parsed/encoded later (terms, dates, categoricals, loan_status).
STRING_COLUMNS = [
    'term', 'grade', 'sub_grade', 'emp_length', 'home_ownership',
    'verification_status', 'issue_d', 'loan_status', 'purpose', 'addr_state',
    'earliest_cr_line', 'application_type',
]
TARGET_COLUMN = 'default'

REQUIRED_COLUMNS = [*NUMERIC_COLUMNS, *STRING_COLUMNS, TARGET_COLUMN]


class SchemaContractError(ValueError):
    """Raised when the curated frame violates the Phase-3 data contract."""


def check_curated(df: pd.DataFrame) -> list[str]:
    """Return a list of contract violations (empty list == valid)."""
    errors: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f'missing required column(s): {", ".join(missing)}')

    if len(df) == 0:
        errors.append('frame has no rows')

    for c in NUMERIC_COLUMNS:
        if c in df.columns and not pdt.is_numeric_dtype(df[c]):
            errors.append(f'{c}: expected numeric, got dtype {df[c].dtype}')

    for c in STRING_COLUMNS:
        if c in df.columns and not (pdt.is_string_dtype(df[c]) or pdt.is_object_dtype(df[c])):
            errors.append(f'{c}: expected string/object, got dtype {df[c].dtype}')

    # An entirely-null column is almost always an upstream break (e.g. a source
    # column got renamed and now reads as all-NaN).
    for c in REQUIRED_COLUMNS:
        if c in df.columns and len(df) and df[c].isna().all():
            errors.append(f'{c}: column is entirely null')

    if TARGET_COLUMN in df.columns:
        vals = set(pd.unique(df[TARGET_COLUMN].dropna()))
        if not vals <= {0, 1}:
            errors.append(f'{TARGET_COLUMN}: must be 0/1, found values {sorted(vals)}')

    return errors


def validate_curated(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the curated frame against the contract; return it unchanged.

    Raises SchemaContractError listing *all* violations at once, so one run
    surfaces every problem rather than failing on the first.
    """
    errors = check_curated(df)
    if errors:
        bullet = '\n'.join(f'  - {e}' for e in errors)
        raise SchemaContractError(
            'Curated data failed the Phase-3 schema contract:\n' + bullet
        )
    return df
