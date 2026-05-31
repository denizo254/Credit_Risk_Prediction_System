"""Tests for contracts.validate_curated — the load -> prepare data contract.

Builds a minimal valid curated frame, then mutates one thing at a time so each
failure points at exactly one rule. Pure / data-free, so it runs in CI.
"""
from __future__ import annotations

import pandas as pd
import pytest

import contracts
from contracts import SchemaContractError, validate_curated


def _valid_frame(n: int = 3) -> pd.DataFrame:
    """A minimal frame satisfying every contract column and type."""
    data = {c: [1.0] * n for c in contracts.NUMERIC_COLUMNS}
    data.update({c: ['x'] * n for c in contracts.STRING_COLUMNS})
    data['default'] = [0, 1, 0][:n]
    return pd.DataFrame(data)


def test_valid_frame_passes():
    df = _valid_frame()
    # Returns the frame unchanged on success.
    assert validate_curated(df) is df


def test_missing_column_raises_and_names_it():
    df = _valid_frame().drop(columns=['annual_inc'])
    with pytest.raises(SchemaContractError, match='annual_inc'):
        validate_curated(df)


def test_numeric_column_as_string_raises():
    df = _valid_frame()
    df['annual_inc'] = ['lots', 'some', 'none']
    with pytest.raises(SchemaContractError, match='annual_inc'):
        validate_curated(df)


def test_string_column_as_numeric_raises():
    df = _valid_frame()
    df['grade'] = [1, 2, 3]
    with pytest.raises(SchemaContractError, match='grade'):
        validate_curated(df)


def test_empty_frame_raises():
    df = _valid_frame(0)
    with pytest.raises(SchemaContractError, match='no rows'):
        validate_curated(df)


def test_all_null_column_raises():
    df = _valid_frame()
    df['dti'] = [pd.NA, pd.NA, pd.NA]
    with pytest.raises(SchemaContractError, match='entirely null'):
        validate_curated(df)


def test_bad_target_values_raise():
    df = _valid_frame()
    df['default'] = [0, 1, 2]   # 2 is not a valid label
    with pytest.raises(SchemaContractError, match='must be 0/1'):
        validate_curated(df)


def test_target_with_nan_is_tolerated():
    # NaN labels are dropped before the value check (censored loans).
    df = _valid_frame()
    df['default'] = pd.Series([0, 1, None], dtype='Float64')
    validate_curated(df)  # should not raise


def test_multiple_violations_all_reported():
    df = _valid_frame().drop(columns=['dti'])
    df['grade'] = [1, 2, 3]          # wrong type
    errors = contracts.check_curated(df)
    joined = ' '.join(errors)
    assert 'dti' in joined and 'grade' in joined
    assert len(errors) >= 2
