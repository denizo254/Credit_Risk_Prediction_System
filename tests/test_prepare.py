"""Unit tests for prepare.py's pure parsing helpers.

These coerce raw LendingClub string columns into numeric types. They're the
deterministic, no-I/O core of Phase 3 — exactly the kind of thing that should
be locked down by tests so the train/test parquets stay reproducible.
"""
from __future__ import annotations

import pandas as pd

import prepare


def test_parse_term_strips_and_extracts_int():
    s = pd.Series([' 36 months', '60 months'])
    out = prepare._parse_term(s)
    assert list(out) == [36, 60]
    assert str(out.dtype) == 'int8'


def test_parse_emp_length_maps_known_levels():
    s = pd.Series(['< 1 year', '1 year', '5 years', '10+ years'])
    out = prepare._parse_emp_length(s)
    assert list(out) == [0, 1, 5, 10]
    assert str(out.dtype) == 'Float32'


def test_parse_emp_length_preserves_missing():
    s = pd.Series(['3 years', None, '10+ years'])
    out = prepare._parse_emp_length(s)
    assert out.iloc[0] == 3
    assert pd.isna(out.iloc[1])
    assert out.iloc[2] == 10


def test_parse_earliest_cr_year_extracts_year():
    s = pd.Series(['Aug-2003', 'Jan-1990', 'Dec-2015'])
    out = prepare._parse_earliest_cr_year(s)
    assert list(out) == [2003, 1990, 2015]
    assert str(out.dtype) == 'Int16'


def test_parse_earliest_cr_year_bad_value_is_nullable_na():
    s = pd.Series(['Aug-2003', 'not-a-date'])
    out = prepare._parse_earliest_cr_year(s)
    assert out.iloc[0] == 2003
    assert pd.isna(out.iloc[1])
