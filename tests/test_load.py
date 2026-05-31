"""Unit tests for load.derive_default_flag — the project's label definition.

This is the single most consequential mapping in the codebase: it decides
which raw loan_status strings count as a default (1), a repayment (0), or a
censored/unknown outcome (NaN, to be dropped). A regression here silently
corrupts every downstream metric, so it gets thorough coverage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import load


def test_default_statuses_map_to_one():
    s = pd.Series([
        'Charged Off',
        'Default',
        'Late (31-120 days)',
        'Late (16-30 days)',
        'Does not meet the credit policy. Status:Charged Off',
    ])
    out = load.derive_default_flag(s)
    assert (out == 1).all()


def test_repay_statuses_map_to_zero():
    s = pd.Series([
        'Fully Paid',
        'Does not meet the credit policy. Status:Fully Paid',
    ])
    out = load.derive_default_flag(s)
    assert (out == 0).all()


def test_censored_statuses_map_to_nan():
    # Open / in-progress loans have unknown outcomes -> NaN (caller drops them).
    s = pd.Series(['Current', 'In Grace Period', 'Issued'])
    out = load.derive_default_flag(s)
    assert out.isna().all()


def test_mixed_series_preserves_order_and_values():
    s = pd.Series(['Fully Paid', 'Charged Off', 'Current', 'Default'])
    out = load.derive_default_flag(s)
    # Compare with NaN-aware equality.
    assert out.iloc[0] == 0
    assert out.iloc[1] == 1
    assert np.isnan(out.iloc[2])
    assert out.iloc[3] == 1
    assert len(out) == len(s)


def test_unknown_string_is_nan_not_error():
    # An unrecognized status must be treated as censored, never crash.
    out = load.derive_default_flag(pd.Series(['Some Future Status']))
    assert out.isna().all()
