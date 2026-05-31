"""Tests for score_batch._missing_required — the malformed-input guard.

Pure column-set logic, no model or dataset needed, so it runs in CI. Proves a
missing base feature is reported (caller errors out) while the derived
interaction columns and any extra columns are correctly exempt.
"""
from __future__ import annotations

import score_batch
from features import INTERACTION_COLS


def test_no_missing_when_all_base_present():
    feature_order = ['loan_amnt', 'annual_inc'] + INTERACTION_COLS
    # Interactions absent from input is fine — they're derived downstream.
    assert score_batch._missing_required(['loan_amnt', 'annual_inc'], feature_order) == []


def test_missing_base_column_is_reported():
    feature_order = ['loan_amnt', 'annual_inc', 'dti'] + INTERACTION_COLS
    assert score_batch._missing_required(['loan_amnt', 'dti'], feature_order) == ['annual_inc']


def test_interaction_columns_are_never_required():
    feature_order = ['loan_amnt'] + INTERACTION_COLS
    missing = score_batch._missing_required(['loan_amnt'], feature_order)
    assert all(c not in missing for c in INTERACTION_COLS)


def test_extra_input_columns_are_ignored():
    feature_order = ['loan_amnt', 'annual_inc'] + INTERACTION_COLS
    cols = ['loan_amnt', 'annual_inc', 'application_id', 'note']
    assert score_batch._missing_required(cols, feature_order) == []


def test_supplied_interactions_also_pass():
    feature_order = ['loan_amnt'] + INTERACTION_COLS
    cols = ['loan_amnt', *INTERACTION_COLS]
    assert score_batch._missing_required(cols, feature_order) == []
