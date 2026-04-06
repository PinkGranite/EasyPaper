"""Tests for reference assignment helpers (no full agent stack)."""
from __future__ import annotations

from tests.ep_imports import load_reference_assignment


def test_claim_matrix_refs_for_section_filters_by_section_type():
    ra = load_reference_assignment()
    rc = {
        "claim_evidence_matrix": [
            {"section_type": "method", "support_refs": ["a", "b"]},
            {"section_type": "related_work", "support_refs": ["c"]},
            {"section_type": "global", "support_refs": ["g"]},
        ]
    }
    assert ra.claim_matrix_refs_for_section(rc, "method") == ["a", "b", "g"]


def test_claim_matrix_refs_empty_when_no_context():
    ra = load_reference_assignment()
    assert ra.claim_matrix_refs_for_section(None, "method") == []


def test_claim_matrix_ignores_malformed_rows():
    ra = load_reference_assignment()
    rc = {"claim_evidence_matrix": [None, {"section_type": "m"}, {"support_refs": "bad"}]}
    assert ra.claim_matrix_refs_for_section(rc, "m") == []
