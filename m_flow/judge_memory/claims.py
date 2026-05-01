"""
Judge Memory Claims

Claim extraction and management linked to evidence.
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from m_flow.judge_memory.models import ClaimRecord


def generate_claim_id() -> str:
    """Generate unique claim ID."""
    return f"cl_{uuid.uuid4().hex[:16]}"


class ClaimsManager:
    """
    Manage claims linked to evidence.

    Every claim must have valid evidence_id.
    Claim status derives from source profile unless explicitly set.
    """

    def __init__(self, storage, source_registry=None):
        self.storage = storage
        self.source_registry = source_registry

    async def add_claim(
        self,
        evidence_id: str,
        claim_text: str,
        claim_type: str = "fact",
        subject: Optional[str] = None,
        case_id: Optional[str] = None,
        judge_id: Optional[str] = None,
        person_id: Optional[str] = None,
        event_date: Optional[datetime] = None,
        jurisdiction: Optional[str] = None,
        confidence: float = 0.5,
        claim_status: Optional[str] = None,
        source_span: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ClaimRecord:
        """
        Add a claim linked to evidence.

        Raises if evidence_id is not found.
        Sets claim_status from source profile if not provided.
        """
        # Verify evidence exists
        evidence = self.storage.get_evidence(evidence_id)
        if not evidence:
            from m_flow.judge_memory.exceptions import EvidenceNotFoundError
            raise EvidenceNotFoundError(f"Evidence not found: {evidence_id}")

        # Get source profile for default claim status
        if claim_status is None and self.source_registry:
            try:
                profile = await self.source_registry.get_source_profile(evidence.source_type)
                claim_status = profile.default_claim_status
            except Exception:
                claim_status = "needs_verification"
        elif claim_status is None:
            claim_status = "needs_verification"

        # Create claim record
        claim = ClaimRecord(
            claim_id=generate_claim_id(),
            evidence_id=evidence_id,
            claim_text=claim_text,
            claim_type=claim_type,
            subject=subject,
            claim_status=claim_status,
            confidence=confidence,
            jurisdiction=jurisdiction or evidence.jurisdiction,
            case_id=case_id,
            judge_id=judge_id,
            person_id=person_id,
            event_date=event_date,
            source_span=source_span,
            metadata=metadata or {},
        )

        # Store in DB
        self.storage.store_claim(claim)

        return claim

    def get_claim(self, claim_id: str) -> Optional[ClaimRecord]:
        """Get claim by ID."""
        return self.storage.get_claim(claim_id)

    def get_claims_for_evidence(self, evidence_id: str) -> list:
        """Get all claims for an evidence record."""
        return self.storage.get_claims_by_evidence(evidence_id)

    def get_claims_for_case(self, case_id: str) -> list:
        """Get all claims for a case."""
        return self.storage.get_claims_by_case(case_id)

    def get_claims_for_judge(self, judge_id: str) -> list:
        """Get all claims for a judge."""
        return self.storage.get_claims_by_judge(judge_id)
