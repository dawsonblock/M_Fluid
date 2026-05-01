"""
Judge Memory Search

Simple keyword search with fallback.
Works without vector/graph DB for initial integration.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime

from m_flow.judge_memory.models import JudgeMemorySearchResult, EvidenceRecord, ClaimRecord
from m_flow.judge_memory.storage import JudgeMemoryStorage


class JudgeMemorySearch:
    """
    Search implementation for Judge memory.

    Fallback to SQLite keyword search when vector/graph unavailable.
    Includes source profile and claim status in results.
    """

    def __init__(self, storage: JudgeMemoryStorage, source_registry=None):
        self.storage = storage
        self.source_registry = source_registry

    async def search(
        self,
        query: str,
        limit: int = 10,
        jurisdiction: Optional[str] = None,
        entity_id: Optional[str] = None,
        case_id: Optional[str] = None,
        judge_id: Optional[str] = None,
        include_conflicted: bool = True,
    ) -> List[JudgeMemorySearchResult]:
        """
        Search memory.

        Returns results ranked by relevance.
        Never hides conflicted evidence by default.
        """
        results = []

        # Search evidence
        evidence_results = self.storage.search_evidence(query, limit)
        for evidence in evidence_results:
            # Get source profile for trust score
            source_trust = None
            if self.source_registry:
                try:
                    profile = await self.source_registry.get_source_profile(evidence.source_type)
                    source_trust = profile.derive_trust()
                except Exception:
                    pass

            result = JudgeMemorySearchResult(
                result_id=evidence.evidence_id,
                result_type="evidence",
                title=evidence.source_title or evidence.evidence_id,
                summary=evidence.raw_text[:200] + "..." if len(evidence.raw_text) > 200 else evidence.raw_text,
                score=1.0,  # Base score for exact match
                source_trust=source_trust,
                evidence_ids=[evidence.evidence_id],
                explanation={
                    "source_type": evidence.source_type,
                    "jurisdiction": evidence.jurisdiction,
                },
                is_conflicted=False,
            )
            results.append(result)

        # Search claims if case/judge/person specified
        if case_id:
            claims = self.storage.get_claims_by_case(case_id)
            for claim in claims:
                if query.lower() in claim.claim_text.lower():
                    result = JudgeMemorySearchResult(
                        result_id=claim.claim_id,
                        result_type="claim",
                        title=f"Claim: {claim.claim_text[:50]}...",
                        summary=claim.claim_text,
                        score=claim.confidence,
                        claim_status=claim.claim_status,
                        evidence_ids=[claim.evidence_id],
                        explanation={
                            "claim_type": claim.claim_type,
                            "confidence": claim.confidence,
                        },
                        is_conflicted=claim.claim_status == "conflicted",
                    )
                    if include_conflicted or not result.is_conflicted:
                        results.append(result)

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        return results[:limit]

    async def get_timeline(
        self,
        case_id: Optional[str] = None,
        judge_id: Optional[str] = None,
        entity_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get timeline for case/judge/entity.

        Chronological view of events.
        """
        events = []

        # Build timeline from claims with dates
        if case_id:
            claims = self.storage.get_claims_by_case(case_id)
            for claim in claims:
                if claim.event_date:
                    events.append({
                        "event_id": claim.claim_id,
                        "event_type": "claim",
                        "event_date": claim.event_date,
                        "title": f"Claim: {claim.claim_text[:50]}...",
                        "summary": claim.claim_text,
                        "evidence_ids": [claim.evidence_id],
                        "confidence": claim.confidence,
                        "claim_status": claim.claim_status,
                    })

        # Sort by date
        events.sort(key=lambda e: e.get("event_date") or datetime.min, reverse=True)

        return events
