"""Judge Memory Search

Search functionality for evidence and claims.
"""

from typing import Optional, List, Tuple, Any

from judge_memory.storage import JudgeMemoryStorage
from judge_memory.models import JudgeMemorySearchResult, EvidenceRecord, ClaimRecord
from judge_memory.source_registry import DEFAULT_REGISTRY
from judge_memory._logger import get_logger

logger = get_logger(__name__)


class JudgeMemorySearch:
    """Search engine for Judge Memory.

    Provides BM25-ranked full-text search with filters for case, judge,
    jurisdiction, entity, and fluid score integration.
    """

    def __init__(
        self, storage: JudgeMemoryStorage, source_registry=None, fluid_adapter=None
    ):
        self.storage = storage
        self.source_registry = source_registry
        self.fluid_adapter = fluid_adapter

    def _get_source_authority(self, source_type: Optional[str]) -> float:
        """Get authority weight for source type."""
        registry = self.source_registry or DEFAULT_REGISTRY
        return registry.get_authority(source_type)

    async def _get_fluid_score(self, record_id: str) -> float:
        """Get fluid activation score for a record if available."""
        if not self.fluid_adapter:
            return 0.0
        try:
            state = await self.fluid_adapter.get_state(record_id)
            if state and hasattr(state, "activation"):
                return float(state.activation)
            elif isinstance(state, dict):
                return float(state.get("activation", 0.0))
        except Exception:
            pass
        return 0.0

    def _compute_final_score(
        self,
        base_score: float,
        source_type: Optional[str],
        fluid_score: float = 0.0,
        claim_confidence: Optional[float] = None,
    ) -> float:
        """Compute final search score blending BM25, source authority, and fluid scores.

        Formula: base_score * (0.6 + 0.3 * source_authority + 0.2 * fluid_score)
        For claims: also factor in claim confidence
        """
        authority = self._get_source_authority(source_type)

        # Base blend: BM25 score weighted by source authority
        score = base_score * (0.6 + 0.3 * authority)

        # Add fluid score boost if available (up to 10% boost)
        if fluid_score > 0:
            score = score * (1.0 + min(0.1, fluid_score * 0.1))

        # For claims, blend with confidence
        if claim_confidence is not None:
            score = score * 0.7 + claim_confidence * 0.3

        return min(1.0, max(0.0, score))

    async def search(
        self,
        query: str,
        entity_id: Optional[str] = None,
        case_id: Optional[str] = None,
        judge_id: Optional[str] = None,
        person_id: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        source_type: Optional[str] = None,
        claim_status: Optional[str] = None,
        limit: int = 20,
        use_fts: bool = True,
    ) -> List[JudgeMemorySearchResult]:
        """Search evidence and claims with BM25 ranking and fluid score blending.

        Args:
            query: Search query string
            entity_id: Filter by entity ID
            case_id: Filter by case ID
            judge_id: Filter by judge ID
            person_id: Filter by person ID
            jurisdiction: Filter by jurisdiction
            source_type: Filter by source type
            claim_status: Filter by claim status
            limit: Maximum results to return
            use_fts: Use FTS5/BM25 ranking if available

        Returns:
            List of search results sorted by relevance score
        """
        results: List[JudgeMemorySearchResult] = []

        # Search evidence with FTS5 if query provided
        if query and use_fts:
            evidence_results = self.storage.search_evidence_fts(
                query=query,
                source_type=source_type,
                jurisdiction=jurisdiction,
                limit=limit,
            )
        else:
            evidence_results = [
                (ev, 0.5)
                for ev in self.storage.search_evidence(
                    query=query if query else None,
                    source_type=source_type,
                    jurisdiction=jurisdiction,
                    limit=limit,
                )
            ]

        for evidence, bm25_score in evidence_results:
            preview_text = evidence.content_preview or evidence.source_title or ""
            fluid_score = await self._get_fluid_score(evidence.evidence_id)
            final_score = self._compute_final_score(
                base_score=bm25_score,
                source_type=evidence.source_type,
                fluid_score=fluid_score,
            )

            metadata = {
                "bm25_score": bm25_score,
                "source_authority": self._get_source_authority(evidence.source_type),
            }
            # Only include fluid_score if it's meaningful (> 0)
            if fluid_score > 0:
                metadata["fluid_score"] = fluid_score

            results.append(
                JudgeMemorySearchResult(
                    result_type="evidence",
                    record_id=evidence.evidence_id,
                    title=evidence.source_title or "Untitled Evidence",
                    content_preview=self._preview(preview_text, 200),
                    source_type=evidence.source_type,
                    jurisdiction=evidence.jurisdiction,
                    confidence=None,
                    status=None,
                    score=final_score,
                    metadata=metadata,
                )
            )

        # Search claims with FTS5 if query provided
        # Use search_claims_with_evidence to get full evidence metadata
        claim_results = self.storage.search_claims_with_evidence(
            query=query if query else "",
            case_id=case_id,
            judge_id=judge_id,
            person_id=person_id,
            entity_id=entity_id,
            status=claim_status,
            limit=limit,
            use_fts=use_fts and bool(query),
        )

        for claim, evidence, bm25_score in claim_results:
            fluid_score = await self._get_fluid_score(claim.claim_id)
            final_score = self._compute_final_score(
                base_score=bm25_score,
                source_type=evidence.source_type,
                fluid_score=fluid_score,
                claim_confidence=claim.confidence,
            )

            metadata = {
                "bm25_score": bm25_score,
                "claim_confidence": claim.confidence,
                "evidence_id": evidence.evidence_id,
                "evidence_title": evidence.source_title,
                "source_authority": self._get_source_authority(evidence.source_type),
                "search_method": "fts5" if (use_fts and query) else "fallback_like",
            }
            # Only include fluid_score if it's meaningful (> 0)
            if fluid_score > 0:
                metadata["fluid_score"] = fluid_score

            results.append(
                JudgeMemorySearchResult(
                    result_type="claim",
                    record_id=claim.claim_id,
                    title=f"Claim: {claim.claim_text[:50]}...",
                    content_preview=self._preview(claim.claim_text, 200),
                    source_type=evidence.source_type,  # Now from evidence!
                    jurisdiction=evidence.jurisdiction,  # Now from evidence!
                    confidence=claim.confidence,
                    status=claim.status,
                    score=final_score,
                    metadata=metadata,
                )
            )

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:limit]

    def get_timeline(
        self,
        entity_id: Optional[str] = None,
        case_id: Optional[str] = None,
        judge_id: Optional[str] = None,
        person_id: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        limit: int = 50,
    ) -> List[JudgeMemorySearchResult]:
        """Get timeline events for entity/case/judge.

        Args:
            entity_id: Filter by entity ID
            case_id: Filter by case ID
            judge_id: Filter by judge ID
            person_id: Filter by person ID
            jurisdiction: Filter by jurisdiction
            limit: Maximum results

        Returns:
            Timeline events as search results
        """
        events = self.storage.get_timeline_events(
            case_id=case_id,
            judge_id=judge_id,
            person_id=person_id,
            entity_id=entity_id,
            jurisdiction=jurisdiction,
            limit=limit,
        )

        results = []
        for event in events:
            results.append(
                JudgeMemorySearchResult(
                    result_type="timeline",
                    record_id=event.event_id,
                    title=f"{event.event_type}: {event.description[:50]}...",
                    content_preview=self._preview(event.description, 200),
                    source_type=None,
                    jurisdiction=event.jurisdiction,
                    score=0.7,  # Timeline events have higher relevance
                )
            )

        return results

    def _preview(self, text: str, max_length: int = 200) -> str:
        """Create preview text.

        Args:
            text: Full text
            max_length: Maximum preview length

        Returns:
            Truncated preview
        """
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."
