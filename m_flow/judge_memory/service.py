"""
Judge Memory Service

Main service interface for Judge app integration.
Clean, bounded API with safe defaults.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any

from m_flow.judge_memory.config import JudgeMemoryConfig
from m_flow.judge_memory.models import (
    EvidenceRecord,
    ClaimRecord,
    JudgeMemorySearchResult,
    TimelineEvent,
    SourcePacket,
)
from m_flow.judge_memory.storage import JudgeMemoryStorage
from m_flow.judge_memory.evidence import EvidenceStorage, compute_content_hash
from m_flow.judge_memory.claims import ClaimsManager
from m_flow.judge_memory.search import JudgeMemorySearch
from m_flow.judge_memory.fluid_adapter import FluidMemoryAdapter
from m_flow.judge_memory.exceptions import EvidenceNotFoundError


class JudgeMemoryService:
    """
    Main service interface for Judge app memory integration.

    Provides:
    - Evidence ingestion (immutable, hash-deduplicated)
    - Claim management (linked to evidence)
    - Search (keyword fallback, fluid optional)
    - Timeline (chronological view)
    - Source packets (explainable trust profiles)

    All external dependencies optional. SQLite-only mode works out of box.
    """

    def __init__(self, config: JudgeMemoryConfig):
        self.config = config

        # Initialize storage
        self.storage = JudgeMemoryStorage(config)
        self.evidence_storage = EvidenceStorage(config)

        # Initialize managers
        self.claims_manager = ClaimsManager(
            self.storage, source_registry=None
        )
        self.search_engine = JudgeMemorySearch(
            self.storage, source_registry=None
        )

        # Initialize fluid adapter (may be disabled)
        self.fluid = FluidMemoryAdapter(
            config, source_registry=None
        )

    # -------------------------------------------------------------------------
    # Evidence operations
    # -------------------------------------------------------------------------

    async def ingest_evidence(
        self,
        raw_text: str,
        source_type: str,
        source_url: Optional[str] = None,
        source_title: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        published_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvidenceRecord:
        """
        Ingest evidence into memory.

        Immutable storage - never overwrites existing evidence.
        Duplicate content (by hash) returns existing record.
        """
        # Compute hash for deduplication
        content_hash = compute_content_hash(raw_text)

        # Check for existing evidence
        existing = self.storage.get_evidence_by_hash(content_hash)
        if existing:
            return existing

        # Save to file storage
        file_record = self.evidence_storage.save_evidence(
            raw_text=raw_text,
            source_type=source_type,
            source_url=source_url,
            source_title=source_title,
            jurisdiction=jurisdiction,
            published_at=published_at,
            metadata=metadata,
        )

        # Store in SQLite
        db_record = self.storage.store_evidence(file_record)

        # Touch fluid state if enabled
        await self.fluid.touch_evidence(
            evidence_id=db_record.evidence_id,
            source_type=source_type,
            jurisdiction=jurisdiction,
        )

        return db_record

    async def get_source_packet(
        self, evidence_id: str
    ) -> Optional[SourcePacket]:
        """
        Get source packet with trust profile.

        Returns explainable source metadata for legal review.
        """
        evidence = self.storage.get_evidence(evidence_id)
        if not evidence:
            raise EvidenceNotFoundError(f"Evidence not found: {evidence_id}")

        # Get source profile
        authority = 0.5
        verifiability = 0.5
        originality = 0.5
        independence = 0.5
        legal_status_label = "unverified"
        legal_status_weight = 0.5
        default_claim_status = "needs_verification"

        # Try to get from source registry
        try:
            from m_flow.memory.fluid.source_registry import _HARDCODED_FALLBACK

            profile = _HARDCODED_FALLBACK.get(
                evidence.source_type,
                _HARDCODED_FALLBACK["unknown"]
            )
            authority = profile.authority
            verifiability = profile.verifiability
            originality = profile.originality
            independence = profile.independence
            legal_status_label = profile.legal_status_label
            legal_status_weight = profile.legal_status_weight
            default_claim_status = profile.default_claim_status
        except Exception:
            pass

        # Get preview
        preview = evidence.raw_text[:500]

        return SourcePacket(
            evidence_id=evidence.evidence_id,
            source_type=evidence.source_type,
            source_url=evidence.source_url,
            source_title=evidence.source_title,
            retrieved_at=evidence.retrieved_at,
            content_hash=evidence.content_hash,
            raw_text_preview=preview,
            authority=authority,
            verifiability=verifiability,
            originality=originality,
            independence=independence,
            legal_status_label=legal_status_label,
            legal_status_weight=legal_status_weight,
            default_claim_status=default_claim_status,
        )

    # -------------------------------------------------------------------------
    # Claim operations
    # -------------------------------------------------------------------------

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
        source_span: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ClaimRecord:
        """
        Add a claim linked to evidence.

        Requires valid evidence_id.
        Claim status derives from source profile unless explicitly set.
        """
        claim = await self.claims_manager.add_claim(
            evidence_id=evidence_id,
            claim_text=claim_text,
            claim_type=claim_type,
            subject=subject,
            case_id=case_id,
            judge_id=judge_id,
            person_id=person_id,
            event_date=event_date,
            jurisdiction=jurisdiction,
            confidence=confidence,
            source_span=source_span,
            metadata=metadata,
        )

        # Touch fluid state if enabled
        evidence = self.storage.get_evidence(evidence_id)
        if evidence:
            await self.fluid.touch_claim(
                claim_id=claim.claim_id,
                evidence_id=evidence_id,
                source_type=evidence.source_type,
                claim_status=claim.claim_status,
            )

        return claim

    async def get_claims(
        self,
        evidence_id: Optional[str] = None,
        case_id: Optional[str] = None,
        judge_id: Optional[str] = None,
        entity_id: Optional[str] = None,
    ) -> List[ClaimRecord]:
        """Get claims by various filters."""
        if evidence_id:
            return self.storage.get_claims_by_evidence(evidence_id)
        elif case_id:
            return self.storage.get_claims_by_case(case_id)
        elif judge_id:
            return self.storage.get_claims_by_judge(judge_id)
        elif entity_id:
            # Treat entity_id as person_id for now
            return self.storage.get_claims_by_judge(entity_id)
        return []

    # -------------------------------------------------------------------------
    # Search operations
    # -------------------------------------------------------------------------

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

        Works without vector/graph DB.
        Never hides conflicted evidence by default.
        """
        return await self.search_engine.search(
            query=query,
            limit=limit,
            jurisdiction=jurisdiction,
            entity_id=entity_id,
            case_id=case_id,
            judge_id=judge_id,
            include_conflicted=include_conflicted,
        )

    # -------------------------------------------------------------------------
    # Memory views
    # -------------------------------------------------------------------------

    async def get_judge_memory(self, judge_id: str) -> Dict[str, Any]:
        """Get memory summary for a judge."""
        claims = self.storage.get_claims_by_judge(judge_id)
        timeline = await self.search_engine.get_timeline(judge_id=judge_id)

        return {
            "judge_id": judge_id,
            "claim_count": len(claims),
            "claims": [c.claim_id for c in claims],
            "timeline_event_count": len(timeline),
            "timeline": timeline[:10],  # Last 10 events
        }

    async def get_case_memory(self, case_id: str) -> Dict[str, Any]:
        """Get memory summary for a case."""
        claims = self.storage.get_claims_by_case(case_id)
        timeline = await self.search_engine.get_timeline(case_id=case_id)

        # Get unique evidence
        evidence_ids = set(c.evidence_id for c in claims)
        evidence_list = []
        for eid in evidence_ids:
            ev = self.storage.get_evidence(eid)
            if ev:
                evidence_list.append({
                    "evidence_id": ev.evidence_id,
                    "source_type": ev.source_type,
                    "source_title": ev.source_title,
                })

        return {
            "case_id": case_id,
            "claim_count": len(claims),
            "evidence_count": len(evidence_list),
            "evidence": evidence_list,
            "timeline_event_count": len(timeline),
            "timeline": timeline[:10],
        }

    async def get_entity_memory(self, entity_id: str) -> Dict[str, Any]:
        """Get memory summary for an entity (person)."""
        # Treat as person_id
        claims = self.storage.get_claims_by_judge(entity_id)

        return {
            "entity_id": entity_id,
            "claim_count": len(claims),
            "claims": [c.claim_id for c in claims],
        }

    async def get_timeline(
        self,
        case_id: Optional[str] = None,
        judge_id: Optional[str] = None,
        entity_id: Optional[str] = None,
    ) -> List[TimelineEvent]:
        """Get timeline for case/judge/entity."""
        events_data = await self.search_engine.get_timeline(
            case_id=case_id,
            judge_id=judge_id,
            entity_id=entity_id,
        )

        # Convert to TimelineEvent objects
        events = []
        for e in events_data:
            events.append(TimelineEvent(
                event_id=e["event_id"],
                event_type=e["event_type"],
                event_date=e.get("event_date"),
                title=e["title"],
                summary=e.get("summary", ""),
                evidence_ids=e.get("evidence_ids", []),
                confidence=e.get("confidence", 0.5),
                claim_status=e.get("claim_status", "needs_verification"),
            ))

        return events

    # -------------------------------------------------------------------------
    # Health check
    # -------------------------------------------------------------------------

    async def healthcheck(self) -> Dict[str, Any]:
        """Check service health."""
        storage_health = self.storage.healthcheck()

        return {
            "status": storage_health.get("status", "unknown"),
            "fluid_enabled": self.config.enable_fluid_memory,
            "storage": storage_health,
            "config": {
                "data_dir": str(self.config.data_dir),
                "evidence_dir": str(self.config.evidence_dir),
                "sqlite_path": str(self.config.sqlite_path),
            },
        }
