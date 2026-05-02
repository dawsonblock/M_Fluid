"""Judge Memory Service

Main service interface for Judge app integration.
Clean, bounded API with safe defaults.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any

from judge_memory.config import JudgeMemoryConfig
from judge_memory.models import (
    EvidenceRecord,
    ClaimRecord,
    JudgeMemorySearchResult,
    TimelineEvent,
    SourcePacket,
)
from judge_memory.storage import JudgeMemoryStorage
from judge_memory.evidence import EvidenceStorage, compute_content_hash
from judge_memory.claims import ClaimsManager
from judge_memory.search import JudgeMemorySearch
from judge_memory.fluid_adapter import FluidMemoryAdapter
from judge_memory.exceptions import EvidenceNotFoundError
from judge_memory._logger import get_logger

logger = get_logger(__name__)


class JudgeMemoryService:
    """Main service interface for Judge app memory integration.
    
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
        """Ingest evidence into memory.
        
        Immutable storage - never overwrites existing evidence.
        Duplicate content (by hash) returns existing record.
        
        Args:
            raw_text: Evidence content
            source_type: Type (court_record, government_data, etc.)
            source_url: Optional source URL
            source_title: Optional title
            jurisdiction: Optional jurisdiction code
            published_at: Optional publication date
            metadata: Optional additional metadata
            
        Returns:
            EvidenceRecord
        """
        # Compute hash for deduplication
        content_hash = compute_content_hash(raw_text)
        
        # Check for existing evidence
        existing = self.storage.get_evidence_by_hash(content_hash)
        if existing:
            logger.info(f"Returning existing evidence: {existing.evidence_id}")
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
        
        logger.info(f"Evidence ingested: {db_record.evidence_id}")
        return db_record
    
    async def get_source_packet(
        self, evidence_id: str
    ) -> Optional[SourcePacket]:
        """Get source packet with trust profile.
        
        Returns explainable source metadata for legal review.
        
        Args:
            evidence_id: Evidence ID
            
        Returns:
            SourcePacket or None
        """
        evidence = self.storage.get_evidence(evidence_id)
        if not evidence:
            raise EvidenceNotFoundError(f"Evidence not found: {evidence_id}")
        
        # Get source profile from fluid adapter (no m_flow dependency)
        profile = self.fluid.get_source_profile(evidence.source_type)
        
        return SourcePacket(
            evidence_id=evidence_id,
            authority=profile["authority"],
            verifiability=profile["verifiability"],
            originality=profile["originality"],
            independence=profile["independence"],
            legal_status_label=profile["legal_status_label"],
            legal_status_weight=profile["legal_status_weight"],
            default_claim_status=profile["default_claim_status"],
            source_type=evidence.source_type,
            source_url=evidence.source_url,
        )
    
    async def get_evidence_content(self, evidence_id: str) -> str:
        """Get raw evidence content.
        
        Args:
            evidence_id: Evidence ID
            
        Returns:
            Content as string
        """
        evidence = self.storage.get_evidence(evidence_id)
        if not evidence:
            raise EvidenceNotFoundError(f"Evidence not found: {evidence_id}")
        
        if evidence.file_path:
            return self.evidence_storage.read_evidence(evidence.file_path)
        
        return ""
    
    # -------------------------------------------------------------------------
    # Claim operations
    # -------------------------------------------------------------------------
    
    async def add_claim(
        self,
        evidence_id: str,
        claim_text: str,
        claim_type: str = "fact",
        case_id: Optional[str] = None,
        judge_id: Optional[str] = None,
        person_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        confidence: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> ClaimRecord:
        """Add claim linked to evidence.
        
        Args:
            evidence_id: Parent evidence ID
            claim_text: Claim content
            claim_type: Type (fact, ruling, opinion, etc.)
            case_id: Optional case ID
            judge_id: Optional judge ID
            person_id: Optional person ID
            entity_id: Optional entity ID
            confidence: Optional confidence (defaults to source profile)
            tags: Optional tags
            
        Returns:
            ClaimRecord
        """
        # Get default confidence from source profile
        if confidence is None:
            evidence = self.storage.get_evidence(evidence_id)
            if evidence:
                profile = self.fluid.get_source_profile(evidence.source_type)
                confidence = profile["authority"] * 0.8  # Derive from authority
            else:
                confidence = 0.5
        
        claim = self.claims_manager.create_claim(
            evidence_id=evidence_id,
            claim_text=claim_text,
            claim_type=claim_type,
            case_id=case_id,
            judge_id=judge_id,
            person_id=person_id,
            entity_id=entity_id,
            confidence=confidence,
            tags=tags,
        )
        
        logger.info(f"Claim added: {claim.claim_id}")
        return claim
    
    async def review_claim(
        self,
        claim_id: str,
        status: str,
        reviewed_by: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> ClaimRecord:
        """Review and update claim status.
        
        Args:
            claim_id: Claim ID
            status: New status (active, under_review, retracted, confirmed)
            reviewed_by: Optional reviewer identifier
            notes: Optional review notes
            
        Returns:
            Updated ClaimRecord
        """
        claim = self.claims_manager.update_claim_status(
            claim_id=claim_id,
            status=status,
        )
        
        # Add review metadata
        if reviewed_by or notes:
            if "reviews" not in claim.metadata:
                claim.metadata["reviews"] = []
            
            claim.metadata["reviews"].append({
                "status": status,
                "reviewed_by": reviewed_by,
                "notes": notes,
                "at": datetime.utcnow().isoformat(),
            })
            
            self.storage.store_claim(claim)
        
        logger.info(f"Claim reviewed: {claim_id} -> {status}")
        return claim
    
    # -------------------------------------------------------------------------
    # Search operations
    # -------------------------------------------------------------------------
    
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
    ) -> List[JudgeMemorySearchResult]:
        """Search evidence and claims.
        
        Args:
            query: Search query
            entity_id: Filter by entity
            case_id: Filter by case
            judge_id: Filter by judge
            person_id: Filter by person
            jurisdiction: Filter by jurisdiction
            source_type: Filter by source type
            claim_status: Filter by claim status
            limit: Max results
            
        Returns:
            Search results
        """
        return self.search_engine.search(
            query=query,
            entity_id=entity_id,
            case_id=case_id,
            judge_id=judge_id,
            person_id=person_id,
            jurisdiction=jurisdiction,
            source_type=source_type,
            claim_status=claim_status,
            limit=limit,
        )
    
    async def get_timeline(
        self,
        entity_id: Optional[str] = None,
        case_id: Optional[str] = None,
        judge_id: Optional[str] = None,
        person_id: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        limit: int = 50,
    ) -> List[JudgeMemorySearchResult]:
        """Get timeline for entity/case/judge.
        
        Args:
            entity_id: Filter by entity
            case_id: Filter by case
            judge_id: Filter by judge
            person_id: Filter by person
            jurisdiction: Filter by jurisdiction
            limit: Max results
            
        Returns:
            Timeline events
        """
        return self.search_engine.get_timeline(
            entity_id=entity_id,
            case_id=case_id,
            judge_id=judge_id,
            person_id=person_id,
            jurisdiction=jurisdiction,
            limit=limit,
        )
    
    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------
    
    async def close(self) -> None:
        """Close service and release resources."""
        logger.info("JudgeMemoryService closed")
