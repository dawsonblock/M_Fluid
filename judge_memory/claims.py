"""Claims Manager

Manages claim records linked to evidence.
"""

import uuid
from datetime import datetime
from typing import Optional, List

from judge_memory.storage import JudgeMemoryStorage
from judge_memory.models import ClaimRecord
from judge_memory.exceptions import ClaimNotFoundError, StorageError
from judge_memory._logger import get_logger

logger = get_logger(__name__)


class ClaimsManager:
    """Manages claim creation and lifecycle.
    
    Claims are mutable interpretations linked to immutable evidence.
    """
    
    def __init__(self, storage: JudgeMemoryStorage, source_registry=None):
        self.storage = storage
        self.source_registry = source_registry
    
    def create_claim(
        self,
        evidence_id: str,
        claim_text: str,
        claim_type: str = "fact",
        case_id: Optional[str] = None,
        judge_id: Optional[str] = None,
        person_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        confidence: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> ClaimRecord:
        """Create a new claim linked to evidence.
        
        Args:
            evidence_id: Parent evidence record ID
            claim_text: The claim content
            claim_type: Type of claim (fact, ruling, opinion, etc.)
            case_id: Optional associated case ID
            judge_id: Optional associated judge ID
            person_id: Optional associated person ID
            entity_id: Optional associated entity ID
            confidence: Initial confidence score (0.0-1.0)
            tags: Optional categorization tags
            
        Returns:
            Created ClaimRecord
        """
        claim_id = f"claim_{uuid.uuid4().hex[:16]}"
        
        now = datetime.utcnow()
        
        claim = ClaimRecord(
            claim_id=claim_id,
            evidence_id=evidence_id,
            claim_text=claim_text,
            claim_type=claim_type,
            case_id=case_id,
            judge_id=judge_id,
            person_id=person_id,
            entity_id=entity_id,
            confidence=confidence,
            status="active",
            tags=tags or [],
            created_at=now,
            updated_at=now,
        )
        
        try:
            self.storage.store_claim(claim)
            logger.info(f"Claim created: {claim_id}")
            return claim
        except StorageError as e:
            logger.error(f"Failed to create claim: {e}")
            raise
    
    def get_claim(self, claim_id: str) -> ClaimRecord:
        """Get claim by ID.
        
        Args:
            claim_id: Claim ID
            
        Returns:
            ClaimRecord
            
        Raises:
            ClaimNotFoundError: If claim not found
        """
        claim = self.storage.get_claim(claim_id)
        if not claim:
            raise ClaimNotFoundError(f"Claim not found: {claim_id}")
        return claim
    
    def update_claim_status(
        self,
        claim_id: str,
        status: str,
        confidence: Optional[float] = None,
    ) -> ClaimRecord:
        """Update claim status and optionally confidence.
        
        Args:
            claim_id: Claim ID
            status: New status (active, under_review, retracted, confirmed)
            confidence: Optional new confidence score
            
        Returns:
            Updated ClaimRecord
        """
        claim = self.get_claim(claim_id)
        
        claim.status = status
        if confidence is not None:
            claim.confidence = max(0.0, min(1.0, confidence))
        claim.updated_at = datetime.utcnow()
        
        try:
            self.storage.store_claim(claim)
            logger.info(f"Claim updated: {claim_id} -> {status}")
            return claim
        except StorageError as e:
            logger.error(f"Failed to update claim: {e}")
            raise
    
    def contradict_claim(
        self,
        claim_id: str,
        contradicting_evidence_id: str,
    ) -> ClaimRecord:
        """Mark claim as contradicted by other evidence.
        
        Reduces confidence and updates status.
        
        Args:
            claim_id: Claim to contradict
            contradicting_evidence_id: Evidence ID that contradicts
            
        Returns:
            Updated ClaimRecord
        """
        claim = self.get_claim(claim_id)
        
        # Reduce confidence
        claim.confidence = max(0.0, claim.confidence - 0.2)
        claim.status = "under_review"
        claim.updated_at = datetime.utcnow()
        
        # Add contradiction reference to metadata
        if "contradictions" not in claim.metadata:
            claim.metadata["contradictions"] = []
        claim.metadata["contradictions"].append(contradicting_evidence_id)
        
        try:
            self.storage.store_claim(claim)
            logger.info(f"Claim contradicted: {claim_id}")
            return claim
        except StorageError as e:
            logger.error(f"Failed to contradict claim: {e}")
            raise
    
    def get_claims_for_evidence(self, evidence_id: str) -> List[ClaimRecord]:
        """Get all claims linked to evidence.
        
        Args:
            evidence_id: Evidence ID
            
        Returns:
            List of ClaimRecords
        """
        try:
            return self.storage.get_claims_for_evidence(evidence_id)
        except StorageError as e:
            logger.error(f"Failed to get claims for evidence {evidence_id}: {e}")
            raise
