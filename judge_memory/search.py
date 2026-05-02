"""Judge Memory Search

Search functionality for evidence and claims.
"""

from typing import Optional, List

from judge_memory.storage import JudgeMemoryStorage
from judge_memory.models import JudgeMemorySearchResult, EvidenceRecord, ClaimRecord
from judge_memory._logger import get_logger

logger = get_logger(__name__)


class JudgeMemorySearch:
    """Search engine for Judge Memory.
    
    Provides keyword-based search with filters for case, judge,
    jurisdiction, entity, and other legal-specific dimensions.
    """
    
    def __init__(self, storage: JudgeMemoryStorage, source_registry=None):
        self.storage = storage
        self.source_registry = source_registry
    
    def search(
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
            query: Search query string
            entity_id: Filter by entity ID
            case_id: Filter by case ID
            judge_id: Filter by judge ID
            person_id: Filter by person ID
            jurisdiction: Filter by jurisdiction
            source_type: Filter by source type
            claim_status: Filter by claim status
            limit: Maximum results to return
            
        Returns:
            List of search results
        """
        results = []
        
        # Search evidence
        evidence_results = self.storage.search_evidence(
            query=query if query else None,
            source_type=source_type,
            jurisdiction=jurisdiction,
            limit=limit,
        )
        
        for evidence in evidence_results:
            results.append(
                JudgeMemorySearchResult(
                    result_type="evidence",
                    record_id=evidence.evidence_id,
                    title=evidence.source_title,
                    content_preview=self._preview(
                        evidence.source_title or "", 200
                    ),
                    source_type=evidence.source_type,
                    jurisdiction=evidence.jurisdiction,
                    score=0.5,  # Base score
                )
            )
        
        # TODO: Add claim search when storage method available
        
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
        # TODO: Implement timeline query in storage
        logger.warning("Timeline search not yet fully implemented")
        return []
    
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
