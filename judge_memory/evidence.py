"""Evidence Storage

Immutable file storage for evidence content with SHA256 hashing.
"""

import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from judge_memory.config import JudgeMemoryConfig
from judge_memory.models import EvidenceRecord
from judge_memory.exceptions import StorageError
from judge_memory._logger import get_logger

logger = get_logger(__name__)


def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content.
    
    Args:
        content: Text content to hash
        
    Returns:
        SHA256 hash string (64 hex characters)
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class EvidenceStorage:
    """File-based storage for evidence content.
    
    Stores evidence files immutably in a directory structure.
    Files are never overwritten - duplicates detected by hash.
    """
    
    def __init__(self, config: JudgeMemoryConfig):
        self.config = config
        self.evidence_dir = config.evidence_dir
        self._init_storage()
    
    def _init_storage(self) -> None:
        """Initialize evidence storage directory."""
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Evidence storage initialized: {self.evidence_dir}")
    
    def save_evidence(
        self,
        raw_text: str,
        source_type: str,
        source_url: Optional[str] = None,
        source_title: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        published_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvidenceRecord:
        """Save evidence to file storage.
        
        Creates file in evidence directory with hash-based deduplication.
        
        Args:
            raw_text: Evidence content
            source_type: Type of source (court_record, etc.)
            source_url: Optional source URL
            source_title: Optional human-readable title
            jurisdiction: Optional jurisdiction code
            published_at: Optional publication date
            metadata: Optional additional metadata
            
        Returns:
            EvidenceRecord with file path
        """
        # Compute hash for deduplication
        content_hash = compute_content_hash(raw_text)
        
        # Generate content preview for search (first 1000 chars)
        content_preview = raw_text[:1000] if raw_text else None
        
        # Generate evidence ID
        evidence_id = f"ev_{uuid.uuid4().hex[:16]}"
        
        # Create filename from hash (first 16 chars)
        filename = f"{content_hash[:16]}.txt"
        file_path = self.evidence_dir / filename
        
        # Check for existing file
        if file_path.exists():
            logger.info(f"Evidence file already exists: {file_path}")
        else:
            # Write content to file
            try:
                file_path.write_text(raw_text, encoding="utf-8")
                logger.info(f"Evidence saved: {file_path}")
            except IOError as e:
                logger.error(f"Failed to save evidence file: {e}")
                raise StorageError(f"Failed to save evidence file: {e}")
        
        return EvidenceRecord(
            evidence_id=evidence_id,
            content_hash=content_hash,
            source_type=source_type,
            source_url=source_url,
            source_title=source_title,
            content_preview=content_preview,
            jurisdiction=jurisdiction,
            published_at=published_at,
            file_path=str(file_path),
            metadata=metadata or {},
        )
    
    def read_evidence(self, file_path: str) -> str:
        """Read evidence content from file.
        
        Args:
            file_path: Path to evidence file
            
        Returns:
            File content as string
        """
        try:
            return Path(file_path).read_text(encoding="utf-8")
        except IOError as e:
            logger.error(f"Failed to read evidence file: {e}")
            raise StorageError(f"Failed to read evidence file: {e}")
    
    def verify_hash(self, file_path: str, expected_hash: str) -> bool:
        """Verify file content matches expected hash.
        
        Args:
            file_path: Path to evidence file
            expected_hash: Expected SHA256 hash
            
        Returns:
            True if hash matches, False otherwise
        """
        try:
            content = Path(file_path).read_text(encoding="utf-8")
            actual_hash = compute_content_hash(content)
            return actual_hash == expected_hash
        except IOError:
            return False
