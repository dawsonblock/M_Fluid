"""Evidence Storage

Immutable file storage for evidence content with SHA256 hashing.
Uses vault abstraction for local or remote storage.
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
from judge_memory.vault import EvidenceVault, create_vault

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

    Stores evidence files immutably using a vault backend.
    Files are never overwritten - duplicates detected by hash.
    """

    def __init__(self, config: JudgeMemoryConfig, vault: Optional[EvidenceVault] = None):
        self.config = config
        self.evidence_dir = config.evidence_dir
        self._vault = vault
        self._init_storage()

    def _init_storage(self) -> None:
        """Initialize evidence storage."""
        if self._vault is None:
            # Create local vault for backward compatibility
            self._vault = create_vault({
                "type": "local",
                "base_path": str(self.evidence_dir),
            })
            logger.info(f"Evidence storage initialized with local vault: {self.evidence_dir}")
        else:
            logger.info(f"Evidence storage initialized with external vault")

    @property
    def vault(self) -> EvidenceVault:
        """Get the vault instance."""
        return self._vault
    
    async def save_evidence(
        self,
        raw_text: str,
        source_type: str,
        source_url: Optional[str] = None,
        source_title: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        published_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvidenceRecord:
        """Save evidence to vault storage.

        Stores content in vault with hash-based deduplication.

        Args:
            raw_text: Evidence content
            source_type: Type of source (court_record, etc.)
            source_url: Optional source URL
            source_title: Optional human-readable title
            jurisdiction: Optional jurisdiction code
            published_at: Optional publication date
            metadata: Optional additional metadata

        Returns:
            EvidenceRecord with storage URI
        """
        # Compute hash for deduplication
        content_hash = compute_content_hash(raw_text)

        # Generate content preview for search (first 1000 chars)
        content_preview = raw_text[:1000] if raw_text else None

        # Generate evidence ID
        evidence_id = f"ev_{uuid.uuid4().hex[:16]}"

        # Store in vault
        content_bytes = raw_text.encode("utf-8")
        try:
            file_uri = await self._vault.store(content_hash, content_bytes)
            logger.info(f"Evidence saved: {file_uri}")
        except Exception as e:
            logger.error(f"Failed to save evidence to vault: {e}")
            raise StorageError(f"Failed to save evidence to vault: {e}")

        return EvidenceRecord(
            evidence_id=evidence_id,
            content_hash=content_hash,
            source_type=source_type,
            source_url=source_url,
            source_title=source_title,
            content_preview=content_preview,
            jurisdiction=jurisdiction,
            published_at=published_at,
            file_path=file_uri,  # URI from vault
            metadata=metadata or {},
        )
    
    async def read_evidence(self, file_uri: str) -> str:
        """Read evidence content from vault.

        Args:
            file_uri: Storage URI from save_evidence()

        Returns:
            Content as string
        """
        try:
            content_bytes = await self._vault.retrieve(file_uri)
            if content_bytes is None:
                raise StorageError(f"Evidence not found: {file_uri}")
            return content_bytes.decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to read evidence from vault: {e}")
            raise StorageError(f"Failed to read evidence from vault: {e}")
    
    async def verify_hash(self, file_uri: str, expected_hash: str) -> bool:
        """Verify vault content matches expected hash.

        Args:
            file_uri: Storage URI
            expected_hash: Expected SHA256 hash

        Returns:
            True if hash matches, False otherwise
        """
        try:
            content_bytes = await self._vault.retrieve(file_uri)
            if content_bytes is None:
                return False
            content = content_bytes.decode("utf-8")
            actual_hash = compute_content_hash(content)
            return actual_hash == expected_hash
        except Exception:
            return False
