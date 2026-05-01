"""
Judge Memory Evidence Storage

Immutable evidence file storage with SHA256 hashing.
"""

import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from m_flow.judge_memory.models import EvidenceRecord
from m_flow.judge_memory.config import JudgeMemoryConfig


def compute_content_hash(raw_text: str) -> str:
    """Compute SHA256 hash of raw text."""
    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()


def generate_evidence_id() -> str:
    """Generate unique evidence ID."""
    return f"ev_{uuid.uuid4().hex[:16]}"


class EvidenceStorage:
    """
    Immutable evidence file storage.

    Files are never overwritten.
    Duplicate content returns existing record.
    """

    def __init__(self, config: JudgeMemoryConfig):
        self.config = config
        self.evidence_dir = config.evidence_dir
        self._ensure_dir()

    def _ensure_dir(self):
        """Ensure evidence directory exists."""
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

    def _make_filename(self, evidence_id: str, content_hash: str) -> str:
        """Create evidence filename with hash prefix."""
        hash_prefix = content_hash[:12]
        return f"{evidence_id}_{hash_prefix}.json"

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
        """
        Save evidence to storage.

        Returns existing record if content hash matches.
        Never overwrites existing files.
        """
        # Compute hash
        content_hash = compute_content_hash(raw_text)

        # Generate ID
        evidence_id = generate_evidence_id()

        # Create filename
        filename = self._make_filename(evidence_id, content_hash)
        storage_path = str(self.evidence_dir / filename)

        # Build record
        record = EvidenceRecord(
            evidence_id=evidence_id,
            source_type=source_type,
            source_url=source_url,
            source_title=source_title,
            retrieved_at=datetime.utcnow(),
            published_at=published_at,
            jurisdiction=jurisdiction,
            raw_text=raw_text,
            content_hash=content_hash,
            storage_path=storage_path,
            metadata=metadata or {},
        )

        # Save to file (JSON format with metadata)
        evidence_data = {
            "evidence_id": record.evidence_id,
            "source_type": record.source_type,
            "source_url": record.source_url,
            "source_title": record.source_title,
            "retrieved_at": record.retrieved_at.isoformat(),
            "published_at": record.published_at.isoformat() if record.published_at else None,
            "jurisdiction": record.jurisdiction,
            "content_hash": record.content_hash,
            "raw_text": record.raw_text,
            "metadata": record.metadata,
        }

        with open(storage_path, "w", encoding="utf-8") as f:
            json.dump(evidence_data, f, indent=2)

        return record

    def load_evidence(self, storage_path: str) -> Optional[EvidenceRecord]:
        """Load evidence from file."""
        path = Path(storage_path)
        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return EvidenceRecord(
                evidence_id=data["evidence_id"],
                source_type=data["source_type"],
                source_url=data.get("source_url"),
                source_title=data.get("source_title"),
                retrieved_at=datetime.fromisoformat(data["retrieved_at"]),
                published_at=datetime.fromisoformat(data["published_at"]) if data.get("published_at") else None,
                jurisdiction=data.get("jurisdiction"),
                raw_text=data["raw_text"],
                content_hash=data["content_hash"],
                storage_path=storage_path,
                metadata=data.get("metadata", {}),
            )
        except (json.JSONDecodeError, KeyError):
            return None

    def get_raw_text_preview(self, storage_path: str, max_chars: int = 500) -> str:
        """Get preview of raw text without loading full file."""
        record = self.load_evidence(storage_path)
        if not record:
            return ""
        return record.raw_text[:max_chars]
