"""
Judge Memory - Integration-ready local memory subsystem for Judge app.

This is a bounded, importable memory module with:
- SQLite local storage (no external DB required)
- Immutable evidence preservation
- Claims linked to evidence
- Basic search fallback
- Optional fluid memory scoring
- Safe defaults (fluid disabled by default)
- Explainable source profiles

Usage:
    from m_flow.judge_memory import JudgeMemoryService, JudgeMemoryConfig

    config = JudgeMemoryConfig(
        data_dir="./judge_memory_data",
        enable_fluid_memory=False,  # Safe default
    )
    memory = JudgeMemoryService(config)

    # Ingest evidence
    evidence = await memory.ingest_evidence(
        raw_text="Court ruling text...",
        source_type="court_record",
        source_title="Smith v. Jones",
        jurisdiction="US-TX",
    )

    # Add claim
    claim = await memory.add_claim(
        evidence_id=evidence.evidence_id,
        claim_text="The court ruled in favor of the plaintiff",
        claim_type="ruling",
        case_id="case_123",
    )

    # Search
    results = await memory.search("court ruling plaintiff")

    # Get source packet
    packet = await memory.get_source_packet(evidence.evidence_id)
    print(f"Authority: {packet.authority}, Legal status: {packet.legal_status_label}")

Note: This is an integration-ready local subsystem, not production-grade.
For production use, enable fluid memory and connect vector/graph DB.
"""

from m_flow.judge_memory.config import JudgeMemoryConfig
from m_flow.judge_memory.service import JudgeMemoryService
from m_flow.judge_memory.models import (
    EvidenceRecord,
    ClaimRecord,
    JudgeMemorySearchResult,
    TimelineEvent,
    SourcePacket,
)
from m_flow.judge_memory.exceptions import (
    JudgeMemoryError,
    EvidenceNotFoundError,
    ClaimNotFoundError,
    DuplicateEvidenceError,
    InvalidSourceTypeError,
    StorageError,
)

__all__ = [
    # Main API
    "JudgeMemoryService",
    "JudgeMemoryConfig",
    # Models
    "EvidenceRecord",
    "ClaimRecord",
    "JudgeMemorySearchResult",
    "TimelineEvent",
    "SourcePacket",
    # Exceptions
    "JudgeMemoryError",
    "EvidenceNotFoundError",
    "ClaimNotFoundError",
    "DuplicateEvidenceError",
    "InvalidSourceTypeError",
    "StorageError",
]

__version__ = "0.1.0"
