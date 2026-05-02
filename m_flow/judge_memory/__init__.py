"""
Judge Memory - Compatibility shim for backward compatibility.

This module now re-exports from the top-level judge_memory package.
The implementation has been moved to judge_memory/ for clean isolation.

New import path (recommended):
    from judge_memory import JudgeMemoryService, JudgeMemoryConfig

Legacy import path (still works):
    from m_flow.judge_memory import JudgeMemoryService, JudgeMemoryConfig
"""

# Re-export from top-level package
from judge_memory import (
    JudgeMemoryConfig,
    JudgeMemoryService,
    EvidenceRecord,
    ClaimRecord,
    JudgeMemorySearchResult,
    TimelineEvent,
    SourcePacket,
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
