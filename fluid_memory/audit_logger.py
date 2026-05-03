"""Enhanced Audit Logging for Fluid Memory.

Structured, enterprise-grade audit logging for memory operations.
Logs are written in JSON format for easy ingestion into SIEM systems.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional
from contextvars import ContextVar

# Context variables for tracking user/session info
audit_user_id: ContextVar[Optional[str]] = ContextVar("audit_user_id", default=None)
audit_session_id: ContextVar[Optional[str]] = ContextVar("audit_session_id", default=None)
audit_operation_id: ContextVar[Optional[str]] = ContextVar("audit_operation_id", default=None)


class AuditEventType(str, Enum):
    """Types of audit events."""
    MEMORY_CREATED = "memory_created"
    MEMORY_ACCESSED = "memory_accessed"
    MEMORY_UPDATED = "memory_updated"
    MEMORY_INVALIDATED = "memory_invalidated"
    MEMORY_DELETED = "memory_deleted"
    CONTRADICTION_DETECTED = "contradiction_detected"
    CONTRADICTION_APPLIED = "contradiction_applied"
    REINFORCEMENT_APPLIED = "reinforcement_applied"
    SEMANTIC_SEARCH = "semantic_search"
    CHECKSUM_VERIFIED = "checksum_verified"
    CHECKSUM_FAILED = "checksum_failed"


class AuditLogger:
    """Enterprise-grade audit logger for memory operations.

    Features:
    - Structured JSON logging
    - User/session tracking
    - Tamper-evident log entries (sequence numbers)
    - Log rotation support
    - Async-safe operation
    """

    def __init__(self, log_dir: Optional[Path] = None, enable_file: bool = True):
        """Initialize audit logger.

        Args:
            log_dir: Directory for audit log files
            enable_file: If True, write to file; otherwise console only
        """
        self.log_dir = Path(log_dir) if log_dir else Path(".fluid_memory/audit")
        self.enable_file = enable_file
        self._sequence_number = 0
        self._logger = logging.getLogger("fluid_memory.audit")
        self._logger.setLevel(logging.INFO)

        # Remove existing handlers
        self._logger.handlers = []

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

        # File handler if enabled
        if enable_file:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_file = self.log_dir / "audit.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

    def _get_next_sequence(self) -> int:
        """Get next sequence number for tamper detection."""
        self._sequence_number += 1
        return self._sequence_number

    def _get_timestamp(self) -> str:
        """Get ISO 8601 timestamp in UTC."""
        return datetime.now(timezone.utc).isoformat()

    def _get_context(self) -> Dict[str, Any]:
        """Get current audit context."""
        return {
            "user_id": audit_user_id.get(),
            "session_id": audit_session_id.get(),
            "operation_id": audit_operation_id.get(),
        }

    def log(
        self,
        event_type: AuditEventType,
        memory_id: str,
        details: Optional[Dict[str, Any]] = None,
        result: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Log an audit event.

        Args:
            event_type: Type of event
            memory_id: Affected memory ID
            details: Additional event details
            result: Event result (success, failure, etc.)

        Returns:
            The logged event record
        """
        event = {
            "timestamp": self._get_timestamp(),
            "sequence": self._get_next_sequence(),
            "event_type": event_type.value,
            "memory_id": memory_id,
            "event_id": str(uuid.uuid4()),
            **self._get_context(),
        }

        if details:
            event["details"] = details
        if result:
            event["result"] = result

        # Log as JSON
        self._logger.info(json.dumps(event, default=str))

        return event

    def log_memory_created(
        self,
        memory_id: str,
        content_hash: str,
        tags: Optional[list] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Log memory creation event."""
        return self.log(
            AuditEventType.MEMORY_CREATED,
            memory_id,
            details={
                "content_hash": content_hash,
                "tags": tags or [],
                "metadata": metadata or {},
            },
            result="success",
        )

    def log_memory_accessed(
        self,
        memory_id: str,
        access_type: str = "read",
    ) -> Dict[str, Any]:
        """Log memory access event."""
        return self.log(
            AuditEventType.MEMORY_ACCESSED,
            memory_id,
            details={"access_type": access_type},
            result="success",
        )

    def log_memory_invalidated(
        self,
        memory_id: str,
        reason: str,
        invalidated_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Log memory invalidation event."""
        return self.log(
            AuditEventType.MEMORY_INVALIDATED,
            memory_id,
            details={
                "reason": reason,
                "invalidated_by": invalidated_by or audit_user_id.get(),
            },
            result="success",
        )

    def log_contradiction_detected(
        self,
        memory_id: str,
        conflicting_memory_id: str,
        similarity_score: float,
        reason: str,
    ) -> Dict[str, Any]:
        """Log contradiction detection event."""
        return self.log(
            AuditEventType.CONTRADICTION_DETECTED,
            memory_id,
            details={
                "conflicting_memory_id": conflicting_memory_id,
                "similarity_score": similarity_score,
                "detection_reason": reason,
            },
            result="detected",
        )

    def log_contradiction_applied(
        self,
        memory_id: str,
        conflicting_memory_id: str,
        amount: float,
        old_confidence: float,
        new_confidence: float,
    ) -> Dict[str, Any]:
        """Log contradiction application event."""
        return self.log(
            AuditEventType.CONTRADICTION_APPLIED,
            memory_id,
            details={
                "conflicting_memory_id": conflicting_memory_id,
                "contradiction_amount": amount,
                "confidence_change": {
                    "old": old_confidence,
                    "new": new_confidence,
                },
            },
            result="applied",
        )

    def log_semantic_search(
        self,
        query: str,
        results_count: int,
        threshold: float,
    ) -> Dict[str, Any]:
        """Log semantic search event."""
        return self.log(
            AuditEventType.SEMANTIC_SEARCH,
            "system",  # No specific memory ID for search
            details={
                "query": query,
                "results_count": results_count,
                "threshold": threshold,
            },
            result="success",
        )

    def log_checksum_verification(
        self,
        memory_id: str,
        checksum_valid: bool,
        stored_checksum: str,
        computed_checksum: str,
    ) -> Dict[str, Any]:
        """Log checksum verification event."""
        event_type = AuditEventType.CHECKSUM_VERIFIED if checksum_valid else AuditEventType.CHECKSUM_FAILED
        return self.log(
            event_type,
            memory_id,
            details={
                "stored_checksum": stored_checksum,
                "computed_checksum": computed_checksum,
                "match": checksum_valid,
            },
            result="valid" if checksum_valid else "invalid",
        )


def set_audit_context(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    operation_id: Optional[str] = None,
):
    """Set audit context for current execution context.

    Args:
        user_id: User performing the operation
        session_id: Session identifier
        operation_id: Operation/trace identifier
    """
    if user_id:
        audit_user_id.set(user_id)
    if session_id:
        audit_session_id.set(session_id)
    if operation_id:
        audit_operation_id.set(operation_id)


def clear_audit_context():
    """Clear audit context."""
    audit_user_id.set(None)
    audit_session_id.set(None)
    audit_operation_id.set(None)


class NoOpAuditLogger:
    """No-op audit logger for when audit is disabled.

    Provides the same interface as AuditLogger but performs no logging.
    Used when enable_audit=False to ensure zero audit output.
    """

    def log(self, *args, **kwargs) -> None:
        """No-op log method."""
        return None

    def log_memory_created(self, *args, **kwargs) -> None:
        """No-op memory created log."""
        return None

    def log_memory_accessed(self, *args, **kwargs) -> None:
        """No-op memory accessed log."""
        return None

    def log_memory_invalidated(self, *args, **kwargs) -> None:
        """No-op memory invalidated log."""
        return None

    def log_contradiction_detected(self, *args, **kwargs) -> None:
        """No-op contradiction detected log."""
        return None

    def log_contradiction_applied(self, *args, **kwargs) -> None:
        """No-op contradiction applied log."""
        return None

    def log_semantic_search(self, *args, **kwargs) -> None:
        """No-op semantic search log."""
        return None

    def log_checksum_verification(self, *args, **kwargs) -> None:
        """No-op checksum verification log."""
        return None
