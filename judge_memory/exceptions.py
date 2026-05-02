"""Judge Memory Exceptions"""


class JudgeMemoryError(Exception):
    """Base exception for Judge Memory operations."""
    pass


class EvidenceNotFoundError(JudgeMemoryError):
    """Raised when evidence is not found."""
    pass


class ClaimNotFoundError(JudgeMemoryError):
    """Raised when claim is not found."""
    pass


class DuplicateEvidenceError(JudgeMemoryError):
    """Raised when duplicate evidence is ingested."""
    pass


class InvalidSourceTypeError(JudgeMemoryError):
    """Raised when source type is invalid."""
    pass


class StorageError(JudgeMemoryError):
    """Raised when storage operation fails."""
    pass
