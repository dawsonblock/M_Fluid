"""
Judge Memory Exceptions

Custom exceptions for the Judge memory subsystem.
"""


class JudgeMemoryError(Exception):
    """Base exception for Judge memory errors."""
    pass


class EvidenceNotFoundError(JudgeMemoryError):
    """Raised when evidence is not found."""
    pass


class ClaimNotFoundError(JudgeMemoryError):
    """Raised when claim is not found."""
    pass


class DuplicateEvidenceError(JudgeMemoryError):
    """Raised when evidence with same hash already exists."""
    pass


class InvalidSourceTypeError(JudgeMemoryError):
    """Raised when source type is invalid."""
    pass


class StorageError(JudgeMemoryError):
    """Raised when storage operation fails."""
    pass
