"""
Fluid Memory Core Exceptions

Custom exceptions for the fluid memory system.
"""


class FluidMemoryError(Exception):
    """Base exception for all fluid memory errors."""
    pass


class MemoryNotFoundError(FluidMemoryError):
    """Raised when a memory ID is not found."""
    pass


class DuplicateMemoryError(FluidMemoryError):
    """Raised when attempting to add a memory with duplicate content hash."""
    pass


class InvalidStateError(FluidMemoryError):
    """Raised when an operation would result in invalid memory state."""
    pass


class StorageError(FluidMemoryError):
    """Raised when a storage operation fails."""
    pass


class MutationError(FluidMemoryError):
    """Raised when a mutation operation fails or is rejected."""
    pass
