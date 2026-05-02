"""Fluid Memory Exceptions"""


class FluidMemoryError(Exception):
    """Base exception for fluid memory errors."""
    pass


class StorageError(FluidMemoryError):
    """Raised when storage operations fail."""
    pass


class MemoryNotFoundError(FluidMemoryError):
    """Raised when a memory item is not found."""
    pass


class DecayError(FluidMemoryError):
    """Raised when decay calculations fail."""
    pass
