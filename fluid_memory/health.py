"""Health checks for Fluid Memory.

Provides health status for storage, checksum integrity, and overall system.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from time import time


@dataclass
class HealthStatus:
    """Health check result."""
    healthy: bool
    component: str
    message: str
    details: Dict[str, Any]
    timestamp: float


class FluidMemoryHealth:
    """Health checker for Fluid Memory engine."""
    
    def __init__(self, engine):
        self.engine = engine
        self._last_check: Optional[HealthStatus] = None
    
    def check_storage(self) -> HealthStatus:
        """Check storage connectivity and basic operations."""
        try:
            # Try a simple operation
            count = self.engine.storage.count_memories()
            return HealthStatus(
                healthy=True,
                component="storage",
                message="Storage accessible",
                details={"memory_count": count},
                timestamp=time()
            )
        except Exception as e:
            return HealthStatus(
                healthy=False,
                component="storage",
                message=f"Storage error: {e}",
                details={"error": str(e)},
                timestamp=time()
            )
    
    def check_checksums(self, sample_size: int = 10) -> HealthStatus:
        """Check checksum integrity on a sample of memories."""
        try:
            result = self.engine.storage.verify_all_checksums()
            total = result["total"]
            invalid = result["invalid"]
            
            # If too many memories, just check sample
            if total > sample_size:
                # Check health based on overall stats
                healthy = invalid == 0
            else:
                healthy = invalid == 0
            
            return HealthStatus(
                healthy=healthy,
                component="checksums",
                message=f"Checksums: {total - invalid}/{total} valid" if healthy else f"{invalid} invalid checksums detected",
                details=result,
                timestamp=time()
            )
        except Exception as e:
            return HealthStatus(
                healthy=False,
                component="checksums",
                message=f"Checksum check failed: {e}",
                details={"error": str(e)},
                timestamp=time()
            )
    
    def check_all(self) -> Dict[str, HealthStatus]:
        """Run all health checks."""
        checks = {
            "storage": self.check_storage(),
            "checksums": self.check_checksums(),
        }
        return checks
    
    def is_healthy(self) -> bool:
        """Quick health check - returns True if all components healthy."""
        checks = self.check_all()
        return all(check.healthy for check in checks.values())
