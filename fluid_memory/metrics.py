"""Metrics collection for Fluid Memory.

Lightweight metrics for monitoring memory system performance and health.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
from time import time
from collections import deque


@dataclass
class OperationMetric:
    """Metric for a single operation."""
    operation: str
    duration_ms: float
    success: bool
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)


class FluidMemoryMetrics:
    """Metrics collector for Fluid Memory."""
    
    def __init__(self, max_history: int = 1000):
        self._operations: deque = deque(maxlen=max_history)
        self._counters: Dict[str, int] = {}
        self._start_times: Dict[str, float] = {}
    
    def start_operation(self, operation_id: str):
        """Mark start of an operation."""
        self._start_times[operation_id] = time()
    
    def end_operation(
        self,
        operation_id: str,
        operation_type: str,
        success: bool = True,
        details: Dict[str, Any] = None
    ):
        """Mark end of an operation and record metric."""
        start_time = self._start_times.pop(operation_id, None)
        if start_time:
            duration_ms = (time() - start_time) * 1000
            metric = OperationMetric(
                operation=operation_type,
                duration_ms=duration_ms,
                success=success,
                timestamp=time(),
                details=details or {}
            )
            self._operations.append(metric)
        
        # Update counter
        counter_key = f"{operation_type}_{'success' if success else 'failure'}"
        self._counters[counter_key] = self._counters.get(counter_key, 0) + 1
    
    def increment(self, counter: str, amount: int = 1):
        """Increment a simple counter."""
        self._counters[counter] = self._counters.get(counter, 0) + amount
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        total_ops = len(self._operations)
        if total_ops == 0:
            return {
                "total_operations": 0,
                "counters": self._counters.copy(),
            }
        
        avg_duration = sum(op.duration_ms for op in self._operations) / total_ops
        success_rate = sum(1 for op in self._operations if op.success) / total_ops
        
        # Group by operation type
        by_type: Dict[str, Dict[str, Any]] = {}
        for op in self._operations:
            if op.operation not in by_type:
                by_type[op.operation] = {"count": 0, "total_ms": 0, "success": 0}
            by_type[op.operation]["count"] += 1
            by_type[op.operation]["total_ms"] += op.duration_ms
            if op.success:
                by_type[op.operation]["success"] += 1
        
        # Calculate averages
        for op_type, stats in by_type.items():
            stats["avg_ms"] = stats["total_ms"] / stats["count"]
            stats["success_rate"] = stats["success"] / stats["count"]
        
        return {
            "total_operations": total_ops,
            "avg_duration_ms": avg_duration,
            "overall_success_rate": success_rate,
            "by_operation": by_type,
            "counters": self._counters.copy(),
        }
    
    def reset(self):
        """Reset all metrics."""
        self._operations.clear()
        self._counters.clear()
        self._start_times.clear()
