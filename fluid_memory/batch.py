"""Batch operations for Fluid Memory.

Efficient bulk operations for memory management.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from fluid_memory.models import MemoryItem


@dataclass
class BatchResult:
    """Result of a batch operation."""
    success_count: int
    failure_count: int
    errors: List[Tuple[str, str]]  # (memory_id, error_message)
    details: List[Any]  # Success results


class BatchOperations:
    """Batch operations for memory management."""
    
    def __init__(self, engine):
        self.engine = engine
    
    def add_memories(
        self,
        contents: List[str],
        tags_list: Optional[List[List[str]]] = None,
        source_refs_list: Optional[List[List[str]]] = None,
        detect_contradictions: bool = True
    ) -> BatchResult:
        """Add multiple memories in batch."""
        errors = []
        details = []
        
        for i, content in enumerate(contents):
            try:
                tags = tags_list[i] if tags_list and i < len(tags_list) else None
                source_refs = source_refs_list[i] if source_refs_list and i < len(source_refs_list) else None
                
                memory = self.engine.add_memory(
                    content=content,
                    tags=tags,
                    source_refs=source_refs,
                    detect_contradictions=detect_contradictions
                )
                details.append(memory)
            except Exception as e:
                errors.append((f"batch_{i}", str(e)))
        
        return BatchResult(
            success_count=len(details),
            failure_count=len(errors),
            errors=errors,
            details=details
        )
    
    def invalidate_memories(
        self,
        memory_ids: List[str],
        reason: str = ""
    ) -> BatchResult:
        """Invalidate multiple memories."""
        errors = []
        details = []
        
        for memory_id in memory_ids:
            try:
                memory = self.engine.invalidate_memory(memory_id, reason=reason)
                details.append(memory)
            except Exception as e:
                errors.append((memory_id, str(e)))
        
        return BatchResult(
            success_count=len(details),
            failure_count=len(errors),
            errors=errors,
            details=details
        )
    
    def verify_memories(
        self,
        memory_ids: List[str]
    ) -> BatchResult:
        """Verify checksums for multiple memories."""
        errors = []
        details = []
        
        for memory_id in memory_ids:
            try:
                is_valid = self.engine.verify_memory(memory_id)
                details.append({"memory_id": memory_id, "valid": is_valid})
                if not is_valid:
                    errors.append((memory_id, "Checksum invalid"))
            except Exception as e:
                errors.append((memory_id, str(e)))
        
        return BatchResult(
            success_count=len(details),
            failure_count=len(errors),
            errors=errors,
            details=details
        )
    
    def link_memories(
        self,
        source_id: str,
        target_ids: List[str],
        link_type: str = "related",
        strength: float = 0.5
    ) -> BatchResult:
        """Link one memory to multiple targets."""
        errors = []
        details = []
        
        for target_id in target_ids:
            try:
                self.engine.link_memories(source_id, target_id, link_type, strength)
                details.append({"source": source_id, "target": target_id})
            except Exception as e:
                errors.append((target_id, str(e)))
        
        return BatchResult(
            success_count=len(details),
            failure_count=len(errors),
            errors=errors,
            details=details
        )
