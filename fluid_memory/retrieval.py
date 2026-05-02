"""
Fluid Memory Retrieval

Retrieval with fluid state scoring.
"""

from typing import Optional, List

from fluid_memory.models import MemoryItem, RetrievalResult
from fluid_memory.scoring import compute_retrieval_score
from fluid_memory.storage import MemoryStorage


def retrieve(
    storage: MemoryStorage,
    query: Optional[str] = None,
    tags: Optional[List[str]] = None,
    limit: int = 10,
) -> List[RetrievalResult]:
    """
    Retrieve memories with fluid state scoring.
    
    Search stored memories, compute retrieval score for each candidate,
    and sort by score descending.
    
    Args:
        storage: Storage backend
        query: Text query to match
        tags: Tags to filter by
        limit: Maximum results to return
        
    Returns:
        List of RetrievalResult sorted by score descending
    """
    # Search for candidates
    candidates = storage.search_memories(query=query, tags=tags, limit=limit * 2)
    
    # Score each candidate
    results = []
    for memory in candidates:
        score = compute_retrieval_score(memory, query=query, tags=tags)
        
        # Determine match type
        if query and tags:
            if query.lower() in memory.content.lower():
                match_type = "combined"
            else:
                match_type = "tag"
        elif query and query.lower() in memory.content.lower():
            match_type = "text"
        elif tags:
            match_type = "tag"
        else:
            match_type = "salience"
        
        results.append(RetrievalResult(
            memory=memory,
            score=score,
            match_type=match_type,
        ))
    
    # Sort by score descending
    results.sort(key=lambda r: r.score, reverse=True)
    
    # Return top results
    return results[:limit]
