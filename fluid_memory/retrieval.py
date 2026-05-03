"""
Fluid Memory Retrieval

Enhanced retrieval with temporal boost, deduplication, and diversity.
"""

from typing import Optional, List, Callable
from time import time
import hashlib

from fluid_memory.models import MemoryItem, RetrievalResult
from fluid_memory.scoring import compute_retrieval_score
from fluid_memory.storage import MemoryStorage


def _content_hash(content: str) -> str:
    """Generate hash for deduplication."""
    return hashlib.sha256(content.lower().strip().encode()).hexdigest()[:16]


def _compute_similarity(content1: str, content2: str) -> float:
    """Simple Jaccard similarity for content deduplication."""
    words1 = set(content1.lower().split())
    words2 = set(content2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)


def temporal_boost(memory: MemoryItem, current_time: float = None) -> float:
    """Compute temporal boost factor based on memory age.
    
    Recent memories get a boost, older memories decay in retrieval priority.
    """
    if current_time is None:
        current_time = time()
    
    age_seconds = current_time - memory.created_at
    # Boost memories less than 1 day old
    if age_seconds < 86400:
        return 1.2
    # Boost memories less than 1 week old
    elif age_seconds < 604800:
        return 1.1
    # Normal for memories 1 week to 1 month
    elif age_seconds < 2592000:
        return 1.0
    # Slight penalty for older memories
    else:
        return 0.9


def remove_duplicates(
    results: List[RetrievalResult],
    similarity_threshold: float = 0.85
) -> List[RetrievalResult]:
    """Remove near-duplicate results based on content similarity."""
    if not results:
        return []
    
    filtered = []
    seen_hashes = set()
    
    for result in results:
        content_hash = _content_hash(result.memory.content)
        
        # Exact duplicate check
        if content_hash in seen_hashes:
            continue
        
        # Near-duplicate check
        is_duplicate = False
        for existing in filtered:
            sim = _compute_similarity(result.memory.content, existing.memory.content)
            if sim >= similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered.append(result)
            seen_hashes.add(content_hash)
    
    return filtered


def maximal_marginal_relevance(
    results: List[RetrievalResult],
    query: str,
    lambda_param: float = 0.5,
    limit: int = 10
) -> List[RetrievalResult]:
    """Apply Maximal Marginal Relevance for result diversity.
    
    Balances relevance to query with diversity from already selected results.
    
    Args:
        results: Initial ranked results
        query: Original query
        lambda_param: Trade-off parameter (0=diversity only, 1=relevance only)
        limit: Maximum results to return
        
    Returns:
        Re-ranked results with diversity consideration
    """
    if not results:
        return []
    
    selected = []
    remaining = list(results)
    
    while remaining and len(selected) < limit:
        max_mmr_score = -float('inf')
        best_result = None
        
        for result in remaining:
            # Relevance component
            relevance = result.score
            
            # Diversity component (max similarity to already selected)
            max_sim = 0.0
            for sel in selected:
                sim = _compute_similarity(result.memory.content, sel.memory.content)
                max_sim = max(max_sim, sim)
            
            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            
            if mmr_score > max_mmr_score:
                max_mmr_score = mmr_score
                best_result = result
        
        if best_result:
            selected.append(best_result)
            remaining.remove(best_result)
        else:
            break
    
    return selected


def retrieve(
    storage: MemoryStorage,
    query: Optional[str] = None,
    tags: Optional[List[str]] = None,
    limit: int = 10,
    enable_temporal_boost: bool = True,
    enable_deduplication: bool = True,
    enable_mmr: bool = False,
    mmr_lambda: float = 0.5,
) -> List[RetrievalResult]:
    """
    Retrieve memories with enhanced scoring and diversity.
    
    Args:
        storage: Storage backend
        query: Text query to match
        tags: Tags to filter by
        limit: Maximum results to return
        enable_temporal_boost: Weight recent memories higher
        enable_deduplication: Remove near-duplicate results
        enable_mmr: Use Maximal Marginal Relevance for diversity
        mmr_lambda: MMR trade-off parameter (0=diversity, 1=relevance)
        
    Returns:
        List of RetrievalResult sorted by score descending
    """
    # Search for candidates (get more for processing)
    candidates = storage.search_memories(query=query, tags=tags, limit=limit * 3)
    
    current_time = time()
    results = []
    
    for memory in candidates:
        # Base score
        score = compute_retrieval_score(memory, query=query, tags=tags)
        
        # Temporal boost
        if enable_temporal_boost:
            score *= temporal_boost(memory, current_time)
        
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
    
    # Apply deduplication
    if enable_deduplication:
        results = remove_duplicates(results)
    
    # Apply MMR for diversity
    if enable_mmr:
        results = maximal_marginal_relevance(results, query or "", mmr_lambda, limit)
    
    # Return top results
    return results[:limit]
