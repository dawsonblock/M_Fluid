"""
Fluid Memory Scoring

Pure functions for computing memory state scores.
"""

from typing import Optional, List
from fluid_memory.models import MemoryItem
from fluid_memory.state import clamp01


def compute_salience(
    base_salience: float,
    access_count: int,
    reinforcement_count: int,
    contradiction_count: int,
    access_boost: float = 0.02,
    reinforcement_boost: float = 0.1,
) -> float:
    """
    Compute salience based on access and reinforcement.
    
    Args:
        base_salience: Current salience value
        access_count: Number of times accessed
        reinforcement_count: Number of reinforcements
        contradiction_count: Number of contradictions (may increase salience)
        access_boost: Salience increase per access
        reinforcement_boost: Salience increase per reinforcement
        
    Returns:
        Updated salience in [0.0, 1.0]
    """
    # Access increases salience slightly
    access_bonus = min(access_count * access_boost, 0.2)  # Cap at 0.2
    
    # Reinforcement increases salience
    reinforcement_bonus = min(reinforcement_count * reinforcement_boost, 0.3)  # Cap at 0.3
    
    # Contradictions may increase salience (attention-grabbing)
    contradiction_bonus = min(contradiction_count * 0.05, 0.1)  # Cap at 0.1
    
    new_salience = base_salience + access_bonus + reinforcement_bonus + contradiction_bonus
    return clamp01(new_salience)


def compute_confidence(
    base_confidence: float,
    reinforcement_count: int,
    contradiction_count: int,
    reinforcement_boost: float = 0.1,
    contradiction_penalty: float = 0.1,
) -> float:
    """
    Compute confidence based on reinforcements and contradictions.
    
    Args:
        base_confidence: Current confidence value
        reinforcement_count: Number of reinforcements
        contradiction_count: Number of contradictions
        reinforcement_boost: Confidence increase per reinforcement
        contradiction_penalty: Confidence decrease per contradiction
        
    Returns:
        Updated confidence in [0.0, 1.0]
    """
    # Reinforcement increases confidence
    reinforcement_bonus = min(reinforcement_count * reinforcement_boost, 0.3)
    
    # Contradiction decreases confidence
    contradiction_penalty_total = min(contradiction_count * contradiction_penalty, 0.5)
    
    new_confidence = base_confidence + reinforcement_bonus - contradiction_penalty_total
    return clamp01(new_confidence)


def compute_volatility(
    base_volatility: float,
    contradiction_count: int,
    stability: float,
    contradiction_increase: float = 0.05,
) -> float:
    """
    Compute volatility based on contradictions and stability.
    
    Higher volatility = more likely to change.
    
    Args:
        base_volatility: Current volatility value
        contradiction_count: Number of contradictions
        stability: Stability value (higher = less volatile)
        contradiction_increase: Volatility increase per contradiction
        
    Returns:
        Updated volatility in [0.0, 1.0]
    """
    # Contradictions increase volatility
    contradiction_bonus = min(contradiction_count * contradiction_increase, 0.3)
    
    # Stability reduces volatility
    stability_reduction = stability * 0.2
    
    new_volatility = base_volatility + contradiction_bonus - stability_reduction
    return clamp01(new_volatility)


def compute_stability(
    base_stability: float,
    reinforcement_count: int,
    contradiction_count: int,
    reinforcement_increase: float = 0.03,
    contradiction_decrease: float = 0.05,
) -> float:
    """
    Compute stability based on reinforcements and contradictions.
    
    Higher stability = more resistant to decay and mutation.
    
    Args:
        base_stability: Current stability value
        reinforcement_count: Number of reinforcements
        contradiction_count: Number of contradictions
        reinforcement_increase: Stability increase per reinforcement
        contradiction_decrease: Stability decrease per contradiction
        
    Returns:
        Updated stability in [0.0, 1.0]
    """
    # Reinforcement increases stability
    reinforcement_bonus = min(reinforcement_count * reinforcement_increase, 0.2)
    
    # Contradiction decreases stability
    contradiction_penalty = min(contradiction_count * contradiction_decrease, 0.3)
    
    new_stability = base_stability + reinforcement_bonus - contradiction_penalty
    return clamp01(new_stability)


def compute_retrieval_score(
    memory: MemoryItem,
    query: Optional[str] = None,
    tags: Optional[List[str]] = None,
    text_weight: float = 0.4,
    tag_weight: float = 0.2,
    salience_weight: float = 0.2,
    confidence_weight: float = 0.1,
    recency_weight: float = 0.1,
) -> float:
    """
    Compute overall retrieval score for a memory.
    
    Combines text match, tag match, salience, confidence, and recency.
    
    Args:
        memory: The memory item to score
        query: Text query to match against content
        tags: Tags to match
        text_weight: Weight for text matching
        tag_weight: Weight for tag matching
        salience_weight: Weight for salience
        confidence_weight: Weight for confidence
        recency_weight: Weight for recency (last_accessed_at)
        
    Returns:
        Score in [0.0, 1.0]
    """
    score = 0.0
    
    # Text match
    if query and query.lower() in memory.content.lower():
        score += text_weight
    
    # Tag match
    if tags:
        matching_tags = len(set(memory.tags) & set(tags))
        if matching_tags > 0:
            tag_match_ratio = matching_tags / len(tags)
            score += tag_weight * tag_match_ratio
    
    # Salience
    score += salience_weight * memory.salience
    
    # Confidence (lower confidence = lower score)
    score += confidence_weight * memory.confidence
    
    # Recency (higher access count = more recent use)
    recency_score = min(memory.access_count / 10, 1.0)  # Normalize by 10
    score += recency_weight * recency_score
    
    return clamp01(score)
