"""
Test Fluid Memory scoring functions.
"""

import pytest
from m_flow.memory.fluid.scoring import (
    compute_salience,
    compute_confidence,
    compute_volatility,
    compute_stability,
    compute_retrieval_score,
)
from m_flow.memory.fluid.models import MemoryItem


def test_compute_salience_access_boost():
    """Test that access increases salience."""
    base = 0.5
    
    s0 = compute_salience(base, 0, 0, 0)
    s1 = compute_salience(base, 5, 0, 0)  # 5 accesses
    s2 = compute_salience(base, 10, 0, 0)  # 10 accesses
    
    assert s1 > s0
    assert s2 > s1
    assert s2 <= 1.0  # Capped at 1.0


def test_compute_salience_reinforcement_boost():
    """Test that reinforcement increases salience."""
    base = 0.5
    
    s0 = compute_salience(base, 0, 0, 0)
    s1 = compute_salience(base, 0, 3, 0)  # 3 reinforcements
    
    assert s1 > s0


def test_compute_salience_contradiction_boost():
    """Test that contradictions may increase salience (attention)."""
    base = 0.5
    
    s0 = compute_salience(base, 0, 0, 0)
    s1 = compute_salience(base, 0, 0, 2)  # 2 contradictions
    
    assert s1 > s0  # Contradictions are attention-grabbing


def test_compute_salience_bounds():
    """Test that salience stays in bounds."""
    # High values should be clamped
    s = compute_salience(0.9, 100, 100, 100)
    assert 0.0 <= s <= 1.0


def test_compute_confidence_reinforcement():
    """Test that reinforcement increases confidence."""
    base = 0.5
    
    c0 = compute_confidence(base, 0, 0)
    c1 = compute_confidence(base, 5, 0)  # 5 reinforcements
    
    assert c1 > c0


def test_compute_confidence_contradiction():
    """Test that contradictions decrease confidence."""
    base = 0.5
    
    c0 = compute_confidence(base, 0, 0)
    c1 = compute_confidence(base, 0, 3)  # 3 contradictions
    
    assert c1 < c0


def test_compute_confidence_bounds():
    """Test that confidence stays in bounds."""
    c = compute_confidence(0.9, 100, 0)  # Many reinforcements
    assert 0.0 <= c <= 1.0
    
    c2 = compute_confidence(0.1, 0, 100)  # Many contradictions
    assert 0.0 <= c2 <= 1.0


def test_compute_volatility_contradiction():
    """Test that contradictions increase volatility."""
    base = 0.3
    
    v0 = compute_volatility(base, 0, 0.5)
    v1 = compute_volatility(base, 3, 0.5)  # 3 contradictions
    
    assert v1 > v0


def test_compute_volatility_stability():
    """Test that stability reduces volatility."""
    base = 0.5
    contradictions = 2
    
    v_high_stability = compute_volatility(base, contradictions, 0.9)
    v_low_stability = compute_volatility(base, contradictions, 0.1)
    
    assert v_high_stability < v_low_stability


def test_compute_stability_reinforcement():
    """Test that reinforcement increases stability."""
    base = 0.5
    
    s0 = compute_stability(base, 0, 0)
    s1 = compute_stability(base, 5, 0)  # 5 reinforcements
    
    assert s1 > s0


def test_compute_stability_contradiction():
    """Test that contradictions decrease stability."""
    base = 0.5
    
    s0 = compute_stability(base, 0, 0)
    s1 = compute_stability(base, 0, 3)  # 3 contradictions
    
    assert s1 < s0


def test_compute_retrieval_score_text_match():
    """Test that text match increases score."""
    memory = MemoryItem(content="Python programming language")
    
    score_match = compute_retrieval_score(memory, query="Python")
    score_no_match = compute_retrieval_score(memory, query="JavaScript")
    
    assert score_match > score_no_match


def test_compute_retrieval_score_tag_match():
    """Test that tag match increases score."""
    memory = MemoryItem(content="Content", tags=["python", "coding"])
    
    score_match = compute_retrieval_score(memory, tags=["python"])
    score_no_match = compute_retrieval_score(memory, tags=["javascript"])
    
    assert score_match > score_no_match


def test_compute_retrieval_score_salience():
    """Test that higher salience increases score."""
    m1 = MemoryItem(content="Test", salience=0.9)
    m2 = MemoryItem(content="Test", salience=0.1)
    
    s1 = compute_retrieval_score(m1)
    s2 = compute_retrieval_score(m2)
    
    assert s1 > s2


def test_compute_retrieval_score_confidence():
    """Test that higher confidence increases score."""
    m1 = MemoryItem(content="Test", confidence=0.9)
    m2 = MemoryItem(content="Test", confidence=0.1)
    
    s1 = compute_retrieval_score(m1)
    s2 = compute_retrieval_score(m2)
    
    assert s1 > s2


def test_compute_retrieval_score_bounds():
    """Test that retrieval score stays in bounds."""
    memory = MemoryItem(content="Test")
    
    score = compute_retrieval_score(memory)
    assert 0.0 <= score <= 1.0
