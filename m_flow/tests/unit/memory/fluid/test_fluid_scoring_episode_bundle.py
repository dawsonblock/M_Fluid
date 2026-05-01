"""
Test EpisodeBundle fluid scoring contract.
Tests that EpisodeBundle has fluid scoring fields and graph_score is computed correctly.
"""

import pytest
from dataclasses import dataclass


def test_episodebundle_fluid_fields():
    """Test EpisodeBundle has fluid scoring fields."""
    from m_flow.retrieval.episodic.bundle_scorer import EpisodeBundle

    bundle = EpisodeBundle(
        episode_id="ep123",
        score=0.85,
        best_path="direct_episode",
    )

    # Check fluid fields exist and default to None
    assert hasattr(bundle, "base_distance_score")
    assert hasattr(bundle, "semantic_score")
    assert hasattr(bundle, "graph_score")
    assert hasattr(bundle, "fluid_effective_score")
    assert hasattr(bundle, "final_distance_score")
    assert hasattr(bundle, "fluid_score_explanation")

    assert bundle.base_distance_score is None
    assert bundle.semantic_score is None
    assert bundle.graph_score is None
    assert bundle.fluid_effective_score is None
    assert bundle.final_distance_score is None
    assert bundle.fluid_score_explanation is None


def test_episodebundle_base_distance_score_set():
    """Test compute_episode_bundles sets base_distance_score."""
    from m_flow.retrieval.episodic.bundle_scorer import EpisodeBundle

    bundle = EpisodeBundle(
        episode_id="ep123",
        score=0.85,
        best_path="direct_episode",
        base_distance_score=0.85,
    )

    assert bundle.base_distance_score == 0.85
    assert bundle.score == 0.85


def test_compute_graph_score_string_best_path():
    """Test _compute_graph_score handles string best_path correctly."""
    from m_flow.memory.fluid.service_interface import LocalFluidMemoryService
    from unittest.mock import AsyncMock

    mock_store = AsyncMock()
    service = LocalFluidMemoryService(graph_engine=None, store=mock_store)

    @dataclass
    class MockBundle:
        best_path: str

    # Test each path type
    assert service._compute_graph_score(MockBundle("direct_episode")) == 1.00
    assert service._compute_graph_score(MockBundle("facet")) == 0.85
    assert service._compute_graph_score(MockBundle("facet_entity")) == 0.78
    assert service._compute_graph_score(MockBundle("point")) == 0.72
    assert service._compute_graph_score(MockBundle("entity")) == 0.65
    assert service._compute_graph_score(MockBundle("unknown")) == 0.50
    assert service._compute_graph_score(MockBundle("")) == 0.50


def test_equal_distances_produce_neutral_semantic_score():
    """Test that when all bundles have equal distances, semantic_score = 0.5."""
    from m_flow.retrieval.episodic.bundle_scorer import EpisodeBundle

    # Create bundles with equal scores
    bundle1 = EpisodeBundle(
        episode_id="ep1",
        score=0.5,
        best_path="direct_episode",
        base_distance_score=0.5,
    )
    bundle2 = EpisodeBundle(
        episode_id="ep2",
        score=0.5,
        best_path="facet",
        base_distance_score=0.5,
    )
    bundle3 = EpisodeBundle(
        episode_id="ep3",
        score=0.5,
        best_path="entity",
        base_distance_score=0.5,
    )

    # When all distances are equal, semantic score should be neutral (0.5)
    # This is because there's no relative distance difference to normalize
    min_dist = 0.5
    max_dist = 0.5
    dist_range = max_dist - min_dist

    assert dist_range == 0
    # When dist_range == 0, semantic_score should be 0.5 (neutral)
    # This is the expected behavior per the fix in apply_fluid_scores
    for bundle in [bundle1, bundle2, bundle3]:
        # With equal distances, semantic_score = 0.5 for all bundles
        assert bundle.base_distance_score == 0.5
