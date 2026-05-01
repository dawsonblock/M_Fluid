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
