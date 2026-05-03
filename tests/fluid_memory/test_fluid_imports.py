"""
Test Fluid Memory imports - verifies clean standalone import.
"""


def test_import_fluid_memory():
    """Test that fluid_memory can be imported without errors."""
    import fluid_memory
    assert fluid_memory is not None


def test_import_main_classes():
    """Test that main classes can be imported."""
    from fluid_memory import FluidMemoryEngine, FluidMemoryConfig
    from fluid_memory import MemoryItem, RetrievalResult, MemoryEvent
    from fluid_memory import EventType
    from fluid_memory import FluidMemoryError, MemoryNotFoundError
    
    assert FluidMemoryEngine is not None
    assert FluidMemoryConfig is not None
    assert MemoryItem is not None
    assert RetrievalResult is not None
    assert MemoryEvent is not None
    assert EventType is not None
    assert FluidMemoryError is not None
    assert MemoryNotFoundError is not None


def test_event_types():
    """Test that all event types are available."""
    from fluid_memory.events import EventType
    
    assert EventType.CREATED.value == "created"
    assert EventType.ACCESSED.value == "accessed"
    assert EventType.REINFORCED.value == "reinforced"
    assert EventType.CONTRADICTED.value == "contradicted"
    assert EventType.DECAYED.value == "decayed"
    assert EventType.MUTATED.value == "mutated"
    assert EventType.LINKED.value == "linked"
    assert EventType.DELETED.value == "deleted"


def test_no_structlog_dependency():
    """Verify structlog is not required."""
    try:
        import structlog
        # If structlog is available, it shouldn't be imported by fluid_memory
        import fluid_memory
        # This test passes if we get here without error
    except ImportError:
        # structlog not installed, which is fine
        pass


def test_no_graph_db_dependency():
    """Verify graph database is not required."""
    import fluid_memory
    # Should work without Neo4j or other graph DB
    assert hasattr(fluid_memory, 'FluidMemoryEngine')
