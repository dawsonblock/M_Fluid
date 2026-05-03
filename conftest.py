"""
Root conftest.py for pytest configuration.
Ensures proper Python path setup for test discovery.

This file uses pytest_configure which is one of the earliest hooks,
executed before any test collection begins.
"""

import sys
from pathlib import Path

# Add project root to Python path IMMEDIATELY when this file is loaded
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def pytest_configure(config):
    """
    Called after command line options have been parsed and all plugins
    and initial conftest files been loaded.

    This is one of the earliest hooks, executed before test collection.
    """
    # Ensure project root is in sys.path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def pytest_sessionfinish(session, exitstatus):
    """Clean up after test session."""
    print("Running teardown with pytest sessionfinish...")
