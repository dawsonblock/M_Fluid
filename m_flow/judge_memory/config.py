"""
Judge Memory Configuration

Configuration for the Judge memory subsystem.
All settings have safe defaults.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class JudgeMemoryConfig:
    """
    Configuration for Judge Memory Service.

    All defaults are safe and conservative.
    Fluid memory is disabled by default.
    External drive paths are supported.

    Example:
        config = JudgeMemoryConfig(
            data_dir=Path("/Volumes/JudgeMemory"),
            evidence_dir=Path("/Volumes/JudgeMemory/evidence"),
            sqlite_path=Path("/Volumes/JudgeMemory/judge_memory.sqlite"),
            enable_fluid_memory=False,  # Safe default
        )
    """

    # Storage paths
    data_dir: Path = field(default_factory=lambda: Path("./judge_memory_data"))
    evidence_dir: Optional[Path] = None
    sqlite_path: Optional[Path] = None

    # Feature flags - all disabled by default for safety
    enable_fluid_memory: bool = False
    enable_graph_retrieval: bool = False
    enable_vector_retrieval: bool = False
    enable_llm_contradiction: bool = False

    # Safety settings
    require_review_for_legal_claims: bool = True
    allow_raw_cypher: bool = False
    allow_mutation_tools: bool = False

    # Search settings
    max_search_results: int = 25

    def __post_init__(self):
        """Set derived paths if not provided."""
        if self.evidence_dir is None:
            self.evidence_dir = self.data_dir / "evidence"
        if self.sqlite_path is None:
            self.sqlite_path = self.data_dir / "judge_memory.sqlite"

        # Ensure paths are Path objects
        self.data_dir = Path(self.data_dir)
        self.evidence_dir = Path(self.evidence_dir)
        self.sqlite_path = Path(self.sqlite_path)
