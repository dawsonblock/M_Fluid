"""Fluid Memory Configuration"""

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class FluidMemoryConfig(BaseModel):
    """Configuration for Fluid Memory engine.

    All parameters have safe defaults for standalone use.
    """

    data_dir: Path = Field(
        default=Path("./fluid_memory_data"),
        description="Directory for SQLite database and storage"
    )

    db_path: Optional[Path] = None

    # Decay configuration
    default_decay_rate: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Default daily decay rate for salience"
    )

    legal_decay_rate: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Decay rate for legal lane (slower)"
    )

    trust_decay_rate: float = Field(
        default=0.03,
        ge=0.0,
        le=1.0,
        description="Decay rate for trust lane"
    )

    interest_decay_rate: float = Field(
        default=0.08,
        ge=0.0,
        le=1.0,
        description="Decay rate for interest lane (faster)"
    )

    attention_decay_rate: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Decay rate for attention lane (fastest)"
    )

    # Reinforcement
    reinforcement_boost: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Salience boost per reinforcement"
    )

    # Contradiction
    contradiction_penalty: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Confidence penalty per contradiction"
    )

    # Retrieval
    retrieval_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum score for retrieval results"
    )

    max_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum retrieval results"
    )

    def model_post_init(self, __context) -> None:
        """Set up derived paths after initialization."""
        if self.db_path is None:
            self.db_path = self.data_dir / "fluid_memory.db"
        self.data_dir.mkdir(parents=True, exist_ok=True)
