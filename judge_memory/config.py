"""Judge Memory Configuration"""

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class JudgeMemoryConfig(BaseModel):
    """Configuration for Judge Memory subsystem.
    
    Self-contained configuration with no external dependencies
    except Pydantic for validation.
    
    Attributes:
        data_dir: Directory for SQLite database and evidence files
        sqlite_path: Optional explicit SQLite database path
        enable_fluid_memory: Whether to enable fluid memory scoring
        min_claim_confidence: Minimum confidence threshold for claims
        max_claims_per_evidence: Maximum claims linked to one evidence
        enable_audit: Whether to write audit events
    """
    
    data_dir: str = Field(
        default="./judge_memory_data",
        description="Directory for database and evidence storage"
    )
    sqlite_path: Optional[str] = Field(
        default=None,
        description="Optional explicit SQLite database path"
    )
    enable_fluid_memory: bool = Field(
        default=False,
        description="Enable fluid memory scoring (requires m_flow)"
    )
    min_claim_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for claims"
    )
    max_claims_per_evidence: int = Field(
        default=100,
        ge=1,
        description="Maximum claims per evidence record"
    )
    enable_audit: bool = Field(
        default=True,
        description="Enable audit event logging"
    )
    
    @property
    def db_path(self) -> str:
        """Get the SQLite database path."""
        if self.sqlite_path:
            return self.sqlite_path
        return str(Path(self.data_dir) / "judge_memory.db")
    
    @property
    def evidence_dir(self) -> Path:
        """Get the evidence file storage directory."""
        return Path(self.data_dir) / "evidence"
