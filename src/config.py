"""
Centralized configuration management for the time-aware RAG pipeline.

This module provides a single source of truth for all configuration parameters,
making the system more maintainable and allowing for easy experimentation.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import os

@dataclass
class ModelConfig:
    """Configuration for embedding models."""
    name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    dtype: str = "float16"  # or "float32"
    batch_size: int = 32

@dataclass
class ClusteringConfig:
    """Configuration for clustering parameters."""
    default_k: int = 6
    auto_k: bool = False
    k_min: int = 2
    k_max: int = 10
    cos_thresh: float = 0.5
    drift_cos_drop: float = 0.2
    term_jaccard_drop: float = 0.60
    emerge_min: int = 30
    min_events_per_week: int = 5

@dataclass
class TimeAwareConfig:
    """Configuration for time-aware retrieval."""
    half_life_days: float = 14.0
    alpha: float = 0.7  # Balance between similarity and recency

@dataclass
class PathConfig:
    """Configuration for file paths."""
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    
    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"
    
    @property
    def data_dir(self) -> Path:
        return self.root / "data"
    
    @property
    def results_dir(self) -> Path:
        return self.root / "results"
    
    @property
    def events_path(self) -> Path:
        return self.data_dir / "events.parquet"
    
    @property
    def embedded_events_path(self) -> Path:
        return self.data_dir / "events_embedded.parquet"

@dataclass
class PipelineConfig:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    time_aware: TimeAwareConfig = field(default_factory=TimeAwareConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load configuration from YAML file."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = yaml.safe_load(f) or {}
            return cls(**data)
        return cls()
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        # Convert to dict, handling Path objects
        data = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, Path):
                data[field_name] = str(field_value)
            elif hasattr(field_value, '__dict__'):
                data[field_name] = field_value.__dict__
            else:
                data[field_name] = field_value
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)

# Global configuration instance
config = PipelineConfig.from_yaml("config.yaml")