"""
Data validation utilities for the time-aware RAG pipeline.

Provides validation functions to ensure data quality and consistency
throughout the pipeline, with meaningful error messages for debugging.
"""
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

from .logger import setup_logger, DataIngestionError, EmbeddingError, ClusteringError

logger = setup_logger(__name__)

def validate_raw_events(events: List[Dict[str, Any]]) -> List[str]:
    """
    Validate raw event data before normalization.
    
    Args:
        events: List of raw event dictionaries
    
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    if not events:
        errors.append("No events found")
        return errors
    
    # Check for required fields in at least some events
    required_fields = ['ts', 'msg']
    field_coverage = {field: 0 for field in required_fields}
    
    for i, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append(f"Event {i} is not a dictionary")
            continue
            
        for field in required_fields:
            if field in event and event[field] is not None:
                field_coverage[field] += 1
    
    total_events = len(events)
    for field, count in field_coverage.items():
        coverage_pct = (count / total_events) * 100
        if coverage_pct < 50:  # Less than 50% coverage
            errors.append(f"Field '{field}' missing in {100-coverage_pct:.1f}% of events")
    
    return errors

def validate_normalized_events(df: pd.DataFrame) -> List[str]:
    """
    Validate normalized event DataFrame.
    
    Args:
        df: Normalized events DataFrame
    
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    if df.empty:
        errors.append("DataFrame is empty")
        return errors
    
    # Check required columns
    required_cols = ['event_id', 'ts', 'product', 'event_type', 'asset_id', 'msg']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    
    # Check for null values in critical fields
    for col in ['event_id', 'ts']:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                errors.append(f"Column '{col}' has {null_count} null values")
    
    # Validate timestamp format
    if 'ts' in df.columns:
        try:
            pd.to_datetime(df['ts'], utc=True, errors='coerce')
        except Exception as e:
            errors.append(f"Invalid timestamp format: {str(e)}")
    
    # Check for duplicate event IDs
    if 'event_id' in df.columns:
        dup_count = df['event_id'].duplicated().sum()
        if dup_count > 0:
            errors.append(f"Found {dup_count} duplicate event IDs")
    
    return errors

def validate_embeddings(df: pd.DataFrame, expected_dim: int) -> List[str]:
    """
    Validate embedded events DataFrame.
    
    Args:
        df: Embedded events DataFrame
        expected_dim: Expected embedding dimension
    
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    if df.empty:
        errors.append("DataFrame is empty")
        return errors
    
    if 'vector' not in df.columns:
        errors.append("Missing 'vector' column")
        return errors
    
    # Check embedding dimensions
    sample_vectors = df['vector'].dropna().head(100)
    if len(sample_vectors) == 0:
        errors.append("No valid vectors found")
        return errors
    
    for i, vec in enumerate(sample_vectors):
        if not isinstance(vec, (list, np.ndarray)):
            errors.append(f"Vector {i} is not a list or array")
            continue
            
        if len(vec) != expected_dim:
            errors.append(f"Vector {i} has dimension {len(vec)}, expected {expected_dim}")
            break
    
    # Check for NaN/inf values in embeddings
    try:
        vectors_array = np.array(df['vector'].tolist())
        nan_count = np.isnan(vectors_array).sum()
        inf_count = np.isinf(vectors_array).sum()
        
        if nan_count > 0:
            errors.append(f"Found {nan_count} NaN values in embeddings")
        if inf_count > 0:
            errors.append(f"Found {inf_count} infinite values in embeddings")
    except Exception as e:
        errors.append(f"Error validating embedding values: {str(e)}")
    
    return errors

def validate_clustering_results(
    clusters_df: pd.DataFrame, 
    trends_df: pd.DataFrame,
    min_cluster_size: int = 1
) -> List[str]:
    """
    Validate clustering and trend detection results.
    
    Args:
        clusters_df: Weekly clusters DataFrame
        trends_df: Trends summary DataFrame
        min_cluster_size: Minimum expected cluster size
    
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Validate clusters DataFrame
    if clusters_df.empty:
        errors.append("Clusters DataFrame is empty")
    else:
        required_cluster_cols = ['week', 'cluster_id', 'size', 'terms']
        missing_cols = [col for col in required_cluster_cols if col not in clusters_df.columns]
        if missing_cols:
            errors.append(f"Missing cluster columns: {missing_cols}")
        
        # Check for reasonable cluster sizes
        if 'size' in clusters_df.columns:
            small_clusters = (clusters_df['size'] < min_cluster_size).sum()
            if small_clusters > 0:
                errors.append(f"Found {small_clusters} clusters smaller than {min_cluster_size}")
    
    # Validate trends DataFrame
    if trends_df.empty:
        errors.append("Trends DataFrame is empty")
    else:
        required_trend_cols = ['week', 'type', 'cluster_id']
        missing_cols = [col for col in required_trend_cols if col not in trends_df.columns]
        if missing_cols:
            errors.append(f"Missing trend columns: {missing_cols}")
        
        # Check for valid trend types
        if 'type' in trends_df.columns:
            valid_types = {'emergence', 'stable', 'drift', 'growth', 'decay'}
            invalid_types = set(trends_df['type'].unique()) - valid_types
            if invalid_types:
                errors.append(f"Invalid trend types found: {invalid_types}")
    
    return errors

def validate_pipeline_outputs(results_dir: Path) -> Tuple[bool, List[str]]:
    """
    Validate that all expected pipeline outputs exist and are valid.
    
    Args:
        results_dir: Path to results directory
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    if not results_dir.exists():
        errors.append(f"Results directory does not exist: {results_dir}")
        return False, errors
    
    # Check for required output files
    required_files = [
        "clusters_weekly.csv",
        "trends_summary.csv", 
        "trends_summary.jsonl"
    ]
    
    for filename in required_files:
        filepath = results_dir / filename
        if not filepath.exists():
            errors.append(f"Missing required output file: {filename}")
        elif filepath.stat().st_size == 0:
            errors.append(f"Output file is empty: {filename}")
    
    # Check assignments directory
    assignments_dir = results_dir / "assignments"
    if not assignments_dir.exists():
        errors.append("Missing assignments directory")
    elif not list(assignments_dir.glob("*.parquet")):
        errors.append("No assignment files found in assignments directory")
    
    return len(errors) == 0, errors

def check_data_quality_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate data quality metrics for monitoring.
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary of quality metrics
    """
    if df.empty:
        return {"error": "DataFrame is empty"}
    
    metrics = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "null_percentages": (df.isnull().sum() / len(df) * 100).to_dict(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
    }
    
    # Add timestamp-specific metrics if available
    if 'ts' in df.columns:
        ts_series = pd.to_datetime(df['ts'], errors='coerce')
        metrics.update({
            "date_range": {
                "start": ts_series.min().isoformat() if ts_series.min() else None,
                "end": ts_series.max().isoformat() if ts_series.max() else None,
            },
            "invalid_timestamps": ts_series.isnull().sum(),
        })
    
    # Add week distribution if available
    if 'week' in df.columns:
        week_counts = df['week'].value_counts().to_dict()
        metrics["events_per_week"] = {
            "min": min(week_counts.values()) if week_counts else 0,
            "max": max(week_counts.values()) if week_counts else 0,
            "mean": sum(week_counts.values()) / len(week_counts) if week_counts else 0,
        }
    
    return metrics