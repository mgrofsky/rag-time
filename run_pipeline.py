#!/usr/bin/env python3
"""
End-to-end pipeline orchestrator for the time-aware RAG system.

This script provides a single entry point to run the complete pipeline
with proper error handling, logging, and validation at each step.

Usage:
    python run_pipeline.py --help
    python run_pipeline.py --config config.yaml --stage all
    python run_pipeline.py --stage embed --validate
"""
import argparse
import sys
from pathlib import Path
from typing import Optional, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import PipelineConfig
from src.logger import setup_logger, log_pipeline_step, log_timing
from src.validation import (
    validate_pipeline_outputs, 
    check_data_quality_metrics,
    validate_normalized_events,
    validate_embeddings,
    validate_clustering_results
)


logger = setup_logger("pipeline_orchestrator")


@log_timing
def run_ingestion(config: PipelineConfig, validate: bool = True) -> bool:
    """Run log ingestion step."""
    try:
        # Import and run log ingestion
        from src.log_ingest import main as ingest_main
        logger.info("Starting log ingestion...")
        ingest_main()
        
        if validate:
            import pandas as pd
            df = pd.read_parquet(config.paths.events_path)
            errors = validate_normalized_events(df)
            if errors:
                logger.warning(f"Validation warnings after ingestion: {errors}")
            
            metrics = check_data_quality_metrics(df)
            logger.info(f"Data quality metrics: {metrics}")
        
        return True
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        return False


@log_timing
def run_embedding(config: PipelineConfig, validate: bool = True) -> bool:
    """Run embedding generation step."""
    try:
        from src.embed_events import main as embed_main
        logger.info("Starting embedding generation...")
        embed_main(dtype=config.model.dtype)
        
        if validate:
            import pandas as pd
            df = pd.read_parquet(config.paths.embedded_events_path)
            errors = validate_embeddings(df, config.model.dimension)
            if errors:
                logger.warning(f"Validation warnings after embedding: {errors}")
        
        return True
    except Exception as e:
        logger.error(f"Embedding failed: {str(e)}")
        return False


@log_timing
def run_clustering(config: PipelineConfig, validate: bool = True) -> bool:
    """Run clustering and trend detection step."""
    try:
        from src.cluster_trends import main as cluster_main
        logger.info("Starting clustering and trend detection...")
        
        cluster_main(
            k=config.clustering.default_k,
            auto_k=config.clustering.auto_k,
            cos_thresh=config.clustering.cos_thresh,
            drift_cos_drop=config.clustering.drift_cos_drop,
            emerge_min=config.clustering.emerge_min,
            term_jaccard_drop=config.clustering.term_jaccard_drop
        )
        
        if validate:
            import pandas as pd
            clusters_df = pd.read_csv(config.paths.results_dir / "clusters_weekly.csv")
            trends_df = pd.read_csv(config.paths.results_dir / "trends_summary.csv")
            
            errors = validate_clustering_results(clusters_df, trends_df)
            if errors:
                logger.warning(f"Validation warnings after clustering: {errors}")
            
            # Validate all pipeline outputs
            is_valid, output_errors = validate_pipeline_outputs(config.paths.results_dir)
            if not is_valid:
                logger.warning(f"Pipeline output validation warnings: {output_errors}")
        
        return True
    except Exception as e:
        logger.error(f"Clustering failed: {str(e)}")
        return False


def run_evaluation(config: PipelineConfig, dataset: str = "synthetic") -> bool:
    """Run evaluation step."""
    try:
        if dataset == "synthetic":
            from src.eval_metrics_synth import main as eval_main
            logger.info("Running synthetic data evaluation...")
            # Note: This would need to be adapted to work with the main() function
        elif dataset == "logon":
            from src.eval_metrics_logon import main as eval_main
            logger.info("Running logon data evaluation...")
        else:
            logger.error(f"Unknown evaluation dataset: {dataset}")
            return False
        
        # eval_main()  # Would need to adapt the evaluation scripts
        logger.info("✓ Evaluation completed (placeholder)")
        return True
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return False


def run_dashboard(config: PipelineConfig) -> bool:
    """Launch the Streamlit dashboard."""
    try:
        import subprocess
        logger.info("Launching Streamlit dashboard...")
        
        # Launch streamlit in a subprocess
        cmd = ["streamlit", "run", "streamlit_app.py", "--server.headless", "true"]
        process = subprocess.Popen(cmd)
        
        logger.info("✓ Dashboard launched successfully")
        logger.info("Access the dashboard at: http://localhost:8501")
        return True
    except Exception as e:
        logger.error(f"Dashboard launch failed: {str(e)}")
        return False


def main():
    """Main pipeline orchestrator."""
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline orchestrator for time-aware RAG system"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "ingest", "embed", "cluster", "evaluate", "dashboard"],
        default="all",
        help="Pipeline stage to run"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation checks after each stage"
    )
    
    parser.add_argument(
        "--eval-dataset",
        type=str,
        choices=["synthetic", "logon"],
        default="synthetic",
        help="Dataset to use for evaluation"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    global logger
    logger = setup_logger("pipeline_orchestrator", level=args.log_level)
    
    # Load configuration
    config = PipelineConfig.from_yaml(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Ensure directories exist
    config.paths.data_dir.mkdir(parents=True, exist_ok=True)
    config.paths.results_dir.mkdir(parents=True, exist_ok=True)
    
    success = True
    
    # Run pipeline stages
    if args.stage in ["all", "ingest"]:
        logger.info("=== STAGE 1: Log Ingestion ===")
        if not run_ingestion(config, validate=args.validate):
            success = False
            if args.stage != "all":
                sys.exit(1)
    
    if args.stage in ["all", "embed"] and success:
        logger.info("=== STAGE 2: Embedding Generation ===")
        if not run_embedding(config, validate=args.validate):
            success = False
            if args.stage != "all":
                sys.exit(1)
    
    if args.stage in ["all", "cluster"] and success:
        logger.info("=== STAGE 3: Clustering & Trend Detection ===")
        if not run_clustering(config, validate=args.validate):
            success = False
            if args.stage != "all":
                sys.exit(1)
    
    if args.stage in ["all", "evaluate"] and success:
        logger.info("=== STAGE 4: Evaluation ===")
        if not run_evaluation(config, dataset=args.eval_dataset):
            success = False
            if args.stage != "all":
                logger.warning("Evaluation failed, continuing...")
    
    if args.stage in ["all", "dashboard"] and success:
        logger.info("=== STAGE 5: Dashboard ===")
        run_dashboard(config)
    
    if success:
        logger.info("🎉 Pipeline completed successfully!")
    else:
        logger.error("❌ Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()