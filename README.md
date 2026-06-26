# Time-Aware Embeddings for Log Trend Detection

A research pipeline for detecting temporal trends in security logs using semantic embeddings and time-aware retrieval. This system automatically clusters log events by week, detects trends like emergence, drift, and decay, and provides time-scoped search capabilities.

## Overview

This project implements a complete pipeline for analyzing log data with temporal awareness:

1. **Log Normalization** - Converts raw logs (JSONL/CSV) into a standardized schema
2. **Semantic Embedding** - Generates vector representations using SentenceTransformer
3. **Weekly Clustering** - Automatically determines optimal cluster counts using elbow method
4. **Trend Detection** - Identifies emergence, drift, decay, and growth patterns
5. **Time-Aware Retrieval** - Balances semantic similarity with temporal recency

## Datasets

The CERT logon dataset used in this research is available at: https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247/1

## Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd time_aware_embeddings_log_trends

# Create and activate virtual environment
python -m venv .timeawarecyber
source .timeawarecyber/bin/activate  # On Windows: .timeawarecyber\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

#### 1. Generate Synthetic Data (Optional)
```bash
python src/sample_logs.py
```

#### 2. Process Your Logs
```bash
# Normalize raw logs (put your JSONL/CSV files in logs/ directory)
python src/log_ingest.py

# Generate semantic embeddings
python src/embed_events.py

# Run clustering with automatic k-selection
python src/cluster_trends.py --auto_k
```

#### 3. Evaluate Results
```bash
# For synthetic data
python src/eval_metrics_synth.py

# For real logon data
python src/eval_metrics_logon.py
```

#### 4. Launch Web Dashboard
```bash
streamlit run streamlit_app.py
```

## Detailed Usage

### Log Ingestion

Place your log files in the `logs/` directory. Supported formats:
- **JSONL**: One JSON object per line
- **CSV**: Comma-separated with headers

The system normalizes various field names to a standard schema:
- `ts`/`@timestamp`/`time` → `ts` (timestamp)
- `product`/`source`/`vendor` → `product`
- `event_type`/`type`/`action` → `event_type`
- `asset_id`/`host`/`device` → `asset_id`
- `msg`/`message` → `msg`

### Clustering and Trend Detection

The system automatically:
- Determines optimal cluster counts per week using the elbow method
- Clusters events using K-means with cosine similarity
- Matches clusters across weeks using one-to-one assignment
- Detects trends based on:
  - **Cosine drift**: Centroid movement between weeks
  - **Term drift**: Changes in cluster vocabulary (Jaccard similarity)
  - **Volume trends**: Growth/decay based on cluster sizes

### Time-Aware Retrieval

The system supports time-scoped queries that balance semantic similarity with temporal recency:

```python
# Fused score = α × cosine_similarity + (1-α) × time_weight
# Where time_weight decays exponentially with age
```

## Command Line Options

### Clustering Pipeline
```bash
python src/cluster_trends.py [options]

Options:
  --k K                    Number of clusters per week (default: 6)
  --auto_k                 Automatically select optimal k using elbow method
  --cos_thresh FLOAT       Cosine similarity threshold for cluster matching (default: 0.5)
  --drift_cos_drop FLOAT   Threshold for cosine-based drift detection (default: 0.2)
  --emerge_min INT         Minimum size for growth trend classification (default: 30)
  --term_jaccard_drop FLOAT Threshold for term-based drift detection (default: 0.60)
```

### Evaluation Scripts
```bash
python src/eval_metrics_synth.py [options]
python src/eval_metrics_logon.py [options]

Options:
  --k_min INT              Minimum k for elbow method (default: 2)
  --k_max INT              Maximum k for elbow method (default: 10)
  --no_plots               Skip generating elbow plots
  --force_rerun            Force rerun clustering even if results exist
  --no_sensitivity         Skip alpha sensitivity analysis
```

## Output Files

The pipeline generates several output files in the `results/` directory:

- **`clusters_weekly.csv`** - Weekly cluster summaries with terms and sizes
- **`trends_summary.csv`** - Trend classifications for each cluster
- **`trends_summary.jsonl`** - Same data in JSONL format
- **`assignments/`** - Per-week event-to-cluster assignments (Parquet)
- **`elbow_week_*.png`** - Elbow plots for each week
- **`alpha_sensitivity_*.csv`** - Sensitivity analysis results

## Web Dashboard

The Streamlit dashboard provides an interactive interface for exploring results:

- **As-of filtering** - Simulate historical analysis by selecting cutoff weeks
- **Trend exploration** - Filter and visualize different trend types
- **Cluster inspection** - Browse weekly clusters and their characteristics

## Research Applications

This system is designed for research in:
- **Security log analysis** - Detecting evolving attack patterns
- **IT operations monitoring** - Identifying infrastructure changes
- **Temporal data mining** - Understanding how data patterns change over time
- **Time-aware information retrieval** - Balancing relevance with recency

## Configuration

### Embedding Model
The system uses `all-MiniLM-L6-v2` by default (384 dimensions). To change:
```python
# In src/embed_events.py
MODEL_NAME = 'your-preferred-model'
```

### Time Decay Parameters
Adjust temporal weighting in evaluation scripts:
```python
# Half-life for time decay (days)
half_life_days = 14

# Balance between similarity and recency
alpha = 0.7  # 0.0 = pure recency, 1.0 = pure similarity
```

## Troubleshooting

### Common Issues

**Missing dependencies:**
```bash
pip install -r requirements.txt
```

**No input logs found:**
- Ensure log files are in `logs/` directory
- Check file extensions (.jsonl or .csv)

**Memory issues with large datasets:**
- Use `--dtype float16` in `embed_events.py`
- Process data in smaller time windows

**Poor clustering results:**
- Try different k ranges with `--k_min` and `--k_max`
- Adjust drift detection thresholds
- Check data quality and preprocessing

### Performance Tips

- Use `--no_plots` for faster evaluation runs
- Use `--no_sensitivity` to skip alpha analysis
- Process data in weekly batches for very large datasets
- Consider using `float16` embeddings to reduce memory usage

## Citation

If you use this system in your research, please cite:

```bibtex
@misc{rag_time_2025,
  title={Time-Aware Embeddings for Log Trend Detection},
  author={Matt Grofsky},
  year={2025},
  howpublished={\url{https://github.com/mattgrofsky/rag-time}},
  note={GitHub repository}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
