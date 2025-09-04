"""
Event embedding pipeline using SentenceTransformer for semantic vectorization.

This module processes normalized event data and generates semantic embeddings using
a pre-trained transformer model. The embeddings capture semantic meaning rather than
just token-based features, enabling better similarity matching and clustering.

The pipeline:
1. Loads normalized events from Parquet
2. Builds text representations from event fields
3. Generates semantic embeddings using SentenceTransformer
4. Saves results with configurable precision (float16/float32)

Dependencies:
    sentence-transformers: For semantic embedding generation
    scikit-learn: For vector normalization
"""
from __future__ import annotations
import json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

# Import sentence-transformers with helpful error message if missing
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers not found. Install with: pip install sentence-transformers"
    )

ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "data" / "events.parquet"
OUT = ROOT / "data" / "events_embedded.parquet"

# Using all-MiniLM-L6-v2: good balance of performance and speed
MODEL_NAME = 'all-MiniLM-L6-v2'
MODEL_DIM = 384  # Fixed embedding dimension for this model

def build_text_repr(row: pd.Series) -> str:
    """
    Build a standardized text representation from event data.
    
    Combines key event fields into a single string that captures the essential
    information for semantic embedding. This includes product, event type, asset,
    message, and contextual metadata.
    
    Args:
        row: Pandas Series containing event data with fields like 'product',
             'event_type', 'asset_id', 'msg', 'context', 'tech', 'attack', 'risk_tag'
    
    Returns:
        String representation suitable for embedding
    """
    parts = [
        f"[source={row['product']}]",
        f"[type={row['event_type']}]",
        f"[entity={row['asset_id']}]",
        f"msg={row['msg'] or ''}",
    ]
    try:
        ctx = json.loads(row["context"]) if isinstance(row["context"], str) else row["context"]
    except Exception:
        ctx = {}
    if isinstance(ctx, dict):
        for k, v in list(ctx.items())[:6]:
            parts.append(f"{k}={v}")
    if isinstance(row.get("tech"), list) and row["tech"]:
        parts.append("tech=" + ",".join(map(str, row["tech"])))
    if isinstance(row.get("attack"), list) and row["attack"]:
        parts.append("attack=" + ",".join(map(str, row["attack"])))
    if isinstance(row.get("risk_tag"), list) and row["risk_tag"]:
        parts.append("risk=" + ",".join(map(str, row["risk_tag"])))
    return " ".join(parts)

def main(dtype: str = "float16") -> None:
    """
    Main pipeline function for event embedding.
    
    Loads normalized events, builds text representations, generates semantic
    embeddings using SentenceTransformer, and saves the results to Parquet.
    
    Args:
        dtype: Data type for storing embeddings ('float16' or 'float32').
               float16 saves space but may have precision loss.
    """
    if not INP.exists():
        print(f"Missing {INP}. Run: python src/log_ingest.py")
        return

    print("Loading normalized events...")
    df = pd.read_parquet(INP)
    
    print("Building text representations...")
    df["text_repr"] = df.apply(build_text_repr, axis=1)

    print(f"Loading sentence transformer model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding events into semantic vectors (this may take a moment)...")
    texts = df["text_repr"].tolist()
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    np_dtype = np.float16 if dtype == "float16" else np.float32
    embeddings = embeddings.astype(np_dtype)

    df["vector"] = [v.tolist() for v in embeddings]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False, compression="zstd")
    print(f"Wrote {len(df)} semantically embedded events → {OUT} (dim={MODEL_DIM}, dtype={dtype})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate semantic embeddings for normalized events using SentenceTransformer"
    )
    parser.add_argument(
        "--dtype", 
        type=str, 
        default="float16", 
        choices=["float16", "float32"],
        help="Precision for storing embeddings (float16 saves space, float32 more precise)"
    )
    args = parser.parse_args()
    main(dtype=args.dtype)
