"""
Log ingestion and normalization pipeline.

This module processes raw log files (JSONL/CSV) and normalizes them into a
standardized schema suitable for downstream analysis. It handles various
timestamp formats, field mappings, and data type conversions to create a
consistent event representation.

The normalization process:
1. Loads raw log files from the logs/ directory
2. Maps fields to a standard schema (product, event_type, asset_id, etc.)
3. Normalizes timestamps to UTC
4. Handles various data formats (JSON, CSV, nested objects)
5. Generates stable event IDs for deduplication
6. Saves normalized data to Parquet format

Supported input formats:
- JSONL: One JSON object per line
- CSV: Comma-separated values with headers
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
OUTP = ROOT / "data" / "events.parquet"

# --- Helper functions for data normalization ---

def _coerce_ts(ts):
    """
    Convert various timestamp formats to UTC datetime.
    
    Handles Unix timestamps (seconds/milliseconds), ISO strings, and other
    common timestamp formats, ensuring all are converted to timezone-aware
    UTC datetime objects.
    """
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        # Handle Unix timestamps (both seconds and milliseconds)
        ts = float(ts)
        if ts > 1e12:   # Likely milliseconds, convert to seconds
            ts /= 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    
    try:
        # Use pandas for flexible timestamp parsing, enforce UTC
        return pd.to_datetime(ts, utc=True)
    except Exception:
        return None

def _stable_id(*parts: str, length: int = 16) -> str:
    """
    Generate a stable, deterministic ID from multiple parts.
    
    Creates a consistent hash-based ID that can be used for deduplication
    across runs. The ID is deterministic based on the input parts.
    """
    key = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:length]

def normalize_record(rec: dict) -> dict:
    """
    Normalize a raw log record into standardized schema.
    
    Maps various field names to a consistent schema, handles data type
    conversions, and generates stable event IDs for deduplication.
    
    Args:
        rec: Raw log record dictionary
    
    Returns:
        Normalized record with standard field names and types
    """
    # Extract timestamp with fallback to common field variants
    ts = rec.get("ts") or rec.get("@timestamp") or rec.get("time") or rec.get("timestamp")
    ts = _coerce_ts(ts)

    # Map to standard field names with fallbacks
    product   = rec.get("product") or rec.get("source") or rec.get("vendor") or "unknown"
    event_type= rec.get("event_type") or rec.get("type")   or rec.get("action") or "unknown"
    asset_id  = rec.get("asset_id")  or rec.get("host")    or rec.get("dst_host") or rec.get("device") or "unknown"
    msg       = rec.get("msg")       or rec.get("message") or ""

    # Handle nested/array fields
    context = rec.get("context") or {}
    tech    = rec.get("tech")    or []
    attack  = rec.get("attack")  or rec.get("attack_techniques") or []
    risk_tag= rec.get("risk_tag")or rec.get("tags") or []

    # Ensure list types for array fields
    if not isinstance(tech, list):     tech = [str(tech)]
    if not isinstance(attack, list):   attack = [str(attack)]
    if not isinstance(risk_tag, list): risk_tag = [str(risk_tag)]
    if not isinstance(context, (dict, list)):
        context = {"raw": str(context)}

    # Generate stable event ID if not present
    event_id = rec.get("event_id") or rec.get("id") or rec.get("_id")
    if not event_id:
        event_id = _stable_id(ts, product, event_type, asset_id, msg)

    return {
        "event_id": str(event_id),
        "ts": ts,
        "product": str(product),
        "event_type": str(event_type),
        "asset_id": str(asset_id),
        "msg": str(msg),
        "context": json.dumps(context, ensure_ascii=False),
        "tech": tech,
        "attack": attack,
        "risk_tag": risk_tag,
    }

def load_jsonl(p: Path) -> list[dict]:
    """
    Load and normalize records from a JSONL file.
    
    Reads one JSON object per line, normalizes each record, and returns
    a list of normalized dictionaries. Skips malformed lines.
    """
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(normalize_record(json.loads(line)))
            except Exception:
                # Skip malformed JSON lines
                continue
    return rows

def load_csv(p: Path) -> list[dict]:
    """
    Load and normalize records from a CSV file.
    
    Reads CSV data, converts each row to a dictionary, and normalizes
    the records using the same schema as JSONL files.
    """
    df = pd.read_csv(p)
    return [normalize_record(r._asdict() if hasattr(r, "_asdict") else dict(r)) for _, r in df.iterrows()]

def main() -> None:
    """
    Main ingestion pipeline.
    
    Loads all log files from the logs/ directory, normalizes them into
    a standard schema, and saves the results to Parquet format.
    """
    rows = []
    
    # Process all JSONL files
    for p in LOGS.glob("*.jsonl"):
        rows.extend(load_jsonl(p))
    
    # Process all CSV files  
    for p in LOGS.glob("*.csv"):
        rows.extend(load_csv(p))

    if not rows:
        print(f"No input logs found in {LOGS} (.jsonl or .csv).")
        return

    # Convert to DataFrame and clean up timestamps
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    # Save normalized data
    OUTP.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTP, index=False, compression="zstd")
    print(f"Wrote {len(df)} events → {OUTP}")

if __name__ == "__main__":
    main()
