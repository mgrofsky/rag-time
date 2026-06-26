"""
Weekly clustering and trend detection pipeline.

This module performs time-series clustering analysis on embedded events, detecting
trends like emergence, drift, decay, and growth across weekly time windows. It uses
K-means clustering with optional automatic k-selection and one-to-one cluster
matching between consecutive weeks.

Key features:
- Weekly clustering with configurable or auto-selected k values
- One-to-one cluster matching to track evolution over time
- Trend detection based on cosine similarity, term overlap, and volume changes
- Support for emergence, drift, decay, and growth trend types

The pipeline processes embedded events and outputs:
- clusters_weekly.csv: Weekly cluster summaries with terms and sizes
- trends_summary.csv: Trend classifications for each cluster
- assignments/: Per-week event-to-cluster assignments
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "data" / "events_embedded.parquet"
OUT = ROOT / "results"

# ----------------------------- Helper Functions -----------------------------

def week_key(ts) -> str:
    """
    Convert timestamp to ISO week format (YYYY-WNN).
    
    Args:
        ts: Timestamp (string, datetime, or pandas timestamp)
    
    Returns:
        ISO week string like "2025-W13"
    """
    iso = pd.to_datetime(ts, utc=True).isocalendar()
    return f"{int(iso.year)}-W{int(iso.week):02d}"

def top_terms(texts: list[str], n: int = 6) -> list[str]:
    """
    Extract top N most frequent terms from a collection of texts.
    
    Uses TF-IDF vectorization to identify the most important terms, filtering
    out common English stop words.
    
    Args:
        texts: List of text strings to analyze
        n: Number of top terms to return
    
    Returns:
        List of top N terms by frequency
    """
    vec = CountVectorizer(max_features=3000, stop_words="english")
    X = vec.fit_transform(texts)
    sums = np.asarray(X.sum(axis=0)).ravel()
    idxs = sums.argsort()[::-1][:n]
    vocab = np.array(vec.get_feature_names_out())
    return vocab[idxs].tolist()

def pick_k(X: np.ndarray, k_min: int = 2, k_max: int = 10) -> int:
    """
    Automatically select optimal k for K-means clustering using silhouette score.
    
    Performs a grid search over k values and selects the one with the highest
    cosine-based silhouette score, which measures cluster quality.
    
    Args:
        X: Feature matrix for clustering
        k_min: Minimum k to test
        k_max: Maximum k to test
    
    Returns:
        Optimal k value based on silhouette score
    """
    best_k, best_s = k_min, -1.0
    n = len(X)
    if n <= k_min:
        return max(1, n)
    for k in range(k_min, min(k_max, n) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        labs = km.labels_
        if len(set(labs)) < 2:
            continue
        try:
            s = silhouette_score(X, labs, metric="cosine")
        except Exception:
            s = -1.0
        if s > best_s:
            best_s, best_k = s, k
    return best_k

def cluster_week(df_week: pd.DataFrame, k: int = 6, auto_k: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """
    Perform K-means clustering on a week's worth of events.
    
    Clusters the embedded vectors for events in a single week, optionally
    using automatic k-selection. Returns both the labeled dataframe and
    cluster metadata including centroids and top terms.
    
    Args:
        df_week: DataFrame containing events for one week with 'vector' column
        k: Number of clusters (ignored if auto_k=True)
        auto_k: Whether to automatically select optimal k
    
    Returns:
        Tuple of (labeled_dataframe, cluster_metadata_list)
    """
    X = np.stack(df_week["vector"].to_numpy())
    if auto_k:
        k = pick_k(X, 2, 10)
    k = min(k, len(df_week)) if len(df_week) > 0 else 1
    if k <= 1:
        labels = np.zeros(len(df_week), dtype=int)
        centroids = np.mean(X, axis=0, keepdims=True)
    else:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        centroids = km.cluster_centers_

    df_week = df_week.copy()
    df_week["label"] = labels

    rows = []
    for lab in sorted(set(labels)):
        sub = df_week[df_week["label"] == lab]
        rows.append({
            "label": int(lab),
            "size": int(len(sub)),
            "terms": top_terms(sub["text_repr"].tolist()),
            "centroid": centroids[lab if k > 1 else 0].tolist(),
        })
    return df_week, rows

def match_clusters_one_to_one(curr_rows: list[dict], prev_rows: list[dict], cos_thresh: float = 0.5) -> tuple[dict, list[float]]:
    """
    Match current week clusters to previous week clusters using cosine similarity.
    
    Performs one-to-one matching to prevent duplicate assignments. Uses a greedy
    approach that prioritizes highest similarity matches above the threshold.
    
    Args:
        curr_rows: Current week cluster metadata with 'centroid' field
        prev_rows: Previous week cluster metadata with 'centroid' field  
        cos_thresh: Minimum cosine similarity threshold for matching
    
    Returns:
        Tuple of (label_mapping_dict, similarity_scores_list)
    """
    if not prev_rows:
        return {r["label"]: None for r in curr_rows}, [1.0] * len(curr_rows)
    C = np.array([r["centroid"] for r in curr_rows])
    P = np.array([r["centroid"] for r in prev_rows])
    S = cosine_similarity(C, P)  # Current x Previous similarity matrix

    mapping = {r["label"]: None for r in curr_rows}
    scores = np.zeros(len(curr_rows), dtype=float)
    taken_prev = set()
    order = np.argsort(-S, axis=None)  # Sort by similarity (descending)

    for flat in order:
        ci, pj = divmod(flat, S.shape[1])
        if mapping[curr_rows[ci]["label"]] is None and pj not in taken_prev and S[ci, pj] >= cos_thresh:
            mapping[curr_rows[ci]["label"]] = prev_rows[pj]["label"]
            scores[ci] = float(S[ci, pj])
            taken_prev.add(pj)
    return mapping, scores.tolist()

# ------------------------------ Main Pipeline ------------------------------

def main(k: int = 6, auto_k: bool = False, cos_thresh: float = 0.5, 
         drift_cos_drop: float = 0.2, emerge_min: int = 30, 
         term_jaccard_drop: float = 0.60) -> None:
    """
    Main clustering and trend detection pipeline.
    
    Processes embedded events week by week, performing clustering and detecting
    trends like emergence, drift, decay, and growth. Saves results to CSV files
    and per-week assignment files.
    
    Args:
        k: Number of clusters per week (ignored if auto_k=True)
        auto_k: Whether to automatically select optimal k per week
        cos_thresh: Cosine similarity threshold for cluster matching
        drift_cos_drop: Threshold for cosine-based drift detection
        emerge_min: Minimum size for growth trend classification
        term_jaccard_drop: Threshold for term-based drift detection
    """
    if not INP.exists():
        print(f"Missing {INP}. Run: python src/embed_events.py")
        return
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INP)
    df["week"] = df["ts"].apply(week_key)

    weeks = sorted(df["week"].unique())
    prev_rows = None
    clusters_all, trends = [], []

    for wk in weeks:
        sub = df[df["week"] == wk].reset_index(drop=True)
        if len(sub) < max(5, k):
            continue

        sub_df, rows = cluster_week(sub, k=k, auto_k=auto_k)
        mapping, scores = match_clusters_one_to_one(rows, prev_rows or [], cos_thresh=cos_thresh)

        # Save per-event cluster assignments for evaluation
        cid_map = {int(r["label"]): f"{wk}_c{int(r['label'])}" for r in rows}
        assign_out = OUT / "assignments"
        assign_out.mkdir(parents=True, exist_ok=True)
        assign_df = sub_df.copy()
        assign_df["cluster_id"] = assign_df["label"].astype(int).map(cid_map)
        keep_cols = [c for c in ["event_id", "ts", "product", "event_type", "asset_id", "text_repr", "cluster_id"] if c in assign_df.columns]
        assign_df = assign_df[keep_cols]
        assign_df.to_parquet(assign_out / f"{wk}.parquet", index=False)

        # Build mapping of previous cluster terms for drift detection
        prev_terms_map = {}
        if prev_rows:
            for pr in prev_rows:
                prev_terms_map[int(pr["label"])] = set(pr["terms"])

        for i, r in enumerate(rows):
            prev_label = mapping[r["label"]]
            size = r["size"]
            label = f"{wk}_c{r['label']}"

            if prev_rows is None or prev_label is None:
                trend = "emergence"
                drift_amt = 0.0
                delta = size
            else:
                prev = next((x for x in prev_rows if x["label"] == prev_label), None)
                prev_size = prev["size"] if prev else 0
                sim = scores[i]
                drift_amt = 1.0 - float(sim)
                delta = size - prev_size
                trend = "stable"

                # Check for cosine-based drift (centroid movement)
                if drift_amt >= drift_cos_drop:
                    trend = "drift"

                # Check for term-based drift (semantic change) if still stable
                if trend == "stable":
                    prev_terms = prev_terms_map.get(int(prev_label), set())
                    curr_terms = set(r["terms"])
                    if prev_terms or curr_terms:
                        inter = len(prev_terms & curr_terms)
                        union = len(prev_terms | curr_terms) or 1
                        term_jacc = 1.0 - (inter / union)
                        if term_jacc >= term_jaccard_drop:
                            trend = "drift"

                # Check for volume-based trends (can override drift)
                if size < 0.5 * prev_size:
                    trend = "decay"
                if size > 1.5 * prev_size and size >= emerge_min:
                    trend = "growth"

            clusters_all.append({"week": wk, "cluster_id": label, "size": size, "terms": r["terms"]})
            trends.append({
                "week": wk,
                "type": trend,
                "cluster_id": label,
                "delta_size": int(delta),
                "drift": round(drift_amt, 3),
                "label": " ".join(r["terms"][:4]),
            })

        prev_rows = rows

    pd.DataFrame(clusters_all).to_csv(OUT / "clusters_weekly.csv", index=False)
    pd.DataFrame(trends).to_csv(OUT / "trends_summary.csv", index=False)
    with (OUT / "trends_summary.jsonl").open("w", encoding="utf-8") as f:
        for r in trends:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT/'clusters_weekly.csv'} and {OUT/'trends_summary.csv'}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--auto_k", action="store_true")
    p.add_argument("--cos_thresh", type=float, default=0.5)
    p.add_argument("--drift_cos_drop", type=float, default=0.2)
    p.add_argument("--emerge_min", type=int, default=30)
    p.add_argument("--term_jaccard_drop", type=float, default=0.60)
    args = p.parse_args()
    main(
        k=args.k,
        auto_k=args.auto_k,
        cos_thresh=args.cos_thresh,
        drift_cos_drop=args.drift_cos_drop,
        emerge_min=args.emerge_min,
        term_jaccard_drop=args.term_jaccard_drop,
    )
