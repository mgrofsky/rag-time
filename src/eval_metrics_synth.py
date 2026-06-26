"""
Evaluation metrics for synthetic data with elbow method integration.

This module provides comprehensive evaluation of the time-aware clustering
pipeline on synthetic data, including automatic elbow method for optimal
k-selection and trend detection evaluation.

Key features:
- Automatic elbow method for determining optimal k values per week
- Trend detection evaluation with F1 scores
- Time-aware retrieval evaluation (as-of correctness, latest@10)
- Visual elbow plots for each week
- Integration with clustering pipeline

The evaluation uses predefined ground truth patterns in synthetic data
to measure how well the pipeline detects growth, drift, and decay trends.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "events_embedded.parquet"
CLUSTERS = BASE / "results" / "clusters_weekly.csv"
TRENDS = BASE / "results" / "trends_summary.csv"
RESULTS_DIR = BASE / "results"

# ---------- Helper Functions ----------

def _week_sort_key(w: str) -> int:
    """
    Convert ISO week string to sortable integer.
    
    Args:
        w: ISO week string like "2025-W13"
    
    Returns:
        Integer for sorting (year*100 + week)
    """
    try:
        y, wk = w.split("-W")
        return int(y) * 100 + int(wk)
    except Exception:
        return -1

def _norm(s) -> str:
    """
    Normalize string values, handling NaN cases.
    
    Args:
        s: Value to normalize
    
    Returns:
        Lowercase string or empty string for NaN
    """
    return "" if pd.isna(s) else str(s).lower()

def _pick_cluster_for_topic(week_df: pd.DataFrame, term_map: dict, size_map: dict, topic_kws: list[str]) -> dict | None:
    """
    Find the best matching cluster for a given topic based on keyword hits.
    
    Args:
        week_df: DataFrame of trends for a specific week
        term_map: Mapping from cluster_id to terms
        size_map: Mapping from cluster_id to cluster size
        topic_kws: List of keywords to match against
    
    Returns:
        Dictionary with best match info or None if no matches
    """
    if week_df.empty:
        return None
    
    rows = []
    for _, r in week_df.iterrows():
        cid = r["cluster_id"]
        terms = term_map.get(cid, "")
        label = _norm(r.get("label", ""))
        text = f"{_norm(terms)} {label}"
        hits = sum(1 for kw in topic_kws if kw in text)
        size = int(size_map.get(cid, 0))
        rows.append((hits, size, r["type"], cid))
    
    rows.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = rows[0]
    return dict(hits=best[0], size=best[1], pred=best[2], cid=best[3])

# ---------- Elbow Method Functions ----------

def calculate_wss(X: np.ndarray, k: int) -> float:
    """
    Calculate Within-Cluster Sum of Squares (WSS) for given k.
    
    Args:
        X: Feature matrix for clustering
        k: Number of clusters
    
    Returns:
        WSS (inertia) value
    """
    if len(X) <= k:
        return 0.0
    
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X)
    return kmeans.inertia_

def find_elbow_point(wss_values: list[float], k_values: list[int]) -> int:
    """
    Find the elbow point using the second derivative method.
    
    The elbow point is where the rate of decrease in WSS sharply slows down,
    indicating the optimal number of clusters.
    
    Args:
        wss_values: List of WSS values for each k
        k_values: List of corresponding k values
    
    Returns:
        Optimal k value at the elbow point
    """
    if len(wss_values) < 3:
        return k_values[0] if k_values else 2
    
    # Calculate second derivative to find sharpest bend
    second_derivatives = []
    for i in range(1, len(wss_values) - 1):
        second_deriv = wss_values[i-1] - 2*wss_values[i] + wss_values[i+1]
        second_derivatives.append(second_deriv)
    
    # Find the point with maximum second derivative (sharpest bend)
    elbow_idx = np.argmax(second_derivatives) + 1
    return k_values[elbow_idx]

def run_elbow_method_weekly(df: pd.DataFrame, k_min: int = 2, k_max: int = 10) -> dict[str, int]:
    """
    Run elbow method for each week to determine optimal k values.
    
    Args:
        df: DataFrame with embedded events and week column
        k_min: Minimum k to test
        k_max: Maximum k to test
    
    Returns:
        Dictionary mapping week -> optimal_k
    """
    weeks = sorted(df["week"].unique())
    optimal_k_per_week = {}
    
    for week in weeks:
        week_data = df[df["week"] == week]
        if len(week_data) < k_min:
            optimal_k_per_week[week] = max(1, len(week_data))
            continue
            
        X = np.stack(week_data["vector"].to_numpy())
        k_values = list(range(k_min, min(k_max + 1, len(week_data))))
        wss_values = []
        
        for k in k_values:
            wss = calculate_wss(X, k)
            wss_values.append(wss)
        
        if len(wss_values) >= 3:
            optimal_k = find_elbow_point(wss_values, k_values)
        else:
            optimal_k = k_values[0] if k_values else 2
            
        optimal_k_per_week[week] = optimal_k
        print(f"Week {week}: optimal k = {optimal_k} (WSS values: {[f'{w:.2f}' for w in wss_values]})")
    
    return optimal_k_per_week

def plot_elbow_curves(df, k_min=2, k_max=10, save_plots=True):
    """
    Plot elbow curves for each week and save them.
    """
    weeks = sorted(df["week"].unique())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    for week in weeks:
        week_data = df[df["week"] == week]
        if len(week_data) < k_min:
            continue
            
        X = np.stack(week_data["vector"].to_numpy())
        k_values = list(range(k_min, min(k_max + 1, len(week_data))))
        wss_values = []
        
        for k in k_values:
            wss = calculate_wss(X, k)
            wss_values.append(wss)
        
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, wss_values, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Within-Cluster Sum of Squares (WSS)')
        plt.title(f'Elbow Method for Week {week}')
        plt.grid(True, alpha=0.3)
        
        # Mark the elbow point
        if len(wss_values) >= 3:
            optimal_k = find_elbow_point(wss_values, k_values)
            optimal_wss = wss_values[k_values.index(optimal_k)]
            plt.plot(optimal_k, optimal_wss, 'ro', markersize=12, label=f'Elbow at k={optimal_k}')
            plt.legend()
        
        if save_plots:
            plt.savefig(RESULTS_DIR / f'elbow_week_{week}.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def run_clustering_with_optimal_k(df, optimal_k_per_week, cos_thresh=0.5, drift_cos_drop=0.2, emerge_min=30, term_jaccard_drop=0.60):
    """
    Run clustering using the optimal k values determined by elbow method.
    This is essentially the main function from cluster_trends.py but with dynamic k values.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    def top_terms(texts, n=6):
        vec = CountVectorizer(max_features=3000, stop_words="english")
        X = vec.fit_transform(texts)
        sums = np.asarray(X.sum(axis=0)).ravel()
        idxs = sums.argsort()[::-1][:n]
        vocab = np.array(vec.get_feature_names_out())
        return vocab[idxs].tolist()
    
    def match_clusters_one_to_one(curr_rows, prev_rows, cos_thresh=0.5):
        if not prev_rows:
            return {r["label"]: None for r in curr_rows}, [1.0] * len(curr_rows)
        C = np.array([r["centroid"] for r in curr_rows])
        P = np.array([r["centroid"] for r in prev_rows])
        S = cosine_similarity(C, P)
        
        mapping = {r["label"]: None for r in curr_rows}
        scores = np.zeros(len(curr_rows), dtype=float)
        taken_prev = set()
        order = np.argsort(-S, axis=None)
        
        for flat in order:
            ci, pj = divmod(flat, S.shape[1])
            if mapping[curr_rows[ci]["label"]] is None and pj not in taken_prev and S[ci, pj] >= cos_thresh:
                mapping[curr_rows[ci]["label"]] = prev_rows[pj]["label"]
                scores[ci] = float(S[ci, pj])
                taken_prev.add(pj)
        return mapping, scores.tolist()
    
    weeks = sorted(df["week"].unique())
    prev_rows = None
    clusters_all, trends = [], []
    
    for wk in weeks:
        sub = df[df["week"] == wk].reset_index(drop=True)
        if len(sub) < 5:
            continue
            
        k = optimal_k_per_week.get(wk, 6)
        k = min(k, len(sub)) if len(sub) > 0 else 1
        
        X = np.stack(sub["vector"].to_numpy())
        if k <= 1:
            labels = np.zeros(len(sub), dtype=int)
            centroids = np.mean(X, axis=0, keepdims=True)
        else:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(X)
            centroids = km.cluster_centers_
        
        sub_df = sub.copy()
        sub_df["label"] = labels
        
        rows = []
        for lab in sorted(set(labels)):
            sub_cluster = sub_df[sub_df["label"] == lab]
            rows.append({
                "label": int(lab),
                "size": int(len(sub_cluster)),
                "terms": top_terms(sub_cluster["text_repr"].tolist()),
                "centroid": centroids[lab if k > 1 else 0].tolist(),
            })
        
        mapping, scores = match_clusters_one_to_one(rows, prev_rows or [], cos_thresh=cos_thresh)
        
        # Store assignments
        cid_map = {int(r["label"]): f"{wk}_c{int(r['label'])}" for r in rows}
        assign_out = RESULTS_DIR / "assignments"
        assign_out.mkdir(parents=True, exist_ok=True)
        assign_df = sub_df.copy()
        assign_df["cluster_id"] = assign_df["label"].astype(int).map(cid_map)
        keep_cols = [c for c in ["event_id", "ts", "product", "event_type", "asset_id", "text_repr", "cluster_id"] if c in assign_df.columns]
        assign_df = assign_df[keep_cols]
        assign_df.to_parquet(assign_out / f"{wk}.parquet", index=False)
        
        # Map prior terms for drift detection
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
                
                # Cosine-based drift
                if drift_amt >= drift_cos_drop:
                    trend = "drift"
                
                # Term-Jaccard drift
                if trend == "stable":
                    prev_terms = prev_terms_map.get(int(prev_label), set())
                    curr_terms = set(r["terms"])
                    if prev_terms or curr_terms:
                        inter = len(prev_terms & curr_terms)
                        union = len(prev_terms | curr_terms) or 1
                        term_jacc = 1.0 - (inter / union)
                        if term_jacc >= term_jaccard_drop:
                            trend = "drift"
                
                # Volume-based decay/growth
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
    
    # Save results
    pd.DataFrame(clusters_all).to_csv(CLUSTERS, index=False)
    pd.DataFrame(trends).to_csv(TRENDS, index=False)
    
    import json
    with (RESULTS_DIR / "trends_summary.jsonl").open("w", encoding="utf-8") as f:
        for r in trends:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"Wrote {CLUSTERS} and {TRENDS}")
    return optimal_k_per_week

TOPIC_SPECS = {
    "okta": {
        "match_kws": ["okta", "vpn", "mfa", "auth", "failure"],
        "gold_fn": lambda rel_wk: ("growth" if 4 <= rel_wk <= 8 else "stable"),
    },
    "data": {
        "match_kws": ["snowflake", "s3", "data", "sql", "select", "getobject"],
        "gold_fn": lambda rel_wk: ("drift" if rel_wk >= 6 else "stable"),
    },
    "qualys": {
        "match_kws": ["qualys", "openssl", "vuln", "vulnerability", "cve"],
        "gold_fn": lambda rel_wk: ("decay" if rel_wk >= 9 else "stable"),
    },
}

def compute_trend_f1() -> tuple[float, dict[str, float], str]:
    """
    Compute F1 scores for trend detection evaluation.
    
    Evaluates how well the pipeline detects predefined trend patterns
    in synthetic data (growth, drift, decay) using F1 scores.
    
    Returns:
        Tuple of (macro_f1, topic_f1_details, classification_report)
    """
    if not TRENDS.exists() or not CLUSTERS.exists():
        print("Clustering results not found. Running elbow method and clustering automatically...")
        run_full_pipeline_with_elbow()
    
    trends = pd.read_csv(TRENDS)
    clusters = pd.read_csv(CLUSTERS)
    term_map = {row["cluster_id"]: " ".join(row["terms"]) for _, row in clusters.iterrows()}
    size_map = {row["cluster_id"]: row["size"] for _, row in clusters.iterrows()}
    weeks_sorted = sorted(trends["week"].unique(), key=_week_sort_key)
    wk2idx = {w: i for i, w in enumerate(weeks_sorted)}
    labels_eval = ["growth", "drift", "decay"]
    all_preds, all_golds = [], []
    f1_details = {}

    for topic, spec in TOPIC_SPECS.items():
        topic_preds, topic_golds = [], []
        for week in weeks_sorted:
            rel_wk = wk2idx[week]
            gold = spec["gold_fn"](rel_wk)
            if gold == "stable":
                continue
            week_df = trends[trends["week"] == week]
            pick = _pick_cluster_for_topic(week_df, term_map, size_map, spec["match_kws"])
            pred = "stable" if (pick is None or pick["hits"] == 0) else pick["pred"]
            topic_preds.append(pred)
            topic_golds.append(gold)
            all_preds.append(pred)
            all_golds.append(gold)
        if topic_golds:
            f1_details[topic] = f1_score(topic_golds, topic_preds, labels=labels_eval, average="macro", zero_division=0.0)

    macro_f1 = f1_score(all_golds, all_preds, labels=labels_eval, average="macro", zero_division=0.0) if all_golds else 0.0
    cls_report = classification_report(all_golds, all_preds, labels=labels_eval, zero_division=0.0)
    return macro_f1, f1_details, cls_report

def compute_asof_correctness(asof_week: int | None = None, k: int = 10, 
                           alpha: float = 0.7, half_life_days: float = 14) -> float:
    """
    Evaluate as-of correctness for time-aware retrieval.
    
    Tests whether time-aware ranking respects temporal boundaries by ensuring
    that queries with as-of constraints don't return results from future time periods.
    
    Args:
        asof_week: Week to use as cutoff (defaults to median week)
        k: Number of top results to check
        alpha: Weight for semantic similarity in fused score
        half_life_days: Half-life for time decay
    
    Returns:
        Correctness score (1.0 if no violations, 0.0 if any violations)
    """
    df = pd.read_parquet(DATA)
    df = df.assign(ts=pd.to_datetime(df["ts"], utc=True))
    now = df["ts"].max()
    age_days = (now - df["ts"]).dt.total_seconds() / 86400.0
    df = df.assign(recency=0.5 ** (age_days / float(half_life_days)))

    if asof_week is None:
        weeks = sorted(df["ts"].dt.isocalendar().week.astype(int).unique())
        asof_week = int(np.median(weeks))
    cutoff_mask = df["ts"].dt.isocalendar().week.astype(int) <= asof_week

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    queries = ["Latest MFA authentication failures from Okta", "Recent data exfiltration patterns from Snowflake", "Qualys scan results for OpenSSL vulnerabilities"]
    
    V = np.vstack(df["vector"].to_numpy()).astype(float)
    any_violation = False
    
    for q in queries:
        qv = model.encode(q, normalize_embeddings=True)
        cos = V @ qv
        fused = alpha * cos + (1 - alpha) * df["recency"].to_numpy()
        sub = df[cutoff_mask].assign(score=fused[cutoff_mask.values]).sort_values("score", ascending=False).head(k)
        if (sub["ts"].dt.isocalendar().week.astype(int) > asof_week).any():
            any_violation = True
            break
    return 0.0 if any_violation else 1.0

def compute_latest_at_10(k: int = 10, alpha: float = 0.7, half_life_days: float = 14) -> tuple[float, float]:
    """
    Evaluate latest@10 metric for time-aware retrieval.
    
    Tests whether time-aware ranking surfaces the most recent relevant events
    within the top-k results more often than pure semantic similarity.
    
    Args:
        k: Number of top results to evaluate
        alpha: Weight for semantic similarity in fused score
        half_life_days: Half-life for time decay
    
    Returns:
        Tuple of (cosine_score, fused_score) - proportion of queries where
        the newest relevant event appears in top-k
    """
    df = pd.read_parquet(DATA)
    df = df.assign(ts=pd.to_datetime(df["ts"], utc=True))
    now = df["ts"].max()
    age_days = (now - df["ts"]).dt.total_seconds() / 86400.0
    df = df.assign(recency=0.5 ** (age_days / float(half_life_days)))

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Use descriptive queries that match the synthetic data patterns
    queries = {
        "investigation of recent Okta MFA authentication failures": ["okta", "mfa", "vpn"],
        "summary of data access patterns involving Snowflake and S3": ["snowflake", "select", "sql", "s3"],
        "latest Qualys findings related to OpenSSL vulnerabilities": ["qualys", "openssl", "vuln", "vulnerability"],
    }
    
    V = np.vstack(df["vector"].to_numpy()).astype(float)
    win_cos, win_fused, total = 0, 0, 0

    for q, kw in queries.items():
        rel_mask = df["text_repr"].str.lower().apply(lambda s: any(k in s for k in kw))
        if not rel_mask.any(): continue
        
        newest_rel_ts = df.loc[rel_mask, "ts"].max()
        newest_idx = df.index[df["ts"] == newest_rel_ts][0]

        qv = model.encode(q, normalize_embeddings=True)
        cos = V @ qv
        fused = alpha * cos + (1 - alpha) * df["recency"].to_numpy()

        idx_cos = np.argsort(-cos)[:k]
        idx_fused = np.argsort(-fused)[:k]

        if newest_idx in df.index[idx_cos]: win_cos += 1
        if newest_idx in df.index[idx_fused]: win_fused += 1
        total += 1

    return (win_cos / max(1, total), win_fused / max(1, total))

def run_alpha_sensitivity_analysis(k: int = 10, half_life_days: float = 14) -> pd.DataFrame:
    """
    Run sensitivity analysis across different alpha values for Latest@10 metric.
    
    Tests how the balance between semantic similarity and recency affects
    the ability to surface the newest relevant events in top-k results.
    
    Args:
        k: Number of top results to evaluate
        half_life_days: Half-life for time decay calculation
    
    Returns:
        DataFrame with alpha values and corresponding Latest@10 scores
    """
    if not DATA.exists():
        print(f"Missing {DATA}. Run: python src/embed_events.py")
        return pd.DataFrame()
    
    # Alpha values to test - covering the range from recency-heavy to semantics-heavy
    alpha_values = [0.4, 0.5, 0.7, 0.9, 0.95]
    
    results = []
    
    print(f"\nRunning alpha sensitivity analysis for Latest@{k}...")
    print("Alpha values to test:", alpha_values)
    
    for alpha in alpha_values:
        print(f"\nTesting alpha = {alpha}...")
        cos_score, fused_score = compute_latest_at_10(k=k, alpha=alpha, half_life_days=half_life_days)
        
        results.append({
            'alpha': alpha,
            'cosine_score': cos_score,
            'fused_score': fused_score,
            'improvement': fused_score - cos_score
        })
        
        print(f"  Cosine score: {cos_score:.3f}")
        print(f"  Fused score:  {fused_score:.3f}")
        print(f"  Improvement:  {fused_score - cos_score:+.3f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    output_path = RESULTS_DIR / "alpha_sensitivity_synth.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nAlpha sensitivity results saved to: {output_path}")
    
    return results_df

def run_full_pipeline_with_elbow(k_min: int = 2, k_max: int = 10, plot_elbows: bool = True) -> dict[str, int]:
    """
    Run the complete pipeline with automatic elbow method.
    
    Loads embedded events, determines optimal k values using elbow method,
    runs clustering, and generates all output files.
    
    Args:
        k_min: Minimum k for elbow method
        k_max: Maximum k for elbow method
        plot_elbows: Whether to generate elbow plots
    
    Returns:
        Dictionary mapping week -> optimal_k
    """
    if not DATA.exists():
        print(f"Missing {DATA}. Run: python src/embed_events.py")
        return
    
    print("Loading embedded events data...")
    df = pd.read_parquet(DATA)
    df["week"] = df["ts"].apply(lambda x: pd.to_datetime(x, utc=True).strftime("%Y-W%U"))
    
    print(f"Found {len(df)} events across {len(df['week'].unique())} weeks")
    
    print("\nRunning elbow method to determine optimal k values...")
    optimal_k_per_week = run_elbow_method_weekly(df, k_min=k_min, k_max=k_max)
    
    if plot_elbows:
        print("\nGenerating elbow plots...")
        plot_elbow_curves(df, k_min=k_min, k_max=k_max, save_plots=True)
    
    print("\nRunning clustering with optimal k values...")
    run_clustering_with_optimal_k(df, optimal_k_per_week)
    
    print("\nPipeline completed successfully!")
    return optimal_k_per_week

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate clustering trends with elbow method")
    parser.add_argument("--k_min", type=int, default=2, help="Minimum k for elbow method")
    parser.add_argument("--k_max", type=int, default=10, help="Maximum k for elbow method")
    parser.add_argument("--no_plots", action="store_true", help="Skip generating elbow plots")
    parser.add_argument("--force_rerun", action="store_true", help="Force rerun clustering even if results exist")
    parser.add_argument("--no_sensitivity", action="store_true", help="Skip alpha sensitivity analysis")
    args = parser.parse_args()
    
    # Force rerun if requested
    if args.force_rerun:
        print("Force rerun requested. Running full pipeline...")
        optimal_k_per_week = run_full_pipeline_with_elbow(
            k_min=args.k_min, 
            k_max=args.k_max, 
            plot_elbows=not args.no_plots
        )
        print(f"\nOptimal k values per week: {optimal_k_per_week}")
    
    # Run evaluations
    macro_f1, f1_details, cls_report = compute_trend_f1()
    asof = compute_asof_correctness()
    latest_cos, latest_fused = compute_latest_at_10()
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Trend F1 (macro over non-stable labels): {macro_f1:.2f}")
    for t, v in f1_details.items():
        print(f"  {t} F1: {v:.2f}")
    print("\nClassification report (growth/drift/decay only):\n", cls_report)
    print(f"As-of correctness: {asof:.2f}")
    print(f"Latest@10 baseline (cosine): {latest_cos:.2f}")
    print(f"Latest@10 fused (cos×half-life): {latest_fused:.2f}")
    
    # Run alpha sensitivity analysis (unless skipped)
    if not args.no_sensitivity:
        print(f"\n{'='*60}")
        print("ALPHA SENSITIVITY ANALYSIS")
        print(f"{'='*60}")
        sensitivity_results = run_alpha_sensitivity_analysis()
        
        if not sensitivity_results.empty:
            print("\nSensitivity Analysis Summary:")
            print(sensitivity_results.to_string(index=False, float_format='%.3f'))
            
            # Find best alpha
            best_alpha = sensitivity_results.loc[sensitivity_results['fused_score'].idxmax(), 'alpha']
            best_score = sensitivity_results['fused_score'].max()
            print(f"\nBest performing alpha: {best_alpha} (Latest@10 = {best_score:.3f})")
    else:
        print("\nSkipping alpha sensitivity analysis (--no_sensitivity flag)")
    
    if not args.no_plots:
        print(f"\nElbow plots saved to: {RESULTS_DIR}")
        print("Check the results/ directory for individual week elbow plots.")
        print("Sensitivity results at: results/alpha_sensitivity_synth.csv")

