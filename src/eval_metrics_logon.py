# src/eval_logon.py
# Eval & visualization for CERT logon/logoff data using your temporal clustering pipeline.
# Usage:
#   python src/eval_logon.py --k_min 2 --k_max 10 --force_rerun
#   python src/eval_logon.py --no_plots    # if you don’t want elbow PNGs

from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Paths ----------
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "events_embedded.parquet"
RESULTS_DIR = BASE / "results"
CLUSTERS = RESULTS_DIR / "clusters_weekly.csv"
TRENDS = RESULTS_DIR / "trends_summary.csv"

# ---------- Helpers ----------
def _week_sort_key(w: str) -> int:
    try:
        y, wk = w.split("-W")
        return int(y) * 100 + int(wk)
    except Exception:
        return -1

def _to_iso_week(ts: pd.Series) -> pd.Series:
    iso = pd.to_datetime(ts, utc=True).dt.isocalendar()
    return (iso["year"].astype(int).astype(str) + "-W" + iso["week"].astype(int).map(lambda x: f"{x:02d}"))

def _top_terms(texts, n=6):
    vec = CountVectorizer(max_features=3000, stop_words="english")
    X = vec.fit_transform(texts)
    sums = np.asarray(X.sum(axis=0)).ravel()
    idxs = np.argsort(-sums)[:n]
    vocab = np.array(vec.get_feature_names_out())
    return vocab[idxs].tolist()

def _match_clusters_one_to_one(curr_rows, prev_rows, cos_thresh=0.5):
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

# ---------- Elbow ----------
def _calculate_wss(X, k):
    if len(X) <= k:
        return 0.0
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X)
    return float(km.inertia_)

def _find_elbow_point(wss_values, k_values):
    if len(wss_values) < 3:
        return k_values[0] if k_values else 2
    second = [wss_values[i-1] - 2*wss_values[i] + wss_values[i+1] for i in range(1, len(wss_values)-1)]
    elbow_idx = int(np.argmax(second)) + 1
    return k_values[elbow_idx]

def run_elbow_method_weekly(df, k_min=2, k_max=10):
    weeks = sorted(df["week"].unique())
    optimal_k_per_week = {}
    for week in weeks:
        W = df[df["week"] == week]
        if len(W) < k_min:
            optimal_k_per_week[week] = max(1, len(W))
            continue
        X = np.stack(W["vector"].to_numpy())
        k_values = list(range(k_min, min(k_max + 1, len(W))))
        wss_values = [_calculate_wss(X, k) for k in k_values]
        optimal_k_per_week[week] = _find_elbow_point(wss_values, k_values) if len(wss_values) >= 3 else (k_values[0] if k_values else 2)
        print(f"Week {week}: optimal k = {optimal_k_per_week[week]} (WSS: {[f'{w:.2f}' for w in wss_values]})")
    return optimal_k_per_week

def plot_elbow_curves(df, k_min=2, k_max=10, save_plots=True):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for week in sorted(df["week"].unique()):
        W = df[df["week"] == week]
        if len(W) < k_min:
            continue
        X = np.stack(W["vector"].to_numpy())
        k_values = list(range(k_min, min(k_max + 1, len(W))))
        wss_values = [_calculate_wss(X, k) for k in k_values]
        plt.figure(figsize=(9, 5))
        plt.plot(k_values, wss_values, "o-")
        plt.xlabel("k")
        plt.ylabel("WSS")
        plt.title(f"Elbow — Week {week}")
        plt.grid(True, alpha=0.3)
        if len(wss_values) >= 3:
            k_elbow = _find_elbow_point(wss_values, k_values)
            plt.plot(k_elbow, wss_values[k_values.index(k_elbow)], "ro", label=f"Elbow @ k={k_elbow}")
            plt.legend()
        if save_plots:
            plt.savefig(RESULTS_DIR / f"elbow_week_{week}.png", dpi=200, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

# ---------- Weekly clustering + trend detection ----------
def run_clustering_with_optimal_k(df, optimal_k_per_week, cos_thresh=0.5, drift_cos_drop=0.2, emerge_min=30, term_jaccard_drop=0.60):
    weeks = sorted(df["week"].unique())
    prev_rows = None
    clusters_all, trends = [], []

    for wk in weeks:
        sub = df[df["week"] == wk].reset_index(drop=True)
        if len(sub) < 5:
            continue

        k = min(optimal_k_per_week.get(wk, 6), len(sub)) if len(sub) > 0 else 1
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
                "terms": _top_terms(sub_cluster["text_repr"].tolist()),
                "centroid": centroids[lab if k > 1 else 0].tolist(),
            })

        mapping, scores = _match_clusters_one_to_one(rows, prev_rows or [], cos_thresh=cos_thresh)

        # Save per-event assignments for this week
        assign_out = RESULTS_DIR / "assignments"
        assign_out.mkdir(parents=True, exist_ok=True)
        cid_map = {int(r["label"]): f"{wk}_c{int(r['label'])}" for r in rows}
        assign_df = sub_df.copy()
        assign_df["cluster_id"] = assign_df["label"].astype(int).map(cid_map)
        keep_cols = [c for c in ["event_id", "ts", "product", "event_type", "asset_id", "text_repr", "cluster_id"] if c in assign_df.columns]
        assign_df = assign_df[keep_cols]
        assign_df.to_parquet(assign_out / f"{wk}.parquet", index=False)

        # Drift / growth / decay determination
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

                if drift_amt >= drift_cos_drop:
                    trend = "drift"

                if trend == "stable":
                    prev_terms = prev_terms_map.get(int(prev_label), set())
                    curr_terms = set(r["terms"])
                    if prev_terms or curr_terms:
                        inter = len(prev_terms & curr_terms)
                        union = len(prev_terms | curr_terms) or 1
                        term_jacc = 1.0 - (inter / union)
                        if term_jacc >= term_jaccard_drop:
                            trend = "drift"

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

    pd.DataFrame(clusters_all).to_csv(CLUSTERS, index=False)
    pd.DataFrame(trends).to_csv(TRENDS, index=False)
    with (RESULTS_DIR / "trends_summary.jsonl").open("w", encoding="utf-8") as f:
        for r in trends:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {CLUSTERS} and {TRENDS}")
    return clusters_all, trends

# ---------- Logon/logoff–centric utilities ----------
def summarize_logon_topics():
    """Quick weekly timelines for Logon/Logoff terms based on cluster labels/terms."""
    if not CLUSTERS.exists():
        print("clusters_weekly.csv not found; run with --force_rerun first.")
        return None
    clusters = pd.read_csv(CLUSTERS)
    clusters = clusters.sort_values("week", key=lambda s: s.map(_week_sort_key))
    def _has_term(s, kw): return kw.lower() in str(s).lower()
    series = {}
    for kw in ["logon", "logoff"]:
        s = (clusters
             .assign(has_kw=clusters["terms"].apply(lambda t: _has_term(t, kw)))
             .groupby("week", as_index=True)["has_kw"].sum())
        series[kw] = s
    out = pd.DataFrame(series).fillna(0).astype(int)
    out.to_csv(RESULTS_DIR / "logon_topic_timeline.csv")
    print("Wrote results/logon_topic_timeline.csv")
    return out

def compute_latest_at_10(k=10, alpha=0.3, half_life_days=3):
    """
    Latest@10 (logon/logoff): among events relevant to each query, does
    the fused (cosine × recency) ranking surface the newest relevant row
    within the top-k more often than cosine alone?

    alpha smaller => more recency weight. half_life_days smaller => faster time decay.
    """
    if not DATA.exists():
        print(f"Missing {DATA}. Run: python src/embed_events.py")
        return (0.0, 0.0)

    df = pd.read_parquet(DATA)
    df = df.assign(ts=pd.to_datetime(df["ts"], utc=True))
    now = df["ts"].max()
    age_days = (now - df["ts"]).dt.total_seconds() / 86400.0
    df = df.assign(recency=0.5 ** (age_days / float(half_life_days)))

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Queries + keyword filters that exist in your text_repr
    # (embed_events builds text like "[source=...] [type=Logon] [entity=PC-...] msg=..." ):contentReference[oaicite:0]{index=0}
    queries = {
        "recent workstation logon events":  ["logon", "pc", "workstation"],
        "recent workstation logoff events": ["logoff", "pc", "workstation"],
    }

    V = np.vstack(df["vector"].to_numpy()).astype(float)
    win_cos = win_fused = total = 0

    for q, kw in queries.items():
        # restrict to relevant rows first (evaluation within-slice)
        rel_mask = df["text_repr"].str.lower().apply(lambda s: any(k in s for k in kw))
        if not rel_mask.any():
            continue

        rel = df[rel_mask].copy()
        # newest relevant row by timestamp
        newest_rel_ts = rel["ts"].max()
        newest_idx = rel.index[rel["ts"] == newest_rel_ts][0]

        # rank *within* the relevant slice
        qv = model.encode(q, normalize_embeddings=True)
        V_rel = V[rel.index]                      # restrict vectors to relevant rows
        cos_rel = V_rel @ qv
        fused_rel = alpha * cos_rel + (1 - alpha) * rel["recency"].to_numpy()

        idx_cos = np.argsort(-cos_rel)[:k]
        idx_fused = np.argsort(-fused_rel)[:k]

        if newest_idx in rel.index[idx_cos]:   win_cos += 1
        if newest_idx in rel.index[idx_fused]: win_fused += 1
        total += 1

    return (win_cos / max(1, total), win_fused / max(1, total))

def run_alpha_sensitivity_analysis(k: int = 10, half_life_days: float = 3) -> pd.DataFrame:
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
    output_path = RESULTS_DIR / "alpha_sensitivity_logon.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nAlpha sensitivity results saved to: {output_path}")
    
    return results_df

# ---------- Orchestrator ----------
def run_full_pipeline_with_elbow(k_min=2, k_max=10, plot_elbows=True):
    if not DATA.exists():
        print(f"Missing {DATA}. Run: python src/embed_events.py")
        return {}

    print("Loading embedded events...")
    df = pd.read_parquet(DATA)
    df["week"] = _to_iso_week(df["ts"])

    print(f"Found {len(df)} events across {len(df['week'].unique())} ISO weeks")
    print("\nRunning elbow method to determine optimal k values...")
    optimal_k_per_week = run_elbow_method_weekly(df, k_min=k_min, k_max=k_max)

    if plot_elbows:
        print("\nGenerating elbow plots...")
        plot_elbow_curves(df, k_min=k_min, k_max=k_max, save_plots=True)

    print("\nRunning clustering with optimal k values...")
    run_clustering_with_optimal_k(df, optimal_k_per_week)

    print("\nPipeline completed.")
    return optimal_k_per_week

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Logon/logoff evaluation on CERT data")
    parser.add_argument("--k_min", type=int, default=2, help="Minimum k for elbow method")
    parser.add_argument("--k_max", type=int, default=10, help="Maximum k for elbow method")
    parser.add_argument("--no_plots", action="store_true", help="Skip generating elbow plots")
    parser.add_argument("--force_rerun", action="store_true", help="Recompute elbow + clustering even if results exist")
    parser.add_argument("--no_sensitivity", action="store_true", help="Skip alpha sensitivity analysis")
    args = parser.parse_args()

    if args.force_rerun or not (CLUSTERS.exists() and TRENDS.exists()):
        optimal = run_full_pipeline_with_elbow(k_min=args.k_min, k_max=args.k_max, plot_elbows=not args.no_plots)
        print(f"\nOptimal k per week: {optimal}")

    # Summaries tailored to logon/logoff
    timeline = summarize_logon_topics()
    latest_cos, latest_fused = compute_latest_at_10()

    print("\n" + "="*60)
    print("LOGON EVALUATION SUMMARY")
    print("="*60)
    if timeline is not None:
        print(timeline.tail(min(10, len(timeline))))
    print(f"Latest@10 baseline (cosine):     {latest_cos:.2f}")
    print(f"Latest@10 fused (cos×half-life): {latest_fused:.2f}")
    
    # Run alpha sensitivity analysis (unless skipped)
    if not args.no_sensitivity:
        print("\n" + "="*60)
        print("ALPHA SENSITIVITY ANALYSIS")
        print("="*60)
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
        print("Trend CSVs at: results/clusters_weekly.csv, results/trends_summary.csv")
        print("Sensitivity results at: results/alpha_sensitivity_logon.csv")
