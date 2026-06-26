"""
Figure generation for time-aware embeddings research paper.

This module creates publication-quality figures from the clustering and trend
detection pipeline results. It generates visualizations showing:

1. Pipeline overview diagram
2. Trend types over time
3. Topic-specific timelines  
4. Weekly cluster snapshots
5. Time-aware ranking demonstrations

The figures are saved in both PDF and PNG formats for publication use.
"""

from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

# --- Path setup (works whether file is in src/ or at repo root) ---
_THIS = Path(__file__).resolve()
BASE = _THIS.parents[1] if _THIS.parent.name == "src" else _THIS.parent
RESULTS = BASE / "results"
FIGS = BASE / "figs"
FIGS.mkdir(exist_ok=True)

def _require(path: Path, hint: str) -> None:
    """
    Check if required file exists, exit with helpful message if not.
    
    Args:
        path: File path to check
        hint: Additional context about what's missing
    """
    if not path.exists():
        raise SystemExit(
            f"[make_figs] Missing {path}. {hint}\n"
            "Run the pipeline in order:\n"
            "  1) python src/log_ingest.py\n"
            "  2) python src/embed_events.py --dim 512 --dtype float16\n"
            "  3) python src/cluster_trends.py --k 6 --auto_k\n"
        )

# --- Helper functions ---

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

def _contains_kw(terms_str, kw: str) -> bool:
    """
    Check if keyword appears in terms string (case-insensitive).
    
    Args:
        terms_str: String containing terms
        kw: Keyword to search for
    
    Returns:
        True if keyword found, False otherwise
    """
    return kw.lower() in str(terms_str).lower()

def _short_label(terms_str: str) -> str:
    """
    Create short label from cluster terms for display.
    
    Cleans up term strings and returns first 3 meaningful terms.
    
    Args:
        terms_str: Raw terms string from cluster data
    
    Returns:
        Shortened label for display
    """
    s = str(terms_str).replace("["," ").replace("]"," ").replace("'"," ")
    toks = [tok.strip() for tok in s.replace(",", " ").split() if tok.strip()]
    stop = {"msg","type","source","entity","risk","tech","attack"}
    toks = [t for t in toks if t not in stop]
    return " ".join(toks[:3]) if toks else "cluster"

# -------------------- Figure 1: Pipeline Overview --------------------

def fig_pipeline() -> None:
    """
    Create pipeline overview diagram showing the four main stages.
    
    Generates a clean diagram showing the flow from raw logs through
    normalization, embedding, clustering, and time-aware retrieval.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 2.6), dpi=200)
    ax = plt.gca()
    ax.axis("off")
    
    # Set coordinate system to [0,1] for consistent layout
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Define box layout parameters
    x0, w, h, gap = 0.04, 0.20, 0.60, 0.04
    y = 0.20
    boxes = [
        (x0 + 0*(w+gap), y, "Normalize\n(JSONL/CSV → Parquet)"),
        (x0 + 1*(w+gap), y, "Embed\n(text → vector)"),
        (x0 + 2*(w+gap), y, "Weekly Cluster\n+ Match"),
        (x0 + 3*(w+gap), y, "Time-aware\nRetrieval"),
    ]

    # Draw boxes and labels
    for (bx, by, label) in boxes:
        ax.add_patch(plt.Rectangle(
            (bx, by), w, h, fill=False, linewidth=1.5,
            transform=ax.transAxes, clip_on=False
        ))
        ax.text(bx + w/2, by + h/2, label, ha="center", va="center",
                transform=ax.transAxes)

    # Draw arrows between boxes
    for i in range(len(boxes) - 1):
        x_start = boxes[i][0] + w
        x_end   = boxes[i+1][0]
        y_mid   = y + h/2
        ax.annotate(
            "", xy=(x_end - 0.006, y_mid), xytext=(x_start + 0.006, y_mid),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", lw=1.2)
        )

    fig.tight_layout()
    fig.savefig(FIGS / "fig_pipeline.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig_pipeline.png", bbox_inches="tight")
    plt.close(fig)


# ---------------- Figure 2: Trend Types Over Time -------------------

def fig_trend_types() -> None:
    """
    Create line plot showing trend type counts over time.
    
    Visualizes how different trend types (emergence, drift, decay, growth)
    change across weeks, helping identify temporal patterns.
    """
    trends_csv = RESULTS / "trends_summary.csv"
    _require(trends_csv, "Did you run cluster_trends.py?")
    trends = pd.read_csv(trends_csv)
    trend_counts = trends.groupby(["week", "type"]).size().unstack(fill_value=0)
    trend_counts = trend_counts.reindex(index=sorted(trend_counts.index, key=_week_sort_key))

    fig = plt.figure(figsize=(7.5, 3.6), dpi=200)
    ax = plt.gca()
    for col in trend_counts.columns:
        ax.plot(trend_counts.index, trend_counts[col], marker="o", label=col)

    ax.set_xlabel("ISO Week")
    ax.set_ylabel("Number of clusters")
    ax.set_title("Trend Types Over Time", pad=8)

    # Position legend outside plot area
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, fontsize=8, borderaxespad=0.0)

    # Adjust layout to accommodate legend and rotated labels
    fig.subplots_adjust(right=0.80, bottom=0.25)

    plt.xticks(rotation=45, ha="right")
    fig.savefig(FIGS / "fig_trend_types_by_week.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig_trend_types_by_week.png", bbox_inches="tight")
    plt.close(fig)


# -------- Figure 3: Topic Timeline -----

def fig_topic_timeline(keyword: str = "qualys") -> None:
    """
    Create timeline showing cluster size for a specific topic keyword.
    
    Shows how cluster sizes related to a particular topic (e.g., "qualys")
    change over time, useful for identifying topic-specific trends.
    
    Args:
        keyword: Topic keyword to track over time
    """
    clusters_csv = RESULTS / "clusters_weekly.csv"
    _require(clusters_csv, "Did you run cluster_trends.py?")
    clusters = pd.read_csv(clusters_csv)
    clusters = clusters.sort_values("week", key=lambda s: s.map(_week_sort_key))
    series = (clusters
              .assign(has_kw=clusters["terms"].apply(lambda s: _contains_kw(s, keyword)))
              .groupby("week", as_index=True)
              .apply(lambda g: g.loc[g["has_kw"], "size"].sum()))
    fig = plt.figure(figsize=(7.5, 3.2), dpi=200)
    ax = plt.gca()
    ax.plot(series.index, series.values, marker="o")
    ax.set_xlabel("ISO Week")
    ax.set_ylabel("Cluster size (topic-related)")
    ax.set_title(f"Topic Timeline: '{keyword}'")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_topic_timeline.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig_topic_timeline.png", bbox_inches="tight")
    plt.close(fig)

# -------- Figure 4: Weekly Cluster Snapshot -----

def fig_weekly_snapshot() -> None:
    """
    Create bar chart showing cluster sizes for the most recent week.
    
    Provides a snapshot view of the current week's clusters, showing
    their relative sizes and helping identify dominant patterns.
    """
    clusters_csv = RESULTS / "clusters_weekly.csv"
    _require(clusters_csv, "Did you run cluster_trends.py?")
    clusters = pd.read_csv(clusters_csv)
    clusters["wk_sort"] = clusters["week"].map(_week_sort_key)
    last_week = clusters.loc[clusters["wk_sort"].idxmax(), "week"]
    snap = clusters[clusters["week"] == last_week].copy()
    snap["short"] = snap["terms"].apply(_short_label)
    snap = snap.sort_values("size", ascending=False)
    fig = plt.figure(figsize=(7.5, 3.2), dpi=200)
    ax = plt.gca()
    ax.bar(snap["short"], snap["size"])
    ax.set_xlabel(f"Clusters in {last_week}")
    ax.set_ylabel("Size")
    ax.set_title("Weekly Clusters (Snapshot)")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_weekly_clusters_snapshot.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig_weekly_clusters_snapshot.png", bbox_inches="tight")
    plt.close(fig)


def fig_rerank_demo(query: str = "qualys openssl", alpha: float = 0.7, 
                   half_life_days: float = 14, k: int = 8) -> None:
    """
    Demonstrate time-aware ranking vs. pure semantic similarity.
    
    Shows how combining semantic similarity with temporal recency
    affects ranking results, comparing cosine similarity alone vs.
    the fused score approach.
    
    Args:
        query: Search query to demonstrate ranking
        alpha: Weight for semantic similarity (0-1)
        half_life_days: Half-life for time decay
        k: Number of top results to show
    """
    import numpy as np
    from utils import tokenize, hashing_vectorizer
    INP = BASE / "data" / "events_embedded.parquet"
    _require(INP, "Did you run embed_events.py?")
    df = pd.read_parquet(INP)

    dim = len(df.iloc[0]["vector"])
    qv = np.asarray(hashing_vectorizer(tokenize(query), dim=dim), dtype=float)

    def _cos(v): return float(np.dot(qv, np.asarray(v, dtype=float)))
    df["cos"] = df["vector"].apply(_cos)

    ts = pd.to_datetime(df["ts"], utc=True)
    now = ts.max()
    age_days = (now - ts).dt.total_seconds() / 86400.0
    recency = 0.5 ** (age_days / float(half_life_days))
    df["fused"] = alpha * df["cos"] + (1 - alpha) * recency

    top = df.sort_values("cos", ascending=False).head(k).copy()
    wk = top["ts"].dt.isocalendar().week.astype(int).astype(str)
    lbl = "[W" + wk + "] " + top["product"].astype(str)

    fig = plt.figure(figsize=(7.5, 3.0), dpi=200)
    ax = plt.gca()
    x = np.arange(len(top))
    ax.bar(x - 0.2, top["cos"],   width=0.4, label="cosine")
    ax.bar(x + 0.2, top["fused"], width=0.4, label="cosine × half-life")

    ax.set_xticks(x)
    ax.set_xticklabels(lbl, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"Re-rank demo: query='{query}' (α={alpha}, h={half_life_days}d)", pad=8)

    # Position legend outside plot area
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, fontsize=8, borderaxespad=0.0)

    # Adjust layout for legend and rotated labels
    fig.subplots_adjust(right=0.78, bottom=0.28)

    fig.savefig(FIGS / "fig_rerank_demo.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig_rerank_demo.png", bbox_inches="tight")
    plt.close(fig)



if __name__ == "__main__":
    """
    Generate all figures for the research paper.
    
    Creates publication-quality figures showing the pipeline overview,
    trend analysis, topic timelines, and time-aware ranking demonstrations.
    """
    fig_pipeline()
    fig_trend_types()
    fig_topic_timeline(keyword="qualys")   # Can change to "snowflake" or "vpn" 
    fig_weekly_snapshot()
    fig_rerank_demo(query="qualys openssl", alpha=0.7, half_life_days=14, k=8)
    print("Figures written to:", FIGS.resolve())
