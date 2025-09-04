"""
Streamlit web application for exploring time-aware clustering results.

This interactive dashboard allows users to explore the results of the
time-aware embeddings pipeline, including trend analysis, cluster
visualizations, and time-scoped queries.

Features:
- As-of date filtering for time-scoped analysis
- Trend type filtering and visualization
- Weekly cluster exploration
- Interactive data tables

The app demonstrates how time-aware retrieval can provide contextually
relevant results based on both semantic similarity and temporal recency.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

# Setup paths and page configuration
ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"

st.set_page_config(page_title="Temporal Trends (Time-Aware RAG Demo)", layout="wide")
st.title("Temporal Trends (Time-Aware RAG Demo)")

# Load data files
clusters_path = RES / "clusters_weekly.csv"
trends_path = RES / "trends_summary.csv"

if not clusters_path.exists() or not trends_path.exists():
    st.warning("Run the pipeline first:\n1) log_ingest → 2) embed_events → 3) cluster_trends")
else:
    clusters = pd.read_csv(clusters_path)
    trends = pd.read_csv(trends_path)

    # ---- Time-scoped filtering (demonstrates as-of queries) ----
    st.subheader("As-of Filter")
    st.write("Select a cutoff week to simulate historical analysis. Only data from that week and earlier will be shown.")
    
    all_weeks = sorted(clusters["week"].unique())
    asof_week = st.selectbox("Answer as-of week (inclusive)", all_weeks, index=len(all_weeks)-1)
    allowed_weeks = [w for w in all_weeks if w <= asof_week]

    # ---- Trend analysis with filtering ----
    st.subheader("Trend Analysis")
    st.write("Explore different types of trends detected across time periods.")
    
    sel_types = st.multiselect(
        "Filter trend types",
        sorted(trends["type"].unique()),
        default=list(trends["type"].unique()),
        help="Select which trend types to display"
    )
    
    # Apply filters and display trends
    tview = trends[trends["week"].isin(allowed_weeks)]
    tview = tview[tview["type"].isin(sel_types)].sort_values(["week","type"])
    st.dataframe(tview, use_container_width=True, height=320)

    # ---- Weekly cluster exploration ----
    st.subheader("Weekly Clusters")
    st.write(f"Cluster breakdown for the selected as-of week: {asof_week}")
    week_clusters = clusters[clusters["week"]==asof_week].sort_values("size", ascending=False)
    st.write(week_clusters)

    # ---- Pipeline information ----
    st.caption(
        "Pipeline: Normalize → Embed → Weekly Cluster/Match → Time-aware Retrieval "
        "(retrieval re-ranks by cosine × recency outside this demo)."
    )

# ---- Sidebar with pipeline instructions ----
st.sidebar.header("How to regenerate")
st.sidebar.write("To update the data shown in this dashboard, run the following commands:")
st.sidebar.code("""# 1) Normalize raw logs (put JSONL/CSV in logs/)
python src/log_ingest.py

# 2) Embed to vectors (float16 by default; use --dtype float32 if preferred)
python src/embed_events.py --dim 512 --dtype float16

# 3) Cluster & detect trends (auto-K optional)
python src/cluster_trends.py --k 6 --auto_k --emerge_min 30

# 4) Launch this app
streamlit run streamlit_app.py
""")
