"""
Utility functions for text processing, vectorization, and time-aware scoring.

This module provides core utilities for the time-aware embeddings pipeline:
- Text tokenization with domain-specific handling
- Hashing-based vectorization for fast, dependency-light embeddings  
- Time-aware scoring functions for recency-weighted retrieval

The hashing vectorizer is particularly useful when you need fast, lightweight
embeddings without external model dependencies, while the time-aware scoring
functions enable temporal ranking of search results.
"""
from __future__ import annotations
import re, hashlib
import numpy as np
from datetime import datetime, timezone

# Regex to preserve domain-specific tokens like IPs, URLs, file paths
_TOKEN_RE = re.compile(r"\b[\w.\-/:]+\b", flags=re.UNICODE)

# Common English stop words to filter out
_STOP = {
    "a","an","and","are","as","at","be","by","for","from","has","he","in","is","it",
    "its","of","on","that","the","to","was","will","with"
}

def tokenize(text: str) -> list[str]:
    """
    Tokenize text into meaningful terms for security/IT domain.
    
    Preserves domain-specific tokens like IP addresses, URLs, and file paths
    while filtering out common stop words and very short terms.
    
    Args:
        text: Input text to tokenize
    
    Returns:
        List of filtered tokens
    """
    if not text:
        return []
    toks = [t.lower() for t in _TOKEN_RE.findall(text)]
    return [t for t in toks if len(t) > 2 and t not in _STOP]

def _h32(s: str) -> int:
    """
    Generate stable 32-bit hash from string using SHA1.
    
    Uses first 8 hex characters of SHA1 hash for consistent, deterministic
    hashing across runs.
    """
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16)

def hashing_vectorizer(tokens: list[str],
                       dim: int = 512,
                       use_sign: bool = True,
                       l2: bool = True,
                       dtype = np.float32) -> np.ndarray:
    """
    Generate fixed-dimensional embeddings using feature hashing.
    
    Fast, dependency-light vectorization that maps tokens to vector dimensions
    using hashing. Uses signed hashing to reduce collision effects and optional
    L2 normalization for unit vectors.
    
    Args:
        tokens: List of token strings to vectorize
        dim: Target vector dimension
        use_sign: Whether to use signed hashing (+1/-1) to reduce collisions
        l2: Whether to L2-normalize the resulting vector
        dtype: Output data type
    
    Returns:
        Vector of specified dimension and type
    """
    v = np.zeros(dim, dtype=np.float32)  # Use float32 for numerical stability
    if not tokens:
        return v.astype(dtype, copy=False)
    
    for tok in tokens:
        h = _h32(tok)
        idx = h % dim
        if use_sign:
            # Use hash parity for +1/-1 to reduce collision effects
            v[idx] += 1.0 if (h & 1) == 0 else -1.0
        else:
            v[idx] += 1.0
    
    if l2:
        n = float(np.linalg.norm(v))
        if n > 0:
            v /= n
    
    return v.astype(dtype, copy=False)

# -------- Time-aware scoring functions --------

def time_weight(ts: datetime,
                now: datetime | None = None,
                half_life_days: float = 14.0) -> float:
    """
    Calculate exponential decay weight based on timestamp age.
    
    Returns a weight in [0,1] where the weight halves every half_life_days.
    More recent timestamps get higher weights, with exponential decay over time.
    
    Args:
        ts: Timestamp to calculate weight for
        now: Reference time (defaults to current UTC time)
        half_life_days: Days for weight to decay to 0.5
    
    Returns:
        Weight between 0 and 1, with 1 being most recent
    """
    if ts is None:
        return 0.0
    if now is None:
        now = datetime.now(timezone.utc)
    
    # Ensure timezone-aware comparison
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    
    age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
    return 0.5 ** (age_days / float(half_life_days))

def fused_score(similarity: float,
                ts: datetime,
                alpha: float = 0.7,
                half_life_days: float = 14.0,
                now: datetime | None = None) -> float:
    """
    Combine semantic similarity with temporal recency into a single score.
    
    Creates a weighted combination of semantic similarity and time-based recency.
    Higher alpha values favor semantic similarity, lower values favor recency.
    
    Args:
        similarity: Semantic similarity score (e.g., cosine similarity)
        ts: Timestamp of the item being scored
        alpha: Weight for similarity (0-1), (1-alpha) is weight for recency
        half_life_days: Half-life for time decay calculation
        now: Reference time for recency calculation
    
    Returns:
        Combined score: alpha * similarity + (1-alpha) * time_weight
    """
    rec = time_weight(ts, now=now, half_life_days=half_life_days)
    return float(alpha) * float(similarity) + (1.0 - float(alpha)) * float(rec)
