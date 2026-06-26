"""
Unit tests for utility functions in the time-aware RAG pipeline.

Tests core functionality like tokenization, hashing vectorization,
and time-aware scoring to ensure correctness and consistency.
"""
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import tokenize, hashing_vectorizer, time_weight, fused_score, _h32


class TestTokenization:
    """Tests for text tokenization functionality."""
    
    def test_basic_tokenization(self):
        """Test basic tokenization with common text."""
        text = "User login failed from 192.168.1.1"
        tokens = tokenize(text)
        expected = ["user", "login", "failed", "from", "192.168.1.1"]
        assert tokens == expected
    
    def test_stop_word_filtering(self):
        """Test that stop words are properly filtered."""
        text = "The user is in the system"
        tokens = tokenize(text)
        # Should exclude stop words like "the", "is", "in"
        assert "the" not in tokens
        assert "is" not in tokens
        assert "in" not in tokens
        assert "user" in tokens
        assert "system" in tokens
    
    def test_domain_specific_tokens(self):
        """Test preservation of domain-specific tokens."""
        text = "Access to /var/log/auth.log on host-01.example.com"
        tokens = tokenize(text)
        assert "/var/log/auth.log" in tokens
        assert "host-01.example.com" in tokens
    
    def test_empty_text(self):
        """Test handling of empty or None text."""
        assert tokenize("") == []
        assert tokenize(None) == []
    
    def test_short_token_filtering(self):
        """Test that tokens shorter than 3 characters are filtered."""
        text = "A big system ok"
        tokens = tokenize(text)
        assert "big" in tokens
        assert "system" in tokens
        # "A" and "ok" should be filtered (too short)
        assert "a" not in tokens
        assert "ok" not in tokens


class TestHashingVectorizer:
    """Tests for hashing-based vectorization."""
    
    def test_basic_vectorization(self):
        """Test basic vectorization functionality."""
        tokens = ["user", "login", "failed"]
        vector = hashing_vectorizer(tokens, dim=100)
        
        assert len(vector) == 100
        assert vector.dtype == np.float32
        assert not np.all(vector == 0)  # Should have non-zero values
    
    def test_empty_tokens(self):
        """Test handling of empty token list."""
        vector = hashing_vectorizer([], dim=50)
        assert len(vector) == 50
        assert np.all(vector == 0)  # Should be zero vector
    
    def test_l2_normalization(self):
        """Test L2 normalization option."""
        tokens = ["test", "tokens", "here"]
        vector = hashing_vectorizer(tokens, dim=100, l2=True)
        
        # Check that vector is approximately unit length
        norm = np.linalg.norm(vector)
        assert abs(norm - 1.0) < 1e-6
    
    def test_deterministic_output(self):
        """Test that vectorization is deterministic."""
        tokens = ["consistent", "hashing", "test"]
        vector1 = hashing_vectorizer(tokens, dim=64)
        vector2 = hashing_vectorizer(tokens, dim=64)
        
        np.testing.assert_array_equal(vector1, vector2)
    
    def test_signed_hashing(self):
        """Test signed vs unsigned hashing."""
        tokens = ["test", "signed", "hashing"]
        
        signed_vec = hashing_vectorizer(tokens, dim=100, use_sign=True)
        unsigned_vec = hashing_vectorizer(tokens, dim=100, use_sign=False)
        
        # Vectors should be different
        assert not np.array_equal(signed_vec, unsigned_vec)
        
        # Unsigned should have no negative values
        assert np.all(unsigned_vec >= 0)
    
    def test_dtype_parameter(self):
        """Test different data type options."""
        tokens = ["dtype", "test"]
        
        vec_f16 = hashing_vectorizer(tokens, dtype=np.float16)
        vec_f32 = hashing_vectorizer(tokens, dtype=np.float32)
        
        assert vec_f16.dtype == np.float16
        assert vec_f32.dtype == np.float32


class TestTimeAwareScoring:
    """Tests for time-aware scoring functions."""
    
    def test_time_weight_calculation(self):
        """Test basic time weight calculation."""
        now = datetime(2025, 1, 15, tzinfo=timezone.utc)
        recent = datetime(2025, 1, 14, tzinfo=timezone.utc)  # 1 day ago
        old = datetime(2025, 1, 1, tzinfo=timezone.utc)  # 14 days ago
        
        recent_weight = time_weight(recent, now=now, half_life_days=14.0)
        old_weight = time_weight(old, now=now, half_life_days=14.0)
        
        # Recent should have higher weight
        assert recent_weight > old_weight
        
        # 14 days old should have weight ≈ 0.5 (half-life)
        assert abs(old_weight - 0.5) < 0.1
    
    def test_time_weight_edge_cases(self):
        """Test edge cases for time weight calculation."""
        now = datetime.now(timezone.utc)
        
        # None timestamp should return 0
        assert time_weight(None, now=now) == 0.0
        
        # Same timestamp should return 1
        assert time_weight(now, now=now) == 1.0
        
        # Future timestamp should return 1 (clamped)
        future = now + timedelta(days=1)
        assert time_weight(future, now=now) >= 1.0
    
    def test_timezone_handling(self):
        """Test timezone-aware timestamp handling."""
        now_utc = datetime(2025, 1, 15, tzinfo=timezone.utc)
        naive_dt = datetime(2025, 1, 14)  # No timezone
        
        # Should handle naive datetime by assuming UTC
        weight = time_weight(naive_dt, now=now_utc)
        assert 0 < weight <= 1
    
    def test_fused_score_calculation(self):
        """Test fused scoring with similarity and recency."""
        now = datetime(2025, 1, 15, tzinfo=timezone.utc)
        recent = datetime(2025, 1, 14, tzinfo=timezone.utc)
        
        similarity = 0.8
        alpha = 0.7
        
        fused = fused_score(similarity, recent, alpha=alpha, now=now)
        
        # Score should be between 0 and 1
        assert 0 <= fused <= 1
        
        # With high alpha, similarity should dominate
        time_only = fused_score(0.0, recent, alpha=0.0, now=now)  # Pure recency
        sim_only = fused_score(similarity, recent, alpha=1.0, now=now)  # Pure similarity
        
        assert sim_only == similarity
        assert time_only < sim_only  # Assuming recent timestamp has weight < similarity
    
    def test_fused_score_alpha_extremes(self):
        """Test fused score with extreme alpha values."""
        now = datetime.now(timezone.utc)
        ts = now - timedelta(days=7)
        similarity = 0.9
        
        # Alpha = 0 should give pure time weight
        pure_time = fused_score(similarity, ts, alpha=0.0, now=now)
        expected_time = time_weight(ts, now=now)
        assert abs(pure_time - expected_time) < 1e-6
        
        # Alpha = 1 should give pure similarity
        pure_sim = fused_score(similarity, ts, alpha=1.0, now=now)
        assert abs(pure_sim - similarity) < 1e-6


class TestHashFunction:
    """Tests for the internal hash function."""
    
    def test_hash_deterministic(self):
        """Test that hash function is deterministic."""
        text = "test string"
        hash1 = _h32(text)
        hash2 = _h32(text)
        assert hash1 == hash2
    
    def test_hash_different_inputs(self):
        """Test that different inputs produce different hashes."""
        hash1 = _h32("string1")
        hash2 = _h32("string2")
        assert hash1 != hash2
    
    def test_hash_returns_int(self):
        """Test that hash function returns integer."""
        result = _h32("test")
        assert isinstance(result, int)
        assert 0 <= result < 2**32  # Should be 32-bit unsigned int


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])