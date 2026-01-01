import pytest
import pandas as pd
from predictive_model import RecommenderEngine, Recommendation

def test_recommendation_schema():
    """Ensure the output follows the Pydantic schema."""
    rec = Recommendation(item_id=101, score=0.85)
    assert rec.item_id == 101
    assert rec.score <= 1.0

def test_empty_db_handling():
    """Test that engine handles missing database gracefully."""
    engine = RecommenderEngine("non_existent.db")
    assert engine.train() is False

def test_recommendation_logic():
    """Test if the engine returns the correct number of recommendations."""
    # Mocking similarity matrix logic could be done here for a full suite
    pass