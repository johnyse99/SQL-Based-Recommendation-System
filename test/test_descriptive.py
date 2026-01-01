import pytest
import pandas as pd
from descriptive_analysis import DatabaseManager, RatingRecord
from pydantic import ValidationError

def test_rating_validation_success():
    """Test that valid data passes the schema."""
    data = {"user_id": 1, "item_id": 100, "rating": 4.5}
    assert RatingRecord(**data).rating == 4.5

def test_rating_validation_failure():
    """Test that invalid data (out of range) raises error."""
    invalid_data = {"user_id": 1, "item_id": 100, "rating": 10.0}
    with pytest.raises(ValidationError):
        RatingRecord(**invalid_data)

def test_db_ingestion_empty_df():
    """Ensure the DB manager handles empty data gracefully."""
    db = DatabaseManager(":memory:") # Use RAM for testing
    result = db.ingest_data(pd.DataFrame(), "test_table")
    assert result is False