"""
Project: SQL-Based Recommendation System - Descriptive Phase (Senior Level)
Description: Data ingestion, SQL-based EDA, and visualization.
Inputs: Raw CSV data or dictionaries.
Outputs: SQLite database and Streamlit Dashboard.
Architecture: Modular Class-based approach with Pydantic validation and logging.
"""

import sqlite3
import logging
import pandas as pd
import streamlit as st
import plotly.express as px
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError

# --- CONFIGURATION & LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

CONFIG: Dict[str, Any] = {
    "DB_NAME": "data/recommendation_system.db",
    "APP_TITLE": "Enterprise Recommendation Insights",
    "SCHEMA": {
        "RATINGS_TABLE": "ratings",
        "ITEMS_TABLE": "items"
    }
}

# --- DATA VALIDATION (PYDANTIC) ---
class RatingRecord(BaseModel):
    """Validates the structure of a single rating entry."""
    user_id: int = Field(gt=0)
    item_id: int = Field(gt=0)
    rating: float = Field(ge=0.0, le=5.0)

# --- CORE LOGIC ---
class DatabaseManager:
    """Handles all database interactions with robustness."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def __enter__(self):
        self._conn = sqlite3.connect(self.db_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._conn:
            self._conn.close()

    def ingest_data(self, df: pd.DataFrame, table_name: str) -> bool:
        """Safe ingestion of dataframes into SQL."""
        try:
            if df.empty:
                raise ValueError("The provided DataFrame is empty.")
            
            with sqlite3.connect(self.db_path) as conn:
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                logger.info(f"Successfully ingested {len(df)} rows into {table_name}")
                return True
        except Exception as e:
            logger.error(f"Failed to ingest data: {e}")
            return False

    def execute_query(self, query: str) -> pd.DataFrame:
        """Executes a SQL query and returns a DataFrame."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"SQL Execution Error: {e}")
            return pd.DataFrame()

# --- STREAMLIT UI ---
def run_dashboard():
    """Main UI Logic for Streamlit."""
    st.set_page_config(page_title=CONFIG["APP_TITLE"], layout="wide")
    st.title(f"🚀 {CONFIG['APP_TITLE']}")

    db = DatabaseManager(CONFIG["DB_NAME"])

    # 1. Mock Data Injection (Simulating real-world input)
    raw_data = [
        {"user_id": 1, "item_id": 101, "rating": 5.0},
        {"user_id": 2, "item_id": 101, "rating": 4.5},
        {"user_id": 3, "item_id": 102, "rating": 3.0}
    ]

    # Senior Check: Data Validation
    try:
        validated_data = [RatingRecord(**item).model_dump() for item in raw_data]
        df_validated = pd.DataFrame(validated_data)
        db.ingest_data(df_validated, CONFIG["SCHEMA"]["RATINGS_TABLE"])
    except ValidationError as e:
        st.error(f"Data Schema Mismatch: {e}")
        return

    # 2. SQL Descriptive Analysis
    st.header("Business Performance Metrics")
    
    query = f"""
        SELECT item_id, 
               COUNT(user_id) as total_interactions, 
               AVG(rating) as avg_score
        FROM {CONFIG["SCHEMA"]["RATINGS_TABLE"]}
        GROUP BY item_id
        ORDER BY total_interactions DESC
    """
    
    metrics_df = db.execute_query(query)

    if not metrics_df.empty:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(metrics_df.style.highlight_max(axis=0))
        with col2:
            fig = px.bar(metrics_df, x='item_id', y='avg_score', 
                         title="Item Popularity vs Rating", color='avg_score')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data found to display.")

if __name__ == "__main__":
    run_dashboard()