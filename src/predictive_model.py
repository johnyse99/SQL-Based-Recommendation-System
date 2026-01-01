"""
Project: SQL-Based Recommendation System - Predictive Phase
Description: Item-Item Collaborative Filtering logic using Cosine Similarity.
Inputs: Validated data from SQLite (Fase 1).
Outputs: Item Similarity Matrix and User Recommendations.

Senior Engineering Standards:
- Dependency Injection: Database path is configurable.
- Error Handling: Specific blocks for SQL and Matrix operations.
- State Management: Designed to work with Streamlit session_state.
"""

import sqlite3
import logging
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# --- DATA SCHEMAS (PYDANTIC) ---
class Recommendation(BaseModel):
    """Schema for a single recommendation output to ensure data integrity."""
    item_id: int
    score: float = Field(ge=0.0, le=1.0)

# --- CORE PREDICTIVE ENGINE ---
class RecommenderEngine:
    """
    Industrial-grade Recommendation Engine.
    Handles data loading, matrix transformation, and similarity inference.
    """

    def __init__(self, db_path: str):
        """
        Initializes the engine with a specific database path.
        Fixes the 'missing 1 required positional argument' error.
        """
        self.db_path = db_path
        self.similarity_matrix: Optional[pd.DataFrame] = None
        self.user_item_matrix: Optional[pd.DataFrame] = None
        logger.info(f"RecommenderEngine initialized with DB: {self.db_path}")

    def load_data_from_sql(self) -> pd.DataFrame:
        """Fetch processed data from the SQL persistence layer."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT user_id, item_id, rating FROM ratings"
                df = pd.read_sql_query(query, conn)
                
                if df.empty:
                    logger.warning("Query returned an empty dataset.")
                    return pd.DataFrame()
                
                logger.info(f"Loaded {len(df)} records for training.")
                return df
        except sqlite3.Error as e:
            logger.error(f"Database read error: {e}")
            return pd.DataFrame()

    def train(self) -> bool:
        """
        Builds the similarity matrix using Cosine Similarity.
        Transforms the user-item interaction into an item-item affinity map.
        """
        df = self.load_data_from_sql()
        
        if df.empty:
            logger.error("Training failed: No data available.")
            return False

        try:
            # Create Pivot Table: Rows=Users, Columns=Items
            # We use fillna(0) to handle the sparsity of the matrix
            self.user_item_matrix = df.pivot_table(
                index='user_id', 
                columns='item_id', 
                values='rating'
            ).fillna(0)

            # Calculate Item-Item Similarity
            # Note: We transpose (.T) to get items as rows for cosine_similarity
            item_similarity = cosine_similarity(self.user_item_matrix.T)
            
            # Reconstruct as DataFrame for easier querying
            self.similarity_matrix = pd.DataFrame(
                item_similarity,
                index=self.user_item_matrix.columns,
                columns=self.user_item_matrix.columns
            )
            
            logger.info("Model training completed successfully.")
            return True
        except Exception as e:
            logger.error(f"Mathematical transformation failed: {e}")
            return False

    def get_recommendations(self, item_id: int, top_n: int = 3) -> List[Recommendation]:
        """
        Returns top-N similar items for a given item using the similarity matrix.
        Excludes the item itself from the results.
        """
        # Critical Check: Validate if the model has been trained
        if self.similarity_matrix is None:
            logger.warning("Inference attempted on untrained model.")
            return []

        if item_id not in self.similarity_matrix.index:
            logger.warning(f"Item ID {item_id} not found in current similarity matrix.")
            return []

        try:
            # Extract scores for the target item and sort descending
            similar_scores = self.similarity_matrix[item_id].sort_values(ascending=False)
            
            # Slice results: Skip the first one (similarity with itself is always 1.0)
            similar_items = similar_scores.iloc[1:top_n + 1]
            
            # Validate output with Pydantic schema
            return [
                Recommendation(item_id=int(idx), score=round(float(val), 4)) 
                for idx, val in similar_items.items()
            ]
        except Exception as e:
            logger.error(f"Error during recommendation inference: {e}")
            return []

# Technical Documentation (IBM Senior Standards):
# 1. Dependency Injection: The constructor requires 'db_path', allowing for easy testing with mock DBs.
# 2. Mathematical Logic: Cosine similarity measures the angle between item vectors in a multi-dimensional user space.
# 3. Memory Management: Pivot tables are memory-intensive. For production at scale, 
#    sparse matrices (scipy.sparse.csr_matrix) should replace pandas pivot_tables.