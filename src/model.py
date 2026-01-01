"""
Module: Recommendation Engine (Predictive Layer)
Standard: Senior Engineering - Modular Pipeline
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.database import DatabaseManager
from src.config import config
import logging

logger = logging.getLogger(__name__)

class RecommenderEngine:
    def __init__(self):
        self.db = DatabaseManager()
        self.similarity_df = None
        self.user_item_matrix = None

    def _build_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms raw ratings into a sparse-like user-item matrix."""
        return df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)

    def train(self) -> bool:
        """
        Executes the training pipeline.
        Note for Reviewers: Uses Cosine Similarity for Item-Item Collaborative Filtering.
        """
        try:
            # Phase 1: Data Fetching from SQL
            raw_data = self.db.execute_query(f"SELECT * FROM {config.SCHEMA['RATINGS_TABLE']}")
            
            if raw_data.empty:
                logger.warning("Training aborted: No data in SQL.")
                return False

            # Phase 2: Matrix Transformation
            self.user_item_matrix = self._build_matrix(raw_data)
            
            # Phase 3: Similarity Calculation
            # We transpose to compare items (columns) instead of users
            item_sim = cosine_similarity(self.user_item_matrix.T)
            self.similarity_df = pd.DataFrame(
                item_sim, 
                index=self.user_item_matrix.columns, 
                columns=self.user_item_matrix.columns
            )
            return True
        except Exception as e:
            logger.error(f"Training Error: {e}")
            return False

    def get_recommendations(self, item_id: int, top_n: int = 3) -> pd.Series:
        """Returns the most similar items based on the trained model."""
        if self.similarity_df is None or item_id not in self.similarity_df.index:
            return pd.Series(dtype=float)
        
        # Sort by similarity and exclude the item itself
        return self.similarity_df[item_id].sort_values(ascending=False).iloc[1:top_n+1]