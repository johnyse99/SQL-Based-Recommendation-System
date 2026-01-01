import sqlite3
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str): # Asegúrate de que acepte este argumento
        self.db_path = db_path

    def execute_query(self, query: str) -> pd.DataFrame:
        try:
            with sqlite3.connect(self.db_path) as conn:
                return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Error en SQL: {e}")
            return pd.DataFrame()

    def get_performance_metrics(self):
        # Lógica para la Fase Descriptiva
        query = """
            SELECT item_id, COUNT(user_id) as total_interactions, AVG(rating) as avg_score
            FROM ratings GROUP BY item_id ORDER BY total_interactions DESC
        """
        return self.execute_query(query)