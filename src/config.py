"""
Module: Configuration Manager
Description: Loads environment variables and defines global constants.
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Ruta relativa desde la raíz del proyecto
    DB_PATH = os.getenv("DB_PATH", "data/recommendation_system.db")
    APP_TITLE = "IBM Recommendation System"

config = Config()