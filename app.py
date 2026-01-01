"""
Main Application - Recommendation System Dashboard
Standard: Senior AI Engineering
Description: Orchestrates Descriptive, Predictive, and Prescriptive phases.
"""

import streamlit as st
import logging
from src.config import config 
from src.database import DatabaseManager
from src.predictive_model import RecommenderEngine
from src.prescriptive_analysis import PrescriptiveEngine

# Logging configuration for execution tracking
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    # Page configuration
    st.set_page_config(page_title=config.APP_TITLE, layout="wide")

    # Sidebar Navigation
    st.sidebar.title("Project Navigation")
    page = st.sidebar.radio(
        "Go to Phase:", 
        ["📊 Descriptive Analysis", "🤖 Predictive Phase", "🎯 Prescriptive Strategy"]
    )

    # ENGINE INITIALIZATION
    # Utilizing st.session_state for model persistence across fragments
    if 'engine' not in st.session_state:
        try:
            # Explicit Dependency Injection of DB_PATH
            st.session_state.engine = RecommenderEngine(db_path=config.DB_PATH)
            logger.info(f"Engine successfully initialized using DB: {config.DB_PATH}")
        except Exception as e:
            logger.error(f"Failed to initialize RecommenderEngine: {e}")
            st.error("Critical Error: Could not initialize the Recommendation Engine.")

    # --- PAGE ROUTING LOGIC ---
    
    if page == "📊 Descriptive Analysis":
        st.header("Enterprise Recommendation Insights")
        st.subheader("Business Performance Metrics")
        
        # SQL Dependency Injection
        db = DatabaseManager(db_path=config.DB_PATH)
        data = db.get_performance_metrics()
        
        if not data.empty:
            st.dataframe(data, use_container_width=True)
        else:
            st.warning("No data found in the database. Please check your data source.")

    elif page == "🤖 Predictive Phase":
        st.header("Predictive Phase: Collaborative Filtering")
        engine = st.session_state.engine
        
        if st.button("🚀 Train Model"):
            with st.spinner("Processing Item-Item Similarity Matrix..."):
                if engine.train():
                    st.success("Model trained and loaded into memory successfully!")
                else:
                    st.error("Training failed. Ensure the SQL database is populated.")

        st.divider()

        # Recommendation UI (Inference)
        if engine.similarity_matrix is not None:
            available_items = engine.similarity_matrix.index.tolist()
            item_id = st.selectbox("Select an Item ID to find matches:", available_items)
            
            if st.button("Get Recommendations"):
                recs = engine.get_recommendations(item_id)
                if recs:
                    # Professional display of results
                    st.table([
                        {"Item ID": r.item_id, "Confidence Score": f"{r.score*100:.2f}%"} 
                        for r in recs
                    ])
                else:
                    st.info("No similar items found for the selected ID.")
        else:
            st.info("The model is not trained yet. Click the button above to begin.")

    elif page == "🎯 Prescriptive Strategy":
        st.header("Prescriptive Phase: AI Strategy Generator")
        engine = st.session_state.engine
        
        # Check if Predictive results are available to generate prescriptions
        if engine.similarity_matrix is not None:
            pres_engine = PrescriptiveEngine(db_path=config.DB_PATH)
            
            target_id = st.selectbox(
                "Select a product to generate an actionable strategy:", 
                engine.similarity_matrix.index
            )
            
            if st.button("Generate Action Plan"):
                # Fetch predictive results to feed the prescriptive engine
                recs = engine.get_recommendations(target_id)
                
                if recs:
                    for r in recs:
                        strategy = pres_engine.generate_business_strategy(r.item_id, r.score)
                        
                        # Professional Card UI for Business Actions
                        with st.container(border=True):
                            st.subheader(f"Strategy for Item {r.item_id}")
                            st.write(f"**Match Confidence:** {r.score*100:.2f}%")
                            st.info(f"**Action Plan:** {strategy.action_plan}")
                            st.warning(f"**Priority Level:** {strategy.priority}")
                else:
                    st.error("Could not generate strategy: No similar items found.")
        else:
            st.warning("⚠️ Predictive results required. Please train the model in the 'Predictive Phase' first.")

if __name__ == "__main__":
    main()