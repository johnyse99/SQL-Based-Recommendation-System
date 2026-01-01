# SQL-Based Recommendation System

This project implements a complete Data Science lifecycle focused on **Applied Artificial Intelligence**. It transitions from raw SQL data to a prescriptive strategy engine.

## 🏗️ System Architecture
The project is modularized into three distinct phases to ensure scalability and maintainability (SOLID principles):

1.  **Descriptive Phase**: SQL-based performance metrics extraction and business health visualization.
2.  **Predictive Phase**: Item-Item Collaborative Filtering using Cosine Similarity to identify hidden patterns in user behavior.
3.  **Prescriptive Phase**: An AI-driven strategy engine that transforms similarity scores into actionable executive plans.

## 🛠️ Tech Stack
* **Language**: Python 3.12 (Strict Type Hinting)
* **Database**: SQLite (Relational Persistence)
* **Core Libraries**: Pandas, Scikit-Learn, Pydantic, Logging
* **Interface**: Streamlit (Dashboarding)
* **DevOps**: Dotenv (Config Management), Pytest (Quality Assurance)

## 🚀 How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Configure your `.env` file (see `src/config.py`).
4. (Optional) Run `python seed_data.py` to populate the database.
5. Launch the dashboard: `streamlit run app.py`.

---

## ❓ Interview FAQ

**Q: Why use Cosine Similarity for the recommendation engine?**
A: Cosine similarity is effective for recommendation systems because it measures the orientation (angle) between item vectors rather than their magnitude. This makes it robust against variations in the number of ratings per user.

**Q: How does the system handle the 'Cold Start' problem?**
A: In this senior implementation, the Prescriptive Phase includes a "Fallthrough" logic. If the similarity score is 0.00% due to lack of data (sparsity), the system shifts from collaborative filtering to popularity-based recommendations (Descriptive Phase).

**Q: Why use Pydantic schemas in a Data Science project?**
A: To ensure data integrity. By validating the output of the recommendation engine before it reaches the UI, we prevent runtime errors and ensure that the business strategy always receives the expected parameters.

---

📄 **License**
This project is distributed under the MIT license. Its purpose is strictly educational and research-based, developed as an Applied Data Science solution.

**Note for recruiters:**
This project demonstrates advanced skills in Software Engineering applied to Artificial Intelligence. Modularity, dependency injection for database management, and state persistence in web applications (Streamlit) were prioritized. It provides a solid foundation for scaling to ML microservices or integrations with LLMs.
