# steps/s10_conclusion.py
import streamlit as st
import numpy as np

def show_conclusion():
    st.header("🏁 10. Conclusion: Tying It All Together")

    # --- Section 1: Recap of Journey and Results ---
    st.subheader("Project Recap: From Data to Insights")

    st.markdown("""
    Our journey began with a common business challenge: **understanding and predicting customer churn** in a telecommunications company. We aimed to build a tool that could identify customers likely to leave, enabling proactive retention efforts.

    **The Data:** We used the well-known **Telco Customer Churn dataset**, which contains information about individual customers – their demographics (like senior citizen status, partners, dependents), account details (tenure, contract type, payment method), the specific services they subscribe to (phone, internet type, online security, tech support, streaming, etc.), and their monthly and total charges. Crucially, it included the 'Churn' label, indicating whether a customer had left the company ('Yes' or 'No').

    **The Process:**
    We followed a standard, structured machine learning workflow:
    1.  **Loading & Understanding the Data:** We first loaded the raw data and performed initial checks to understand its size, structure, data types, and identify obvious issues like missing values (specifically in 'TotalCharges' for new customers).
    2.  **Exploring the Data (EDA):** We visualized the data to uncover patterns. We looked at distributions (like how many customers had long tenure vs. short tenure) and, importantly, how different factors related to actual churn. This exploratory phase revealed key insights – for example, customers on month-to-month contracts with high monthly charges and fewer additional services were more likely to churn. We also confirmed the dataset was 'imbalanced' (fewer churners than non-churners), a critical factor for modeling.
    3.  **Preparing the Data:** We cleaned the data (handling missing values, dropping irrelevant IDs) and then **engineered new features**. This meant creating more informative signals from the existing data, like grouping tenure into categories or calculating the number of optional services used, based on hypotheses formed during EDA.
    4.  **Splitting the Data:** We divided the prepared data into a 'training' set (to teach the models) and a 'test' set (kept separate for a final, unbiased evaluation).
    5.  **Preprocessing:** We transformed the data into a format suitable for machine learning models, mainly by scaling numerical features (so large numbers didn't dominate) and converting categorical features (like 'Contract' type) into numerical representations (using one-hot encoding).
    6.  **Modeling & Tuning:** We trained several different types of classification models (like Logistic Regression, Random Forest, XGBoost, LightGBM), experimenting with techniques (SMOTE, class weights) to handle the data imbalance. We used automated tuning (`RandomizedSearchCV`) to find the best settings (hyperparameters) for each model based on its performance during cross-validation on the training data.
    7.  **Evaluation:** We evaluated the tuned models on the unseen 'test' set using metrics like Accuracy (overall correctness) and ROC AUC (ability to distinguish churners from non-churners). We compared their performance using classification reports and confusion matrices.
    8.  **Ensemble (Optional):** We explored combining the predictions of the top individual models (using a Voting Classifier) to see if the collective 'wisdom' could yield even better, more stable results.
    9.  **Interpretation (SHAP):** Finally, for the best-performing *individual* model, we used SHAP analysis to understand *why* it was making specific predictions – identifying which features were most influential for predicting churn overall and for individual customers.

    """)

    # --- Section 2: Results Summary ---
    artifacts = st.session_state.get('artifacts', {})
    final_model_name = artifacts.get('final_model_name', 'N/A')
    evaluation_results = artifacts.get('evaluation_results', {})
    final_model_results = evaluation_results.get(str(final_model_name), {})
    final_accuracy = final_model_results.get('accuracy', np.nan)
    final_roc_auc = final_model_results.get('roc_auc', np.nan)

    st.subheader("Key Result")
    st.success(f"""
    This systematic process culminated in the selection of the **'{final_model_name}'** model. On the unseen test data, it demonstrated strong predictive capability:
    *   **Accuracy:** **{final_accuracy:.4f}** (Correctly predicted the outcome for ~{final_accuracy*100:.1f}% of test customers)
    *   **ROC AUC:** **{final_roc_auc:.4f}** (Shows a strong ability to differentiate between customers who will churn and those who won't, significantly better than random guessing which has an AUC of 0.5)

    This result directly addresses the initial problem: we now have a validated model capable of identifying customers likely to churn with a quantifiable level of accuracy.
    """)

    st.markdown("---") # Separator before the detailed perspectives

    # --- Section 3: Business Value & Technical Perspectives ---
    # (Keep the previously generated SA and MLE sections here)

    st.subheader("📈 Business Value & Strategic Impact")
    st.markdown("""
    Predicting churn is not just an academic exercise; it's a strategic tool for customer retention and business growth.

    *   **Reduced Revenue Loss:** Proactively identifying at-risk customers allows for targeted retention campaigns (e.g., personalized offers, support outreach) *before* they leave, directly mitigating revenue leakage associated with churn. This is typically far more cost-effective than acquiring new customers.
    *   **Improved Customer Lifetime Value (CLTV):** By reducing churn and extending customer tenure, the overall CLTV increases, boosting long-term profitability.
    *   **Enhanced Customer Experience:** Understanding *why* customers churn (via feature importance/SHAP) provides actionable insights to improve products, services, or support processes, leading to higher overall satisfaction even for non-churning customers.
    *   **Optimized Resource Allocation:** Instead of generic, expensive retention campaigns, resources (marketing budget, support agent time) can be focused on customers with the highest churn risk *and* potentially highest value, maximizing the ROI of retention efforts.
    *   **Competitive Advantage:** A lower churn rate compared to competitors signifies a healthier business, stronger customer loyalty, and potentially a better market reputation.
    """)

    st.subheader("🏗️ Solutions Architect Perspective: Integration & Scalability")
    st.markdown("""
    From an SA viewpoint, the focus is on integrating this ML capability seamlessly and reliably into the existing business ecosystem.

    *   **Deployment Strategy:** The selected pipeline (`final_pipeline.joblib`) can be deployed as a scalable **API endpoint** (e.g., using Flask/FastAPI within a Docker container, hosted on Cloud Functions, AWS Lambda, or Kubernetes). This allows real-time predictions for individual customers (e.g., triggered by a support call) or small batches.
    *   **Batch Scoring:** For large-scale campaigns, the model can be integrated into **batch processing workflows** (e.g., using Airflow, AWS Batch, Databricks Jobs) to score the entire customer base periodically (e.g., weekly/monthly), feeding results into marketing automation platforms or CRM systems.
    *   **System Integration:** The prediction API or batch output needs integration points with:
        *   **CRM:** To flag at-risk customers for sales/support teams.
        *   **Marketing Automation:** To trigger targeted email/SMS campaigns or personalized offers.
        *   **Data Warehouse/Lake:** To store prediction history for analysis and monitoring.
    *   **Data Pipelines:** Robust pipelines are needed to reliably extract, transform (using the `preprocessor.joblib`), and feed the required features to the model endpoint or batch job, ensuring data consistency between training and inference.
    *   **Scalability & Performance:** The deployment architecture must handle expected load (e.g., number of API calls, size of batch jobs) cost-effectively. Cloud-native solutions offer auto-scaling capabilities. Latency requirements for real-time predictions need consideration.
    *   **Monitoring & Alerting:** Infrastructure monitoring (CPU, memory, endpoint availability) and logging are crucial for operational health. Alerts should trigger on system failures or performance degradation.
    """)

    st.subheader("⚙️ Machine Learning Engineer Perspective: Model Lifecycle & Maintainability")
    st.markdown("""
    An MLE focuses on the model's performance, robustness, and the end-to-end lifecycle required to maintain its value over time.

    *   **Actionable Explainability (SHAP):** The use of SHAP is critical. It moves beyond *what* the model predicts to *why*. This allows business users to understand the key drivers for a specific customer's churn risk (e.g., "High risk due to recent price increase + short tenure") and tailor interventions accordingly, building trust in the model.
    *   **Performance Monitoring:** Model metrics (Accuracy, AUC, Precision/Recall, F1) must be tracked **continuously** on new predictions (if ground truth becomes available). Degradation below acceptable thresholds should trigger investigation or retraining. Monitoring the distribution of prediction scores is also vital.
    *   **Concept & Data Drift Detection:** The underlying customer behavior or data characteristics might change over time (drift). Monitoring input feature distributions and prediction distributions compared to the training data is essential to detect drift early, indicating a need for model retraining or rebuilding.
    *   **Retraining Strategy:** A defined strategy for retraining is necessary (e.g., scheduled retraining every quarter, or triggered by performance degradation/drift detection). This requires automating the entire workflow demonstrated here (data extraction, preprocessing, training, evaluation, deployment).
    *   **Experimentation & Improvement:** This model serves as a baseline. Further improvements can involve:
        *   Engineering more sophisticated features (e.g., usage patterns, support ticket analysis).
        *   Trying different algorithms or ensemble techniques.
        *   Fine-tuning based on the specific cost/benefit of false positives vs. false negatives for the business.
    *   **Versioning & Reproducibility:** Both code (feature engineering, training script) and artifacts (preprocessor, model, evaluation results) need version control (e.g., Git, DVC, MLflow) to ensure reproducibility and enable rollbacks if needed.
    *   **CI/CD for ML:** Implementing Continuous Integration/Continuous Deployment pipelines automates testing, training, and deployment, ensuring faster iterations and more reliable updates.

    """)

    st.info("""
    **Synergy:** Successfully operationalizing this churn model requires close collaboration between SAs (designing the robust infrastructure and integration) and MLEs (building, monitoring, and improving the model itself) to deliver sustained business value.
    """)

    st.markdown("---")
    st.markdown("**Thank you for exploring this project journey!**")