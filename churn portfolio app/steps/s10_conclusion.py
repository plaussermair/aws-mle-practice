# steps/s10_conclusion.py
import streamlit as st
import numpy as np

def show_conclusion():
    st.header("🏁 9. Conclusion & Next Steps")

    # Retrieve final model name and results from session state if available
    # This avoids passing the whole artifacts dict down
    artifacts = st.session_state.get('artifacts', {})
    final_model_name = artifacts.get('final_model_name', 'N/A')
    evaluation_results = artifacts.get('evaluation_results', {})

    final_model_results = evaluation_results.get(str(final_model_name), {}) # Ensure key is string
    final_accuracy = final_model_results.get('accuracy', np.nan)
    final_roc_auc = final_model_results.get('roc_auc', np.nan)

    st.markdown(f"""
    This project successfully navigated the machine learning workflow to build a model predicting Telco customer churn.

    **Key Outcomes & Insights:**
    *   EDA effectively identified key factors influencing churn, such as short **tenure**, **Month-to-Month contracts**, **Fiber optic** service, lack of **support services**, and **Electronic check** payments.
    *   Feature engineering created valuable signals (e.g., `TenureGroup`, `NumOptionalServices`) that likely aided model performance.
    *   Addressing **class imbalance** via SMOTE or class weights was confirmed as essential.
    *   Gradient boosting models (**LightGBM**, **XGBoost**) and **Random Forest** generally performed well, often benefiting from careful hyperparameter tuning.
    *   An **Ensemble (Voting Classifier)** was explored, potentially offering more robustness.
    *   The final selected model (**{final_model_name}**) demonstrated strong predictive capability on unseen test data, achieving:
        *   **Accuracy:** {final_accuracy:.4f} {'(N/A)' if np.isnan(final_accuracy) else ''}
        *   **ROC AUC:** {final_roc_auc:.4f} {'(N/A)' if np.isnan(final_roc_auc) else ''}
    *   **SHAP analysis** provided crucial model interpretability, pinpointing the main drivers behind individual predictions and confirming the importance of features like `tenure`, `Contract`, and `MonthlyCharges`.

    **Business Value & Next Steps:**
    *   The model can proactively identify customers at high risk of churning, enabling targeted retention campaigns.
    *   Insights from feature importance and SHAP can inform business strategy (e.g., improving support for Fiber users, simplifying payment options, offering incentives for longer contracts).
    *   **Future Work:**
        *   Deploy the model as a microservice/API for integration into operational systems.
        *   Continuously monitor model performance (drift) and retrain periodically with new data.
        *   Explore more advanced features (e.g., time-series analysis of charges, customer interaction logs if available).
        *   A/B test different retention strategies based on model predictions.

    Thank you for following this project journey!
    """)