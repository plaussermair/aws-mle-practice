# steps/s7_model_evaluation.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score # Base metrics
from helpers import plot_confusion_matrix, plot_roc_curve_comparison # Import helpers

def show_model_evaluation(evaluation_results, X_test, y_test):
    st.header("📈 7. Model Evaluation on Test Set")
    st.markdown("""
    This is the crucial step where the trained and tuned models are evaluated on the **unseen test set**. Performance here gives the best estimate of how the models would perform on new, real-world data.
    """)
    if not evaluation_results:
        st.error("Evaluation results artifact not loaded. Cannot display evaluation.")
        return

    st.subheader("Performance Metrics Summary")
    st.markdown("""
    Comparing key metrics across all trained model variants (SMOTE vs. Class Weight):
    *   **Accuracy:** Overall correctness.
    *   **ROC AUC:** Ability to distinguish between classes (higher is better).
    *   *(Precision, Recall, F1-Score are detailed in the classification reports below).*
    """)
    metrics_df = pd.DataFrame([evaluation_results])
    st.dataframe(metrics_df.style.format(precision=3))

    results_df_list = []
    for name, result in evaluation_results.items():
        acc = result.get('accuracy', np.nan)
        roc_auc = result.get('roc_auc', np.nan)
        f1_churn = np.nan
        report_str = result.get('report', '')
        # Attempt to parse F1 score for the positive class ('Churn')
        if report_str:
            # Look for the line starting with 'Churn' (adjust if target names differ)
            match = re.search(r"^\s*Churn\s+(\d\.\d+)\s+(\d\.\d+)\s+(\d\.\d+)", report_str, re.MULTILINE)
            if match:
                f1_churn = float(match.group(3))
            else:
                # Fallback: Check for class '1' if target names weren't used
                match_1 = re.search(r"^\s*1\s+(\d\.\d+)\s+(\d\.\d+)\s+(\d\.\d+)", report_str, re.MULTILINE)
                if match_1:
                    f1_churn = float(match_1.group(3))

        results_df_list.append({'Model': name, 'Accuracy': acc, 'ROC AUC': roc_auc, 'F1 (Churn)': f1_churn})

    results_summary_df = pd.DataFrame(results_df_list).sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
    st.dataframe(results_summary_df.style.format({
        'Accuracy': '{:.4f}', 'ROC AUC': '{:.4f}', 'F1 (Churn)': '{:.4f}'
    }), use_container_width=True)

    st.subheader("Detailed View & Confusion Matrix")
    # Ensure evaluation_results keys are strings if they aren't already
    model_options = list(map(str, evaluation_results.keys()))
    selected_model = st.selectbox("Select Model for Detailed Report:", options=model_options, key="eval_model_select")

    if selected_model and selected_model in evaluation_results: # Check if key exists
        result = evaluation_results[selected_model]
        st.markdown(f"--- \n ### Details for: **{selected_model}**")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.metric("Test Accuracy", f"{result.get('accuracy', np.nan):.4f}")
            st.metric("Test ROC AUC", f"{result.get('roc_auc', np.nan):.4f}")
            st.text("Classification Report:")
            st.text(result.get('report', 'Report not available.'))
        with col2:
            cm = result.get('confusion_matrix')
            if cm is not None:
                 st.markdown("**Confusion Matrix:**")
                 plot_confusion_matrix(cm, ['No Churn', 'Churn'], f'CM: {selected_model}', 'Greens') # Use helper
                 st.caption(f"""
                 *   **Top-Left (TN):** Correctly predicted 'No Churn'.
                 *   **Bottom-Right (TP):** Correctly predicted 'Churn'.
                 *   **Top-Right (FP):** Incorrectly predicted 'Churn' (False Alarm).
                 *   **Bottom-Left (FN):** Incorrectly predicted 'No Churn' (Missed Churn).
                 """)
            else:
                 st.warning("Confusion matrix not found.")

    st.subheader("ROC Curve Comparison")
    st.markdown("Visual comparison of model performance in distinguishing classes across different thresholds.")
    plot_roc_curve_comparison(evaluation_results, X_test, y_test) # Use helper