# steps/s7_model_evaluation.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score # Base metrics
from helpers import plot_confusion_matrix, plot_roc_curve_comparison # Import helpers

def format_classification_report(report_str):
    """Convert classification report string to DataFrame"""
    # Split the report string into lines
    lines = report_str.split('\n')
    # Remove empty lines
    lines = [line for line in lines if line.strip()]
    
    # Extract headers and data
    headers = ['Class', 'Precision', 'Recall', 'F1-score', 'Support']
    data = []
    
    for line in lines[1:-3]:  # Skip header and avg/total rows
        if line.strip():
            # Split line and clean values
            row_values = line.split()
            if len(row_values) == 5:  # Normal class row
                class_name, precision, recall, f1, support = row_values
            else:  # Row with class name containing spaces
                class_name = ' '.join(row_values[:-4])
                precision, recall, f1, support = row_values[-4:]
            data.append([
                class_name,
                float(precision),
                float(recall),
                float(f1),
                int(support)
            ])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=headers)
    return df

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
    
    # Create a more readable DataFrame from evaluation_results
    metrics_rows = []
    for model_name, metrics in evaluation_results.items():
        row = {
            'Model': model_name,
            'Accuracy': metrics.get('accuracy', np.nan),
            'ROC AUC': metrics.get('roc_auc', np.nan)
        }
        metrics_rows.append(row)
    
    # Create DataFrame and sort by accuracy
    metrics_df = pd.DataFrame(metrics_rows).sort_values(by='Accuracy', ascending=False)
    
    # Display formatted DataFrame
    st.dataframe(
        metrics_df.style.format({
            'Accuracy': '{:.3f}',
            'ROC AUC': '{:.3f}',
            'Precision (Churn)': '{:.3f}',
            'Recall (Churn)': '{:.3f}',
            'F1 (Churn)': '{:.3f}'
        }),
        use_container_width=True
    )

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
            
            # Format and display classification report
            report_str = result.get('report', '')
            if report_str:
                st.markdown("### Classification Report:")
                df_report = format_classification_report(report_str)
                st.dataframe(
                    df_report.style.format({
                        'Precision': '{:.3f}',
                        'Recall': '{:.3f}',
                        'F1-score': '{:.3f}',
                        'Support': '{:,.0f}'
                    }),
                    use_container_width=True
                )
                
                # Add metric definitions
                st.markdown("""
                **Metric Definitions:**
                * **Precision:** Of all customers predicted to churn, what percentage actually churned
                    * *High precision = Few false alarms*
                * **Recall:** Of all actual churners, what percentage did we catch
                    * *High recall = Few missed churners*
                * **F1-score:** Harmonic mean of precision and recall (balance between false alarms and missed churners)
                    * *Higher is better (max=1.0)*
                * **Support:** Number of samples in each class
                    * *Shows class distribution in test set*
                """)
            else:
                st.warning("Classification report not available.")
        with col2:
            cm = result.get('confusion_matrix')
            if cm is not None:
                 st.markdown("**Confusion Matrix:**")
                 plot_confusion_matrix(cm, ['No Churn', 'Churn'], f'CM: {selected_model}', 'Greens') # Use helper
                 st.caption(f"""
                 *   **Top-Left (True Negative):** Correctly predicted 'No Churn'.
                 *   **Bottom-Right (True Positive):** Correctly predicted 'Churn'.
                 *   **Top-Right (False Positive):** Incorrectly predicted 'Churn' (False Alarm).
                 *   **Bottom-Left (False Negative):** Incorrectly predicted 'No Churn' (Missed Churn).
                 """)
            else:
                 st.warning("Confusion matrix not found.")

    st.subheader("ROC Curve Comparison")
    st.markdown("Visual comparison of model performance in distinguishing classes across different thresholds.")
    plot_roc_curve_comparison(evaluation_results, X_test, y_test) # Use helper