# steps/s8_ensemble_model.py
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from helpers import plot_confusion_matrix # Import helper

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

def show_ensemble_model(voting_pipeline, X_test, y_test, evaluation_results, best_individual_name):
    st.header(" 8. Ensemble Model (Voting Classifier)")
    st.markdown("""
    Ensemble methods combine predictions from multiple models. A **Voting Classifier** was tested, using 'soft' voting (averaging probabilities) on the **top 3** individual models (based on test accuracy). The goal is to potentially achieve more robust and slightly better performance than any single model.
    """)
    if voting_pipeline is None:
        st.info("The Voting Classifier artifact was not found or loaded, possibly because it didn't outperform the best single model or wasn't generated. Skipping this section.")
        return

    try:
        # Basic check if the loaded object is likely fitted
        if not hasattr(voting_pipeline, 'estimators_') or not hasattr(voting_pipeline, 'predict'):
             st.warning("Loaded Voting Classifier object doesn't seem to be fitted or is invalid. Cannot evaluate.")
             return

        y_pred_voting = voting_pipeline.predict(X_test)
        # Ensure predict_proba exists before calling
        if hasattr(voting_pipeline, 'predict_proba'):
             y_pred_proba_voting = voting_pipeline.predict_proba(X_test)[:, 1]
             roc_auc_voting = roc_auc_score(y_test, y_pred_proba_voting)
        else:
             y_pred_proba_voting = None
             roc_auc_voting = np.nan # Cannot calculate ROC AUC
             st.warning("Voting classifier does not support 'predict_proba'. ROC AUC cannot be calculated.")


        acc_voting = accuracy_score(y_test, y_pred_voting)
        report_voting = classification_report(y_test, y_pred_voting, target_names=['No Churn', 'Churn'])
        cm_voting = confusion_matrix(y_test, y_pred_voting)

        st.subheader("Voting Classifier Performance on Test Set")
        col1, col2 = st.columns([1, 1])
        with col1:
             st.metric("Voting Accuracy", f"{acc_voting:.4f}")
             st.metric("Voting ROC AUC", f"{roc_auc_voting:.4f}" if not np.isnan(roc_auc_voting) else "N/A")
             
             # Format and display classification report
             if report_voting:
                 st.markdown("### Classification Report:")
                 df_report = format_classification_report(report_voting)
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
             st.markdown("**Voting Confusion Matrix:**")
             plot_confusion_matrix(cm_voting, ['No Churn', 'Churn'], 'CM: Voting Classifier', 'Purples')
             st.caption("""
             *   **Top-Left (True Negative):** Correctly predicted 'No Churn'.
             *   **Bottom-Right (True Positive):** Correctly predicted 'Churn'.
             *   **Top-Right (False Positive):** Incorrectly predicted 'Churn' (False Alarm).
             *   **Bottom-Left (False Negative):** Incorrectly predicted 'No Churn' (Missed Churn).
             """)

        # Comparison section
        st.markdown("---")
        st.subheader("Comparison vs. Best Single Model")
        
        # Create comparison DataFrame
        comparison_data = {
            'Model': ['Voting Classifier', str(best_individual_name)],
            'Accuracy': [acc_voting, evaluation_results.get(str(best_individual_name), {}).get('accuracy', np.nan)],
            'ROC AUC': [roc_auc_voting, evaluation_results.get(str(best_individual_name), {}).get('roc_auc', np.nan)]
        }
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display formatted comparison
        st.dataframe(
            comparison_df.style.format({
                'Accuracy': '{:.3f}',
                'ROC AUC': '{:.3f}'
            }),
            use_container_width=True
        )

        # Add performance comparison message
        best_ind_acc = evaluation_results.get(str(best_individual_name), {}).get('accuracy', np.nan) # Ensure name is string
        if not np.isnan(best_ind_acc):
            if acc_voting > best_ind_acc:
                st.success(f"✨ The Voting Classifier outperformed the best individual model by {(acc_voting - best_ind_acc):.4f} accuracy points.")
            else:
                st.info(f"The best individual model remains superior by {(best_ind_acc - acc_voting):.4f} accuracy points.")
        else:
            st.warning(f"Could not retrieve accuracy for the best individual model '{best_individual_name}' to compare.")

    except Exception as e:
        st.error(f"Error evaluating the Voting Classifier: {e}")
        st.exception(e)

def show_model_evaluation(evaluation_results, X_test, y_test):
    # ...existing code to get metrics...
    metrics_df = pd.DataFrame([evaluation_results])
    st.dataframe(metrics_df.style.format(precision=3))