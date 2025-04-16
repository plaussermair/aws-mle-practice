# steps/s8_ensemble_model.py
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from helpers import plot_confusion_matrix # Import helper

def show_ensemble_model(voting_pipeline, X_test, y_test, evaluation_results, best_individual_name):
    st.header(" V. Ensemble Model (Voting Classifier)")
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
             st.text("Voting Classification Report:")
             st.text(report_voting)
        with col2:
             st.markdown("**Voting Confusion Matrix:**")
             plot_confusion_matrix(cm_voting, ['No Churn', 'Churn'], 'CM: Voting Classifier', 'Purples') # Use helper

        # Safely get the best individual model's accuracy
        best_ind_acc = evaluation_results.get(str(best_individual_name), {}).get('accuracy', np.nan) # Ensure name is string

        st.subheader("Comparison vs. Best Single Model")
        if not np.isnan(best_ind_acc):
            if acc_voting > best_ind_acc:
                st.success(f"The Voting Classifier (Accuracy: {acc_voting:.4f}) **outperformed** the best individual model ({best_individual_name}: {best_ind_acc:.4f}).")
            else:
                st.info(f"The Voting Classifier (Accuracy: {acc_voting:.4f}) **did not outperform** the best individual model ({best_individual_name}: {best_ind_acc:.4f}).")
        else:
            st.warning(f"Could not retrieve accuracy for the best individual model '{best_individual_name}' to compare.")

    except Exception as e:
        st.error(f"Error evaluating the Voting Classifier: {e}")
        st.exception(e) # Show traceback for debugging

def show_model_evaluation(evaluation_results, X_test, y_test):
    # ...existing code to get metrics...
    metrics_df = pd.DataFrame([evaluation_results])
    st.dataframe(metrics_df.style.format(precision=3))