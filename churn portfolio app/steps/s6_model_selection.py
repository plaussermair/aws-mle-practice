# steps/s6_model_selection.py
import streamlit as st
import json # For displaying parameters nicely

def show_model_selection(evaluation_results, best_individual_name, best_individual_pipeline):
    st.header("🧠 6. Model Selection & Training")
    st.markdown("""
    With preprocessed data ready, various classification algorithms were trained and evaluated. Key considerations included handling the class imbalance identified in EDA and optimizing model hyperparameters.
    """)
    st.subheader("Addressing Class Imbalance")
    st.markdown("""
    Two primary strategies were employed within the model training pipelines:
    1.  **SMOTE (Synthetic Minority Over-sampling Technique):** Used via `imbalanced-learn`'s `Pipeline`. SMOTE intelligently generates *synthetic* samples of the minority class (Churn='Yes') in the *training data only* during cross-validation fits, helping the model learn minority patterns without simply duplicating existing samples.
    2.  **Class Weights:** Models like Logistic Regression, Random Forest, and LightGBM have a `class_weight='balanced'` parameter. This automatically adjusts weights inversely proportional to class frequencies, effectively penalizing misclassifications of the minority class more heavily during training. XGBoost uses `scale_pos_weight` for a similar purpose.
    """)
    st.subheader("Models Considered")
    st.markdown("""
    *   **Logistic Regression:** Linear model, good baseline for interpretability.
    *   **Random Forest:** Ensemble of decision trees, robust to overfitting, captures interactions.
    *   **XGBoost:** Efficient gradient boosting implementation, often high accuracy.
    *   **LightGBM:** Another gradient boosting model, known for speed and efficiency, especially on large datasets.
    *(Models were evaluated using both SMOTE and Class Weight strategies where applicable.)*
    """)
    st.subheader("Hyperparameter Tuning with RandomizedSearchCV")
    st.markdown("""
    Each model type has hyperparameters (e.g., tree depth, learning rate) that significantly impact performance. `RandomizedSearchCV` was used to efficiently explore different combinations.
    *   It randomly samples a fixed number (`n_iter`) of parameter combinations from specified distributions.
    *   It uses **k-fold cross-validation** (typically 5-fold) on the *training data* for each combination.
    *   The combination yielding the best average cross-validation score (using **Accuracy** as the primary metric here, although others like ROC AUC or F1 could be chosen) was selected for each model/strategy pair.
    """)
    st.subheader("Example: Final Parameters of Best Individual Model")
    if best_individual_name and best_individual_pipeline:
         st.write(f"The best individual model selected was **{best_individual_name}**. Its final classifier parameters (after tuning) were:")
         try:
              # Access the classifier step within the pipeline
              classifier = best_individual_pipeline.named_steps.get('classifier')
              if classifier:
                   clf_params = classifier.get_params()
                   # Pretty print the dictionary using st.json
                   st.json(clf_params, expanded=False)
              else:
                   st.warning(f"Could not find 'classifier' step in the pipeline: {best_individual_pipeline.steps}")
         except Exception as e:
              st.warning(f"Could not retrieve parameters: {e}")
    else:
         st.warning("Could not load best individual model name or pipeline to show parameters.")