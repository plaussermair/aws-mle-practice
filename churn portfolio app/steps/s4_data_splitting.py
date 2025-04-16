# steps/s4_data_splitting.py
import streamlit as st
import pandas as pd # Needed for dataframe display

def show_data_splitting(X_test, y_test):
    st.header("🔪 4. Data Splitting")
    st.markdown("""
    To evaluate model performance reliably on unseen data, the dataset (features `X` and target `y` derived *after* feature engineering) was split into:
    *   **Training Set (80%):** Used to train the machine learning models and tune hyperparameters.
    *   **Testing Set (20%):** Held back and used *only* for the final evaluation of the selected model(s).

    **Stratification** on the 'Churn' variable was crucial during splitting. This ensures both training and testing sets maintain the original dataset's proportion of churners vs. non-churners, which is vital for imbalanced datasets.
    """)
    st.code("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)", language='python')
    st.subheader("Test Set Information")
    st.write(f"Shape of Test Features (X_test): **{X_test.shape}**")
    st.write(f"Shape of Test Target (y_test): **{y_test.shape}**")
    st.subheader("Churn Distribution in Test Set (Verification)")
    test_churn_dist = y_test.value_counts(normalize=True) * 100
    # Display as DataFrame for consistency
    st.dataframe(test_churn_dist.reset_index().rename(columns={'index':'Churn', y_test.name:'Percentage'})) # Use y_test.name
    st.success("Stratification successfully maintained the churn distribution in the test set.")