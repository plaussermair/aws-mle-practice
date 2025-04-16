# steps/s5_preprocessing.py
import streamlit as st
import pandas as pd

def show_preprocessing(preprocessor, model_columns, X_test):
    st.header("✨ 5. Preprocessing Pipeline")
    st.markdown("""
    Machine learning models require numerical input and perform best when features are appropriately scaled. A Scikit-learn `Pipeline` combined with a `ColumnTransformer` automated these steps consistently for both training and testing data.
    """)
    st.subheader("Steps Implemented:")
    st.markdown("""
    1.  **Numerical Scaling:** `StandardScaler` was applied to all numerical features (including engineered ones). This scales data to have zero mean and unit variance, preventing features with naturally larger values (like TotalCharges) from disproportionately influencing distance-based or regularized models.
    2.  **Categorical Encoding:** `OneHotEncoder` transformed categorical features into numerical format. Each category became a new binary (0/1) column. `handle_unknown='ignore'` prevents errors if unseen categories appear in test data (assigns all zeros), and `drop='first'` removes one category per feature to avoid multicollinearity.
    """)
    st.subheader("Preprocessor Object")
    # Displaying the object structure might be complex, show its string representation
    st.code(str(preprocessor))
    st.subheader("Features After Preprocessing")
    if model_columns:
        st.markdown(f"The pipeline transforms the input features into **{len(model_columns)}** numerical features ready for modeling:")
        st.text_area("Model Input Features (Sample)", ", ".join(model_columns[:25]) + "...", height=100)
        # Display sample of processed data
        try:
            # Ensure X_test is a DataFrame for proper transformation if needed by preprocessor
            if not isinstance(X_test, pd.DataFrame):
                 st.warning("X_test is not a DataFrame, attempting transformation anyway.")
                 # Potentially recreate DataFrame if original columns are known, but risky
                 # For now, assume preprocessor handles numpy array input if needed

            X_test_processed = preprocessor.transform(X_test)
            # Try creating DataFrame for display
            df_processed_sample = pd.DataFrame(X_test_processed[:5, :], columns=model_columns)
            st.markdown("**Sample of Processed Test Data (First 5 Rows):**")
            st.dataframe(df_processed_sample, use_container_width=True)
        except ValueError as ve:
             st.error(f"Error displaying processed data: {ve}. Potential mismatch between preprocessor expectations and X_test format or columns.")
             st.markdown("Showing raw processed data (first 5 rows):")
             st.dataframe(X_test_processed[:5, :]) # Show numpy array
        except Exception as e:
            st.warning(f"Could not display sample processed data: {e}")
            st.markdown("Showing raw processed data (first 5 rows):")
            try:
                st.dataframe(preprocessor.transform(X_test)[:5, :])
            except Exception as e2:
                 st.error(f"Failed to show even raw processed data: {e2}")

    else:
        st.warning("Model column names were not loaded. Cannot display the final feature list or processed data sample with names.")