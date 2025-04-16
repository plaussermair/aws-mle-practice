# steps/s3_feature_engineering.py
import streamlit as st
# Note: We don't need pandas here as we're just showing code/explanations

def show_feature_engineering(df_original=None): # df_original is optional now, only for context
    st.header("🛠️ 3. Feature Engineering & Processing")
    st.markdown("""
    Raw data often doesn't capture the full picture. Feature engineering involves creating new, potentially more informative features from the existing ones. This step combines insights from EDA with domain knowledge to help the model learn better. We also handle final data cleaning steps here.
    """)

    st.subheader("A. Data Cleaning & Basic Transformations")
    with st.expander("Details: Cleaning Steps", expanded=True): # Expand by default maybe
        st.markdown("**1. Handling Missing 'TotalCharges':**")
        st.markdown("""
        *   **Issue:** As seen in Load Data, `TotalCharges` was missing for customers with 0 tenure.
        *   **Strategy:** Imputed (filled) these missing values with **0**.
        *   **Rationale:** Customers with zero tenure logically haven't accumulated any total charges yet. This is a direct and interpretable imputation method in this context.
        """)
        st.code("df['TotalCharges'].fillna(0, inplace=True)", language='python')

        st.markdown("**2. Dropping 'customerID':**")
        st.markdown("""
        *   **Issue:** The `customerID` is a unique identifier for each customer.
        *   **Strategy:** Dropped the column.
        *   **Rationale:** Unique IDs generally have no predictive power for the *general* behavior of customers and can sometimes confuse models if not handled correctly.
        """)
        st.code("df.drop('customerID', axis=1, inplace=True)", language='python')

        st.markdown("**3. Encoding Target Variable 'Churn':**")
        st.markdown("""
        *   **Issue:** The target variable 'Churn' was categorical ('Yes'/'No'). Models require numerical targets.
        *   **Strategy:** Mapped 'Yes' to 1 and 'No' to 0.
        *   **Rationale:** Standard practice for binary classification tasks.
        """)
        st.code("df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})", language='python')

    st.subheader("B. Creating New Features (Feature Engineering)")
    st.markdown("Based on EDA insights and hypotheses about customer behavior, several new features were crafted:")

    with st.expander("Feature 1: Tenure Groups"):
        st.markdown("**Feature:** `TenureGroup` (Categorical)")
        st.markdown("**Logic:** Binned numerical `tenure` into discrete groups (e.g., '0-1 Year', '1-2 Years', '5+ Years').")
        st.code("bins = [0, 12, 24, 36, 48, 60, 100]\nlabels = [...]\ndf['TenureGroup'] = pd.cut(df['tenure'], bins=bins, labels=labels, right=False)", language='python')
        st.markdown("**Rationale:** EDA suggested churn rates might vary significantly across different tenure stages (e.g., high churn early on, lower churn later). Creating explicit groups allows models (especially linear ones or simpler trees) to capture these non-linear effects more easily than using raw tenure alone.")

    with st.expander("Feature 2: Number of Optional Services"):
        st.markdown("**Feature:** `NumOptionalServices` (Numerical)")
        st.markdown("**Logic:** Counted how many 'add-on' services (like `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`) a customer subscribes to. Mapped 'Yes' to 1, 'No' and 'No internet service' to 0, then summed across these columns.")
        st.code("# Simplified logic:\noptional_service_cols = [...] \ndf[col] = df[col].map({'Yes': 1, ...: 0})\ndf['NumOptionalServices'] = df[[...]].sum(axis=1)", language='python')
        st.markdown("**Rationale:** Hypothesis: Customers deeply integrated into the ecosystem (using more services) might be less likely to churn. This feature provides a single measure of service engagement.")

    with st.expander("Feature 3: Simplified Payment Method"):
        st.markdown("**Feature:** `PaymentMethodType` (Categorical)")
        st.markdown("**Logic:** Grouped the original `PaymentMethod` categories into 'Automatic' (bank transfer, credit card) and 'Manual' (mailed check, electronic check).")
        st.code("payment_map_simple = {...}\ndf['PaymentMethodType'] = df['PaymentMethod'].map(payment_map_simple)", language='python')
        st.markdown("**Rationale:** EDA showed distinct churn patterns between automatic and manual payments (especially high churn for electronic checks). Simplifying reduces dimensionality and focuses on the key behavioral difference: payment convenience/commitment.")

    with st.expander("Feature 4: Interaction Features"):
        st.markdown("**Features:** e.g., `Tenure_x_Monthly`, `Senior_Fiber` (Numerical/Binary)")
        st.markdown("**Logic:** Created features by combining existing ones:")
        st.code("df['Tenure_x_Monthly'] = df['tenure'] * df['MonthlyCharges']\ndf['Senior_Fiber'] = ((df['SeniorCitizen'] == 1) & (df['InternetService'] == 'Fiber optic')).astype(int)", language='python')
        st.markdown("**Rationale:**")
        st.markdown("*   `Tenure_x_Monthly`: Tests if the *combination* of high tenure AND high monthly charges has a different churn impact than either feature alone. Might capture long-term, high-value customers.")
        st.markdown("*   `Senior_Fiber`: EDA might suggest that senior citizens using Fiber Optic are a specific segment with unique churn behavior (e.g., potentially struggling with new tech or higher costs). This flag explicitly captures that intersection.")

    with st.expander("Feature 5: Ratio & Flag Features"):
        st.markdown("**Features:** e.g., `Monthly_per_Tenure`, `HighMonthlyCharge` (Numerical/Binary)")
        st.code("df['Monthly_per_Tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1e-6) # Handle tenure=0\nhigh_charge_threshold = df['MonthlyCharges'].quantile(0.80)\ndf['HighMonthlyCharge'] = (df['MonthlyCharges'] > high_charge_threshold).astype(int)", language='python')
        st.markdown("**Rationale:**")
        st.markdown("*   `Monthly_per_Tenure`: Normalizes monthly charges by tenure. Might indicate perceived value (high charge for short tenure could be bad).")
        st.markdown("*   `HighMonthlyCharge`: Creates a simple flag for customers in the top percentile of monthly spending. Useful for identifying potentially price-sensitive or premium customers.")

    st.success("**Outcome:** These engineered features, combined with the original cleaned ones, form the input dataset (denoted as `X`) for the subsequent preprocessing and modeling steps.")