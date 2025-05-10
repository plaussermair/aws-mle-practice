# steps/s2_eda.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def show_exploratory_data_analysis(df_original):
    st.header("📊 2. Exploratory Data Analysis (EDA)")
    st.markdown("""
    EDA is where we dive deep into the data to uncover patterns, relationships, and anomalies.
    We visualize distributions and explore how different features relate to customer churn.
    This understanding is crucial for effective feature engineering and model selection.
    """)

    # Prepare data for EDA
    df_eda = df_original.copy()
    df_eda['TotalCharges'] = pd.to_numeric(df_eda['TotalCharges'], errors='coerce').fillna(0)
    df_eda['Churn_numeric'] = df_eda['Churn'].map({'Yes': 1, 'No': 0})

    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = df_eda.select_dtypes(include='object').columns.drop(['customerID', 'Churn'])

    # Tabs for organization
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Target Variable (Churn)",
        "🔢 Numerical Features",
        "🔠 Categorical Features",
        "🔄 Relationships with Churn"
    ])

    with tab1:
        st.subheader("🎯 Target Variable: Churn Distribution")
        st.markdown("""
        Understanding the balance of our target variable is paramount. Imbalanced datasets, where one class heavily outweighs the other, require special handling during modeling.
        """)
        fig_churn, ax_churn = plt.subplots(figsize=(6, 4))
        churn_counts = df_eda['Churn'].value_counts()
        sns.countplot(data=df_eda, x='Churn', ax=ax_churn, palette='pastel', order=churn_counts.index)
        ax_churn.set_title('Customer Churn Distribution')
        for i, count in enumerate(churn_counts):
             ax_churn.text(i, count + 50, str(count), ha='center', va='bottom')
        st.pyplot(fig_churn, use_container_width=False)
        plt.close(fig_churn)

        churn_dist = df_eda['Churn'].value_counts(normalize=True) * 100
        st.write("Churn Percentage:")
        st.dataframe(churn_dist.reset_index().rename(columns={'index':'Churn', 'Churn':'Percentage'}))
        st.warning(f"""
        **Insight:** The dataset is **imbalanced**. Only **{churn_dist.get('Yes', 0):.2f}%** of customers in this dataset churned.
        This means a naive model predicting 'No Churn' for everyone would be ~73% accurate, but useless for identifying churners.
        This imbalance necessitates techniques like Synthetic Minority Oversampling Technique (**SMOTE**) or **class weighting** during modeling.
        """)

    with tab2:
        st.subheader("🔢 Numerical Features Analysis")
        st.markdown("Exploring the distribution and characteristics of continuous numerical features.")

        st.markdown("**Histograms & KDE Plots:** Show the frequency distribution.")
        fig_num_hist, axes_num_hist = plt.subplots(1, 3, figsize=(15, 4))
        for i, col in enumerate(numerical_features):
            sns.histplot(df_eda[col], kde=True, ax=axes_num_hist[i], bins=30)
            axes_num_hist[i].set_title(f'Distribution of {col}')
        plt.tight_layout()
        st.pyplot(fig_num_hist, use_container_width=True)
        plt.close(fig_num_hist)
        st.warning("""
        **Insights:**
        *   **Tenure:** Shows a bi-modal tendency - peaks at very low tenure (new customers) and very high tenure (long-term customers), with fewer customers in between.
        *   **MonthlyCharges:** Broadly distributed, with a peak towards lower charges and a long tail towards higher charges. Suggests different service packages/usage levels.
        *   **TotalCharges:** Strongly right-skewed, influenced by tenure and monthly charges. The spike at 0 corresponds to the zero-tenure customers we handled.
        """)

        st.markdown("**Box Plots:** Identify median, spread (IQR), and potential outliers.")
        fig_num_box, ax_num_box = plt.subplots(figsize=(10, 4))
        sns.boxplot(data=df_eda[numerical_features], ax=ax_num_box, palette='coolwarm')
        ax_num_box.set_title("Box Plots of Numerical Features")
        st.pyplot(fig_num_box, use_container_width=True)
        plt.close(fig_num_box)
        st.warning("""
        **Insights:** Box plots confirm the distributions seen in histograms. 'MonthlyCharges' and 'TotalCharges' don't show significant outliers based on the standard IQR definition, but their skewness is apparent.
        """)

    with tab3:
        st.subheader("🔠 Categorical Features Analysis")
        st.markdown("Examining the frequencies of different categories within each feature.")

        show_all_cats = st.checkbox("Show plots for all categorical features?", value=False)
        if show_all_cats:
            # Plotting logic for all categorical features (as before)
            num_cat_cols = len(categorical_features)
            rows = (num_cat_cols + 1) // 2
            fig_cat, axes_cat = plt.subplots(rows, 2, figsize=(14, 5 * rows))
            axes_cat = axes_cat.flatten()
            for i, col in enumerate(categorical_features):
                order = df_eda[col].value_counts().index
                sns.countplot(data=df_eda, y=col, order=order, ax=axes_cat[i], palette='viridis')
                axes_cat[i].set_title(f'Distribution of {col}')
                axes_cat[i].set_xlabel('Count')
                axes_cat[i].set_ylabel('')
            for j in range(i + 1, len(axes_cat)): fig_cat.delaxes(axes_cat[j])
            plt.tight_layout()
            st.pyplot(fig_cat, use_container_width=True)
            plt.close(fig_cat)
        else:
            # Select box logic (as before)
            selected_cat_feat = st.selectbox("Select a Categorical Feature to View:", options=categorical_features)
            if selected_cat_feat:
                 fig_single_cat, ax_single_cat = plt.subplots(figsize=(7, 5))
                 order = df_eda[selected_cat_feat].value_counts().index
                 sns.countplot(data=df_eda, y=selected_cat_feat, order=order, ax=ax_single_cat, palette='viridis')
                 ax_single_cat.set_title(f'Distribution of {selected_cat_feat}')
                 ax_single_cat.set_xlabel('Count')
                 ax_single_cat.set_ylabel('')
                 st.pyplot(fig_single_cat, use_container_width=True)
                 plt.close(fig_single_cat)

        st.markdown("""
        **Insights (General):**
        *   Many services (like `OnlineSecurity`, `TechSupport`) have a large proportion of 'No' or 'No internet service', indicating potential areas for upselling or different customer segments.
        *   Contract types are dominated by 'Month-to-month'.
        *   Payment methods show a mix, with 'Electronic check' being quite common.
        *(Select specific features above or check the box to see detailed counts for all.)*
        """)

    with tab4:
        st.subheader("🔄 Feature Relationships vs. Churn")
        st.markdown("How do different features correlate with the likelihood of a customer churning? This is key for identifying potential churn drivers.")

        st.markdown("**Numerical Features vs. Churn (Box Plots)**")
        # Plotting logic (as before)
        fig_num_churn, axes_num_churn = plt.subplots(1, 3, figsize=(15, 5))
        for i, col in enumerate(numerical_features):
            sns.boxplot(data=df_eda, x='Churn', y=col, ax=axes_num_churn[i], palette='vlag')
            axes_num_churn[i].set_title(f'{col} vs Churn')
        plt.tight_layout()
        st.pyplot(fig_num_churn, use_container_width=True)
        plt.close(fig_num_churn)
        st.markdown("""
        **Insights:**
        *   **Tenure:** Customers who churn tend to have significantly lower median tenure. Loyalty seems protective against churn.
        *   **MonthlyCharges:** Customers who churn tend to have higher median monthly charges. This could be due to expensive plans, or perhaps perceived lack of value for the cost.
        *   **TotalCharges:** Churning customers generally have lower total charges, likely linked to their shorter tenure.
        """)

        st.markdown("**Categorical Features vs. Churn (Proportion Plots)**")
        key_cats_for_churn = ['Contract', 'InternetService', 'OnlineSecurity', 'TechSupport', 'PaymentMethod', 'Dependents', 'Partner', 'PaperlessBilling']
        selected_cat_churn = st.selectbox("Select Categorical Feature to see Churn Relationship:", options=key_cats_for_churn, key="eda_cat_churn_select") # Unique key
        # Plotting logic (as before)
        if selected_cat_churn:
             fig_cat_churn, ax_cat_churn = plt.subplots(figsize=(8, 5))
             order = df_eda[selected_cat_churn].value_counts().index
             prop_df = (df_eda.groupby([selected_cat_churn])['Churn']
                       .value_counts(normalize=True)
                       .rename('proportion')
                       .reset_index())
             sns.barplot(data=prop_df, y=selected_cat_churn, x='proportion', hue='Churn', ax=ax_cat_churn, order=order, palette='coolwarm')
             ax_cat_churn.set_title(f'{selected_cat_churn} vs Churn Proportion')
             ax_cat_churn.set_xlabel('Proportion of Customers')
             ax_cat_churn.set_ylabel('')
             ax_cat_churn.legend(title='Churn')
             st.pyplot(fig_cat_churn, use_container_width=True)
             plt.close(fig_cat_churn)

        st.markdown("""
        **Insights (Examples - Select feature above):**
        *   **Contract:** Month-to-month contracts have a dramatically higher churn proportion compared to One or Two year contracts. Lack of commitment is a major risk factor.
        *   **Internet Service:** Fiber optic customers churn more often than DSL users. While faster, it might be more expensive or less reliable in some areas, leading to dissatisfaction. Customers with no internet service rarely churn (likely only have phone).
        *   **Online Security / Tech Support:** Lack of these support services correlates strongly with higher churn rates. Customers may feel less secure or unsupported.
        *   **Payment Method:** Electronic check users have a notably higher churn rate compared to automatic payment methods or mailed checks. This might indicate less financial stability or a less "sticky" relationship.
        """)

        st.markdown("**Correlation Heatmap (Numerical Features & Churn)**")
        st.markdown("Shows *linear* correlations. Values range from -1 (perfect negative correlation) to +1 (perfect positive correlation). 0 indicates no linear correlation.")
        # Plotting logic (as before)
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        corr_matrix = df_eda[numerical_features + ['Churn_numeric']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr, linewidths=.5)
        ax_corr.set_title('Numerical Feature Correlation with Churn')
        st.pyplot(fig_corr, use_container_width=True)
        plt.close(fig_corr)
        st.markdown("""
        **Insights:** Confirms visualisations:
        *   `tenure` has a moderate negative correlation with `Churn_numeric` (-0.35).
        *   `MonthlyCharges` has a positive correlation (+0.19).
        *   `TotalCharges` (influenced by tenure) has a negative correlation (-0.20).
        *   `tenure` and `TotalCharges` are highly positively correlated (+0.83), as expected.
        """)