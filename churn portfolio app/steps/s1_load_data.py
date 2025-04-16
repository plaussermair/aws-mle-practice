# steps/s1_load_data.py
import streamlit as st
import pandas as pd
import numpy as np
from helpers import display_dataframe_info_pretty # Import helper

def show_load_data(df_original):
    st.header("💾 1. Load Data & Initial Inspection")
    st.markdown("""
    The journey begins by loading the raw data using Pandas. We then perform essential initial checks to understand its basic structure and identify immediate issues.
    """)

    with st.expander("Show Raw Data Preview (First 5 Rows)", expanded=False):
        st.dataframe(df_original.head())

    st.subheader("Dataset Dimensions")
    st.write(f"The dataset has **{df_original.shape[0]} rows** (each representing a customer) and **{df_original.shape[1]} columns** (features).")

    st.subheader("Column Information (Types & Non-Null Counts)")
    st.markdown("This table provides a concise overview of each column, its data type (`Dtype`), and the number of non-missing values. It helps identify potential type issues or missing data early on.")
    display_dataframe_info_pretty(df_original.copy()) # Use the imported helper

    st.subheader("Numerical Feature Summary")
    st.markdown("Basic statistics (count, mean, standard deviation, min/max, quartiles) for numerical columns give a sense of their distribution and scale.")
    st.dataframe(df_original.describe(include=np.number), use_container_width=True)

    st.subheader("Initial Missing Value Check & 'TotalCharges' Handling")
    st.markdown("""
    A direct check for missing values using `isnull().sum()` might be deceptive.
    """)
    missing_initial = df_original.isnull().sum()
    st.write("Initial `isnull().sum()` counts:")
    st.dataframe(missing_initial[missing_initial > 0])

    st.markdown("""
    As suspected, 'TotalCharges' is initially read as an 'object' (string) type. This often happens when numerical columns contain non-numeric characters (like spaces). We convert it to numeric, forcing errors into `NaN` (Not a Number).
    """)
    df_processed = df_original.copy()
    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    missing_after_conversion = df_processed.isnull().sum()
    st.markdown("**Result after converting 'TotalCharges' to numeric:**")
    st.dataframe(missing_after_conversion[missing_after_conversion > 0])
    st.success("""
    **Insight:** The conversion reveals missing 'TotalCharges' values. These correspond exactly to customers with `tenure = 0`. For modeling, these were imputed (filled) - the specific strategy (e.g., filling with 0) is discussed in the 'Feature Engineering & Processing' step. For EDA, we might fill them temporarily or exclude them depending on the analysis.
    """)