# steps/s0_introduction.py
import streamlit as st

def show_introduction():
    st.header("🌟 Project Introduction")
    st.markdown("""
    Welcome to this interactive showcase demonstrating the end-to-end process of building a **Customer Churn Prediction Model**.
    This project aims to predict whether a customer of a fictional telecommunications company is likely to churn (stop using the services).

    **Why Predict Churn?**
    Acquiring new customers is often significantly more expensive than retaining existing ones. By identifying customers at high risk of churning, the company can implement targeted retention strategies (e.g., special offers, improved support) to reduce revenue loss and improve customer satisfaction.

    **The Dataset:**
    We are using the popular **Telco Customer Churn** dataset.
    *   **Source:** Originally provided by IBM, widely available on platforms like Kaggle.
    *   **Content:** Contains customer demographics, account information (tenure, contract type), services subscribed to (phone, internet, streaming, etc.), charges, and the target variable 'Churn' (Yes/No).
    *   **Link:** You can find the dataset and more details [here on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data).

    **Our Journey:**
    This application guides you through the key stages of the machine learning workflow implemented for this project. Use the 'Next Step' and 'Previous Step' buttons below to navigate. The sidebar indicates your current position in the workflow.
    """)
    st.info("Click 'Next Step' to begin!")