# helpers.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import re
from sklearn.metrics import roc_curve

# Ignore warnings (optional) - Place here if helpers generate warnings
import warnings
warnings.filterwarnings('ignore')

def display_dataframe_info_pretty(df):
    """Captures df.info() and displays it as a clean Streamlit table."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    lines = info_str.split('\n')
    data = []
    pattern = re.compile(r"^\s*(\d+)\s+(.+?)\s+(\d+)\s+non-null\s+(.+)$")
    pattern_no_index = re.compile(r"^\s*(.+?)\s+(\d+)\s+non-null\s+(.+)$")

    for line in lines:
        match = pattern.match(line)
        if match:
            Column = match.group(2).strip()
            Non_Null_Count = int(match.group(3))
            Dtype = match.group(4).strip()
            data.append({"Column": Column, "Non-Null Count": Non_Null_Count, "Dtype": Dtype})
        else:
            match_no_idx = pattern_no_index.match(line)
            if match_no_idx:
                Column = match_no_idx.group(1).strip()
                Non_Null_Count = int(match_no_idx.group(2))
                Dtype = match_no_idx.group(3).strip()
                data.append({"Column": Column, "Non-Null Count": Non_Null_Count, "Dtype": Dtype})

    if data:
        info_df = pd.DataFrame(data)
        st.dataframe(info_df, use_container_width=True)
    else:
        st.text("Could not parse df.info() output into a table. Raw output:")
        st.text(info_str)

    mem_usage_line = [line for line in lines if "memory usage" in line]
    if mem_usage_line:
        st.caption(mem_usage_line[0].strip())


def plot_confusion_matrix(cm, labels, title, cmap):
    """Generates and displays a confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=labels, yticklabels=labels, ax=ax, annot_kws={"size": 12})
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig) # Close the figure to free memory


def plot_roc_curve_comparison(results, X_test, y_test):
    """Plots ROC curves for multiple models."""
    fig, ax = plt.subplots(figsize=(9, 7))
    plotted_roc = False
    for name, result in results.items():
        pipeline = result.get('pipeline')
        roc_auc = result.get('roc_auc')

        if pipeline and roc_auc is not None and not np.isnan(roc_auc) and hasattr(pipeline, "predict_proba"):
            try:
                y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})', linewidth=2)
                plotted_roc = True
            except Exception as e:
                st.warning(f"Could not plot ROC for {name}: {e}")

    if plotted_roc:
        ax.plot([0, 1], [0, 1], 'k--', label='Random Guessing', linewidth=2)
        ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
        ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
        ax.set_title('ROC Curve Comparison on Test Set', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig, use_container_width=True)
    else:
        st.warning("No valid ROC AUC scores available to plot ROC curves.")
    plt.close(fig)