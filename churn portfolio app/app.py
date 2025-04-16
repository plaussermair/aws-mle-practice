# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import sys # To potentially add steps directory to path if needed, though direct import often works

# --- Configuration ---
st.set_page_config(
    page_title="Customer Churn Prediction Showcase",
    page_icon="🚀",
    layout="wide"
)

# --- Constants ---
# Assume artifacts are in the same directory as app.py for simplicity
# If you move artifacts to a subfolder (e.g., "artifacts/"), change ARTIFACTS_DIR
ARTIFACTS_DIR = "."
DATA_FILE = f"{ARTIFACTS_DIR}/WA_Fn-UseC_-Telco-Customer-Churn.csv"
PREPROCESSOR_FILE = f"{ARTIFACTS_DIR}/preprocessor.joblib"
MODEL_COLUMNS_FILE = f"{ARTIFACTS_DIR}/model_columns.txt" # Or .joblib
BEST_INDIVIDUAL_PIPELINE_FILE = f"{ARTIFACTS_DIR}/best_individual_pipeline.joblib"
BEST_INDIVIDUAL_NAME_FILE = f"{ARTIFACTS_DIR}/best_individual_name.joblib"
VOTING_PIPELINE_FILE = f"{ARTIFACTS_DIR}/voting_classifier_pipeline.joblib"
X_TEST_FILE = f"{ARTIFACTS_DIR}/X_test.joblib"
Y_TEST_FILE = f"{ARTIFACTS_DIR}/y_test.joblib"
EVAL_RESULTS_FILE = f"{ARTIFACTS_DIR}/evaluation_results.joblib"
FINAL_PIPELINE_FILE = f"{ARTIFACTS_DIR}/final_pipeline.joblib"
FINAL_MODEL_NAME_FILE = f"{ARTIFACTS_DIR}/final_model_name.joblib"

# --- Caching Functions ---
@st.cache_resource(show_spinner="Loading ML model/preprocessor...")
def load_joblib_resource(file_path):
    """Loads resources like models, pipelines using resource caching."""
    try: return joblib.load(file_path)
    except FileNotFoundError: st.error(f"Error: File not found: {file_path}"); return None
    except Exception as e: st.error(f"Error loading {file_path}: {e}"); return None

@st.cache_data(show_spinner="Loading data...")
def load_csv_data(file_path):
    """Loads CSV data using data caching."""
    try: return pd.read_csv(file_path)
    except FileNotFoundError: st.error(f"Error: Data file not found: {file_path}"); return None
    except Exception as e: st.error(f"Error loading {file_path}: {e}"); return None

@st.cache_data
def load_text_list_data(file_path):
    """Loads text list data using data caching."""
    try:
        with open(file_path, 'r') as f: return [line.strip() for line in f if line.strip()]
    except FileNotFoundError: st.warning(f"Warning: File not found: {file_path}"); return None
    except Exception as e: st.error(f"Error loading {file_path}: {e}"); return None

@st.cache_data(show_spinner="Loading test data...")
def load_joblib_data(file_path):
    """Loads data like X_test, y_test using data caching."""
    try: return joblib.load(file_path)
    except FileNotFoundError: st.error(f"Error: File not found: {file_path}"); return None
    except Exception as e: st.error(f"Error loading {file_path}: {e}"); return None

# --- Load Artifacts ---
# Use a dictionary stored in session state to avoid reloading across reruns within a session
# Only load initially or if not present
if 'artifacts' not in st.session_state:
    print("Loading artifacts into session state...") # For debugging
    st.session_state.artifacts = {}
    artifacts_temp = {} # Load into temp dict first

    # Load essential files
    artifacts_temp['df_original'] = load_csv_data(DATA_FILE)
    artifacts_temp['preprocessor'] = load_joblib_resource(PREPROCESSOR_FILE)
    artifacts_temp['X_test'] = load_joblib_data(X_TEST_FILE) # Cache as data
    artifacts_temp['y_test'] = load_joblib_data(Y_TEST_FILE) # Cache as data
    artifacts_temp['evaluation_results'] = load_joblib_data(EVAL_RESULTS_FILE) # Eval results likely data
    artifacts_temp['final_pipeline'] = load_joblib_resource(FINAL_PIPELINE_FILE)
    artifacts_temp['final_model_name'] = load_joblib_resource(FINAL_MODEL_NAME_FILE) # Name is small, resource ok
    artifacts_temp['best_individual_pipeline'] = load_joblib_resource(BEST_INDIVIDUAL_PIPELINE_FILE)
    artifacts_temp['best_individual_name'] = load_joblib_resource(BEST_INDIVIDUAL_NAME_FILE)

    # Load optional files
    artifacts_temp['model_columns'] = load_text_list_data(MODEL_COLUMNS_FILE)
    artifacts_temp['voting_pipeline'] = load_joblib_resource(VOTING_PIPELINE_FILE) # Can be None

    st.session_state.artifacts = artifacts_temp # Assign to session state
else:
    print("Artifacts already in session state.") # For debugging


# Check if essential artifacts were loaded successfully *within the session state dict*
essential_keys = [
    'df_original', 'preprocessor', 'X_test', 'y_test', 'evaluation_results',
    'final_pipeline', 'final_model_name', 'best_individual_pipeline', 'best_individual_name'
]
essential_loaded = all(st.session_state.artifacts.get(k) is not None for k in essential_keys)

if not essential_loaded:
    st.error("One or more essential artifact files could not be loaded. Stopping execution.")
    # Display which ones failed
    missing = [k for k in essential_keys if st.session_state.artifacts.get(k) is None]
    st.error(f"Missing artifacts: {missing}")
    st.stop() # Stop the script if critical files are missing


# --- Import Step Functions ---
# Ensure the 'steps' directory is in the Python path or use relative imports
try:
    from steps.s0_introduction import show_introduction
    from steps.s1_load_data import show_load_data
    from steps.s2_eda import show_exploratory_data_analysis
    from steps.s3_feature_engineering import show_feature_engineering
    from steps.s4_data_splitting import show_data_splitting
    from steps.s5_preprocessing import show_preprocessing
    from steps.s6_model_selection import show_model_selection
    from steps.s7_model_evaluation import show_model_evaluation
    from steps.s8_ensemble_model import show_ensemble_model
    from steps.s9_interpretation import show_interpretation
    from steps.s10_conclusion import show_conclusion
except ImportError as e:
    st.error(f"Could not import step functions. Ensure 'steps' directory and '__init__.py' exist.")
    st.error(f"Import Error: {e}")
    st.stop()

# --- Define Steps ---
# Pass only the necessary artifacts from session state to each step function
# This avoids passing the large 'artifacts' dict repeatedly
steps = [
    ("Introduction", show_introduction, {}),
    ("Load Data", show_load_data, {'df_original': st.session_state.artifacts['df_original']}),
    ("EDA", show_exploratory_data_analysis, {'df_original': st.session_state.artifacts['df_original']}),
    ("Feature Engineering", show_feature_engineering, {'df_original': st.session_state.artifacts.get('df_original')}), # Pass df for context
    ("Data Splitting", show_data_splitting, {'X_test': st.session_state.artifacts['X_test'], 'y_test': st.session_state.artifacts['y_test']}),
    ("Preprocessing", show_preprocessing, {'preprocessor': st.session_state.artifacts['preprocessor'], 'model_columns': st.session_state.artifacts.get('model_columns'), 'X_test': st.session_state.artifacts['X_test']}),
    ("Model Selection", show_model_selection, {'evaluation_results': st.session_state.artifacts['evaluation_results'], 'best_individual_name': st.session_state.artifacts['best_individual_name'], 'best_individual_pipeline': st.session_state.artifacts['best_individual_pipeline']}),
    ("Model Evaluation", show_model_evaluation, {'evaluation_results': st.session_state.artifacts['evaluation_results'], 'X_test': st.session_state.artifacts['X_test'], 'y_test': st.session_state.artifacts['y_test']}),
    ("Ensemble Model", show_ensemble_model, {'voting_pipeline': st.session_state.artifacts.get('voting_pipeline'), 'X_test': st.session_state.artifacts['X_test'], 'y_test': st.session_state.artifacts['y_test'], 'evaluation_results': st.session_state.artifacts['evaluation_results'], 'best_individual_name': st.session_state.artifacts['best_individual_name']}),
    ("Interpretation", show_interpretation, {'best_individual_pipeline': st.session_state.artifacts['best_individual_pipeline'], 'best_individual_name': st.session_state.artifacts['best_individual_name'], 'preprocessor': st.session_state.artifacts['preprocessor'], 'model_columns': st.session_state.artifacts.get('model_columns'), 'X_test': st.session_state.artifacts['X_test']}),
    ("Conclusion", show_conclusion, {}), # Conclusion pulls from session state internally now
]
total_steps = len(steps)

# --- Initialize Step Counter ---
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

# --- Page Title ---
st.title("🚀 Telco Customer Churn Prediction")
st.subheader("An Interactive ML Workflow Showcase")

# --- Sidebar: Progress Indicator ---
st.sidebar.title("Workflow Progress")
st.sidebar.markdown("Navigate using the buttons below the main content.")

for i, (step_name, _, _) in enumerate(steps):
    is_current = (i == st.session_state.current_step)
    st.sidebar.markdown(f"{'➡️' if is_current else ''} **{i+1}. {step_name}** {'⬅️' if is_current else ''}")

# Add progress bar at the bottom of the sidebar section
st.sidebar.progress((st.session_state.current_step + 1) / total_steps)
st.sidebar.markdown("---") # Separator
# Add portfolio link or name in sidebar
st.sidebar.info("Created by: [Your Name/Portfolio Link]")


# --- Navigation Buttons (Displayed Below Title) ---
st.markdown("---") # Separator before buttons
col1, col2, col3 = st.columns([1, 5, 1]) # Adjust width ratios as needed

with col1:
    if st.session_state.current_step > 0:
        if st.button("⬅️ Previous", use_container_width=True):
            st.session_state.current_step -= 1
            # Reset SHAP calculation flag if moving away from Interpretation step?
            # if steps[st.session_state.current_step + 1][0] == "Interpretation":
            #      st.session_state.shap_values_calculated = False # Reset if leaving SHAP page
            st.rerun()

with col3:
    if st.session_state.current_step < total_steps - 1:
        if st.button("Next ➡️", use_container_width=True):
            st.session_state.current_step += 1
            st.rerun()

st.markdown("---") # Separator after buttons

# --- Display Current Step Content ---
step_name, step_function, step_args = steps[st.session_state.current_step]

# Inject artifacts dictionary into session state if needed by a step (like conclusion)
# This is slightly less clean than passing args but works if a step needs many artifacts
# st.session_state.current_artifacts = st.session_state.artifacts # If needed

# Call the function for the current step
try:
    step_function(**step_args)
except TypeError as te:
     st.error(f"Error calling step function '{step_name}': Argument mismatch.")
     st.error(f"TypeError: {te}")
     st.error(f"Expected arguments based on definition: {step_function.__code__.co_varnames[:step_function.__code__.co_argcount]}")
     st.error(f"Provided arguments via step_args: {list(step_args.keys())}")
except Exception as e:
     st.error(f"An unexpected error occurred in step '{step_name}':")
     st.exception(e)

# --- Auto-scroll to top (Optional, can sometimes be jarring) ---
st.markdown(
    """
    <script>
        // Find the main content section and scroll to top
        const main = window.parent.document.querySelector('.main > div');
        if (main) {
            main.scrollTo(0, 0);
        }
    </script>
    """,
    unsafe_allow_html=True,
)