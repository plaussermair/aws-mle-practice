# steps/s9_interpretation.py
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
# Note: Specific model libraries (xgboost, lightgbm) are not needed here
# as SHAP interacts with the fitted pipeline object directly.

def show_interpretation(best_individual_pipeline, best_individual_name, preprocessor, model_columns, X_test):
    st.header("🔍 9. Final Model Interpretation (SHAP)")

    # Retrieve final model name from session state if passed, otherwise use best individual
    # This assumes final_model_name was stored in artifacts dict passed to app.py
    final_model_name = st.session_state.get('artifacts',{}).get('final_model_name', best_individual_name) # Fallback
    st.success(f"**Final Selected Model:** The model ultimately chosen was **{final_model_name}**.")

    st.markdown(f"""
    To build trust and understand *why* the model makes its predictions, we interpret the best-performing *individual* model (**{best_individual_name}**) using **SHAP (SHapley Additive exPlanations)**. SHAP values explain the contribution of each feature to pushing a prediction away from the baseline (average prediction) towards 'Churn' (positive SHAP) or 'No Churn' (negative SHAP).
    """)

    if best_individual_pipeline is None or best_individual_name is None:
        st.error("Best individual model pipeline or name not loaded. Cannot perform interpretation.")
        return
    if model_columns is None:
        st.warning("Model columns not loaded. SHAP plots might lack feature names.")

    st.info(f"Interpreting the contributions of features for the **{best_individual_name}** model.")

    try:
        interpreter_classifier = best_individual_pipeline.named_steps['classifier']
        # Use the preprocessor associated with *this specific pipeline*
        interpreter_preprocessor = best_individual_pipeline.named_steps['preprocess']
    except KeyError as ke:
         st.error(f"Error accessing pipeline steps ('classifier' or 'preprocess' missing) for interpretation: {ke}")
         return
    except Exception as e:
        st.error(f"Error accessing pipeline steps for interpretation: {e}")
        return

    # --- Feature Importance ---
    st.subheader("Global Feature Importance")
    st.markdown("Which features have the largest impact *overall* across all predictions?")
    importance_plotted = False
    try:
        if hasattr(interpreter_classifier, 'feature_importances_') and model_columns is not None:
            importances = pd.Series(interpreter_classifier.feature_importances_, index=model_columns)
            fig_imp, ax_imp = plt.subplots(figsize=(10, 8))
            importances.nlargest(20).sort_values().plot(kind='barh', ax=ax_imp)
            ax_imp.set_title(f'Feature Importances ({best_individual_name}) - Top 20')
            st.pyplot(fig_imp, use_container_width=True)
            plt.close(fig_imp)
            importance_plotted = True
        elif hasattr(interpreter_classifier, 'coef_') and model_columns is not None:
            coeffs = pd.Series(interpreter_classifier.coef_[0], index=model_columns) # Assumes binary
            fig_coef, ax_coef = plt.subplots(figsize=(10, 8))
            coeffs.abs().nlargest(20).sort_values().plot(kind='barh', ax=ax_coef)
            ax_coef.set_title(f'Top 20 Feature Coefficients (Abs Value) - {best_individual_name}')
            st.pyplot(fig_coef, use_container_width=True)
            plt.close(fig_coef)
            importance_plotted = True
    except IndexError: # Handle cases where coef_ might have unexpected shape
         st.warning(f"Could not plot coefficients for {best_individual_name} due to unexpected shape.")
    except Exception as e:
         st.warning(f"Could not plot feature importance/coefficients: {e}")

    if not importance_plotted:
        st.markdown("*Feature importance/coefficients plot not available or columns missing.*")

    # --- SHAP Plots --- (With Button and Session State Caching)
    st.subheader("SHAP Summary Plots (Detailed Impact)")
    st.markdown("""
    These plots provide more granular insights than standard feature importance.
    *   **Beeswarm:** Shows the SHAP value for every feature for every sample. Reveals not just importance but also the *direction* of the effect and distribution. Color indicates feature value.
    *   **Bar:** Shows the average *magnitude* (mean absolute SHAP value) of each feature's impact.
    """)

    # Initialize SHAP state if not present
    if 'shap_values_calculated' not in st.session_state:
        st.session_state.shap_values_calculated = False
        st.session_state.shap_values_churn = None
        st.session_state.shap_data_for_plot = None
        st.session_state.shap_explainer = None

    # Button to trigger calculation
    if not st.session_state.shap_values_calculated:
        if st.button(f"Calculate SHAP Values for {best_individual_name} (can take time)", key="shap_calc_button"):
            with st.spinner("Calculating SHAP values... Please wait..."):
                try:
                    # Transform data using the *pipeline's* preprocessor
                    X_test_processed_shap = interpreter_preprocessor.transform(X_test)

                    # Create DataFrame with columns if possible
                    if model_columns is not None and X_test_processed_shap.shape[1] == len(model_columns):
                        X_test_processed_shap_df = pd.DataFrame(X_test_processed_shap, columns=model_columns)
                    else:
                        X_test_processed_shap_df = pd.DataFrame(X_test_processed_shap)
                        if model_columns: st.warning("Column name/count mismatch. Using default column names for SHAP.")

                    # Determine explainer type
                    # More robust check: isinstance might be better if model libs are available
                    # For now, rely on attribute check which is fragile
                    use_tree_explainer = hasattr(interpreter_classifier, 'feature_importances_') and not hasattr(interpreter_classifier, 'coef_')

                    if use_tree_explainer:
                        # Remove incompatible parameters and use defaults
                        explainer = shap.TreeExplainer(interpreter_classifier)
                        shap_values_obj = explainer(X_test_processed_shap_df)
                        
                        # Handle different output shapes
                        if hasattr(shap_values_obj, 'values'):
                            if shap_values_obj.values.ndim == 3:
                                # For probability outputs with shape (N, M, 2)
                                shap_values_churn = shap_values_obj.values[:,:,1]
                            else:
                                # For single output with shape (N, M)
                                shap_values_churn = shap_values_obj.values
                        else:
                            # Handle case where shap_values_obj is the values directly
                            shap_values_churn = shap_values_obj

                        data_for_plot = X_test_processed_shap_df
                    else: # KernelExplainer
                        st.info("Using KernelExplainer (slower). Subsampling data...")
                        # Important: KernelExplainer needs predict_proba function
                        if not hasattr(interpreter_classifier, 'predict_proba'):
                             raise AttributeError(f"{type(interpreter_classifier).__name__} model does not have predict_proba needed for KernelExplainer.")

                        # Sample background data (use a small subset of transformed test data as proxy)
                        # Ideally, use transformed training data summary if available
                        background_sample = shap.sample(X_test_processed_shap_df, min(100, X_test_processed_shap_df.shape[0]))

                        explainer = shap.KernelExplainer(interpreter_classifier.predict_proba, background_sample)

                        n_samples_shap = min(200, X_test_processed_shap_df.shape[0]) # Limit samples
                        X_test_sample_shap = shap.sample(X_test_processed_shap_df, n_samples_shap)

                        shap_values = explainer.shap_values(X_test_sample_shap) # Returns list [shap_class0, shap_class1]
                        shap_values_churn = shap_values[1] # SHAP values for P(Churn=1)
                        data_for_plot = X_test_sample_shap

                    # Store results in session state
                    st.session_state.shap_values_calculated = True
                    st.session_state.shap_values_churn = shap_values_churn
                    st.session_state.shap_data_for_plot = data_for_plot
                    st.session_state.shap_explainer = explainer # Store explainer too
                    st.rerun() # Rerun immediately to display plots

                except ImportError as ie: st.error(f"Missing library required for SHAP or model? {ie}")
                except AttributeError as ae: st.error(f"Attribute error during SHAP calculation (often model incompatibility): {ae}")
                except Exception as e_shap: st.error(f"Unexpected error during SHAP calculation: {e_shap}"); st.exception(e_shap)


    # Display plots if calculated
    if st.session_state.get('shap_values_calculated', False):
        if st.session_state.shap_values_churn is not None and st.session_state.shap_data_for_plot is not None:
            # Need to handle potential DataFrame/Numpy array type for data_for_plot
            plot_data = st.session_state.shap_data_for_plot
            if not isinstance(plot_data, pd.DataFrame) and model_columns and plot_data.shape[1] == len(model_columns):
                 plot_data = pd.DataFrame(plot_data, columns=model_columns)


            st.markdown("**SHAP Beeswarm Summary Plot:**")
            try:
                fig_shap_dot, ax_shap_dot = plt.subplots()
                # Ensure shap_values match dimensions of plot_data
                shap_vals_for_plot = st.session_state.shap_values_churn
                if shap_vals_for_plot.shape[0] != plot_data.shape[0]:
                     st.warning(f"SHAP values ({shap_vals_for_plot.shape[0]}) and data ({plot_data.shape[0]}) sample size mismatch. Cannot plot beeswarm.")
                else:
                    shap.summary_plot(shap_vals_for_plot, plot_data, plot_type="dot", show=False, max_display=20)
                    plt.title(f"SHAP Feature Impact on Churn Prediction ({best_individual_name})")
                    st.pyplot(fig_shap_dot, bbox_inches='tight', use_container_width=True)
                    plt.close(fig_shap_dot)
            except Exception as e: st.warning(f"Could not generate SHAP beeswarm plot: {e}")

            st.markdown("**SHAP Bar Summary Plot (Mean Absolute Impact):**")
            try:
                fig_shap_bar, ax_shap_bar = plt.subplots()
                shap.summary_plot(st.session_state.shap_values_churn, plot_data, plot_type="bar", show=False, max_display=20)
                plt.title(f"Average Feature Impact Magnitude ({best_individual_name})")
                st.pyplot(fig_shap_bar, bbox_inches='tight', use_container_width=True)
                plt.close(fig_shap_bar)
            except Exception as e: st.warning(f"Could not generate SHAP bar plot: {e}")
        else:
            st.warning("SHAP values seem calculated but are missing. Please try recalculating.")
            # Add a button to reset calculation state
            if st.button("Reset SHAP Calculation", key="shap_reset"):
                 st.session_state.shap_values_calculated = False
                 st.session_state.shap_values_churn = None
                 st.session_state.shap_data_for_plot = None
                 st.session_state.shap_explainer = None
                 st.rerun()