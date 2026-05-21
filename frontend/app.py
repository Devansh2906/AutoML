import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.model_engine import AutoMLBackend
from frontend.components.eda_dashboard import render_eda
from frontend.components.predict_panel import render_predictions

# 1. Global Page Configurations
st.set_page_config(
    page_title="Instant AutoML Platform", 
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Autonomous AutoML Web Platform")
st.write("Upload any raw dataset and watch the system automatically clean, profile, train, and explain the optimal model.")

# 2. Session State (Maintains app data memory across user interface refreshes)
if 'automl' not in st.session_state:
    st.session_state.automl = AutoMLBackend()
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'leaderboard' not in st.session_state:
    st.session_state.leaderboard = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None

# --- 3. Dataset Ingestion Sidebar ---
st.sidebar.header("📁 Data Source")
uploaded_file = st.sidebar.file_uploader("Drop your dataset here (.csv format)", type=["csv"])

if uploaded_file is not None:
    if st.session_state.df is None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.session_state.model_trained = False
        st.session_state.leaderboard = None
        st.session_state.problem_type = None
    
    df = st.session_state.df

    tabs = st.tabs(["📊 Data Overview & Insights", "⚙️ Model Training Engine", "🔮 Interactive Operational Predictor"])
    
    # ----------------------------------------------------
    # TAB 1: Exploratory Data Analysis & Raw Preview
    # ----------------------------------------------------
    with tabs[0]:
        st.subheader("📋 Raw Data Head Frame Preview")
        st.dataframe(df.head(5), use_container_width=True)
        st.write("---")
        
        render_eda(df)

    # ----------------------------------------------------
    # TAB 2: Machine Learning Optimization Architecture
    # ----------------------------------------------------
    with tabs[1]:
        st.subheader("⚙️ Pipeline Targeting Configurations")
        
        target_col = st.selectbox(
            "Select the specific target variable you want the AI to predict:", 
            options=df.columns,
            index=len(df.columns) - 1 
        )
        st.session_state.target_col = target_col
        
        st.write("---")
        
        if st.button("Launch Autonomous Training Engine", type="primary"):
            with st.spinner("Executing structural feature handling, addressing null matrices, and ranking performance metrics..."):
                try:
                    problem_type, leaderboard = st.session_state.automl.run_pipeline(df, target_col)
                    
                    
                    st.session_state.problem_type = problem_type
                    st.session_state.leaderboard = leaderboard
                    st.session_state.model_trained = True
                  
                    saved_path = st.session_state.automl.save_winning_model()
                    st.toast(f"Model serialized safely to: {saved_path}", icon="💾")
                    
                except Exception as error:
                    st.error(f"Execution Error encountered during preprocessing operations: {str(error)}")
        
        if st.session_state.model_trained:
            st.success(f"### Optimal Strategy Found: Task Classified as **{st.session_state.problem_type.upper()}**")
            
            st.write("### 📊 Model Optimization Leaderboard Matrix")
            
            available_columns = st.session_state.leaderboard.columns
            if st.session_state.problem_type == 'classification':
                target_metrics = [col for col in ['Model', 'Accuracy', 'AUC', 'F1', 'Recall', 'Precision'] if col in available_columns]
            else:
                # Syllabus specified: RMSE, alongside basic R2
                target_metrics = [col for col in ['Model', 'MAE', 'MSE', 'RMSE', 'R2'] if col in available_columns]
                
            st.dataframe(st.session_state.leaderboard[target_metrics], use_container_width=True)
            
            st.write("---")
            st.subheader("💡 Global Model Explainability (Feature Importance)")
            st.write("The matrix map below tracks the relative performance weight of features guiding the top model's conclusions:")
            
            try:
                fig = st.session_state.automl.exp.plot_model(
                    st.session_state.automl.best_model, 
                    plot='feature', 
                    display_format='streamlit'
                )
            except Exception as e:
                st.info("Feature coefficient optimization mapping is not natively supported by this specific champion model type.")

    # ----------------------------------------------------
    # TAB 3: Interactive Real-Time Prediction Sandbox
    # ----------------------------------------------------
    with tabs[2]:
        st.subheader("🔮 Live Workspace Prediction Sandbox")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ No active model weights found inside memory cache. Please navigate to the 'Model Training Engine' tab and launch an optimization pass first.")
        else:
            render_predictions(df, st.session_state.target_col, st.session_state.automl)

else:
    # Fallback splash layout landing page card shown if server instance launches empty
    st.info("👋 Welcome! Please upload a valid structural CSV dataset using the sidebar file portal to launch the AutoML workspace.")