import streamlit as st
import pandas as pd

def render_predictions(df: pd.DataFrame, target: str, automl_backend):
    st.write("Modify the parameter fields below to manually test how changing real-world attributes alters the model's predictions:")
    
    user_inputs = {}
    
    # Create two columns for form layout inputs
    col1, col2 = st.columns(2)
    idx = 0
    
    for col in df.columns:
        if col == target:
            continue 
        
        target_placement = col1 if idx % 2 == 0 else col2
        
        with target_placement:
            if df[col].dtype == 'object' or df[col].dtype == 'bool':
                options = df[col].dropna().unique()
                user_inputs[col] = st.selectbox(f"Attribute: {col}", options)
            else:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                mean_val = float(df[col].mean())
                
                if min_val == max_val:
                    max_val += 1.0
                    
                user_inputs[col] = st.slider(f"Attribute: {col}", min_val, max_val, mean_val)
        idx += 1

    st.write("---")
    
    if st.button("Compute Real-Time Inference Result", type="primary"):
        input_row_df = pd.DataFrame([user_inputs])
        
        prediction_output = automl_backend.exp.predict_model(automl_backend.best_model, data=input_row_df)
        
        # Response value
        predicted_label = prediction_output['prediction_label'][0]
        
        st.success("### Prediction Computation Successful!")
        
        # Classification confidence matrix 
        if 'prediction_score' in prediction_output.columns:
            confidence_score = prediction_output['prediction_score'][0]
            st.metric(label=f"Predicted Output for '{target}'", value=str(predicted_label), delta=f"Confidence Profile: {confidence_score:.2%}")
        else:
            st.metric(label=f"Predicted Target Vector Value", value=f"{predicted_label:,.4f}")