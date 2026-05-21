import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def render_eda(df: pd.DataFrame):
    
    st.subheader("📊 Exploratory Data Analysis & Quality Insights")
    
    # Metadata Overview KPI metrics row
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Row Records", df.shape[0])
    m_col2.metric("Feature Columns", df.shape[1])
    m_col3.metric("Total Missing Records", df.isna().sum().sum())
    
    st.write("---")
    
    # Left col table data, right col for linear corr matrix
    layout_col1, layout_col2 = st.columns([1, 1])
    
    with layout_col1:
        st.write("### Missing Data Value Counts")
        missing_df = df.isna().sum().reset_index()
        missing_df.columns = ['Column Name', 'Missing Cells']
        st.dataframe(missing_df.sort_values(by='Missing Cells', ascending=False), use_container_width=True)
        
    with layout_col2:
        st.write("### Numerical Correlation Heatmap")
        num_cols = df.select_dtypes(include=['number'])
        
        if num_cols.shape[1] > 1:
            fig, ax = plt.subplots(figsize=(6, 4))

            sns.heatmap(num_cols.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax, annot_kws={"size": 8})
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(fontsize=8)
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Not enough numerical attributes present inside this dataset file to generate a cross-correlation matrix visual map.")