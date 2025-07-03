import streamlit as st
import pandas as pd

st.title("Model Interpretation")

# Access saved DataFrame from session
if 'data' in st.session_state:
    df = st.session_state['data']
    
    st.success("Data loaded from session state")
    st.write("Here is your data preview:")
    st.dataframe(df.head())

else:
    st.warning("No data found. Please upload a CSV on the first page.")
