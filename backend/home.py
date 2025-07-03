import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import os
import numpy as np

from agents.cfDNAClassifier_agent import cfDNAClassifierAgent
from agents.utils import load_npz, select_random_row, load_npz_as_df, is_chromosomal

# ------------------------------
# Configuration
# ------------------------------
def setup_page():
    st.set_page_config(page_title="Clinical Dashboard", layout="wide")
    st.title("Clinical Metrics Dashboard")

# ------------------------------
# File Upload Handler
# ------------------------------
def handle_file_upload():
    if 'clinical_data' not in st.session_state:
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            #df.columns = [col.lower() for col in df.columns]
            df.columns = [col for col in df.columns]
            st.session_state['clinical_data'] = df
            st.session_state['filename'] = uploaded_file.name
            st.success(f"Uploaded: {uploaded_file.name}")
            st.rerun()
        else:
            st.stop()

# ------------------------------
# Validate Required Columns
# ------------------------------
def validate_columns(df, required_columns):
    #df.columns = [col.lower() for col in df.columns]
    df.columns = [col for col in df.columns]
    #st.session_state['data'] = df
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.error(f"The following required columns are missing in your file: {', '.join(missing)}")
        st.stop()
    return df

# ------------------------------
# Select Row to View
# ------------------------------
def select_data_row(df):
    if len(df) > 1:
        index = st.selectbox("Select row index to view", df.index.tolist())
    else:
        index = df.index[0]
    return df.loc[index]

# ------------------------------
# Display Demographics
# ------------------------------
def show_demographics(row):
    with st.container():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("#### Demographics")
            st.markdown(f"**Age:** {row['recipient_age']}")
            st.markdown(f"**Sex:** {row['sex']}")
        with col2:
            st.empty()

# ------------------------------
# Display Clinical Metrics
# ------------------------------
def show_clinical_metrics(row, metrics):
    st.markdown("#### Clinical Lab Metrics")
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics):
        value = row[metric]
        label = metric.upper()
        formatted_value = f"<span style='font-size:32px'>{value} <span style='font-size:12px'>U/L</span></span>"
        col.markdown(f"**{label}**<br>{formatted_value}", unsafe_allow_html=True)

# ------------------------------
# Display Charts
# ------------------------------
def show_pie_chart(probabilities_row: pd.Series, label_map: dict):
    """
    Displays a pie chart of class probabilities for a single prediction row.

    Args:
        probabilities_row (pd.Series): A row with class probabilities (e.g., 0.03, 0.1, 0.87).
        label_map (dict): A dict mapping class index to readable names.
                          E.g., {0: "CTRL-LT", 1: "MASH-LT", 2: "TCMR"}
    """
    # Extract class probabilities
    class_probs = probabilities_row[[k for k in label_map.keys()]]

    # Prepare labels and values
    labels = [label_map[int(col)] for col in class_probs.index]
    values = class_probs.values

    # Define colors for each label
    color_map = {
        "CTRL-LT": "#EEB348",
        "MASH-LT": "#BB372F",
        "TCMR": "#3D7EC1"
    }

    # Create pie chart with custom colors
    fig = px.pie(
        names=labels,
        values=values,
        title="Predicted Outcome",
        hole=0.3,
        color=labels,
        color_discrete_map=color_map
    )
    st.plotly_chart(fig, use_container_width=True)

def temp(agent, clinical_df, methyl_df):
    shap_results = agent.run_shap_analysis(clinical_df, methyl_df)
    return shap_results

import matplotlib.pyplot as plt
import numpy as np

def show_shap_chart(shap_array, feature_names):
    """
    Create and display a SHAP diagram in Streamlit,
    showing all clinical features plus the top 2 genomic features.

    Parameters:
    - shap_array: 1D numpy array of SHAP values for the predicted class
    - feature_names: pandas Index, Series, or list matching shap_array
    """
    if not isinstance(feature_names, list):
        feature_names = feature_names.tolist()

    clinical_idxs = [i for i, name in enumerate(feature_names) if not is_chromosomal(name)]
    genomic_idxs  = [i for i, name in enumerate(feature_names) if is_chromosomal(name)]

    top_genomic = sorted(genomic_idxs, key=lambda i: abs(shap_array[i]), reverse=True)[:2]
    selected_idxs = clinical_idxs + top_genomic

    selected_names  = [feature_names[i] for i in selected_idxs]
    selected_values = [shap_array[i] for i in selected_idxs]

    sort_order     = np.argsort(selected_values)[::-1]
    sorted_names   = [selected_names[i] for i in sort_order]
    sorted_values  = [selected_values[i] for i in sort_order]
    colors         = ['#FF0051' if val > 0 else '#008BFB' for val in sorted_values]

    fig, ax = plt.subplots(figsize=(8, len(sorted_names) * 0.6))
    ax.barh(sorted_names, sorted_values, color=colors)
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_title("Feature Contributions to Prediction")
    ax.invert_yaxis()
    fig.tight_layout()

    st.pyplot(fig)



# ------------------------------
# Main App Logic
# ------------------------------
def main():
    # Directories
    base_dir = os.path.dirname(os.path.abspath(__file__))   # current script directory
    #base_dir = os.path.dirname(current_dir)  
    #base_dir = os.path.join(current_dir)

    data_dir = os.path.join(base_dir, "data")
    results_dir = os.path.join(base_dir, "results")
    models_dir = os.path.join(base_dir, "models")
    staging_dir = os.path.join(base_dir, "data", "staging")
    figures_dir = os.path.join(base_dir, "results", "figures")
    features_dir = os.path.join(base_dir, "results", "selected_features")
    
    # Start
    setup_page()
    handle_file_upload()


    if 'clinical_data' in st.session_state:
        df = st.session_state['clinical_data']
        
        # Select Demographic data
        demographic_fields = ["recipient_age", "sex"]
        metric_fields = ["Hgb", "ALP", "ALT", "AST", "Creatinine"]
        all_required = demographic_fields + metric_fields

        # Display data
        df = validate_columns(df, all_required)
        st.session_state['clinical_data'] = df  # Save validated in session state
        row = select_data_row(df)
        show_demographics(row)
        show_clinical_metrics(row, metric_fields)

        # Get preprocessed Methylation data
        npz_path = "data/staging/methylation_processed.npz"

        # Load data
        data = load_npz_as_df(npz_path)

        # Load selected features
        selected_features_path = os.path.join(features_dir, f"boruta_selected_features_fold1.txt")
        if os.path.exists(selected_features_path):
            with open(selected_features_path, "r") as f:
                selected_features = [line.strip() for line in f.readlines()]
                st.session_state['selected_features'] = selected_features # Save in session state
        else:
            raise FileNotFoundError(f"Selected features file not found for fold 1: {selected_features_path}")

        # Filter methylation data based on selected features
        methylation_data = select_random_row(data)
        # Filter features
        methylation_data = methylation_data[selected_features]
        st.session_state['methylation_data'] = methylation_data  # Save in session state

        # Instantiate the agent
        agent = cfDNAClassifierAgent(model_name="combined", model_type="rf", fold=1)

        # Run prediction
        with st.spinner("Running classification..."):
            # Run the model and return the prediction result
            result = agent.run(st.session_state['clinical_data'], st.session_state['methylation_data'])

        label_map = {
            0: "CTRL-LT",
            1: "MASH-LT",
            2: "TCMR"
        }

        show_pie_chart(result.iloc[0], label_map)

        
        # Run SHAP analysis
        shap_results = agent.run_shap_analysis(st.session_state['clinical_data'], st.session_state['methylation_data'])
        predicted_class = result["predicted_class"].iloc[0] # Get predicted class
        
        # Use the only available output # Get SHAP values for the predicted class
        shap_for_predicted_class = shap_results[0, :, predicted_class]
        
        feature_names = agent.get_feature_names()
        show_shap_chart(shap_for_predicted_class, feature_names)

        #st.dataframe(show_shap_chart(agent, st.session_state['clinical_data'], st.session_state['methylation_data']))

# Run the app
if __name__ == "__main__":
    
    main()
