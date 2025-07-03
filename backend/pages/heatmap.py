import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import random
from collections import defaultdict

st.session_state.update(st.session_state)

# ---------------------------
# Function Definitions
# ---------------------------

#@st.cache_resource
def load_npz(path):
    # Load and return contents of the .npz file.
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        st.stop()

    data = np.load(path, allow_pickle=True)
    return {
        "X": data["X"],
        "feature_names": data["feature_names"],
        "ids": data["sample_ids"],
        "y": data["y"]
    }

def load_selected_features(path):
    # Load feature names from a text file (one per line).
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def df_to_np():
    if 'methylation_data' not in st.session_state:
        st.error("No methylation data found in session.")
    else:
        df = st.session_state['methylation_data']

        # Optional: remove non-feature columns like 'sample_id' or 'label' if they exist
        feature_columns = [col for col in df.columns if 'chr' in col]
        return df[feature_columns].to_numpy()


def filter_features(row, feature_names, selected_features):
    # Return a filtered row and feature names based on selected features.
    mask = [name in selected_features for name in feature_names]
    filtered_row = row[mask]
    filtered_names = feature_names[mask]
    return filtered_row, filtered_names

def load_selected_features_by_chr(path):
    # Return a dict: { 'chr3': [feature1, feature2, ...], ... }
    chr_features = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            name = line.strip()
            if not name or ':' not in name:
                continue
            chrom = name.split(':')[0]
            chr_features[chrom].append(name)
    return chr_features

def index_feature_map(feature_names):
    # Return a dict: { feature_name: index }
    return {name: i for i, name in enumerate(feature_names)}

# Select a random sample from the dataset
def select_random_sample(X, ids):
    # Return a random row and its ID.
    idx = random.randint(0, X.shape[0] - 1)
    return X[idx], ids[idx], idx

def chrom_sort_key(chrom):
    # Return a sort key for chromosome names like chr1, chrX, etc.
    if chrom == 'chrX':
        return 1000  # Large number to ensure chrX comes last
    elif chrom.startswith('chr') and chrom[3:].isdigit():
        return int(chrom[3:])
    else:
        return 9999  # For unknown or non-standard chromosomes


def group_features_by_chromosome(feature_names):
    """
    Groups feature names by their chromosome prefix (e.g., chr3).
    Returns a dict: { 'chr3': [...], 'chrX': [...] }
    """
    chr_groups = {}
    for f in feature_names:
        match = re.match(r"^(chr(?:\d+|X)):", f)
        if match:
            chrom = match.group(1)
            chr_groups.setdefault(chrom, []).append(f)
    return chr_groups

def plot_heatmap():
    """
    Plots small heatmaps grouped by chromosome in a vertical layout.
    Uses st.session_state['methylation_data'] which should have 1 row.
    """
    if 'methylation_data' not in st.session_state:
        st.error("No methylation data found in session.")
        return

    methylation_df = st.session_state['methylation_data']
    if methylation_df.empty:
        st.warning("Methylation data is empty.")
        return

    row = methylation_df.iloc[0]
    feature_names = methylation_df.columns.tolist()
    chr_groups = group_features_by_chromosome(feature_names)

    sorted_chroms = sorted(chr_groups.keys(), key=chrom_sort_key)

    for chrom in sorted_chroms:
        features = chr_groups[chrom]

        # Extract and clean values
        values = pd.to_numeric(row[features], errors="coerce").astype(float).values
        if np.isnan(values).all():
            st.warning(f"No usable values in {chrom}. Skipping.")
            continue

        # Plot a narrow heatmap per chromosome
        fig, ax = plt.subplots(figsize=(max(6, len(features) * 0.25), 0.4))
        sns.heatmap([values], cmap="PuBu", xticklabels=features,
                    yticklabels=False, ax=ax, cbar=True)
        ax.set_title(f"{chrom} ({len(features)} features)")
        ax.tick_params(axis='x', labelsize=8, rotation=90)
        st.pyplot(fig)

        plt.close(fig) # Avoid memory leak

# ---------------------------
# Streamlit App Entry Point
# ---------------------------

def main():
    st.set_page_config(page_title="Methylation Heatmap", layout="wide")
    st.title("Random Methylation Sample Heatmap")

    # Get preprocessed
    #npz_path = "data/staging/methylation_processed.npz"

    # Load data
    methylation_data = st.session_state['methylation_data']
    #row, sample_id, idx = select_random_sample(methylation_data["X"], methylation_data["ids"])

    # Load and group selected features
    selected_chr_groups = load_selected_features_by_chr("results/selected_features/boruta_selected_features_fold1.txt")
    #selected_chr_groups = st.session_state['selected_features']

    # Display
    #st.markdown(f"### Sample ID: `{sample_id}` (Index {idx})")
    st.markdown(f"### Methylation Heatmap")
    #methylation_np = df_to_np()
    #plot_heatmap(methylation_np, methylation_np["feature_names"], selected_chr_groups)
    plot_heatmap()


# Only run if this is the main script
if __name__ == "__main__":
    main()
