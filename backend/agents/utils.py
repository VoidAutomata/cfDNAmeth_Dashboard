import re
import os
import random
import numpy as np
import pandas as pd
import streamlit as st

def flatten_dict_values(input_dict):
        flat_dict = {}
        for k, v in input_dict.items():
            if isinstance(v, dict):
                flat_dict[k] = str(v)  # Or extract a key: v.get("label")
            else:
                flat_dict[k] = v
        return flat_dict

# Check if a column name is chromosomal (e.g., "chr1:100-200")
def is_chromosomal(col_name):
    return bool(re.match(r"^chr(\d+|[XY]):\d+-\d+$", col_name))

def select_random_methylation_row(methylation_df: pd.DataFrame, selected_features: list) -> pd.DataFrame:
    """
    Selects a random row from methylation data (filtered to selected features)
    and returns it as a single-row DataFrame.
    """
    # Filter by selected features
    filtered_df = methylation_df[selected_features]

    # Pick a random index
    random_idx = np.random.choice(filtered_df.index)

    # Return single-row DataFrame (keep as DataFrame, not Series)
    return filtered_df.loc[[random_idx]]

# Select a random sample from the dataset (numpy)
def select_random_sample(X, ids):
    # Return a random row and its ID.
    idx = random.randint(0, X.shape[0] - 1)
    return X[idx], ids[idx], idx

# Select a random row from a DataFrame
def select_random_row(df):
    return df.sample(n=1, random_state=None)


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

def load_npz_as_df(path):
    # Load .npz file and return contents as a DataFrame
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        st.stop()

    data = np.load(path, allow_pickle=True)
    X = data["X"]
    feature_names = data["feature_names"]
    sample_ids = data["sample_ids"]
    y = data["y"]

    # Build DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df["sample_ids"] = sample_ids
    df["label"] = y

    return df
