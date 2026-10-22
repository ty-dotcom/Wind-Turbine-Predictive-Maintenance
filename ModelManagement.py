from pathlib import Path
import pandas as pd
import streamlit as st
from Model import Classifier

MAIN_DATA_PATH = Path(__file__).parent / "turbine_5yr_labeled_data.csv"
SAMPLE_DATA_PATH = Path(__file__).parent / "sample_data.csv"
DATA_PATH = MAIN_DATA_PATH if MAIN_DATA_PATH.exists() else SAMPLE_DATA_PATH

@st.cache_data
def load_data():
    """Loads data from CSV/Excel file, dropping the timestamp column."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"File {DATA_PATH} not found.")

    df = pd.read_csv(DATA_PATH)
    df = df.drop("timestamp", axis=1)
    return df


@st.cache_resource
def load_model(data):
    """Train both models and selects the best model via the f1 score and cached for the session."""
    return Classifier(data)
