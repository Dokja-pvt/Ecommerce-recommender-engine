import streamlit as st
import os
import pandas as pd
import joblib  # Added for binary loading

st.set_page_config(
    page_title="Enterprise Product Recommender Dashboard", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Reviews.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "hybrid_recommender.pkl")

@st.cache_resource
def bootstrap_analytics_engine():
    """Loads the pre-trained model binary directly from disk to simulate production performance."""
    if os.path.exists(MODEL_PATH):
        print("[+] Loading pre-trained model binary artifact...")
        engine = joblib.load(MODEL_PATH)
        cleaned_df = engine.cleaned_data
    else:
        print("[!] Saved model not found. Falling back to real-time training sequence...")
        from src.engine import EcommerceRecommenderEngine
        engine = EcommerceRecommenderEngine(alpha=0.6, n_latent_features=15)
        cleaned_df = engine.preprocess_data(filepath=RAW_DATA_PATH, sample_size=25000)
        engine.fit()
    return engine, cleaned_df