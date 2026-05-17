import streamlit as st
import os
import pandas as pd
import joblib
import numpy as np

# Set professional web page configurations
st.set_page_config(
    page_title="Enterprise Product Recommender Dashboard", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Reviews.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "hybrid_recommender.pkl")

def generate_fallback_simulation_data():
    """Generates structural simulation records if raw source data is blocked by gitignore configurations."""
    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        print("[!] Raw dataset missing on cloud server container. Compiling safe mock simulation vectors...")
        
        np.random.seed(42)
        mock_users = [f"USER_{i:04d}" for i in range(1, 120)]
        mock_prods = [f"PROD_{i:04d}" for i in range(1, 60)]
        summaries = [
            "Excellent value performance", "Crisp sound clear highs", 
            "Durable build long cable", "Works out of the box", "Highly recommended product"
        ]
        
        simulated_records = []
        for _ in range(1500):
            simulated_records.append({
                "UserId": np.random.choice(mock_users),
                "ProductId": np.random.choice(mock_prods),
                "Score": int(np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.05, 0.1, 0.3, 0.5])),
                "Summary": np.random.choice(summaries)
            })
        pd.DataFrame(simulated_records).to_csv(RAW_DATA_PATH, index=False)

@st.cache_resource
def bootstrap_analytics_engine():
    """Loads the pre-trained model binary or builds an operational runtime pipeline inline."""
    generate_fallback_simulation_data()
    
    if os.path.exists(MODEL_PATH):
        engine = joblib.load(MODEL_PATH)
        cleaned_df = engine.cleaned_data
    else:
        from src.engine import EcommerceRecommenderEngine
        engine = EcommerceRecommenderEngine(alpha=0.6, n_latent_features=15)
        cleaned_df = engine.preprocess_data(filepath=RAW_DATA_PATH, sample_size=20000)
        engine.fit()
    return engine, cleaned_df

# UI Branding Headers
st.title("🛒 E-Commerce Personalized Recommendation Engine")
st.markdown("##### **Enterprise Architecture Portfolio** | Component: Multimodal Hybrid Inference Pipeline (SVD + NLP)")
st.markdown("---")

try:
    with st.spinner("Initializing calculations, factorizing matrices, and processing text corpora..."):
        engine, cleaned_df = bootstrap_analytics_engine()
    st.sidebar.success("Core Algorithmic Engine: Active")

    # Sidebar Controls
    st.sidebar.header("System Hyperparameters")
    alpha_slider = st.sidebar.slider(
        "Model Balance Weight (Alpha)", 
        min_value=0.0, max_value=1.0, value=0.6, step=0.1,
        help="Higher values favor user behavior history (SVD). Lower values favor product text description matches (NLP)."
    )
    engine.alpha = alpha_slider 

    # Extract unique users for profile lookup
    available_users = sorted(cleaned_df['UserId'].unique()[:50])

    # App View Layout
    col1, col2 = st.columns([2, 3], gap="large")

    with col1:
        st.markdown("### 👤 User Profile Selection")
        selected_user = st.selectbox("Select an enterprise User ID node for live evaluation:", available_users)
        
        user_history = cleaned_df[cleaned_df['UserId'] == selected_user][['ProductId', 'Score', 'Summary']].rename(
            columns={'ProductId': 'Product ID', 'Score': 'Explicit Rating', 'Summary': 'Textual Review'}
        )
        st.markdown("#### Historical Interactions Ledger")
        st.dataframe(user_history, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### 🎯 Live Model Prediction Matrix")
        recommendation_count = st.slider("Target recommendation array length (Top N):", min_value=3, max_value=10, value=5)
        
        if st.button("Compute Hybrid Recommendation Inference", type="primary", use_container_width=True):
            with st.spinner("Processing coordinate tensor dot products..."):
                predictions = engine.recommend(user_id=selected_user, top_n=recommendation_count)
            
            st.markdown("#### Optimized Recommendations for Selection Context:")
            for rank, prod_id in enumerate(predictions, 1):
                with st.container(border=True):
                    c_left, c_right = st.columns([1, 4])
                    c_left.metric(label="Rank", value=f"#{rank}")
                    c_right.markdown(f"##### Product Identifier Link: `{prod_id}`")
                    sample_desc = cleaned_df[cleaned_df['ProductId'] == prod_id]['Summary'].iloc[0]
                    c_right.markdown(f"*Semantic Category/Keywords:* **{sample_desc}**")

except Exception as fatal_err:
    st.error("Application Process Thread Aborted")
    st.info("Check backend terminal traces to debug system graph failures.")
    st.code(str(fatal_err))