import os
import pandas as pd
from src.engine import EcommerceRecommenderEngine

# Define absolute workspace route matrices
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Reviews.csv")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

def verify_and_prep_workspace():
    """Ensures directories exist and validates the structural integrity of the raw data asset."""
    for folder in [PROCESSED_DATA_DIR, MODEL_DIR]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"[+] Operational folder initialized: {folder}")
            
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Missing Critical Data Asset: Drop 'Reviews.csv' into {os.path.dirname(RAW_DATA_PATH)}")

def main():
    print("==================================================================")
    print(" STARTING PIPELINE: HYBRID ECOMMERCE RECOMMENDER SYSTEM")
    print("==================================================================\n")
    
    # Initialize workspace checkpoints
    verify_and_prep_workspace()
    
    # 1. Instantiate Core Engine
    # Assigning alpha=0.6 (60% Collaborative Filtering, 40% NLP Content-Based text matching)
    engine = EcommerceRecommenderEngine(alpha=0.6, n_latent_features=15)
    
    # 2. Ingest and Clean Sparse Interaction Records
    # Using a subset slice of 30,000 interactions for local execution efficiency
    cleaned_df = engine.preprocess_data(filepath=RAW_DATA_PATH, sample_size=30000)
    
    # Checkpoint cleaned data layer to disk
    checkpoint_path = os.path.join(PROCESSED_DATA_DIR, "cleaned_reviews.csv")
    cleaned_df.to_csv(checkpoint_path, index=False)
    print(f"[+] Saved structured checkpoint data to: {checkpoint_path}")
    
    # 3. Train Model Vectors
    engine.fit()
    
    # 4. Run Pipeline Verification Lookups
    try:
        # Pull a highly active user from the matrix to ensure a deep interaction trail
        sample_user = cleaned_df['UserId'].value_counts().index[0]
        print(f"\n[*] Evaluating profile history and computing matrix tracks for User: {sample_user}")
        
        # Display past purchases
        user_history = cleaned_df[cleaned_df['UserId'] == sample_user]['ProductId'].unique()[:3]
        print(f"[*] Confirmed User Purchase History (Samples): {user_history}")
        
        # Compute recommendation array
        suggestions = engine.recommend(user_id=sample_user, top_n=5)
        
        print("\n==================================================================")
        print(f" SUCCESS: VERIFIED TOP-5 HYBRID RECOMMENDATIONS FOR SYSTEM LOGS")
        print("==================================================================")
        for rank, item_id in enumerate(suggestions, 1):
            # Fetch contextual summary text to prove NLP cross-referencing works
            matching_summary = cleaned_df[cleaned_df['ProductId'] == item_id]['Summary'].iloc[0]
            print(f" Rank #{rank} | Product ID: {item_id} | Context: {matching_summary}")
        print("==================================================================\n")
        
    except Exception as pipeline_error:
        print(f"[!] Pipeline Execution Failure: {str(pipeline_error)}")

if __name__ == "__main__":
    main()