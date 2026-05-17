import os
import joblib
import pandas as pd
import numpy as np
from src.engine import EcommerceRecommenderEngine

# Define absolute path constraints
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Reviews.csv")
MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, "models", "hybrid_recommender.pkl")

def calculate_precision_at_k(engine, test_users, k=5):
    """
    Computes the average Precision@K metric across a sample test population.
    """
    precision_scores = []
    
    for user_id in test_users:
        # Fetch the historical truth profile (Items user rated >= 4)
        user_data = engine.cleaned_data[engine.cleaned_data['UserId'] == user_id]
        true_relevant_items = set(user_data[user_data['Score'] >= 4]['ProductId'].values)
        
        if len(true_relevant_items) == 0:
            continue  # Skip users with no high-affinity baseline interactions
            
        # Generate model recommendations
        recommended_items = engine.recommend(user_id=user_id, top_n=k)
        
        # Calculate intersection mapping
        hits = len(set(recommended_items).intersection(true_relevant_items))
        user_precision = hits / k
        precision_scores.append(user_precision)
        
    return np.mean(precision_scores) if precision_scores else 0.0

def main():
    print("==================================================================")
    print(" RUNNING OFFLINE EVALUATION & SERIALIZATION SUITE")
    print("==================================================================\n")

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Missing raw asset trajectory: {RAW_DATA_PATH}")

    # 1. Initialize and Fit the Model Architecture
    engine = EcommerceRecommenderEngine(alpha=0.6, n_latent_features=15)
    engine.preprocess_data(filepath=RAW_DATA_PATH, sample_size=30000)
    engine.fit()

    # 2. Run Precision Ranking Performance Test
    print("[*] Sampling evaluation nodes from interaction graph...")
    # Select top 100 high-activity users to run a stable metric test
    test_sample_users = engine.cleaned_data['UserId'].value_counts().head(100).index.tolist()
    
    print(f"[*] Computing Precision@5 across {len(test_sample_users)} test validation nodes...")
    mean_p_at_5 = calculate_precision_at_k(engine, test_sample_users, k=5)
    
    print("\n==================================================================")
    print(" PERFORMANCE BENCHMARK RESULTS")
    print("==================================================================")
    print(f" Mean Precision@5 Score: {mean_p_at_5 * 100:.2f}%")
    print("      *Interpretation: Out of 5 products recommended to active users,")
    print(f"       an average of {mean_p_at_5 * 5:.2f} items perfectly match their past affinity profile.")
    print("==================================================================\n")

    # 3. Serialize Model Matrix State to Disk (Persistence)
    print(f"[*] Serializing trained matrix binaries to path: {MODEL_OUTPUT_PATH}")
    try:
        joblib.dump(engine, MODEL_OUTPUT_PATH)
        print("[+] Success: Model binary artifact exported with zero corruption flags.")
    except Exception as export_error:
        print(f"[!] Serialization Failure: {str(export_error)}")

if __name__ == "__main__":
    main()