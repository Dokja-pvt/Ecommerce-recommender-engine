import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

class EcommerceRecommenderEngine:
    def __init__(self, alpha=0.6, n_latent_features=15):
        """
        Initializes the Hybrid Recommender Engine.
        :param alpha: Balancing weight for Collaborative Filtering (0.0 to 1.0).
        :param n_latent_features: Target dimensions for SVD matrix decomposition.
        """
        self.alpha = alpha
        self.n_latent_features = n_latent_features
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        
        # Inversion and mapping lookups
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.prod_to_idx = {}
        self.idx_to_prod = {}
        
        # Matrix and vector allocations
        self.user_features = None
        self.item_features = None
        self.content_similarity_matrix = None
        self.interaction_matrix = None
        self.cleaned_data = None

    def preprocess_data(self, filepath, sample_size=30000):
        """
        Loads dataset layers and filters out extreme matrix sparsity.
        """
        print("[+] Ingesting data transactions from source...")
        df = pd.read_csv(filepath, usecols=['UserId', 'ProductId', 'Score', 'Summary'])
        df = df.dropna().drop_duplicates().head(sample_size)
        
        # Density Filtering: Retain active interaction nodes
        user_counts = df['UserId'].value_counts()
        prod_counts = df['ProductId'].value_counts()
        df = df[df['UserId'].isin(user_counts[user_counts >= 2].index)]
        df = df[df['ProductId'].isin(prod_counts[prod_counts >= 2].index)]
        
        self.cleaned_data = df.reset_index(drop=True)
        return self.cleaned_data

    def fit(self):
        """
        Executes concurrent algebraic operations to fit behavior and NLP models.
        """
        if self.cleaned_data is None:
            raise ValueError("Data pipeline uninitialized. Execute preprocess_data before model fitting.")
            
        print("[+] Compiling coordinate index maps...")
        unique_users = self.cleaned_data['UserId'].unique()
        unique_prods = self.cleaned_data['ProductId'].unique()
        
        self.user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        self.idx_to_user = {i: uid for i, uid in enumerate(unique_users)}
        self.prod_to_idx = {pid: i for i, pid in enumerate(unique_prods)}
        self.idx_to_prod = {i: pid for i, pid in enumerate(unique_prods)}
        
        self.cleaned_data['user_idx'] = self.cleaned_data['UserId'].map(self.user_to_idx)
        self.cleaned_data['prod_idx'] = self.cleaned_data['ProductId'].map(self.prod_to_idx)
        
        # Generate Compressed Sparse Row Matrix
        self.interaction_matrix = csr_matrix(
            (self.cleaned_data['Score'], (self.cleaned_data['user_idx'], self.cleaned_data['prod_idx'])),
            shape=(len(unique_users), len(unique_prods)), dtype=np.float64
        )
        
        # Channel 1: Collaborative Filtering Matrix Factorization
        print("[+] Executing Truncated SVD Matrix Factorization...")
        n_components = min(self.n_latent_features, len(unique_prods) - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_features = svd.fit_transform(self.interaction_matrix)
        self.item_features = svd.components_.T 
        
        # Channel 2: Text Feature Extraction Pipeline
        print("[+] Running TF-IDF Text Vectorization on Metadata...")
        prod_text = self.cleaned_data.groupby('prod_idx')['Summary'].apply(lambda x: " ".join(x)).reset_index()
        prod_text = prod_text.sort_values('prod_idx')
        
        tfidf_matrix = self.vectorizer.fit_transform(prod_text['Summary'])
        self.content_similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        print("[+] Model processing state achieved.")

    def recommend(self, user_id, top_n=5):
        """
        Applies linear score blending to generate optimized personalized items.
        """
        # Handling Cold-Start Route
        if user_id not in self.user_to_idx:
            print(f"[!] Target user '{user_id}' missing in historical index. Routing popularity baseline.")
            return self.cleaned_data['ProductId'].value_counts().head(top_n).index.tolist()
            
        user_idx = self.user_to_idx[user_id]
        
        # Extract and scale Collaborative Filtering probabilities
        user_vector = self.user_features[user_idx, :].reshape(1, -1)
        cf_scores = np.dot(user_vector, self.item_features.T).flatten()
        if cf_scores.max() != cf_scores.min():
            cf_scores = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min())
            
        # Extract and scale NLP content similarities based on highly rated items (Score >= 4)
        user_interactions = self.cleaned_data[self.cleaned_data['user_idx'] == user_idx]
        high_rated_interactions = user_interactions[user_interactions['Score'] >= 4]
        
        if not high_rated_interactions.empty:
            liked_indices = high_rated_interactions['prod_idx'].values
            content_scores = self.content_similarity_matrix[liked_indices].mean(axis=0)
        else:
            all_indices = user_interactions['prod_idx'].values
            content_scores = self.content_similarity_matrix[all_indices].mean(axis=0) if len(all_indices) > 0 else np.zeros(len(self.prod_to_idx))

        # Enforce Weighted Hybrid Blending Optimization
        hybrid_scores = (self.alpha * cf_scores) + ((1 - self.alpha) * content_scores)
        
        # Filter items already purchased by the user
        interacted_items = set(user_interactions['prod_idx'].values)
        sorted_indices = np.argsort(hybrid_scores)[::-1]
        
        recommendations = []
        for idx in sorted_indices:
            if idx not in interacted_items:
                recommendations.append(self.idx_to_prod[idx])
            if len(recommendations) == top_n:
                break
                
        return recommendations