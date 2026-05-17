# 🔗 Enterprise E-Commerce Product Recommendation Engine

An end-to-end Machine Learning Engineering (MLE) system designed to optimize personalized discovery pipelines and increase customer lifetime value (LTV). This repository implements a decoupled, production-grade hybrid pipeline that validates raw transactional schemas, runs high-density matrix transformations, discovers latent behavioral signals via matrix factorization, and extracts semantic textual similarities with natural language processing (NLP).

---

## 👔 Business Case & Dataset Source

Modern e-commerce platforms struggle with catalog discoverability due to sparse user-item interaction histories and severe cold-start constraints for newly introduced inventory items. This system blends multi-user behavioral trends with granular descriptive metadata to build an optimized hybrid affinity matrix, allowing for highly targeted personalization and zero runtime execution delay for edge web clients.

- **Dataset Source:** [Amazon Product Reviews (Kaggle)](https://www.kaggle.com/datasets/jillanisofttech/amazon-product-reviews)
- **Core Files:** `Reviews.csv` (contains user identity records, explicit numeric product ratings, and short textual review summaries).

---

## 🛠️ Repository Blueprint

```text
Ecommerce_recommender/
├── data/
│   ├── raw/                  # Source CSV binaries from Kaggle (git-ignored)
│   └── processed/            # Intermediary clean data layer and pipeline checkpoints
├── models/                   # Serialized model states and pre-trained binaries (.pkl)
├── src/
│   ├── __init__.py           # Package initialization and public API exposures
│   └── engine.py             # Core hybrid matrix factorization and NLP engine module
├── app.py                    # Streamlit interactive web deployment application
├── evaluate.py               # Offline precision ranking and validation suite
├── main.py                   # Automated sequential end-to-end training orchestrator
└── requirements.txt          # Production package dependency manager

```

---

## 🛡️ Modular Pipeline Breakdown

- **Data Validation Gate (`src/engine.py` -> `preprocess_data`)** — Filters raw transaction data, checks columns, and drops duplicate strings while removing interaction sparsity to optimize matrix density.
- **Algorithmic Core Engine (`src/engine.py` -> `fit`)** — Executes parallel background calculation sequences across behavioral matrices and semantic text arrays.
- **Pipeline Orchestration (`main.py`)** — Handles automated workspace verification, manages path routing, saves clean checkpoint states, and triggers baseline diagnostics.
- **Offline Model Evaluation (`evaluate.py`)** — Runs precision profiling using top-N ranking metrics over sample test groups, exporting the final model state to disk using `joblib`.
- **Interactive Interface (`app.py`)** — Connects the trained model state with a polished browser frontend, utilizing resource caching to achieve instantaneous recommendation updates.

---

## 🧮 Hybrid Score Blending Mechanics

Raw model probabilities are mapped onto an optimized numerical sequence using a balanced linear score formulation:

- $\text{Score}_{\text{collaborative}} = \text{Dot product of SVD compressed latent matrices}$
- $\text{Score}_{\text{content}} = \text{Cosine similarity mapping of text TF-IDF arrays}$

$$\text{Score}_{\text{hybrid}} = \alpha \cdot \text{Score}_{\text{collaborative}} + (1 - \alpha) \cdot \text{Score}_{\text{content}}$$

### Hyperparameter Tiers & Routing:

- **Active Behavioral Mode ($\alpha \ge 0.6$):** Weights user historical trends and multi-consumer interaction correlation profiles heavily.
- **Semantic Content Mode ($\alpha \le 0.4$):** Prioritizes specific product review summary text, isolating keyword overlap frequencies.
- **Cold-Start Router Fallback:** Automatically activates if a user query does not exist in the coordinate indexes. It bypasses matrix calculations and outputs global item popularity scores to ensure uptime.

---

## 💻 Software Engineering Standards

- **Decoupled System Architecture:** Decouples heavy offline model compilation tasks (`evaluate.py`) from real-time downstream client serving paths (`app.py`) via serialization layers.
- **Object Memory Caching:** Implements Streamlit's `@st.cache_resource` configuration along with `joblib` binary loading mechanics to eliminate compute redundancy and keep dashboard latency at 0ms.
- **Sparsity Mitigation Limits:** Enforces a density pass rule that retains only high-activity items and users with $\ge 2$ entries, removing extreme matrix noise.

---

## ⚡ Setup & Execution Guide

### 1. Workspace Initialization

**Install required production dependencies**

```bash
pip install -r requirements.txt

```

_Note: Download the source data from the Kaggle link above and drop your unzipped CSV files into `data/raw/` as `Reviews.csv`._

### 2. Run the Processing and Training Pipeline

```bash
python main.py
python evaluate.py

```

### 3. Launch the Serving UI

```bash
streamlit run app.py

```

---

## 🤝 Acknowledgements

- **Amazon Web Services / Kaggle:** For making large-scale, anonymized e-commerce transaction matrices accessible for baseline testing.
- **Open Source Contributors:** Special thanks to the engineering maintainers of `scikit-learn`, `SciPy`, `Pandas`, and `Streamlit` for providing the underlying tools for this architecture.
