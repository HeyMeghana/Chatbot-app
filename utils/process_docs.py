import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

VECTOR_STORE_PATH = "models/vector_store.pkl"

def process_document(file_path):
    """Process uploaded document and store vectorized data."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([content])
    
    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "vectors": vectors}, f)

def load_vector_store():
    """Load stored vectorized document data."""
    if os.path.exists(VECTOR_STORE_PATH):
        with open(VECTOR_STORE_PATH, "rb") as f:
            return pickle.load(f)
    return None
