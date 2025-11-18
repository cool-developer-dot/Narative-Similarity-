"""
Baseline Model A: TF-IDF + Logistic Regression
Traditional approach using feature extraction and classical ML
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle
from typing import List, Tuple
import pandas as pd


class TFIDFBaseline:
    """TF-IDF + Logistic Regression baseline"""
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95
        )
        self.classifier = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        
    def create_features(self, anchor: List[str], text_a: List[str], 
                       text_b: List[str]) -> np.ndarray:
        """
        Create feature vectors by computing similarity between:
        - anchor and text_a
        - anchor and text_b
        """
        # Combine all texts for fitting vocabulary
        all_texts = anchor + text_a + text_b
        
        # Fit or transform
        if not hasattr(self.vectorizer, 'vocabulary_'):
            self.vectorizer.fit(all_texts)
        
        # Get TF-IDF vectors
        anchor_vecs = self.vectorizer.transform(anchor).toarray()
        text_a_vecs = self.vectorizer.transform(text_a).toarray()
        text_b_vecs = self.vectorizer.transform(text_b).toarray()
        
        # Compute similarity features
        features = []
        for i in range(len(anchor)):
            # Cosine similarity features
            sim_a = self._cosine_similarity(anchor_vecs[i], text_a_vecs[i])
            sim_b = self._cosine_similarity(anchor_vecs[i], text_b_vecs[i])
            
            # Difference and ratio
            sim_diff = sim_a - sim_b
            sim_ratio = sim_a / (sim_b + 1e-10)
            
            # Length features
            len_anchor = len(anchor[i].split())
            len_a = len(text_a[i].split())
            len_b = len(text_b[i].split())
            len_diff_a = abs(len_anchor - len_a)
            len_diff_b = abs(len_anchor - len_b)
            
            # Combine features
            feature_vec = [sim_a, sim_b, sim_diff, sim_ratio, 
                          len_anchor, len_a, len_b, len_diff_a, len_diff_b]
            features.append(feature_vec)
        
        return np.array(features)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def train(self, train_df: pd.DataFrame):
        """Train the model"""
        print("Training TF-IDF Baseline...")
        
        features = self.create_features(
            train_df['anchor_text'].tolist(),
            train_df['text_a'].tolist(),
            train_df['text_b'].tolist()
        )
        
        labels = train_df['text_a_is_closer'].astype(int).values
        
        self.classifier.fit(features, labels)
        
        # Training accuracy
        train_preds = self.classifier.predict(features)
        train_acc = accuracy_score(labels, train_preds)
        print(f"Training Accuracy: {train_acc:.4f}")
        
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Predict on test data"""
        features = self.create_features(
            test_df['anchor_text'].tolist(),
            test_df['text_a'].tolist(),
            test_df['text_b'].tolist()
        )
        
        return self.classifier.predict(features)
    
    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        features = self.create_features(
            test_df['anchor_text'].tolist(),
            test_df['text_a'].tolist(),
            test_df['text_b'].tolist()
        )
        
        return self.classifier.predict_proba(features)
    
    def evaluate(self, test_df: pd.DataFrame) -> dict:
        """Evaluate model on test data"""
        predictions = self.predict(test_df)
        labels = test_df['text_a_is_closer'].astype(int).values
        
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions),
            'precision': precision_score(labels, predictions),
            'recall': recall_score(labels, predictions)
        }
        
        return metrics
    
    def save(self, path: str):
        """Save model"""
        with open(path, 'wb') as f:
            pickle.dump({'vectorizer': self.vectorizer, 
                        'classifier': self.classifier}, f)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.classifier = data['classifier']
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    from data_loader import SemEvalDataLoader, split_dataset
    
    # Load data
    loader = SemEvalDataLoader()
    df = loader.load_track_a()
    train_df, val_df, test_df = split_dataset(df)
    
    # Train baseline
    model = TFIDFBaseline()
    model.train(train_df)
    
    # Evaluate
    val_metrics = model.evaluate(val_df)
    print("\nValidation Metrics:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    test_metrics = model.evaluate(test_df)
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model
    model.save("models/baseline_tfidf.pkl")

