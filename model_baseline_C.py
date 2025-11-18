"""
Baseline Model C: Sentence-BERT (Bi-encoder)
Uses pre-trained sentence transformers for semantic similarity
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
from typing import List


class SentenceBERTBaseline:
    """Sentence-BERT baseline using bi-encoder architecture"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = None):
        """
        Initialize Sentence-BERT model
        
        Args:
            model_name: Pre-trained model name
                - 'all-MiniLM-L6-v2': Fast and efficient (default)
                - 'all-mpnet-base-v2': Better quality, slower
                - 'multi-qa-mpnet-base-dot-v1': Good for QA tasks
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"Loaded Sentence-BERT: {model_name} on {self.device}")
        
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings
    
    def predict(self, test_df: pd.DataFrame, batch_size: int = 32) -> np.ndarray:
        """
        Predict which text is closer to anchor
        
        Strategy: Compute cosine similarity between:
        - anchor and text_a
        - anchor and text_b
        Predict text_a is closer if sim(anchor, text_a) > sim(anchor, text_b)
        """
        # Encode all texts
        anchors = test_df['anchor_text'].tolist()
        text_a_list = test_df['text_a'].tolist()
        text_b_list = test_df['text_b'].tolist()
        
        print("Encoding anchors...")
        anchor_embeddings = self.encode_texts(anchors, batch_size)
        
        print("Encoding text_a...")
        text_a_embeddings = self.encode_texts(text_a_list, batch_size)
        
        print("Encoding text_b...")
        text_b_embeddings = self.encode_texts(text_b_list, batch_size)
        
        # Compute cosine similarities
        similarities_a = util.cos_sim(anchor_embeddings, text_a_embeddings).diagonal().cpu().numpy()
        similarities_b = util.cos_sim(anchor_embeddings, text_b_embeddings).diagonal().cpu().numpy()
        
        # Predict: 1 if text_a is closer, 0 otherwise
        predictions = (similarities_a > similarities_b).astype(int)
        
        return predictions
    
    def predict_with_scores(self, test_df: pd.DataFrame, batch_size: int = 32) -> tuple:
        """
        Predict and return similarity scores
        
        Returns:
            predictions, similarities_a, similarities_b
        """
        anchors = test_df['anchor_text'].tolist()
        text_a_list = test_df['text_a'].tolist()
        text_b_list = test_df['text_b'].tolist()
        
        anchor_embeddings = self.encode_texts(anchors, batch_size)
        text_a_embeddings = self.encode_texts(text_a_list, batch_size)
        text_b_embeddings = self.encode_texts(text_b_list, batch_size)
        
        similarities_a = util.cos_sim(anchor_embeddings, text_a_embeddings).diagonal().cpu().numpy()
        similarities_b = util.cos_sim(anchor_embeddings, text_b_embeddings).diagonal().cpu().numpy()
        
        predictions = (similarities_a > similarities_b).astype(int)
        
        return predictions, similarities_a, similarities_b
    
    def evaluate(self, test_df: pd.DataFrame, batch_size: int = 32) -> dict:
        """Evaluate model on test data"""
        print("Evaluating Sentence-BERT...")
        predictions = self.predict(test_df, batch_size)
        labels = test_df['text_a_is_closer'].astype(int).values
        
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions),
            'precision': precision_score(labels, predictions),
            'recall': recall_score(labels, predictions)
        }
        
        return metrics
    
    def fine_tune(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                  epochs: int = 3, batch_size: int = 16):
        """
        Fine-tune Sentence-BERT on task-specific data
        Uses contrastive loss with triplets
        """
        from sentence_transformers import InputExample, losses
        from torch.utils.data import DataLoader
        
        print("Fine-tuning Sentence-BERT...")
        
        # Create training examples
        train_examples = []
        for _, row in train_df.iterrows():
            anchor = row['anchor_text']
            text_a = row['text_a']
            text_b = row['text_b']
            
            if row['text_a_is_closer']:
                # text_a is positive, text_b is negative
                train_examples.append(InputExample(
                    texts=[anchor, text_a, text_b]
                ))
            else:
                # text_b is positive, text_a is negative
                train_examples.append(InputExample(
                    texts=[anchor, text_b, text_a]
                ))
        
        train_dataloader = DataLoader(train_examples, batch_size=batch_size, shuffle=True)
        
        # Use triplet loss
        train_loss = losses.TripletLoss(self.model)
        
        # Train
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=int(len(train_dataloader) * 0.1),
            show_progress_bar=True
        )
        
        print("Fine-tuning complete!")
        
        # Evaluate on validation
        val_metrics = self.evaluate(val_df, batch_size)
        print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Validation F1: {val_metrics['f1']:.4f}")
        
        return val_metrics
    
    def save(self, path: str):
        """Save model"""
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model"""
        self.model = SentenceTransformer(path, device=self.device)
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    from data_loader import SemEvalDataLoader, split_dataset
    
    # Load data
    loader = SemEvalDataLoader()
    df = loader.load_track_a()
    train_df, val_df, test_df = split_dataset(df)
    
    # Test with pre-trained model (no fine-tuning)
    print("\n=== Pre-trained Sentence-BERT (no fine-tuning) ===")
    model = SentenceBERTBaseline(model_name='all-MiniLM-L6-v2')
    
    val_metrics = model.evaluate(val_df)
    print("\nValidation Metrics (Pre-trained):")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    test_metrics = model.evaluate(test_df)
    print("\nTest Metrics (Pre-trained):")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Fine-tune on training data
    print("\n=== Fine-tuning Sentence-BERT ===")
    model_ft = SentenceBERTBaseline(model_name='all-MiniLM-L6-v2')
    model_ft.fine_tune(train_df, val_df, epochs=3, batch_size=16)
    
    test_metrics_ft = model_ft.evaluate(test_df)
    print("\nTest Metrics (Fine-tuned):")
    for metric, value in test_metrics_ft.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save fine-tuned model
    model_ft.save("models/baseline_sentence_bert")

