"""
Proposed Model: Hybrid Cross-Encoder with Contrastive Learning
Advanced solution combining multiple techniques for improved performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd


class CrossEncoderDataset(Dataset):
    """Dataset for Cross-Encoder with triplet format"""
    
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Tokenize (anchor, text_a) pair
        pair_a = self.tokenizer(
            row['anchor_text'],
            row['text_a'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize (anchor, text_b) pair
        pair_b = self.tokenizer(
            row['anchor_text'],
            row['text_b'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        label = 1 if row['text_a_is_closer'] else 0
        
        return {
            'pair_a_input_ids': pair_a['input_ids'].squeeze(0),
            'pair_a_attention_mask': pair_a['attention_mask'].squeeze(0),
            'pair_b_input_ids': pair_b['input_ids'].squeeze(0),
            'pair_b_attention_mask': pair_b['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


class HybridCrossEncoder(nn.Module):
    """
    Hybrid Cross-Encoder with advanced features:
    1. Cross-attention between anchor and candidates
    2. Multi-layer perceptron for scoring
    3. Contrastive learning support
    """
    
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2', 
                 dropout: float = 0.1, hidden_dim: int = 512):
        super(HybridCrossEncoder, self).__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # Multi-layer scoring head
        encoder_dim = self.encoder.config.hidden_size
        self.fc1 = nn.Linear(encoder_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 256)
        self.fc3 = nn.Linear(256, 1)  # Similarity score
        
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass to compute similarity score
        
        Returns:
            Similarity score (scalar)
        """
        # Encode with transformer
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token + mean pooling
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        mean_output = torch.mean(outputs.last_hidden_state, dim=1)  # Mean pooling
        
        # Combine both representations
        combined = (cls_output + mean_output) / 2
        combined = self.dropout(combined)
        
        # Multi-layer scoring
        x = self.fc1(combined)
        x = self.layer_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        score = self.fc3(x)  # Raw similarity score
        
        return score


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for triplet learning
    Maximizes similarity between anchor and positive, 
    minimizes similarity between anchor and negative
    """
    
    def __init__(self, margin: float = 0.5, alpha: float = 0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha  # Weight for contrastive vs classification
        
    def forward(self, score_a, score_b, labels):
        """
        Args:
            score_a: Similarity scores for (anchor, text_a)
            score_b: Similarity scores for (anchor, text_b)
            labels: 1 if text_a is closer, 0 if text_b is closer
        """
        # Triplet margin loss component
        # If label=1: we want score_a > score_b + margin
        # If label=0: we want score_b > score_a + margin
        
        triplet_loss = torch.where(
            labels == 1,
            F.relu(self.margin + score_b - score_a),  # text_a should be higher
            F.relu(self.margin + score_a - score_b)   # text_b should be higher
        )
        
        # Classification component: predict which is closer
        logits = torch.cat([score_b, score_a], dim=1)
        classification_loss = F.cross_entropy(logits, labels)
        
        # Combined loss
        total_loss = (1 - self.alpha) * classification_loss + self.alpha * triplet_loss.mean()
        
        return total_loss


class ProposedModelTrainer:
    """Trainer for proposed hybrid cross-encoder model"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2',
                 max_length: int = 256, device: str = None):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = HybridCrossEncoder(model_name).to(self.device)
        self.best_model_state = None
        
        print(f"Initialized Proposed Model: {model_name} on {self.device}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
              epochs: int = 10, batch_size: int = 16, lr: float = 2e-5,
              margin: float = 0.5, alpha: float = 0.5, early_stopping_patience: int = 3):
        """
        Train the proposed model with advanced techniques
        """
        print(f"\n{'='*60}")
        print("Training Proposed Hybrid Cross-Encoder Model")
        print(f"{'='*60}\n")
        
        # Create datasets
        train_dataset = CrossEncoderDataset(train_df, self.tokenizer, self.max_length)
        val_dataset = CrossEncoderDataset(val_df, self.tokenizer, self.max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Optimizer with weight decay
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        
        # Learning rate scheduler with warmup
        total_steps = len(train_loader) * epochs
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss function
        criterion = ContrastiveLoss(margin=margin, alpha=alpha)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }
        
        best_val_f1 = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            # ========== Training ==========
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for batch in pbar:
                # Move to device
                pair_a_ids = batch['pair_a_input_ids'].to(self.device)
                pair_a_mask = batch['pair_a_attention_mask'].to(self.device)
                pair_b_ids = batch['pair_b_input_ids'].to(self.device)
                pair_b_mask = batch['pair_b_attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                score_a = self.model(pair_a_ids, pair_a_mask)
                score_b = self.model(pair_b_ids, pair_b_mask)
                
                # Compute loss
                loss = criterion(score_a, score_b, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Track metrics
                train_loss += loss.item()
                predictions = (score_a > score_b).long().squeeze()
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)
                
                pbar.set_postfix({'loss': loss.item(), 'acc': train_correct/train_total})
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            
            # ========== Validation ==========
            val_metrics = self.evaluate(val_df, batch_size, val_loader)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_metrics.get('loss', 0))
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1'])
            
            print(f"\nEpoch {epoch+1}/{epochs} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            print(f"  Val Loss: {val_metrics.get('loss', 0):.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.4f}, Val Precision: {val_metrics['precision']:.4f}")
            
            # Early stopping and model checkpoint
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                self.best_model_state = self.model.state_dict().copy()
                torch.save(self.model.state_dict(), 'models/proposed_model_best.pt')
                print(f"  ✓ New best model saved (F1: {best_val_f1:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{early_stopping_patience}")
                
                if patience_counter >= early_stopping_patience:
                    print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\n✓ Loaded best model (F1: {best_val_f1:.4f})")
        
        return history
    
    def predict(self, test_df: pd.DataFrame, batch_size: int = 16, 
                test_loader: DataLoader = None) -> np.ndarray:
        """Predict on test data"""
        self.model.eval()
        
        if test_loader is None:
            test_dataset = CrossEncoderDataset(test_df, self.tokenizer, self.max_length)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        predictions = []
        
        with torch.no_grad():
            for batch in test_loader:
                pair_a_ids = batch['pair_a_input_ids'].to(self.device)
                pair_a_mask = batch['pair_a_attention_mask'].to(self.device)
                pair_b_ids = batch['pair_b_input_ids'].to(self.device)
                pair_b_mask = batch['pair_b_attention_mask'].to(self.device)
                
                score_a = self.model(pair_a_ids, pair_a_mask)
                score_b = self.model(pair_b_ids, pair_b_mask)
                
                # Predict 1 if text_a has higher score
                preds = (score_a > score_b).long().squeeze().cpu().numpy()
                predictions.extend(preds if preds.shape else [preds.item()])
        
        return np.array(predictions)
    
    def evaluate(self, test_df: pd.DataFrame, batch_size: int = 16,
                 test_loader: DataLoader = None) -> dict:
        """Evaluate model on test data"""
        predictions = self.predict(test_df, batch_size, test_loader)
        labels = test_df['text_a_is_closer'].astype(int).values
        
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='binary'),
            'precision': precision_score(labels, predictions, average='binary'),
            'recall': recall_score(labels, predictions, average='binary')
        }
        
        return metrics
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'max_length': self.max_length
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    from data_loader import SemEvalDataLoader, split_dataset
    
    # Load data
    loader = SemEvalDataLoader()
    df = loader.load_track_a()
    train_df, val_df, test_df = split_dataset(df)
    
    # Train proposed model
    trainer = ProposedModelTrainer(
        model_name='sentence-transformers/all-mpnet-base-v2'
    )
    
    history = trainer.train(
        train_df, val_df,
        epochs=10,
        batch_size=16,
        lr=2e-5,
        margin=0.5,
        alpha=0.5,
        early_stopping_patience=3
    )
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Final Test Set Evaluation")
    print("="*60 + "\n")
    
    test_metrics = trainer.evaluate(test_df)
    print("Test Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save final model
    trainer.save("models/proposed_model_final.pt")

