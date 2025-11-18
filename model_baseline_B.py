"""
Baseline Model B: BERT-base Fine-tuning
Transformer-based approach using pre-trained BERT
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd


class BERTDataset(Dataset):
    """Dataset for BERT model"""
    
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Concatenate: [CLS] anchor [SEP] text_a [SEP]
        pair_a = self.tokenizer(
            row['anchor_text'],
            row['text_a'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Concatenate: [CLS] anchor [SEP] text_b [SEP]
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


class BERTBaseline(nn.Module):
    """BERT-based baseline model"""
    
    def __init__(self, model_name: str = 'bert-base-uncased', dropout: float = 0.1):
        super(BERTBaseline, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)  # Binary classification
        
    def forward(self, input_ids, attention_mask):
        """Forward pass through BERT"""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token representation
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits


class BERTTrainer:
    """Trainer for BERT baseline"""
    
    def __init__(self, model_name: str = 'bert-base-uncased', 
                 max_length: int = 256, device: str = None):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BERTBaseline(model_name).to(self.device)
        
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
              epochs: int = 5, batch_size: int = 16, lr: float = 2e-5):
        """Train the model"""
        print(f"Training BERT Baseline on {self.device}...")
        
        # Create datasets
        train_dataset = BERTDataset(train_df, self.tokenizer, self.max_length)
        val_dataset = BERTDataset(val_df, self.tokenizer, self.max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                # Process pair_a (anchor, text_a)
                pair_a_ids = batch['pair_a_input_ids'].to(self.device)
                pair_a_mask = batch['pair_a_attention_mask'].to(self.device)
                logits_a = self.model(pair_a_ids, pair_a_mask)
                
                # Process pair_b (anchor, text_b)
                pair_b_ids = batch['pair_b_input_ids'].to(self.device)
                pair_b_mask = batch['pair_b_attention_mask'].to(self.device)
                logits_b = self.model(pair_b_ids, pair_b_mask)
                
                # Compare: if text_a is closer, logits_a should be higher
                # Convert to binary classification
                labels = batch['label'].to(self.device)
                
                # Use similarity scores (class 1 probability)
                score_a = torch.softmax(logits_a, dim=1)[:, 1]
                score_b = torch.softmax(logits_b, dim=1)[:, 1]
                
                # Predict based on which score is higher
                combined_logits = torch.stack([score_b, score_a], dim=1)
                
                loss = criterion(combined_logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            val_metrics = self.evaluate(val_df, batch_size)
            
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
                  f"Val Acc={val_metrics['accuracy']:.4f}, Val F1={val_metrics['f1']:.4f}")
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_metrics.get('loss', 0))
            history['val_acc'].append(val_metrics['accuracy'])
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                torch.save(self.model.state_dict(), 'models/baseline_bert_best.pt')
        
        return history
    
    def predict(self, test_df: pd.DataFrame, batch_size: int = 16) -> np.ndarray:
        """Predict on test data"""
        self.model.eval()
        test_dataset = BERTDataset(test_df, self.tokenizer, self.max_length)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        predictions = []
        
        with torch.no_grad():
            for batch in test_loader:
                pair_a_ids = batch['pair_a_input_ids'].to(self.device)
                pair_a_mask = batch['pair_a_attention_mask'].to(self.device)
                logits_a = self.model(pair_a_ids, pair_a_mask)
                
                pair_b_ids = batch['pair_b_input_ids'].to(self.device)
                pair_b_mask = batch['pair_b_attention_mask'].to(self.device)
                logits_b = self.model(pair_b_ids, pair_b_mask)
                
                score_a = torch.softmax(logits_a, dim=1)[:, 1]
                score_b = torch.softmax(logits_b, dim=1)[:, 1]
                
                # Predict 1 if text_a has higher score
                preds = (score_a > score_b).long().cpu().numpy()
                predictions.extend(preds)
        
        return np.array(predictions)
    
    def evaluate(self, test_df: pd.DataFrame, batch_size: int = 16) -> dict:
        """Evaluate model"""
        predictions = self.predict(test_df, batch_size)
        labels = test_df['text_a_is_closer'].astype(int).values
        
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions),
            'precision': precision_score(labels, predictions),
            'recall': recall_score(labels, predictions)
        }
        
        return metrics


if __name__ == "__main__":
    from data_loader import SemEvalDataLoader, split_dataset
    
    # Load data
    loader = SemEvalDataLoader()
    df = loader.load_track_a()
    train_df, val_df, test_df = split_dataset(df)
    
    # Train baseline
    trainer = BERTTrainer()
    history = trainer.train(train_df, val_df, epochs=3, batch_size=16)
    
    # Evaluate
    test_metrics = trainer.evaluate(test_df)
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")

