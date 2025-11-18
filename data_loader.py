"""
Data Loader for SemEval 2026 Semantic Textual Relatedness Task
Handles loading and parsing of JSONL files for Track A and Track B

Assignment 3 - CS-272: Artificial Intelligence
National University of Sciences and Technology (NUST)
"""

import json
import pandas as pd
from typing import List, Dict, Tuple
from pathlib import Path


class SemEvalDataLoader:
    """Load and parse SemEval 2026 STR data"""
    
    def __init__(self, track_a_path: str = "dev_track_a.jsonl", 
                 track_b_path: str = "dev_track_b.jsonl"):
        self.track_a_path = track_a_path
        self.track_b_path = track_b_path
        
    def load_track_a(self) -> pd.DataFrame:
        """
        Load Track A data: anchor + two candidates with label
        
        Returns:
            DataFrame with columns: anchor_text, text_a, text_b, text_a_is_closer
        """
        data = []
        with open(self.track_a_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append({
                    'anchor_text': item['anchor_text'],
                    'text_a': item['text_a'],
                    'text_b': item['text_b'],
                    'text_a_is_closer': item['text_a_is_closer']
                })
        
        df = pd.DataFrame(data)
        print(f"Loaded Track A: {len(df)} samples")
        print(f"  - text_a closer: {df['text_a_is_closer'].sum()} ({df['text_a_is_closer'].mean()*100:.1f}%)")
        print(f"  - text_b closer: {(~df['text_a_is_closer']).sum()} ({(~df['text_a_is_closer']).mean()*100:.1f}%)")
        return df
    
    def load_track_b(self) -> pd.DataFrame:
        """
        Load Track B data: single texts (for auxiliary training)
        
        Returns:
            DataFrame with column: text
        """
        data = []
        with open(self.track_b_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append({'text': item['text']})
        
        df = pd.DataFrame(data)
        print(f"Loaded Track B: {len(df)} samples")
        return df
    
    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """Compute dataset statistics"""
        stats = {
            'num_samples': len(df),
            'avg_anchor_length': df['anchor_text'].str.split().str.len().mean(),
            'avg_text_a_length': df['text_a'].str.split().str.len().mean(),
            'avg_text_b_length': df['text_b'].str.split().str.len().mean(),
            'class_balance': df['text_a_is_closer'].value_counts(normalize=True).to_dict(),
            'max_text_length': max(
                df['anchor_text'].str.split().str.len().max(),
                df['text_a'].str.split().str.len().max(),
                df['text_b'].str.split().str.len().max()
            )
        }
        return stats
    
    def create_triplets(self, df: pd.DataFrame) -> List[Tuple[str, str, str, int]]:
        """
        Convert to triplet format: (anchor, positive, negative, label)
        where label=1 if text_a is closer, else 0
        """
        triplets = []
        for _, row in df.iterrows():
            anchor = row['anchor_text']
            text_a = row['text_a']
            text_b = row['text_b']
            label = 1 if row['text_a_is_closer'] else 0
            
            triplets.append((anchor, text_a, text_b, label))
        
        return triplets
    
    def create_pairs(self, df: pd.DataFrame) -> List[Tuple[str, str, int]]:
        """
        Convert to pair format: (text1, text2, label)
        Creates two pairs per sample: (anchor, text_a) and (anchor, text_b)
        """
        pairs = []
        for _, row in df.iterrows():
            anchor = row['anchor_text']
            text_a = row['text_a']
            text_b = row['text_b']
            label_a = 1 if row['text_a_is_closer'] else 0
            label_b = 0 if row['text_a_is_closer'] else 1
            
            pairs.append((anchor, text_a, label_a))
            pairs.append((anchor, text_b, label_b))
        
        return pairs


def split_dataset(df: pd.DataFrame, train_ratio: float = 0.8, 
                  val_ratio: float = 0.1, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets
    
    Args:
        df: Input DataFrame
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        random_state: Random seed
        
    Returns:
        train_df, val_df, test_df
    """
    from sklearn.model_selection import train_test_split
    
    # Stratified split to maintain class balance
    train_df, temp_df = train_test_split(
        df, train_size=train_ratio, 
        stratify=df['text_a_is_closer'], 
        random_state=random_state
    )
    
    val_size = val_ratio / (1 - train_ratio)
    val_df, test_df = train_test_split(
        temp_df, train_size=val_size,
        stratify=temp_df['text_a_is_closer'],
        random_state=random_state
    )
    
    print(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df


if __name__ == "__main__":
    # Example usage
    loader = SemEvalDataLoader()
    
    # Load Track A data
    df_a = loader.load_track_a()
    stats = loader.get_statistics(df_a)
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Split dataset
    train_df, val_df, test_df = split_dataset(df_a)
    
    # Create triplets for contrastive learning
    triplets = loader.create_triplets(train_df)
    print(f"\nCreated {len(triplets)} triplets for training")
    
    # Example triplet
    print("\nExample triplet:")
    anchor, text_a, text_b, label = triplets[0]
    print(f"Anchor: {anchor[:100]}...")
    print(f"Text A: {text_a[:100]}...")
    print(f"Text B: {text_b[:100]}...")
    print(f"Label: {'text_a closer' if label == 1 else 'text_b closer'}")

