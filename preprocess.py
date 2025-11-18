"""
Preprocessing utilities for text data
Includes cleaning, normalization, and augmentation

Assignment 3 - CS-272: Artificial Intelligence
National University of Sciences and Technology (NUST)
"""

import re
import unicodedata
from typing import List, Tuple
import random


class TextPreprocessor:
    """Text preprocessing and augmentation"""
    
    def __init__(self, lowercase: bool = True, remove_special: bool = False):
        self.lowercase = lowercase
        self.remove_special = remove_special
        
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Optional: lowercase
        if self.lowercase:
            text = text.lower()
        
        # Optional: remove special characters (keep for semantic tasks)
        if self.remove_special:
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        return text.strip()
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of texts"""
        return [self.clean_text(text) for text in texts]
    
    def simple_augment(self, text: str) -> str:
        """
        Simple text augmentation techniques:
        - Random word order shuffling (slight)
        - Synonym replacement (simulated)
        """
        words = text.split()
        
        # With 30% probability, shuffle a small portion
        if random.random() < 0.3 and len(words) > 5:
            # Shuffle a random window of 2-3 words
            window_size = random.randint(2, 3)
            start_idx = random.randint(0, len(words) - window_size)
            window = words[start_idx:start_idx + window_size]
            random.shuffle(window)
            words[start_idx:start_idx + window_size] = window
        
        return ' '.join(words)
    
    def back_translation_simulation(self, text: str) -> str:
        """
        Simulate back-translation augmentation
        (In production, use actual translation APIs or models)
        For now, we'll use simple paraphrasing rules
        """
        # Common paraphrasing patterns
        replacements = {
            r'\bis\b': 'was',
            r'\bwas\b': 'is',
            r'\bthe\b': 'a',
            r'\ba\b': 'the',
            r'\band\b': 'as well as',
            r'\bbut\b': 'however',
            r'\bvery\b': 'extremely',
            r'\bgood\b': 'excellent',
            r'\bbad\b': 'poor',
        }
        
        augmented = text
        # Apply 1-2 random replacements
        num_replacements = random.randint(1, 2)
        selected = random.sample(list(replacements.items()), 
                                min(num_replacements, len(replacements)))
        
        for pattern, replacement in selected:
            augmented = re.sub(pattern, replacement, augmented, count=1)
        
        return augmented


class TripletDataset:
    """
    Custom dataset for triplet-based training
    Handles (anchor, positive, negative) pairs
    """
    
    def __init__(self, triplets: List[Tuple[str, str, str, int]], 
                 preprocessor: TextPreprocessor = None,
                 augment: bool = False):
        self.triplets = triplets
        self.preprocessor = preprocessor or TextPreprocessor()
        self.augment = augment
        
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        anchor, text_a, text_b, label = self.triplets[idx]
        
        # Preprocess
        anchor = self.preprocessor.clean_text(anchor)
        text_a = self.preprocessor.clean_text(text_a)
        text_b = self.preprocessor.clean_text(text_b)
        
        # Augment if enabled
        if self.augment and random.random() < 0.3:
            anchor = self.preprocessor.simple_augment(anchor)
            text_a = self.preprocessor.simple_augment(text_a)
            text_b = self.preprocessor.simple_augment(text_b)
        
        # Determine positive and negative based on label
        if label == 1:  # text_a is closer
            positive = text_a
            negative = text_b
        else:  # text_b is closer
            positive = text_b
            negative = text_a
        
        return {
            'anchor': anchor,
            'positive': positive,
            'negative': negative,
            'text_a': text_a,
            'text_b': text_b,
            'label': label
        }


class PairDataset:
    """
    Custom dataset for pair-based classification
    Handles (text1, text2, label) pairs
    """
    
    def __init__(self, pairs: List[Tuple[str, str, int]], 
                 preprocessor: TextPreprocessor = None):
        self.pairs = pairs
        self.preprocessor = preprocessor or TextPreprocessor()
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        text1, text2, label = self.pairs[idx]
        
        text1 = self.preprocessor.clean_text(text1)
        text2 = self.preprocessor.clean_text(text2)
        
        return {
            'text1': text1,
            'text2': text2,
            'label': label
        }


def create_augmented_dataset(triplets: List[Tuple[str, str, str, int]], 
                            augmentation_factor: int = 2) -> List[Tuple[str, str, str, int]]:
    """
    Create augmented dataset by generating variations
    
    Args:
        triplets: Original triplets
        augmentation_factor: How many augmented versions per original
        
    Returns:
        Augmented triplets list
    """
    preprocessor = TextPreprocessor(lowercase=False)  # Keep original casing for augmentation
    augmented = list(triplets)  # Keep originals
    
    for anchor, text_a, text_b, label in triplets[:len(triplets) // augmentation_factor]:
        for _ in range(augmentation_factor):
            aug_anchor = preprocessor.back_translation_simulation(anchor)
            aug_text_a = preprocessor.back_translation_simulation(text_a)
            aug_text_b = preprocessor.back_translation_simulation(text_b)
            
            augmented.append((aug_anchor, aug_text_a, aug_text_b, label))
    
    print(f"Augmented dataset: {len(triplets)} -> {len(augmented)} samples")
    return augmented


if __name__ == "__main__":
    # Example usage
    preprocessor = TextPreprocessor()
    
    sample_text = "  This is a SAMPLE text with   extra spaces!  "
    cleaned = preprocessor.clean_text(sample_text)
    print(f"Original: '{sample_text}'")
    print(f"Cleaned: '{cleaned}'")
    
    # Augmentation example
    augmented = preprocessor.simple_augment(cleaned)
    print(f"Augmented: '{augmented}'")
    
    backtrans = preprocessor.back_translation_simulation(cleaned)
    print(f"Back-translation sim: '{backtrans}'")

