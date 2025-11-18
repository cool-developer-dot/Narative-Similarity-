"""
Main experiment runner for Assignment 3
Trains and evaluates all models, generates comparative results

CS-272: Artificial Intelligence - Semester Project
National University of Sciences and Technology (NUST)
"""

import argparse
import json
import time
from pathlib import Path
import pandas as pd
import torch

from data_loader import SemEvalDataLoader, split_dataset
from model_baseline_A import TFIDFBaseline
from model_baseline_B import BERTTrainer
from model_baseline_C import SentenceBERTBaseline
from model_baseline_D import RoBERTaTrainer
from model_proposed import ProposedModelTrainer


def run_baseline_tfidf(train_df, val_df, test_df):
    """Run TF-IDF baseline"""
    print("\n" + "="*80)
    print("BASELINE A: TF-IDF + Logistic Regression")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    model = TFIDFBaseline(max_features=5000, ngram_range=(1, 2))
    model.train(train_df)
    
    val_metrics = model.evaluate(val_df)
    test_metrics = model.evaluate(test_df)
    
    training_time = time.time() - start_time
    
    print("\nValidation Metrics:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nTraining Time: {training_time:.2f} seconds")
    
    # Save model
    model.save("models/baseline_tfidf.pkl")
    
    return {
        'model': 'TF-IDF + LogReg',
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'training_time': training_time
    }


def run_baseline_bert(train_df, val_df, test_df, epochs=5, batch_size=16):
    """Run BERT baseline"""
    print("\n" + "="*80)
    print("BASELINE B: BERT-base Fine-tuning")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    trainer = BERTTrainer(model_name='bert-base-uncased')
    history = trainer.train(train_df, val_df, epochs=epochs, batch_size=batch_size, lr=2e-5)
    
    # Load best model
    trainer.model.load_state_dict(torch.load('models/baseline_bert_best.pt'))
    
    val_metrics = trainer.evaluate(val_df, batch_size)
    test_metrics = trainer.evaluate(test_df, batch_size)
    
    training_time = time.time() - start_time
    
    print("\nValidation Metrics:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nTraining Time: {training_time:.2f} seconds")
    
    return {
        'model': 'BERT-base',
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'training_time': training_time,
        'history': history
    }


def run_baseline_sentence_bert(train_df, val_df, test_df, fine_tune=True, 
                               epochs=3, batch_size=16):
    """Run Sentence-BERT baseline"""
    print("\n" + "="*80)
    print("BASELINE C: Sentence-BERT")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    model = SentenceBERTBaseline(model_name='all-MiniLM-L6-v2')
    
    if fine_tune:
        print("Fine-tuning Sentence-BERT...")
        model.fine_tune(train_df, val_df, epochs=epochs, batch_size=batch_size)
        model.save("models/baseline_sentence_bert")
    else:
        print("Using pre-trained Sentence-BERT (no fine-tuning)")
    
    val_metrics = model.evaluate(val_df, batch_size=32)
    test_metrics = model.evaluate(test_df, batch_size=32)
    
    training_time = time.time() - start_time
    
    print("\nValidation Metrics:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nTime: {training_time:.2f} seconds")
    
    return {
        'model': 'Sentence-BERT' + (' (FT)' if fine_tune else ' (PT)'),
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'training_time': training_time
    }


def run_baseline_roberta(train_df, val_df, test_df, epochs=5, batch_size=16):
    """Run RoBERTa baseline"""
    print("\n" + "="*80)
    print("BASELINE D: RoBERTa Fine-tuning")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    trainer = RoBERTaTrainer(model_name='roberta-base')
    history = trainer.train(train_df, val_df, epochs=epochs, batch_size=batch_size, lr=2e-5)
    
    # Load best model
    trainer.model.load_state_dict(torch.load('models/baseline_roberta_best.pt'))
    
    val_metrics = trainer.evaluate(val_df, batch_size)
    test_metrics = trainer.evaluate(test_df, batch_size)
    
    training_time = time.time() - start_time
    
    print("\nValidation Metrics:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nTraining Time: {training_time:.2f} seconds")
    
    return {
        'model': 'RoBERTa-base',
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'training_time': training_time,
        'history': history
    }


def run_proposed_model(train_df, val_df, test_df, epochs=10, batch_size=16):
    """Run proposed hybrid cross-encoder model"""
    print("\n" + "="*80)
    print("PROPOSED MODEL: Hybrid Cross-Encoder with Contrastive Learning")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    trainer = ProposedModelTrainer(
        model_name='sentence-transformers/all-mpnet-base-v2',
        max_length=256
    )
    
    history = trainer.train(
        train_df, val_df,
        epochs=epochs,
        batch_size=batch_size,
        lr=2e-5,
        margin=0.5,
        alpha=0.5,
        early_stopping_patience=3
    )
    
    # Load best model
    trainer.model.load_state_dict(torch.load('models/proposed_model_best.pt'))
    
    val_metrics = trainer.evaluate(val_df, batch_size)
    test_metrics = trainer.evaluate(test_df, batch_size)
    
    training_time = time.time() - start_time
    
    print("\nValidation Metrics:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nTraining Time: {training_time:.2f} seconds")
    
    return {
        'model': 'Proposed (Hybrid Cross-Encoder)',
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'training_time': training_time,
        'history': history
    }


def save_results(all_results, output_file='results/all_results.json'):
    """Save all results to JSON"""
    Path('results').mkdir(exist_ok=True)
    
    # Convert to serializable format
    serializable_results = []
    for result in all_results:
        serializable = {
            'model': result['model'],
            'val_metrics': result['val_metrics'],
            'test_metrics': result['test_metrics'],
            'training_time': result['training_time']
        }
        serializable_results.append(serializable)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Also save as CSV for easy viewing
    csv_data = []
    for result in serializable_results:
        row = {
            'Model': result['model'],
            'Test Accuracy': result['test_metrics']['accuracy'],
            'Test F1': result['test_metrics']['f1'],
            'Test Precision': result['test_metrics']['precision'],
            'Test Recall': result['test_metrics']['recall'],
            'Training Time (s)': result['training_time']
        }
        csv_data.append(row)
    
    df_results = pd.DataFrame(csv_data)
    df_results.to_csv('results/all_results.csv', index=False)
    print(f"✓ Results saved to results/all_results.csv")
    
    # Print comparison table
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80 + "\n")
    print(df_results.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='Run experiments for Assignment 3')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'tfidf', 'bert', 'sbert', 'roberta', 'proposed'],
                       help='Which model to train')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs for proposed model')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--quick_test', action='store_true',
                       help='Quick test with fewer epochs')
    parser.add_argument('--no_baselines', action='store_true',
                       help='Skip baselines, only run proposed model')
    
    args = parser.parse_args()
    
    # Adjust epochs for quick test
    if args.quick_test:
        args.epochs = 3
        print("\n⚠ Quick test mode: reduced epochs")
    
    # Load data
    print("\nLoading data...")
    loader = SemEvalDataLoader()
    df = loader.load_track_a()
    
    print("\nSplitting dataset...")
    train_df, val_df, test_df = split_dataset(df, train_ratio=0.7, val_ratio=0.15)
    
    # Create directories
    Path('models').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    
    all_results = []
    
    # Run experiments
    if args.model == 'all' or args.model == 'tfidf':
        if not args.no_baselines:
            result = run_baseline_tfidf(train_df, val_df, test_df)
            all_results.append(result)
    
    if args.model == 'all' or args.model == 'bert':
        if not args.no_baselines:
            epochs = 3 if args.quick_test else 5
            result = run_baseline_bert(train_df, val_df, test_df, 
                                      epochs=epochs, batch_size=args.batch_size)
            all_results.append(result)
    
    if args.model == 'all' or args.model == 'sbert':
        if not args.no_baselines:
            epochs = 2 if args.quick_test else 3
            result = run_baseline_sentence_bert(train_df, val_df, test_df,
                                               fine_tune=True, epochs=epochs, 
                                               batch_size=args.batch_size)
            all_results.append(result)
    
    if args.model == 'all' or args.model == 'roberta':
        if not args.no_baselines:
            epochs = 3 if args.quick_test else 5
            result = run_baseline_roberta(train_df, val_df, test_df,
                                          epochs=epochs, batch_size=args.batch_size)
            all_results.append(result)
    
    if args.model == 'all' or args.model == 'proposed':
        result = run_proposed_model(train_df, val_df, test_df,
                                    epochs=args.epochs, batch_size=args.batch_size)
        all_results.append(result)
    
    # Save and display results
    if all_results:
        save_results(all_results)
    
    print("\n" + "="*80)
    print("✓ All experiments completed successfully!")
    print("="*80 + "\n")
    print("Next steps:")
    print("  1. Run: python evaluate_results.py --plot")
    print("  2. Check results/ and plots/ directories")
    print("  3. Review the LaTeX report template")


if __name__ == "__main__":
    main()

