"""
Train a requirement vs. non-requirement classifier - COLAB OPTIMIZED VERSION

This version is optimized for Google Colab free tier with limited GPU memory.

Usage on Colab:
    !python train_requirement_classifier_colab.py --data labeled_train.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
import json

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from data_loader import ClassificationDataset, load_jsonl, records_to_samples


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("labeled_train.jsonl"))
    # QUALITY-OPTIMIZED: Use best multilingual model
    parser.add_argument("--model_name", type=str, default="xlm-roberta-large")
    parser.add_argument("--output_dir", type=Path, default=Path("requirement_classifier"))
    # QUALITY-OPTIMIZED: More epochs, better tuning
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)  # Smaller for larger model
    parser.add_argument("--lr", type=float, default=2e-5)  # Lower LR for stability
    parser.add_argument("--max_length", type=int, default=512)  # Full context
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)  # Larger validation
    parser.add_argument("--patience", type=int, default=3)  # Early stopping
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    
    print(f"Loading data from {args.data}...")
    records = load_jsonl(args.data)
    print(f"Loaded {len(records)} records")
    
    samples = records_to_samples(records)
    print(f"Converted to {len(samples)} samples")
    
    # Check label distribution
    labels = [s.requirement_label for s in samples if s.requirement_label is not None]
    print(f"\nLabel distribution:")
    print(f"  Requirements (1): {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    print(f"  Non-requirements (0): {len(labels)-sum(labels)} ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_samples, eval_samples = train_test_split(
        samples, test_size=args.test_size, random_state=args.seed, stratify=labels
    )
    
    print(f"\nCreating datasets...")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Eval: {len(eval_samples)} samples")
    
    train_ds = ClassificationDataset(train_samples, tokenizer, max_length=args.max_length)
    eval_ds = ClassificationDataset(eval_samples, tokenizer, max_length=args.max_length)
    
    print(f"\nLoading model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        evaluation_strategy="steps",  # Evaluate more frequently
        eval_steps=100,  # Every 100 steps
        save_strategy="steps",
        save_steps=100,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=20,  # More frequent logging
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        # QUALITY optimizations
        fp16=True,
        gradient_accumulation_steps=4,  # Effective batch size 32
        warmup_ratio=0.1,  # 10% warmup
        save_total_limit=5,  # Keep 5 best checkpoints
        # Early stopping
        early_stopping_patience=args.patience,
        # Learning rate scheduling
        lr_scheduler_type="cosine",
        # Regularization
        label_smoothing_factor=0.1,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    trainer.train()
    
    print("\n" + "="*60)
    print("Evaluating on validation set...")
    print("="*60)
    eval_results = trainer.evaluate()
    
    print("\nEvaluation Results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Detailed classification report
    print("\nGenerating detailed classification report...")
    predictions = trainer.predict(eval_ds)
    pred_labels = predictions.predictions.argmax(axis=-1)
    true_labels = predictions.label_ids
    
    print("\nClassification Report:")
    print(classification_report(
        true_labels, pred_labels, 
        target_names=['Non-requirement', 'Requirement'],
        digits=4
    ))
    
    print(f"\nSaving model to {args.output_dir}...")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    
    # Save metrics
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    
    print("\n✅ Training complete!")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
