#!/usr/bin/env python3
"""
Train requirement classifier on Google Colab.
Quality-optimized version with larger model and more epochs.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset


def load_data(jsonl_file: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    print(f"Loading data from {jsonl_file}...")
    records = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records")
    return records


def records_to_samples(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert records to classification samples."""
    samples = []
    for rec in records:
        text = rec.get("text", "")
        # Map requirement_class to binary label
        # requirement_class can be: "requirement", "non_requirement", or specific types
        requirement_class = rec.get("requirement_class", "non_requirement")
        label = 0 if requirement_class == "non_requirement" else 1
        
        samples.append({
            "text": text,
            "label": label
        })
    
    print(f"Converted to {len(samples)} samples")
    return samples


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate accuracy
    accuracy = (predictions == labels).mean()
    
    # Calculate precision, recall, F1 for each class
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to labeled JSONL file")
    parser.add_argument("--output", type=str, default="requirement_classifier", help="Output directory")
    parser.add_argument("--model", type=str, default="xlm-roberta-large", help="Model name")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    args = parser.parse_args()

    # Load data
    records = load_data(args.data)
    samples = records_to_samples(records)
    
    # Print label distribution
    labels = [s["label"] for s in samples]
    print(f"\nLabel distribution:")
    print(f"  Requirements (1): {sum(labels)} ({100*sum(labels)/len(labels):.1f}%)")
    print(f"  Non-requirements (0): {len(labels)-sum(labels)} ({100*(len(labels)-sum(labels))/len(labels):.1f}%)")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Create datasets
    print("\nCreating datasets...")
    train_samples, eval_samples = train_test_split(samples, test_size=0.2, random_state=42, stratify=labels)
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Eval: {len(eval_samples)} samples")
    
    train_dataset = Dataset.from_dict({
        "text": [s["text"] for s in train_samples],
        "label": [s["label"] for s in train_samples]
    })
    
    eval_dataset = Dataset.from_dict({
        "text": [s["text"] for s in eval_samples],
        "label": [s["label"] for s in eval_samples]
    })
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=args.max_length)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    
    # Load model
    print(f"\nLoading model: {args.model}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2
    )
    
    # Training arguments - FIXED: evaluation_strategy -> eval_strategy
    training_args = TrainingArguments(
        output_dir=args.output,
        eval_strategy="steps",  # FIXED: was evaluation_strategy
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=f"{args.output}/logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        report_to=["tensorboard"],
        label_smoothing_factor=0.1,
        lr_scheduler_type="cosine"
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    trainer.train()
    
    # Save final model
    print(f"\nSaving final model to {args.output}/")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    
    # Final evaluation
    print("\n" + "="*80)
    print("Final evaluation on validation set:")
    print("="*80)
    
    eval_results = trainer.evaluate()
    print("\nMetrics:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Detailed classification report
    predictions = trainer.predict(eval_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = predictions.label_ids
    
    print("\nClassification Report:")
    print(classification_report(
        true_labels, 
        pred_labels,
        target_names=["Non-Requirement", "Requirement"],
        digits=4
    ))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, pred_labels))
    
    print(f"\n✓ Training complete! Model saved to {args.output}/")
    print(f"  - Best F1: {eval_results.get('eval_f1', 0):.4f}")
    print(f"  - Best Accuracy: {eval_results.get('eval_accuracy', 0):.4f}")


if __name__ == "__main__":
    main()
