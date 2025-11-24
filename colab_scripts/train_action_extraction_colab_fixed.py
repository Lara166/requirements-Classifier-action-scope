#!/usr/bin/env python3
"""
Train action extraction model on Google Colab.
Quality-optimized version with mt5-base and advanced features.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
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


def action_labels_to_text(action_labels: Dict[str, Any]) -> str:
    """Convert action_labels dict to text for seq2seq target."""
    if not action_labels:
        return ""
    
    parts = []
    # Handle both string and list formats
    action = action_labels.get("action")
    if action:
        if isinstance(action, list):
            parts.append(f"action: {', '.join(action)}")
        else:
            parts.append(f"action: {action}")
    
    actor = action_labels.get("actor")
    if actor:
        if isinstance(actor, list):
            parts.append(f"actor: {', '.join(actor)}")
        else:
            parts.append(f"actor: {actor}")
    
    deadline = action_labels.get("deadline")
    if deadline:
        parts.append(f"deadline: {deadline}")
    
    document = action_labels.get("document")
    if document:
        if isinstance(document, list):
            parts.append(f"document: {', '.join(document)}")
        else:
            parts.append(f"document: {document}")
    
    references = action_labels.get("references")
    if references:
        if isinstance(references, list):
            parts.append(f"references: {', '.join(references)}")
        else:
            parts.append(f"references: {references}")
    
    return " | ".join(parts)


def records_to_samples(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert records to seq2seq samples."""
    samples = []
    for rec in records:
        action_labels = rec.get("action_labels", {})
        if not action_labels:
            continue
        
        # Skip if all values are None or empty
        has_content = False
        for key in ['action', 'actor', 'deadline', 'document', 'references']:
            val = action_labels.get(key)
            if val:  # Not None, not empty string, not empty list
                if isinstance(val, list):
                    if len(val) > 0:
                        has_content = True
                        break
                else:
                    has_content = True
                    break
        
        if not has_content:
            continue  # Skip samples without meaningful action labels
        
        text = rec.get("text", "")
        target = action_labels_to_text(action_labels)
        
        if target and len(target.strip()) > 0:  # Only add if we have a non-empty target
            samples.append({
                "text": text,
                "target": target
            })
    
    return samples


def compute_metrics(eval_pred):
    """Compute BLEU and other metrics."""
    predictions, labels = eval_pred
    
    # Decode predictions and labels
    # predictions are token ids, need to decode them
    # For simplicity, we'll just count exact matches
    
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Replace -100 in labels (used for padding)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Count non-empty predictions
    non_empty = (predictions != tokenizer.pad_token_id).any(axis=1).sum()
    
    return {
        'non_empty_predictions': non_empty / len(predictions)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to labeled JSONL file")
    parser.add_argument("--output", type=str, default="action_extraction", help="Output directory")
    parser.add_argument("--model", type=str, default="google/mt5-base", help="Model name")
    parser.add_argument("--epochs", type=int, default=12, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_source_length", type=int, default=512, help="Max source length")
    parser.add_argument("--max_target_length", type=int, default=256, help="Max target length")
    args = parser.parse_args()

    # Load data
    records = load_data(args.data)
    print(f"Samples with action_labels: {len(records)}")
    
    samples = records_to_samples(records)
    non_empty = sum(1 for s in samples if s["target"])
    print(f"Samples with non-empty action_labels: {non_empty} ({100*non_empty/len(records):.1f}%)")
    
    # Load tokenizer
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Create datasets
    print("\nCreating datasets...")
    train_samples, eval_samples = train_test_split(samples, test_size=0.1, random_state=42)
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Eval: {len(eval_samples)} samples")
    
    train_dataset = Dataset.from_dict({
        "text": [s["text"] for s in train_samples],
        "target": [s["target"] for s in train_samples]
    })
    
    eval_dataset = Dataset.from_dict({
        "text": [s["text"] for s in eval_samples],
        "target": [s["target"] for s in eval_samples]
    })
    
    # Tokenize
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["text"], 
            max_length=args.max_source_length, 
            truncation=True,
            padding="max_length"
        )
        
        # Setup the tokenizer for targets
        labels = tokenizer(
            examples["target"],
            max_length=args.max_target_length,
            truncation=True,
            padding="max_length"
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    
    # Filter out empty samples
    train_dataset = train_dataset.filter(lambda x: len(x["text"]) > 0 and len(x["target"]) > 0)
    eval_dataset = eval_dataset.filter(lambda x: len(x["text"]) > 0 and len(x["target"]) > 0)
    
    print(f"\nActual dataset sizes after filtering:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Eval: {len(eval_dataset)} samples")
    
    # Load model
    print(f"\nLoading model: {args.model}")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    
    # Data collator for handling decoder_input_ids
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        max_length=args.max_target_length
    )
    
    # Training arguments - FIXED: evaluation_strategy -> eval_strategy
    training_args = Seq2SeqTrainingArguments(
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
        metric_for_best_model="loss",
        greater_is_better=False,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        report_to=["tensorboard"],
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        generation_num_beams=4,
        max_grad_norm=1.0,
        lr_scheduler_type="linear"
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
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
    
    # Generate some example predictions
    print("\n" + "="*80)
    print("Sample predictions:")
    print("="*80)
    
    sample_size = min(5, len(eval_dataset))
    for i in range(sample_size):
        sample = eval_dataset[i]
        input_ids = torch.tensor([sample["input_ids"]]).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=args.max_target_length,
                num_beams=5,
                early_stopping=True
            )
        
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        target = tokenizer.decode(sample["labels"], skip_special_tokens=True)
        
        print(f"\nExample {i+1}:")
        print(f"  Target:     {target}")
        print(f"  Prediction: {prediction}")
    
    print(f"\n✓ Training complete! Model saved to {args.output}/")
    print(f"  - Best loss: {eval_results.get('eval_loss', 0):.4f}")


if __name__ == "__main__":
    main()
