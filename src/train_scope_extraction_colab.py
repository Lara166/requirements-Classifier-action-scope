"""
Train scope extraction model - COLAB OPTIMIZED VERSION

This version is optimized for Google Colab free tier.

Usage on Colab:
    !python train_scope_extraction_colab.py --data labeled_train.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
import json

import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    DataCollatorForSeq2Seq, 
    Trainer, 
    TrainingArguments
)

from data_loader import Seq2SeqDataset, load_jsonl, records_to_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("labeled_train.jsonl"))
    # QUALITY-OPTIMIZED: Use larger mT5 model
    parser.add_argument("--model_name", type=str, default="google/mt5-base")
    parser.add_argument("--output_dir", type=Path, default=Path("scope_extraction"))
    # QUALITY-OPTIMIZED: More epochs, better settings
    parser.add_argument("--epochs", type=int, default=15)  # More epochs for seq2seq
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)  # Lower for stability
    parser.add_argument("--max_source_length", type=int, default=512)  # Full context
    parser.add_argument("--max_target_length", type=int, default=256)  # More output space
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_beams", type=int, default=5)  # Beam search for generation
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    
    print(f"Loading data from {args.data}...")
    records = load_jsonl(args.data)
    print(f"Loaded {len(records)} records")
    
    samples = records_to_samples(records)
    
    # Filter samples that have scope_labels
    samples_with_scope = [s for s in samples if s.scope_labels is not None]
    print(f"Samples with scope_labels: {len(samples_with_scope)}")
    
    # Check how many have non-empty scope labels
    non_empty = [s for s in samples_with_scope if any(s.scope_labels.values())]
    print(f"Samples with non-empty scope_labels: {len(non_empty)} ({len(non_empty)/len(samples_with_scope)*100:.1f}%)")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_samples, eval_samples = train_test_split(
        samples, test_size=0.1, random_state=args.seed
    )
    
    print(f"\nCreating datasets...")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Eval: {len(eval_samples)} samples")
    
    train_ds = Seq2SeqDataset(
        train_samples,
        target_attr="scope_labels",
        tokenizer=tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )
    eval_ds = Seq2SeqDataset(
        eval_samples,
        target_attr="scope_labels",
        tokenizer=tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )
    
    print(f"\nActual dataset sizes after filtering:")
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Eval: {len(eval_ds)} samples")
    
    print(f"\nLoading model: {args.model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        evaluation_strategy="steps",
        eval_steps=50,  # Frequent evaluation
        save_strategy="steps",
        save_steps=50,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=10,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        # QUALITY optimizations
        fp16=True,
        gradient_accumulation_steps=8,  # Larger effective batch
        save_total_limit=5,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        # Generation settings
        generation_max_length=args.max_target_length,
        generation_num_beams=args.num_beams,
        # Early stopping
        early_stopping_patience=args.patience,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    trainer.train()
    
    print("\n" + "="*60)
    print("Evaluating...")
    print("="*60)
    eval_results = trainer.evaluate()
    
    print("\nEvaluation Results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nSaving model to {args.output_dir}...")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    
    # Save metrics
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    
    # Test generation on a few examples
    print("\n" + "="*60)
    print("Testing generation on sample inputs...")
    print("="*60)
    
    test_samples = eval_samples[:3]
    for i, sample in enumerate(test_samples, 1):
        input_ids = tokenizer(
            sample.text, 
            max_length=args.max_source_length, 
            truncation=True, 
            return_tensors="pt"
        ).input_ids.to(model.device)
        
        outputs = model.generate(
            input_ids, 
            max_length=args.max_target_length,
            num_beams=args.num_beams,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\nExample {i}:")
        print(f"Input: {sample.text[:150]}...")
        print(f"True scope: {sample.scope_labels}")
        print(f"Predicted: {prediction}")
    
    print("\n✅ Training complete!")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
