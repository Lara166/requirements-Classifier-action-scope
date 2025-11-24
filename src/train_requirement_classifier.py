"""
Train a requirement vs. non-requirement classifier on segments_train.jsonl.

Usage:
    python -m src.train_requirement_classifier --data data/processed/segments_train.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.data_loader import ClassificationDataset, load_jsonl, records_to_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/segments_train.jsonl"))
    parser.add_argument("--model_name", type=str, default="distilbert-base-multilingual-cased")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/requirement_classifier"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    records = load_jsonl(args.data)
    samples = records_to_samples(records)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_samples, eval_samples = train_test_split(samples, test_size=0.1, random_state=args.seed)

    train_ds = ClassificationDataset(train_samples, tokenizer, max_length=args.max_length)
    eval_ds = ClassificationDataset(eval_samples, tokenizer, max_length=args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
