"""
Train a seq2seq model to extract required actions as JSON.

Expected field in data: "action_labels" containing the target JSON object.
Usage:
    python -m src.train_action_extraction --data data/processed/segments_train.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

from src.data_loader import Seq2SeqDataset, load_jsonl, records_to_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/segments_train.jsonl"))
    parser.add_argument("--model_name", type=str, default="google/mt5-small")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/action_extraction"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    records = load_jsonl(args.data)
    samples = records_to_samples(records)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_samples, eval_samples = train_test_split(samples, test_size=0.1, random_state=args.seed)

    train_ds = Seq2SeqDataset(
        train_samples,
        target_attr="action_labels",
        tokenizer=tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )
    eval_ds = Seq2SeqDataset(
        eval_samples,
        target_attr="action_labels",
        tokenizer=tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

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
        predict_with_generate=True,
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
