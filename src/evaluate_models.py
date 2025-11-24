"""
Simple evaluation harness for the three models.

Classification: accuracy/f1.
Seq2Seq: exact-match ratio of generated JSON strings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from src.data_loader import Seq2SeqDataset, load_jsonl, records_to_samples


def evaluate_classifier(model_dir: Path, samples) -> Dict[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    texts = [s.text for s in samples if s.requirement_label is not None]
    labels = [s.requirement_label for s in samples if s.requirement_label is not None]

    preds = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = int(torch.argmax(logits, dim=-1).item())
        preds.append(pred)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


def evaluate_seq2seq(model_dir: Path, samples, target_attr: str) -> Dict[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.eval()

    filtered = [s for s in samples if getattr(s, target_attr) is not None]
    if not filtered:
        raise ValueError(f"No labels present for {target_attr}.")

    gold_texts = [json.dumps(getattr(s, target_attr), ensure_ascii=False) for s in filtered]
    preds: List[str] = []
    for s in filtered:
        inputs = tokenizer(s.text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=256, num_beams=4)
        pred_text = tokenizer.decode(ids[0], skip_special_tokens=True)
        preds.append(pred_text)

    exact = [int(p.strip() == g.strip()) for p, g in zip(preds, gold_texts)]
    return {"exact_match": float(np.mean(exact))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/segments_train.jsonl"))
    parser.add_argument("--classifier_dir", type=Path, default=Path("outputs/requirement_classifier"))
    parser.add_argument("--scope_dir", type=Path, default=Path("outputs/scope_extraction"))
    parser.add_argument("--action_dir", type=Path, default=Path("outputs/action_extraction"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.data)
    samples = records_to_samples(records)

    clf_metrics = evaluate_classifier(args.classifier_dir, samples)
    scope_metrics = evaluate_seq2seq(args.scope_dir, samples, target_attr="scope_labels")
    action_metrics = evaluate_seq2seq(args.action_dir, samples, target_attr="action_labels")

    print("Classifier:", clf_metrics)
    print("Scope:", scope_metrics)
    print("Action:", action_metrics)


if __name__ == "__main__":
    main()
