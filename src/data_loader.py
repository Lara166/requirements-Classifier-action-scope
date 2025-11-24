"""
Dataset utilities for requirement extraction experiments.

The loader keeps JSONL samples in memory and exposes small helper
datasets for classification and seq2seq training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from torch.utils.data import Dataset


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into memory."""
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


@dataclass
class TextSample:
    """Single sample representation."""

    text: str
    requirement_label: Optional[int] = None  # 1 = requirement, 0 = none
    scope_labels: Optional[Dict[str, Any]] = None  # structured scope targets
    action_labels: Optional[Dict[str, Any]] = None  # structured action targets
    meta: Optional[Dict[str, Any]] = None


def records_to_samples(records: Sequence[Dict[str, Any]]) -> List[TextSample]:
    """Map raw records to TextSample objects."""
    samples: List[TextSample] = []
    for rec in records:
        meta = {k: rec.get(k) for k in rec.keys() if k not in ("text",)}
        
        # Map requirement_class to binary label
        # requirement_class can be: requirement_undertaking, requirement_member_state, 
        # requirement_other_actor, or non_requirement
        requirement_label = None
        if "requirement_class" in rec:
            # 1 = requirement (any type), 0 = non_requirement
            requirement_label = 0 if rec["requirement_class"] == "non_requirement" else 1
        elif "contains_obligation" in rec:
            # Fallback for old format
            requirement_label = int(bool(rec.get("contains_obligation")))
        
        samples.append(
            TextSample(
                text=rec["text"],
                requirement_label=requirement_label,
                scope_labels=rec.get("scope_labels"),
                action_labels=rec.get("action_labels"),
                meta=meta,
            )
        )
    return samples


class ClassificationDataset(Dataset):
    """Torch dataset for requirement vs. non-requirement classification."""

    def __init__(self, samples: Sequence[TextSample], tokenizer, max_length: int = 512):
        labeled = [s for s in samples if s.requirement_label is not None]
        if not labeled:
            raise ValueError("No classification labels found.")
        self.samples = labeled
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        encoding = self.tokenizer(
            sample.text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        encoding["labels"] = sample.requirement_label
        return encoding


class Seq2SeqDataset(Dataset):
    """Torch dataset for text-to-JSON generation tasks."""

    def __init__(
        self,
        samples: Sequence[TextSample],
        target_attr: str,
        tokenizer,
        max_source_length: int = 512,
        max_target_length: int = 256,
    ):
        labeled: List[TextSample] = []
        for s in samples:
            target = getattr(s, target_attr)
            if target is not None:
                labeled.append(s)
        if not labeled:
            raise ValueError(f"No labels present for {target_attr}.")
        self.samples = labeled
        self.target_attr = target_attr
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        target_data = getattr(sample, self.target_attr)
        target_text = json.dumps(target_data, ensure_ascii=False)

        model_inputs = self.tokenizer(
            sample.text,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target_text,
                max_length=self.max_target_length,
                padding="max_length",
                truncation=True,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
