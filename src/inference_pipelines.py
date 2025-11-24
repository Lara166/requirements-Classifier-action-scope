"""
End-to-end inference pipelines for requirement extraction.

Pipeline:
1) Requirement classifier filters relevant segments.
2) Scope extractor generates structured scope JSON.
3) Action extractor generates structured action JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)


@dataclass
class RequirementOutput:
    text: str
    requirement: bool
    requirement_score: float
    scope: Optional[Dict[str, Any]] = None
    action: Optional[Dict[str, Any]] = None


def _parse_json_safe(generated: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(generated)
    except Exception:
        return None


class RequirementPipeline:
    def __init__(
        self,
        classifier_dir: Path = Path("outputs/requirement_classifier"),
        scope_dir: Path = Path("outputs/scope_extraction"),
        action_dir: Path = Path("outputs/action_extraction"),
        device: Optional[int] = None,
    ):
        device = device if device is not None else (0 if torch.cuda.is_available() else -1)
        self.classifier = pipeline(
            "text-classification",
            model=str(classifier_dir),
            tokenizer=str(classifier_dir),
            device=device,
            return_all_scores=True,
        )
        self.scope_tokenizer = AutoTokenizer.from_pretrained(scope_dir)
        self.scope_model = AutoModelForSeq2SeqLM.from_pretrained(scope_dir).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.action_tokenizer = AutoTokenizer.from_pretrained(action_dir)
        self.action_model = AutoModelForSeq2SeqLM.from_pretrained(action_dir).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def run(self, texts: List[str], requirement_threshold: float = 0.5) -> List[RequirementOutput]:
        outputs: List[RequirementOutput] = []
        class_preds = self.classifier(texts)

        for text, scores in zip(texts, class_preds):
            # scores is list of dicts with label "LABEL_0"/"LABEL_1"
            score_map = {s["label"]: s["score"] for s in scores}
            req_score = score_map.get("LABEL_1", 0.0)
            is_req = req_score >= requirement_threshold

            scope_json: Optional[Dict[str, Any]] = None
            action_json: Optional[Dict[str, Any]] = None

            if is_req:
                scope_json = self._generate_json(self.scope_model, self.scope_tokenizer, text)
                action_json = self._generate_json(self.action_model, self.action_tokenizer, text)

            outputs.append(
                RequirementOutput(
                    text=text,
                    requirement=is_req,
                    requirement_score=req_score,
                    scope=scope_json,
                    action=action_json,
                )
            )
        return outputs

    def _generate_json(self, model, tokenizer, text: str) -> Optional[Dict[str, Any]]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=4,
            )
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return _parse_json_safe(generated_text)
