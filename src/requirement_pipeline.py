"""
Requirement Pipeline - Phase 2: Segmente → Strukturierte Requirements
======================================================================

Verarbeitet segments.jsonl und erzeugt structured_requirements.jsonl

Architektur:
    1. Lese Segmente aus Phase 1 (Ingest)
    2. Multi-Class Classification (4 Klassen)
    3. Parallel Attribute Extraction
    4. Product vs. Reporting Classification
    5. Erzeuge vollständige StructuredRequirement-Objekte
    6. Markiere low-confidence für manuelles Review
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import re

from src.requirement_schema import StructuredRequirement
from src.requirement_extractor import (
    RequirementClassifier, 
    ParallelAttributeExtractor,
    ScopeLabelsExtractor,
    ActionLabelsExtractor
)


class RequirementPipeline:
    """
    Haupt-Pipeline für Requirement-Extraktion.
    """
    
    def __init__(self, min_confidence_threshold: float = 0.6):
        self.classifier = RequirementClassifier()
        self.extractor = ParallelAttributeExtractor()
        self.scope_extractor = ScopeLabelsExtractor()
        self.action_extractor = ActionLabelsExtractor()
        self.min_confidence = min_confidence_threshold
        self.stats = {
            'total_segments': 0,
            'requirement_undertaking': 0,
            'requirement_member_state': 0,
            'requirement_other_actor': 0,
            'non_requirement': 0,
            'product_requirements': 0,
            'reporting_requirements': 0,
            'needs_manual_review': 0
        }
    
    def _parse_law_info(self, doc_id: str) -> Dict[str, str]:
        """Extrahiert Gesetzesinfo aus doc_id."""
        # Format: CSRD_Corporate_..._2022_CELEX-32022L2464_EN
        parts = doc_id.split('_')
        
        law_name = parts[0] if parts else 'Unknown'
        
        # CELEX-Nummer extrahieren
        celex = None
        for part in parts:
            if part.startswith('CELEX-'):
                celex = part.replace('CELEX-', '')
                break
        
        # Jahr extrahieren (vierstellige Zahl)
        year = None
        for part in parts:
            if re.match(r'^\d{4}$', part):
                year = part
                break
        
        return {
            'law_name': law_name,
            'celex_number': celex,
            'year': year
        }
    
    def _extract_article_refs(self, text: str) -> List[str]:
        """Extrahiert Verweise auf andere Artikel."""
        refs = []
        
        # Englisch: Article X, Article X(Y)
        refs.extend(re.findall(r'Article\s+(\d+[a-z]?(?:\(\d+\))?)', text, re.I))
        
        # Deutsch: Artikel X, § X
        refs.extend(re.findall(r'Artikel\s+(\d+[a-z]?)', text, re.I))
        refs.extend(re.findall(r'§\s*(\d+[a-z]?)', text))
        
        return list(set(refs))  # Deduplizieren
    
    def _check_flags(self, text: str) -> Dict[str, bool]:
        """Prüft spezielle Flags (exception, definition, conditional)."""
        text_lower = text.lower()
        
        return {
            'contains_exception': bool(re.search(
                r'\b(unless|except|exemption|by way of derogation|Ausnahme|außer)\b', 
                text_lower
            )),
            'contains_definition': bool(re.search(
                r'\b(means|for the purposes of|shall mean|Definition|bedeutet)\b',
                text_lower
            )),
            'is_conditional': bool(re.search(
                r'\b(if|where|when|provided that|falls|sofern|wenn)\b',
                text_lower
            ))
        }
    
    def process_segment(self, segment: Dict) -> StructuredRequirement:
        """
        Verarbeitet ein einzelnes Segment.
        
        Returns: StructuredRequirement-Objekt
        """
        text = segment['text']
        language = segment.get('language', 'unknown')
        structure_type = segment.get('structure_type') or 'unspecified'
        
        # Step 1: Multi-Class Classification
        requirement_class, class_confidence = self.classifier.classify(
            text, structure_type, language
        )
        
        # Step 2: Parallel Attribute Extraction
        attributes = self.extractor.extract_all_attributes(
            text, language, structure_type, requirement_class
        )
        
        # Step 2a: Extract Scope Labels and Action Labels (for ML training)
        scope_labels = self.scope_extractor.extract(text)
        action_labels = self.action_extractor.extract(text)
        
        # Step 3: Law Info
        law_info = self._parse_law_info(segment['doc_id'])
        
        # Step 4: Flags & References
        flags = self._check_flags(text)
        article_refs = self._extract_article_refs(text)
        
        # Step 5: Needs Manual Review?
        avg_confidence = (
            class_confidence +
            attributes['modality'].confidence +
            attributes['addressee'].confidence +
            attributes['topic'].confidence +
            attributes['is_product_requirement'].confidence
        ) / 5
        
        needs_review = (
            avg_confidence < self.min_confidence or
            requirement_class == 'non_requirement' and class_confidence < 0.8
        )
        
        # Step 6: Konstruiere requirement_id
        article_num = segment.get('structure_label', 'NA')
        chunk_idx = segment.get('chunk_index', 0)
        requirement_id = f"{law_info['law_name']}#{article_num}#{chunk_idx:04d}"
        
        # Step 7: Baue StructuredRequirement
        req = StructuredRequirement(
            requirement_id=requirement_id,
            doc_id=segment['doc_id'],
            law_name=law_info['law_name'],
            celex_number=law_info.get('celex_number'),
            article_number=str(article_num) if article_num else None,
            structure_type=structure_type,
            
            text=text,
            char_start=segment.get('char_start', 0),
            char_end=segment.get('char_end', 0),
            language=language,
            
            requirement_class=requirement_class,
            classification_confidence=class_confidence,
            
            addressee=attributes['addressee'].value,
            addressee_confidence=attributes['addressee'].confidence,
            
            modality=attributes['modality'].value,
            modality_confidence=attributes['modality'].confidence,
            
            topic=attributes['topic'].value,
            topic_confidence=attributes['topic'].confidence,
            
            obligation_type=attributes['obligation_type'].value,
            scope=attributes['scope'].value,
            
            is_product_requirement=(attributes['is_product_requirement'].value == 'True'),
            product_relevance_confidence=attributes['is_product_requirement'].confidence,
            
            contains_exception=flags['contains_exception'],
            contains_definition=flags['contains_definition'],
            is_conditional=flags['is_conditional'],
            references_other_articles=article_refs,
            
            validity_status=segment.get('validity_status', 'unknown'),
            last_amendment_date=segment.get('last_amendment_date'),
            
            extracted_at=datetime.now(),
            extraction_method='rule_based',
            needs_manual_review=needs_review
        )
        
        # Update Stats
        self.stats['total_segments'] += 1
        self.stats[requirement_class] += 1
        if req.is_product_requirement:
            self.stats['product_requirements'] += 1
        else:
            self.stats['reporting_requirements'] += 1
        if needs_review:
            self.stats['needs_manual_review'] += 1
        
        return req
    
    def run(self, input_file: Path, output_file: Path, batch_size: int = 5000) -> None:
        """
        Verarbeitet alle Segmente in Batches und überschreibt Output.
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)
        # Lösche existierende Output-Datei
        if output_file.exists():
            output_file.unlink()
        print(f"[info] Processing {input_file}...")
        print(f"[info] Min confidence threshold: {self.min_confidence}")
        print(f"[info] Output will be overwritten: {output_file}")
        
        processed_count = 0
        batch = []
        with input_file.open('r', encoding='utf-8') as f_in:
            for line_num, line in enumerate(f_in, 1):
                try:
                    segment = json.loads(line)
                    req = self.process_segment(segment)
                    
                    # Get scope_labels and action_labels
                    scope_labels = self.scope_extractor.extract(segment['text'])
                    action_labels = self.action_extractor.extract(segment['text'])
                    
                    # Convert to dict and add the extra fields
                    req_dict = json.loads(req.model_dump_json())
                    req_dict['scope_labels'] = scope_labels
                    req_dict['action_labels'] = action_labels
                    
                    batch.append(json.dumps(req_dict, ensure_ascii=False))
                    processed_count += 1
                    if processed_count % batch_size == 0:
                        with output_file.open('a', encoding='utf-8') as f_out:
                            f_out.write('\n'.join(batch) + '\n')
                        print(f"[info]   → {processed_count} segments processed and written...")
                        batch = []
                except Exception as e:
                    print(f"[error] Line {line_num}: {e}")
                    continue
            # Schreibe letzten Batch
            if batch:
                with output_file.open('a', encoding='utf-8') as f_out:
                    f_out.write('\n'.join(batch) + '\n')
        print(f"\n[info] ✅ Processing complete!")
        print(f"[info] Total segments: {self.stats['total_segments']}")
        print(f"[info] Classification:")
        print(f"        - requirement_undertaking: {self.stats['requirement_undertaking']}")
        print(f"        - requirement_member_state: {self.stats['requirement_member_state']}")
        print(f"        - requirement_other_actor: {self.stats['requirement_other_actor']}")
        print(f"        - non_requirement: {self.stats['non_requirement']}")
        print(f"[info] Product vs. Reporting:")
        print(f"        - product_requirements: {self.stats['product_requirements']}")
        print(f"        - reporting_requirements: {self.stats['reporting_requirements']}")
        print(f"[info] Needs manual review: {self.stats['needs_manual_review']} ({self.stats['needs_manual_review']/self.stats['total_segments']*100:.1f}%)")
        print(f"[info] Output: {output_file}")


def run_requirement_pipeline(
    input_segments: str = "data/processed/segments.jsonl",
    output_requirements: str = "data/processed/structured_requirements.jsonl",
    min_confidence: float = 0.6
) -> None:
    """CLI-Einstiegspunkt für die Pipeline."""
    
    inp = Path(input_segments)
    outp = Path(output_requirements)
    
    if not inp.exists():
        print(f"[error] Input file not found: {inp}")
        return
    
    pipeline = RequirementPipeline(min_confidence_threshold=min_confidence)
    pipeline.run(inp, outp, batch_size=5000)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "data/processed/structured_requirements.jsonl"
        run_requirement_pipeline(input_file, output_file)
    else:
        run_requirement_pipeline()
