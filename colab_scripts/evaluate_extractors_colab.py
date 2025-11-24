# ============================================================================
# ACTION & SCOPE EXTRACTION EVALUATION - COLAB SCRIPT
# ============================================================================
# Evaluiert beide Extractor-Modelle auf Validation/Test Sets
# Runtime: GPU (T4/A100)
# Voraussetzung: action_extraction/ und scope_extraction/ Ordner vorhanden
#                labeled_validation.jsonl und labeled_test.jsonl in Colab
# ============================================================================

# ============================================================================
# ZELLE 1: GPU Check
# ============================================================================
!nvidia-smi

# ============================================================================
# ZELLE 2: Dependencies
# ============================================================================
!pip install -q rouge-score bert-score

# ============================================================================
# ZELLE 3: Imports & Hilfsfunktionen
# ============================================================================
import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import List, Dict, Any
from collections import defaultdict
import re

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Lade JSONL-Datei"""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def parse_extraction_output(output: str, extraction_type: str) -> Dict[str, Any]:
    """Parse T5 Output zurück in strukturierte Form"""
    result = {}
    
    if extraction_type == "action":
        # Parse: "actor: ... | action: ... | deadline: ..."
        parts = output.split('|')
        for part in parts:
            part = part.strip()
            if ':' in part:
                key, value = part.split(':', 1)
                key = key.strip()
                value = value.strip()
                if key == 'actor':
                    result['actor'] = value
                elif key == 'action':
                    result['action'] = value
                elif key == 'deadline':
                    result['deadline'] = value
    
    elif extraction_type == "scope":
        # Parse: "products: ... | materials: ... | components: ..."
        parts = output.split('|')
        for part in parts:
            part = part.strip()
            if ':' in part:
                key, value = part.split(':', 1)
                key = key.strip()
                value = value.strip()
                if key == 'products':
                    result['product_types'] = [v.strip() for v in value.split(',')]
                elif key == 'materials':
                    result['materials'] = [v.strip() for v in value.split(',')]
                elif key == 'components':
                    result['components'] = [v.strip() for v in value.split(',')]
    
    return result

def exact_match(pred: Dict, gold: Dict, keys: List[str]) -> float:
    """Berechne Exact Match für gegebene Keys"""
    matches = 0
    total = 0
    
    for key in keys:
        pred_val = pred.get(key, '')
        gold_val = gold.get(key, '')
        
        # Normalisiere
        if isinstance(pred_val, list):
            pred_val = sorted([str(v).lower().strip() for v in pred_val])
        else:
            pred_val = str(pred_val).lower().strip()
        
        if isinstance(gold_val, list):
            gold_val = sorted([str(v).lower().strip() for v in gold_val])
        else:
            gold_val = str(gold_val).lower().strip()
        
        total += 1
        if pred_val == gold_val:
            matches += 1
    
    return matches / total if total > 0 else 0.0

def field_f1(pred: Dict, gold: Dict, keys: List[str]) -> Dict[str, float]:
    """Berechne F1 pro Feld"""
    results = {}
    
    for key in keys:
        pred_val = pred.get(key, '')
        gold_val = gold.get(key, '')
        
        # Convert to sets of tokens
        if isinstance(pred_val, list):
            pred_tokens = set(' '.join(pred_val).lower().split())
        else:
            pred_tokens = set(str(pred_val).lower().split())
        
        if isinstance(gold_val, list):
            gold_tokens = set(' '.join(gold_val).lower().split())
        else:
            gold_tokens = set(str(gold_val).lower().split())
        
        if not pred_tokens and not gold_tokens:
            results[key] = 1.0
        elif not pred_tokens or not gold_tokens:
            results[key] = 0.0
        else:
            overlap = len(pred_tokens & gold_tokens)
            precision = overlap / len(pred_tokens) if pred_tokens else 0
            recall = overlap / len(gold_tokens) if gold_tokens else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            results[key] = f1
    
    return results

print("✓ Funktionen geladen")

# ============================================================================
# ZELLE 4: Action Extraction Model laden
# ============================================================================
print("Lade Action Extraction Model...")
action_tokenizer = T5Tokenizer.from_pretrained("./action_extraction")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

action_model = T5ForConditionalGeneration.from_pretrained(
    "./action_extraction",
    dtype=torch.float16 if device.type == "cuda" else torch.float32,
    device_map="auto" if device.type == "cuda" else None
)
if device.type != "cuda":
    action_model.to(device)

print(f"✓ Action Model geladen auf {device}")

# ============================================================================
# ZELLE 5: Scope Extraction Model laden
# ============================================================================
print("Lade Scope Extraction Model...")
scope_tokenizer = T5Tokenizer.from_pretrained("./scope_extraction")

scope_model = T5ForConditionalGeneration.from_pretrained(
    "./scope_extraction",
    dtype=torch.float16 if device.type == "cuda" else torch.float32,
    device_map="auto" if device.type == "cuda" else None
)
if device.type != "cuda":
    scope_model.to(device)

print(f"✓ Scope Model geladen auf {device}")

# ============================================================================
# ZELLE 6: Daten laden
# ============================================================================
print("Lade Validation Set...")
validation_data = load_jsonl("/content/labeled_validation.jsonl")
print(f"✓ {len(validation_data)} Validation Samples")

print("\nLade Test Set...")
test_data = load_jsonl("/content/labeled_test.jsonl")
print(f"✓ {len(test_data)} Test Samples")

# ============================================================================
# ZELLE 7: Action Extraction Evaluation
# ============================================================================
def evaluate_action_extraction(model, tokenizer, records: List[Dict], device, dataset_name: str):
    """Evaluiere Action Extraction"""
    print(f"\n{'='*80}")
    print(f"ACTION EXTRACTION EVALUATION: {dataset_name.upper()}")
    print(f"{'='*80}")
    
    # Filter: nur Requirements mit action_labels
    samples = [r for r in records if r.get('action_labels')]
    print(f"Samples mit Action Labels: {len(samples)}")
    
    if len(samples) == 0:
        print("⚠️ Keine Samples mit Action Labels gefunden!")
        return
    
    # Predictions
    predictions = []
    references = []
    
    model.eval()
    with torch.no_grad():
        for i, rec in enumerate(samples):
            # Input
            input_text = 'extract action: ' + rec['text']
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse
            pred_dict = parse_extraction_output(pred_text, "action")
            gold_dict = rec['action_labels']
            
            predictions.append(pred_dict)
            references.append(gold_dict)
            
            if (i + 1) % 100 == 0:
                print(f"  Verarbeitet: {i+1}/{len(samples)} Samples")
    
    # Metriken
    print(f"\n{'='*80}")
    print("METRIKEN:")
    print(f"{'='*80}")
    
    # Exact Match pro Sample
    exact_matches = [exact_match(pred, gold, ['actor', 'action', 'deadline']) 
                     for pred, gold in zip(predictions, references)]
    avg_exact_match = sum(exact_matches) / len(exact_matches)
    print(f"Average Exact Match (alle 3 Felder): {avg_exact_match:.4f}")
    
    # F1 pro Feld
    field_f1_scores = defaultdict(list)
    for pred, gold in zip(predictions, references):
        scores = field_f1(pred, gold, ['actor', 'action', 'deadline'])
        for key, score in scores.items():
            field_f1_scores[key].append(score)
    
    print("\nF1-Scores pro Feld:")
    for key in ['actor', 'action', 'deadline']:
        scores = field_f1_scores[key]
        avg_f1 = sum(scores) / len(scores)
        print(f"  {key:12s}: {avg_f1:.4f}")
    
    # Makro F1
    macro_f1 = sum([sum(scores)/len(scores) for scores in field_f1_scores.values()]) / len(field_f1_scores)
    print(f"\nMacro F1 (Durchschnitt): {macro_f1:.4f}")
    
    # Beispiele
    print(f"\n{'='*80}")
    print("BEISPIELE (erste 3):")
    print(f"{'='*80}")
    for i in range(min(3, len(samples))):
        print(f"\n--- Beispiel {i+1} ---")
        print(f"Text: {samples[i]['text'][:150]}...")
        print(f"\nGold:")
        print(f"  Actor:    {references[i].get('actor', 'N/A')}")
        print(f"  Action:   {references[i].get('action', 'N/A')}")
        print(f"  Deadline: {references[i].get('deadline', 'N/A')}")
        print(f"\nPredicted:")
        print(f"  Actor:    {predictions[i].get('actor', 'N/A')}")
        print(f"  Action:   {predictions[i].get('action', 'N/A')}")
        print(f"  Deadline: {predictions[i].get('deadline', 'N/A')}")
    
    return {
        'exact_match': avg_exact_match,
        'macro_f1': macro_f1,
        'field_f1': {k: sum(v)/len(v) for k, v in field_f1_scores.items()}
    }

# Validation Set
val_action_metrics = evaluate_action_extraction(
    action_model, action_tokenizer, validation_data, device, "VALIDATION SET"
)

# ============================================================================
# ZELLE 8: Action Extraction Test Set
# ============================================================================
test_action_metrics = evaluate_action_extraction(
    action_model, action_tokenizer, test_data, device, "TEST SET"
)

# ============================================================================
# ZELLE 9: Scope Extraction Evaluation
# ============================================================================
def evaluate_scope_extraction(model, tokenizer, records: List[Dict], device, dataset_name: str):
    """Evaluiere Scope Extraction"""
    print(f"\n{'='*80}")
    print(f"SCOPE EXTRACTION EVALUATION: {dataset_name.upper()}")
    print(f"{'='*80}")
    
    # Filter: nur Requirements mit scope_labels
    samples = [r for r in records if r.get('scope_labels')]
    print(f"Samples mit Scope Labels: {len(samples)}")
    
    if len(samples) == 0:
        print("⚠️ Keine Samples mit Scope Labels gefunden!")
        return
    
    # Predictions
    predictions = []
    references = []
    
    model.eval()
    with torch.no_grad():
        for i, rec in enumerate(samples):
            # Input
            input_text = 'extract scope: ' + rec['text']
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse
            pred_dict = parse_extraction_output(pred_text, "scope")
            gold_dict = rec['scope_labels']
            
            predictions.append(pred_dict)
            references.append(gold_dict)
            
            if (i + 1) % 100 == 0:
                print(f"  Verarbeitet: {i+1}/{len(samples)} Samples")
    
    # Metriken
    print(f"\n{'='*80}")
    print("METRIKEN:")
    print(f"{'='*80}")
    
    # Exact Match pro Sample
    exact_matches = [exact_match(pred, gold, ['product_types', 'materials', 'components']) 
                     for pred, gold in zip(predictions, references)]
    avg_exact_match = sum(exact_matches) / len(exact_matches)
    print(f"Average Exact Match (alle 3 Felder): {avg_exact_match:.4f}")
    
    # F1 pro Feld
    field_f1_scores = defaultdict(list)
    for pred, gold in zip(predictions, references):
        scores = field_f1(pred, gold, ['product_types', 'materials', 'components'])
        for key, score in scores.items():
            field_f1_scores[key].append(score)
    
    print("\nF1-Scores pro Feld:")
    for key in ['product_types', 'materials', 'components']:
        scores = field_f1_scores[key]
        avg_f1 = sum(scores) / len(scores)
        print(f"  {key:15s}: {avg_f1:.4f}")
    
    # Makro F1
    macro_f1 = sum([sum(scores)/len(scores) for scores in field_f1_scores.values()]) / len(field_f1_scores)
    print(f"\nMacro F1 (Durchschnitt): {macro_f1:.4f}")
    
    # Beispiele
    print(f"\n{'='*80}")
    print("BEISPIELE (erste 3):")
    print(f"{'='*80}")
    for i in range(min(3, len(samples))):
        print(f"\n--- Beispiel {i+1} ---")
        print(f"Text: {samples[i]['text'][:150]}...")
        print(f"\nGold:")
        print(f"  Products:   {references[i].get('product_types', [])}")
        print(f"  Materials:  {references[i].get('materials', [])}")
        print(f"  Components: {references[i].get('components', [])}")
        print(f"\nPredicted:")
        print(f"  Products:   {predictions[i].get('product_types', [])}")
        print(f"  Materials:  {predictions[i].get('materials', [])}")
        print(f"  Components: {predictions[i].get('components', [])}")
    
    return {
        'exact_match': avg_exact_match,
        'macro_f1': macro_f1,
        'field_f1': {k: sum(v)/len(v) for k, v in field_f1_scores.items()}
    }

# Validation Set
val_scope_metrics = evaluate_scope_extraction(
    scope_model, scope_tokenizer, validation_data, device, "VALIDATION SET"
)

# ============================================================================
# ZELLE 10: Scope Extraction Test Set
# ============================================================================
test_scope_metrics = evaluate_scope_extraction(
    scope_model, scope_tokenizer, test_data, device, "TEST SET"
)

# ============================================================================
# ZELLE 11: Finale Zusammenfassung
# ============================================================================
print(f"\n{'='*80}")
print("FINALE ZUSAMMENFASSUNG - ALLE METRIKEN")
print(f"{'='*80}")

print("\n" + "="*80)
print("ACTION EXTRACTION")
print("="*80)

if val_action_metrics:
    print(f"\nValidation Set:")
    print(f"  Exact Match: {val_action_metrics['exact_match']:.4f}")
    print(f"  Macro F1:    {val_action_metrics['macro_f1']:.4f}")
    print(f"  Field F1:")
    for field, score in val_action_metrics['field_f1'].items():
        print(f"    {field:12s}: {score:.4f}")

if test_action_metrics:
    print(f"\nTest Set:")
    print(f"  Exact Match: {test_action_metrics['exact_match']:.4f}")
    print(f"  Macro F1:    {test_action_metrics['macro_f1']:.4f}")
    print(f"  Field F1:")
    for field, score in test_action_metrics['field_f1'].items():
        print(f"    {field:12s}: {score:.4f}")

print("\n" + "="*80)
print("SCOPE EXTRACTION")
print("="*80)

if val_scope_metrics:
    print(f"\nValidation Set:")
    print(f"  Exact Match: {val_scope_metrics['exact_match']:.4f}")
    print(f"  Macro F1:    {val_scope_metrics['macro_f1']:.4f}")
    print(f"  Field F1:")
    for field, score in val_scope_metrics['field_f1'].items():
        print(f"    {field:15s}: {score:.4f}")

if test_scope_metrics:
    print(f"\nTest Set:")
    print(f"  Exact Match: {test_scope_metrics['exact_match']:.4f}")
    print(f"  Macro F1:    {test_scope_metrics['macro_f1']:.4f}")
    print(f"  Field F1:")
    for field, score in test_scope_metrics['field_f1'].items():
        print(f"    {field:15s}: {score:.4f}")

print(f"\n{'='*80}")
print("✅ EVALUATION ABGESCHLOSSEN!")
print(f"{'='*80}")
