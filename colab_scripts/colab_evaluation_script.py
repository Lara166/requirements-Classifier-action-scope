# ============================================================================
# REQUIREMENT CLASSIFIER EVALUATION - COLAB SCRIPT
# ============================================================================
# Kopiere diese Code-Blöcke nacheinander in Colab-Zellen
# Runtime: GPU (T4/P100/A100)
# Voraussetzung: requirement_classifier.zip und labeled_validation.zip in Google Drive
# ============================================================================

# ============================================================================
# ZELLE 1: GPU Check
# ============================================================================
!nvidia-smi

# ============================================================================
# ZELLE 2: Dependencies installieren
# ============================================================================
!pip install -q scikit-learn

# ============================================================================
# ZELLE 3: Google Drive mounten
# ============================================================================
from google.colab import drive
drive.mount('/content/drive')

# ============================================================================
# ZELLE 4: ZIPs aus Drive extrahieren
# ============================================================================
import zipfile
import os

# WICHTIG: Passe diesen Pfad an!
# Wenn ZIPs direkt in "Meine Ablage" liegen:
drive_path = "/content/drive/MyDrive/"
# Wenn in Unterordner (z.B. "classifier_eval"):
# drive_path = "/content/drive/MyDrive/classifier_eval/"

print("Verfügbare ZIP-Dateien in Drive:")
!ls -lh {drive_path}*.zip

# Extrahieren
model_zip = drive_path + "requirement_classifier.zip"
data_zip = drive_path + "labeled_validation.zip"

print(f"\nExtrahiere {model_zip}...")
with zipfile.ZipFile(model_zip, 'r') as zip_ref:
    zip_ref.extractall('/content/')
print("✓ Model extrahiert")

print(f"\nExtrahiere {data_zip}...")
with zipfile.ZipFile(data_zip, 'r') as zip_ref:
    zip_ref.extractall('/content/')
print("✓ Daten extrahiert")

# Verifizieren
print("\n" + "="*80)
print("VERIFIKATION:")
print("="*80)
!ls -lh /content/requirement_classifier/ | head -10
!ls -lh /content/*.jsonl

# ============================================================================
# ZELLE 5: PyTorch Check
# ============================================================================
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA verfügbar: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# ZELLE 6: Evaluation-Funktionen definieren
# ============================================================================
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    f1_score, precision_score, recall_score, accuracy_score
)
from typing import List, Dict, Any

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Lade JSONL-Datei"""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def predict_batch(model, tokenizer, texts: List[str], device, batch_size=32):
    """Batch-Vorhersagen"""
    predictions = []
    probabilities = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(device)
            
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Verarbeitet: {i+len(batch_texts)}/{len(texts)} Samples")
    
    return predictions, probabilities

def evaluate_on_dataset(model, tokenizer, records: List[Dict], device, dataset_name: str):
    """Evaluiere auf Dataset"""
    print(f"\n{'='*80}")
    print(f"EVALUATION: {dataset_name.upper()}")
    print(f"{'='*80}")
    
    texts = [rec.get("text", "") for rec in records]
    true_labels = [
        1 if rec.get("requirement_class", "").startswith("requirement_") else 0
        for rec in records
    ]
    
    print(f"Gesamt: {len(texts)} Samples")
    print(f"Requirements: {sum(true_labels)} ({100*sum(true_labels)/len(true_labels):.1f}%)")
    print(f"Non-Requirements: {len(true_labels)-sum(true_labels)} ({100*(len(true_labels)-sum(true_labels))/len(true_labels):.1f}%)")
    
    print("\nMache Vorhersagen...")
    predictions, probabilities = predict_batch(model, tokenizer, texts, device)
    
    # Metriken
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='binary')
    recall = recall_score(true_labels, predictions, average='binary')
    f1 = f1_score(true_labels, predictions, average='binary')
    
    print(f"\n{'='*80}")
    print("METRIKEN:")
    print(f"{'='*80}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Non-Req  Requirement")
    print(f"Actual Non-Req  {cm[0][0]:6d}   {cm[0][1]:6d}")
    print(f"Actual Req      {cm[1][0]:6d}   {cm[1][1]:6d}")
    
    # Classification Report
    print(f"\nClassification Report:")
    print(classification_report(
        true_labels, 
        predictions,
        target_names=['Non-Requirement', 'Requirement'],
        digits=4
    ))
    
    # Fehleranalyse
    false_positives = []
    false_negatives = []
    
    for i, (true_label, pred_label, prob) in enumerate(zip(true_labels, predictions, probabilities)):
        if true_label != pred_label:
            confidence = prob[pred_label]
            if true_label == 0 and pred_label == 1:
                false_positives.append({'text': texts[i], 'confidence': confidence})
            elif true_label == 1 and pred_label == 0:
                false_negatives.append({'text': texts[i], 'confidence': confidence})
    
    print(f"\n{'='*80}")
    print("FEHLERANALYSE:")
    print(f"{'='*80}")
    print(f"\nFalse Positives: {len(false_positives)}")
    if false_positives:
        print("\nTop 3 False Positives (höchste Confidence):")
        false_positives.sort(key=lambda x: x['confidence'], reverse=True)
        for i, fp in enumerate(false_positives[:3], 1):
            print(f"\n{i}. Confidence: {fp['confidence']:.4f}")
            print(f"   Text: {fp['text'][:200]}...")
    
    print(f"\nFalse Negatives: {len(false_negatives)}")
    if false_negatives:
        print("\nTop 3 False Negatives (höchste Confidence):")
        false_negatives.sort(key=lambda x: x['confidence'], reverse=True)
        for i, fn in enumerate(false_negatives[:3], 1):
            print(f"\n{i}. Confidence: {fn['confidence']:.4f}")
            print(f"   Text: {fn['text'][:200]}...")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

print("✓ Funktionen geladen")

# ============================================================================
# ZELLE 7: Modell laden
# ============================================================================
model_path = "/content/requirement_classifier"

print(f"Lade Modell von {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    device_map="auto" if device.type == "cuda" else None
)
if device.type != "cuda":
    model.to(device)

print(f"✓ Modell geladen auf {device}")

# ============================================================================
# ZELLE 8: Daten laden
# ============================================================================
print("Lade Validation Set...")
validation_data = load_jsonl("/content/labeled_validation.jsonl")
print(f"✓ {len(validation_data)} Validation Samples")

print("\nLade Test Set...")
test_data = load_jsonl("/content/labeled_test.jsonl")
print(f"✓ {len(test_data)} Test Samples")

# ============================================================================
# ZELLE 9: Evaluation Validation Set
# ============================================================================
val_metrics = evaluate_on_dataset(model, tokenizer, validation_data, device, "VALIDATION SET")

# ============================================================================
# ZELLE 10: Evaluation Test Set
# ============================================================================
test_metrics = evaluate_on_dataset(model, tokenizer, test_data, device, "TEST SET")

# ============================================================================
# ZELLE 11: Zusammenfassung
# ============================================================================
print(f"\n{'='*80}")
print("FINALE ZUSAMMENFASSUNG")
print(f"{'='*80}")
print(f"\nValidation Set:")
print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
print(f"  Precision: {val_metrics['precision']:.4f}")
print(f"  Recall:    {val_metrics['recall']:.4f}")
print(f"  F1 Score:  {val_metrics['f1']:.4f}")

print(f"\nTest Set:")
print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
print(f"  Precision: {test_metrics['precision']:.4f}")
print(f"  Recall:    {test_metrics['recall']:.4f}")
print(f"  F1 Score:  {test_metrics['f1']:.4f}")

print(f"\n{'='*80}")
print("✅ EVALUATION ABGESCHLOSSEN!")
print(f"{'='*80}")
