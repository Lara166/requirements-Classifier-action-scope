"""
End-to-End Requirement Extraction Pipeline
Kombiniert ML-Classifier mit regelbasierter Action/Scope Extraction
"""
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import regelbasierte Extractors
import sys
sys.path.append('src')
from requirement_extractor import ActionLabelsExtractor, ScopeLabelsExtractor


class HybridRequirementPipeline:
    """
    End-to-End Pipeline für Requirement Extraction.
    
    Architektur:
    1. ML Classifier (XLM-RoBERTa): Requirement Classification
    2. Regelbasiert: Action Labels Extraction
    3. Regelbasiert: Scope Labels Extraction
    """
    
    def __init__(
        self,
        classifier_path: str = "models/requirement_classifier",
        device: Optional[str] = None
    ):
        """
        Initialisiert Pipeline.
        
        Args:
            classifier_path: Pfad zum trainierten Classifier
            device: 'cuda', 'cpu' oder None (auto-detect)
        """
        print("Initialisiere Hybrid Requirement Pipeline...")
        
        # Device Setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"  Device: {self.device}")
        
        # Lade ML Classifier
        print(f"  Lade Classifier von {classifier_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(classifier_path)
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            classifier_path,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        ).to(self.device)
        self.classifier.eval()
        print("  ✓ Classifier geladen")
        
        # Initialisiere regelbasierte Extractors
        print("  Initialisiere regelbasierte Extractors...")
        self.action_extractor = ActionLabelsExtractor()
        self.scope_extractor = ScopeLabelsExtractor()
        print("  ✓ Extractors initialisiert")
        
        # Label Mapping (Fallback wenn nicht im Model)
        if hasattr(self.classifier.config, 'id2label') and self.classifier.config.id2label:
            self.id2label = self.classifier.config.id2label
        else:
            # Manuelles Mapping basierend auf Training
            self.id2label = {
                0: 'non_requirement',
                1: 'requirement_undertaking',
                2: 'requirement_prohibition',
                3: 'requirement_permission',
                4: 'requirement_exemption',
                5: 'requirement_option',
                6: 'requirement_other_actor'
            }
            print(f"  ⚠️  Verwende manuelles Label-Mapping")
        
        print("✓ Pipeline bereit!\n")
    
    def classify(self, text: str) -> Dict:
        """
        Klassifiziert einen Text als Requirement.
        
        Args:
            text: Zu klassifizierender Text
            
        Returns:
            Dict mit predicted_class, confidence, probabilities
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.classifier(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_id].item()
        
        predicted_class = self.id2label[pred_id]
        
        # Alle Probabilities
        all_probs = {
            self.id2label[i]: float(probs[0][i])
            for i in range(len(probs[0]))
        }
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': all_probs,
            'is_requirement': predicted_class.startswith('requirement_')
        }
    
    def extract_action_labels(self, text: str, classification: Dict) -> Dict:
        """
        Extrahiert Action Labels regelbasiert.
        
        Args:
            text: Text für Extraktion
            classification: Classifier-Output
            
        Returns:
            Dict mit actor, action, deadline
        """
        # Nur für Requirements extrahieren
        if not classification['is_requirement']:
            return {'actor': None, 'action': None, 'deadline': None}
        
        # Regelbasierte Extraktion
        action_labels = self.action_extractor.extract(text)
        
        return {
            'actor': action_labels.get('actor'),
            'action': action_labels.get('action'),
            'deadline': action_labels.get('deadline')
        }
    
    def extract_scope_labels(self, text: str, classification: Dict) -> Dict:
        """
        Extrahiert Scope Labels regelbasiert.
        
        Args:
            text: Text für Extraktion
            classification: Classifier-Output
            
        Returns:
            Dict mit product_types, materials, components
        """
        # Nur für Requirements extrahieren
        if not classification['is_requirement']:
            return {
                'product_types': [],
                'materials': [],
                'components': []
            }
        
        # Regelbasierte Extraktion
        scope_labels = self.scope_extractor.extract(text)
        
        return {
            'product_types': scope_labels.get('product_types', []),
            'materials': scope_labels.get('materials', []),
            'components': scope_labels.get('components', [])
        }
    
    def process(self, text: str, include_metadata: bool = True) -> Dict:
        """
        Vollständige Pipeline: Classification + Extraction.
        
        Args:
            text: Zu verarbeitender Text
            include_metadata: Include probabilities etc.
            
        Returns:
            Vollständiges Extraction-Result
        """
        # Step 1: Classification
        classification = self.classify(text)
        
        # Step 2: Action Extraction (nur für Requirements)
        action_labels = self.extract_action_labels(text, classification)
        
        # Step 3: Scope Extraction (nur für Requirements)
        scope_labels = self.extract_scope_labels(text, classification)
        
        # Kombiniere Ergebnisse
        result = {
            'text': text[:200] + '...' if len(text) > 200 else text,
            'requirement_class': classification['predicted_class'],
            'is_requirement': classification['is_requirement'],
            'confidence': classification['confidence'],
            'action_labels': action_labels,
            'scope_labels': scope_labels
        }
        
        if include_metadata:
            result['probabilities'] = classification['probabilities']
        
        return result
    
    def process_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Verarbeitet mehrere Texte in Batches.
        
        Args:
            texts: Liste von Texten
            batch_size: Batch-Größe für Classifier
            show_progress: Zeige Progress
            
        Returns:
            Liste von Extraction-Results
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Batch Classification
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.classifier(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                pred_ids = torch.argmax(probs, dim=-1)
            
            # Process each in batch
            for j, text in enumerate(batch_texts):
                pred_id = pred_ids[j].item()
                confidence = probs[j][pred_id].item()
                predicted_class = self.id2label[pred_id]
                is_requirement = predicted_class.startswith('requirement_')
                
                classification = {
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'is_requirement': is_requirement
                }
                
                # Extract labels
                action_labels = self.extract_action_labels(text, classification)
                scope_labels = self.extract_scope_labels(text, classification)
                
                results.append({
                    'text': text[:200] + '...' if len(text) > 200 else text,
                    'requirement_class': predicted_class,
                    'is_requirement': is_requirement,
                    'confidence': confidence,
                    'action_labels': action_labels,
                    'scope_labels': scope_labels
                })
            
            if show_progress and (i + batch_size) % 100 == 0:
                print(f"  Verarbeitet: {min(i+batch_size, len(texts))}/{len(texts)}")
        
        return results


def demo_pipeline():
    """Demo: Pipeline auf Beispieltexten."""
    print("="*80)
    print("HYBRID REQUIREMENT EXTRACTION PIPELINE - DEMO")
    print("="*80 + "\n")
    
    # Initialisiere Pipeline
    pipeline = HybridRequirementPipeline()
    
    # Beispieltexte
    examples = [
        {
            'name': 'Undertaking (Battery)',
            'text': 'Die Hersteller von Batterien müssen bis zum 31. Dezember 2025 '
                   'einen Nachhaltigkeitsbericht vorlegen, der Informationen über '
                   'den Kobaltgehalt und die Recyclingfähigkeit enthält.'
        },
        {
            'name': 'Prohibition (REACH)',
            'text': 'Die Verwendung von Blei in Verpackungsmaterialien ist verboten, '
                   'es sei denn, der Gehalt liegt unter 100 ppm.'
        },
        {
            'name': 'Non-Requirement',
            'text': 'Diese Verordnung tritt am zwanzigsten Tag nach ihrer Veröffentlichung '
                   'im Amtsblatt der Europäischen Union in Kraft.'
        }
    ]
    
    # Verarbeite Beispiele
    for i, example in enumerate(examples, 1):
        print(f"\n{'='*80}")
        print(f"BEISPIEL {i}: {example['name']}")
        print('='*80)
        print(f"\nText:\n  {example['text']}\n")
        
        result = pipeline.process(example['text'], include_metadata=False)
        
        print(f"Classification:")
        print(f"  Class:      {result['requirement_class']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Is Req:     {result['is_requirement']}")
        
        if result['is_requirement']:
            print(f"\nAction Labels:")
            print(f"  Actor:      {result['action_labels']['actor']}")
            print(f"  Action:     {result['action_labels']['action']}")
            print(f"  Deadline:   {result['action_labels']['deadline']}")
            
            print(f"\nScope Labels:")
            print(f"  Products:   {result['scope_labels']['product_types']}")
            print(f"  Materials:  {result['scope_labels']['materials']}")
            print(f"  Components: {result['scope_labels']['components']}")


def evaluate_pipeline_on_test_set(test_file: str = 'outputs/labeled_test.jsonl'):
    """Evaluiert Pipeline auf Test Set."""
    print("\n" + "="*80)
    print("PIPELINE EVALUATION ON TEST SET")
    print("="*80 + "\n")
    
    # Lade Test Daten
    print(f"Lade Test Set von {test_file}...")
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f if line.strip()]
    print(f"✓ {len(test_data)} Samples geladen\n")
    
    # Initialisiere Pipeline
    pipeline = HybridRequirementPipeline()
    
    # Verarbeite
    print("Starte Pipeline-Verarbeitung...")
    texts = [r['text'] for r in test_data]
    results = pipeline.process_batch(texts, batch_size=32)
    
    print(f"\n✓ Pipeline abgeschlossen: {len(results)} Samples verarbeitet")
    
    # Evaluate Classifier Performance
    print("\n" + "="*80)
    print("CLASSIFIER PERFORMANCE")
    print("="*80)
    
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    
    true_labels = [r['requirement_class'] for r in test_data]
    pred_labels = [r['requirement_class'] for r in results]
    
    accuracy = accuracy_score(true_labels, pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average='macro')
    f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
    
    print(f"\nAccuracy:    {accuracy:.4f}")
    print(f"F1 (Macro):  {f1_macro:.4f}")
    print(f"F1 (Weight): {f1_weighted:.4f}")
    
    print("\n" + "="*80)
    print("✅ EVALUATION ABGESCHLOSSEN")
    print("="*80)
    
    return results


if __name__ == '__main__':
    # Demo
    demo_pipeline()
    
    # Optional: Evaluation auf Test Set
    # evaluate_pipeline_on_test_set()
