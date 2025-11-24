"""
Rule-Based Requirement Classifier - Baseline für ML-Vergleich
Klassifiziert Requirements basierend auf Keywords und Patterns
"""
import re
from typing import Dict, List, Tuple


class RuleBasedClassifier:
    """
    Regelbasierter Classifier für Requirement-Kategorien.
    Dient als Baseline für ML-Modell-Vergleich.
    """
    
    def __init__(self):
        # Patterns für jede Kategorie (Deutsch & Englisch)
        self.patterns = {
            'requirement_undertaking': [
                # Deutsch
                r'\b(muss|müssen|hat zu|haben zu|ist zu|sind zu|verpflichtet|verpflichten)\b',
                r'\bsoll(en)?\b(?!.*\bnicht\b)',
                r'\b(gewährleisten|sicherstellen|durchführen|umsetzen|vorlegen)\b',
                # Englisch
                r'\bshall\b(?!.*\bnot\b)',
                r'\bmust\b(?!.*\bnot\b)',
                r'\b(ensure|guarantee|implement|carry out|provide|submit)\b',
                r'\brequired to\b',
                r'\bobliged to\b'
            ],
            'requirement_prohibition': [
                # Deutsch
                r'\b(nicht|kein|keine|verboten|untersagt|darf nicht|dürfen nicht)\b',
                r'\b(verbietet|verbieten|untersagen)\b',
                r'\b(ausgeschlossen|unzulässig)\b',
                # Englisch
                r'\bshall not\b',
                r'\bmust not\b',
                r'\bprohibited\b',
                r'\bforbidden\b',
                r'\bnot allowed\b',
                r'\bnot permitted\b',
                r'\b(ban|banned)\b'
            ],
            'requirement_permission': [
                # Deutsch
                r'\b(darf|dürfen|kann|können|berechtigt|erlaubt)\b',
                r'\b(gestattet|zulässig)\b',
                r'\b(Berechtigung|Erlaubnis)\b',
                # Englisch
                r'\bmay\b',
                r'\b(allowed|permitted|authorized) to\b',
                r'\b(entitle|permission)\b'
            ],
            'requirement_exemption': [
                # Deutsch
                r'\b(Ausnahme|ausgenommen|gilt nicht|gelten nicht)\b',
                r'\b(entfällt|befreit|Befreiung)\b',
                r'\b(sofern nicht|es sei denn)\b',
                r'\b(Abweichung|abweichen)\b',
                # Englisch
                r'\b(exception|exempt|exemption)\b',
                r'\b(does not apply|do not apply)\b',
                r'\b(unless|except|excluding)\b',
                r'\b(waiver|derogation)\b'
            ],
            'requirement_option': [
                # Deutsch
                r'\b(alternativ|oder|wahlweise)\b',
                r'\b(stattdessen|anstelle)\b',
                r'\b(Wahlmöglichkeit|Option)\b',
                # Englisch
                r'\b(alternative|alternatively|option)\b',
                r'\b(either.*or|instead of)\b',
                r'\bmay choose\b'
            ],
            'non_requirement': [
                # Definitionen
                r'\b(Definition|bedeutet|bezeichnet als|verstanden als)\b',
                r'\b(means|refers to|defined as)\b',
                # Erklärungen
                r'\b(Erklärung|Beschreibung|Zweck|Ziel)\b',
                r'\b(explanation|description|purpose|objective)\b',
                # Beispiele
                r'\b(Beispiel|beispielsweise|etwa|wie|such as)\b',
                r'\b(for example|e\.g\.|such as|including)\b'
            ]
        }
        
        # Gewichtungen für überlappende Matches
        self.category_priority = {
            'requirement_prohibition': 3,     # Höchste Priorität
            'requirement_exemption': 2.5,
            'requirement_undertaking': 2,
            'requirement_permission': 1.5,
            'requirement_option': 1,
            'non_requirement': 0.5
        }
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Klassifiziert einen Text.
        
        Args:
            text: Zu klassifizierender Text
            
        Returns:
            Tuple (predicted_class, confidence)
        """
        text_lower = text.lower()
        
        # Zähle Matches pro Kategorie
        category_scores = {}
        
        for category, patterns in self.patterns.items():
            matches = 0
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    matches += 1
            
            # Score = Anzahl Matches * Priorität
            if matches > 0:
                priority = self.category_priority.get(category, 1.0)
                category_scores[category] = matches * priority
        
        # Keine Matches → non_requirement
        if not category_scores:
            return ('non_requirement', 0.3)
        
        # Höchster Score gewinnt
        predicted_class = max(category_scores.items(), key=lambda x: x[1])[0]
        
        # Confidence basierend auf relativem Score
        total_score = sum(category_scores.values())
        max_score = category_scores[predicted_class]
        confidence = min(max_score / total_score, 0.95)  # Max 95%
        
        return (predicted_class, confidence)
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Batch-Prediction für mehrere Texte."""
        return [self.predict(text) for text in texts]


def evaluate_rule_based_classifier(data_file: str, output_file: str = None):
    """
    Evaluiert regelbasierten Classifier auf gelabelten Daten.
    
    Args:
        data_file: Pfad zu labeled_validation.jsonl oder labeled_test.jsonl
        output_file: Optional - Speichere Ergebnisse
    """
    import json
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
    
    # Lade Daten
    print(f"Lade Daten von {data_file}...")
    records = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"✓ {len(records)} Samples geladen")
    
    # Initialisiere Classifier
    classifier = RuleBasedClassifier()
    
    # Predictions
    print("\nMache Predictions...")
    texts = [r['text'] for r in records]
    true_labels = [r['requirement_class'] for r in records]
    
    predictions = classifier.predict_batch(texts)
    pred_labels = [pred[0] for pred in predictions]
    confidences = [pred[1] for pred in predictions]
    
    # Metriken
    print("\n" + "="*80)
    print("RULE-BASED CLASSIFIER - EVALUATION RESULTS")
    print("="*80)
    
    accuracy = accuracy_score(true_labels, pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average='macro')
    f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:       {accuracy:.4f}")
    print(f"  F1 (Macro):     {f1_macro:.4f}")
    print(f"  F1 (Weighted):  {f1_weighted:.4f}")
    print(f"  Avg Confidence: {sum(confidences)/len(confidences):.4f}")
    
    # Classification Report
    print("\n" + "="*80)
    print("Classification Report:")
    print("="*80)
    print(classification_report(true_labels, pred_labels, digits=4, zero_division=0))
    
    # Confusion Matrix
    print("\n" + "="*80)
    print("Confusion Matrix:")
    print("="*80)
    cm = confusion_matrix(true_labels, pred_labels)
    labels = sorted(set(true_labels + pred_labels))
    
    # Header
    print(f"{'':25s}", end='')
    for label in labels:
        print(f"{label[:15]:>15s}", end='')
    print()
    
    # Rows
    for i, label in enumerate(labels):
        print(f"{label:25s}", end='')
        for j in range(len(labels)):
            if i < len(cm) and j < len(cm[i]):
                print(f"{cm[i][j]:15d}", end='')
            else:
                print(f"{'0':>15s}", end='')
        print()
    
    # Fehleranalyse
    print("\n" + "="*80)
    print("Fehleranalyse (Top 5):")
    print("="*80)
    
    errors = []
    for i, (true, pred, conf) in enumerate(zip(true_labels, pred_labels, confidences)):
        if true != pred:
            errors.append({
                'index': i,
                'text': texts[i][:150],
                'true': true,
                'pred': pred,
                'conf': conf
            })
    
    # Sortiere nach Confidence (höchste Confidence bei falscher Prediction = interessanteste Fehler)
    errors.sort(key=lambda x: x['conf'], reverse=True)
    
    for i, error in enumerate(errors[:5], 1):
        print(f"\n{i}. Confidence: {error['conf']:.2f}")
        print(f"   True: {error['true']}")
        print(f"   Pred: {error['pred']}")
        print(f"   Text: {error['text']}...")
    
    # Speichere Ergebnisse
    if output_file:
        results = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'avg_confidence': float(sum(confidences)/len(confidences)),
            'total_samples': len(records),
            'classification_report': classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Ergebnisse gespeichert in {output_file}")
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'predictions': predictions
    }


if __name__ == '__main__':
    import sys
    
    # Validation Set
    print("\n" + "="*80)
    print("VALIDATION SET")
    print("="*80)
    val_results = evaluate_rule_based_classifier(
        'outputs/labeled_validation.jsonl',
        'outputs/rule_based_classifier_validation.json'
    )
    
    # Test Set
    print("\n\n" + "="*80)
    print("TEST SET")
    print("="*80)
    test_results = evaluate_rule_based_classifier(
        'outputs/labeled_test.jsonl',
        'outputs/rule_based_classifier_test.json'
    )
    
    # Vergleich
    print("\n\n" + "="*80)
    print("ZUSAMMENFASSUNG - RULE-BASED vs. ML")
    print("="*80)
    print("\nRule-Based Classifier:")
    print(f"  Validation F1: {val_results['f1_macro']:.4f}")
    print(f"  Test F1:       {test_results['f1_macro']:.4f}")
    
    print("\nML Classifier (XLM-RoBERTa-Large):")
    print(f"  Validation F1: 0.9028")
    print(f"  Test F1:       0.8806")
    
    print("\nImprovement durch ML:")
    print(f"  Validation: +{(0.9028 - val_results['f1_macro']):.4f} ({((0.9028 / val_results['f1_macro']) - 1) * 100:.1f}%)")
    print(f"  Test:       +{(0.8806 - test_results['f1_macro']):.4f} ({((0.8806 / test_results['f1_macro']) - 1) * 100:.1f}%)")
