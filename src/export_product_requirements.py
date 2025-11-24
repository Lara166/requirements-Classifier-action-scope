"""
Exportiert alle Produktanforderungen als CSV und erstellt ein Mapping auf Produktstruktur (DSM/ADG).
"""

import json
import csv
from pathlib import Path
from typing import List, Dict

# Beispiel-Mapping: Produktkomponenten (DSM/ADG)
PRODUCT_COMPONENTS = {
    'battery': ['capacity', 'labeling', 'safety', 'durability'],
    'washing_machine': ['energy_efficiency', 'labeling'],
    'vehicle': ['emission', 'safety', 'labeling'],
    # ... weitere Komponenten
}


def map_requirement_to_component(requirement_text: str) -> str:
    """
    Einfache Heuristik: Mappt Requirement auf Produktkomponente.
    """
    text = requirement_text.lower()
    for component, keywords in PRODUCT_COMPONENTS.items():
        for kw in keywords:
            if kw in text:
                return component
    return 'unspecified'


def export_product_requirements(
    input_file: str = "data/processed/structured_requirements.jsonl",
    output_csv: str = "outputs/product_requirements.csv"
) -> None:
    """
    Exportiert alle Produktanforderungen als CSV mit Mapping.
    """
    inp = Path(input_file)
    outp = Path(output_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    
    with inp.open('r', encoding='utf-8') as f_in, outp.open('w', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(f_out)
        # Header
        writer.writerow([
            'requirement_id', 'law_name', 'article_number', 'text',
            'addressee', 'modality', 'topic', 'obligation_type', 'scope',
            'product_component', 'classification_confidence', 'product_relevance_confidence'
        ])
        
        for line in f_in:
            req = json.loads(line)
            if req.get('is_product_requirement'):
                component = map_requirement_to_component(req['text'])
                writer.writerow([
                    req.get('requirement_id'),
                    req.get('law_name'),
                    req.get('article_number'),
                    req.get('text'),
                    req.get('addressee'),
                    req.get('modality'),
                    req.get('topic'),
                    req.get('obligation_type'),
                    req.get('scope'),
                    component,
                    req.get('classification_confidence'),
                    req.get('product_relevance_confidence')
                ])
    print(f"[info] Export complete: {output_csv}")


if __name__ == "__main__":
    export_product_requirements()
