"""
Requirement Extractor mit paralleler Attribut-Extraktion
=========================================================

Wissenschaftlich fundiert:
- Multi-Class Classification (4 Klassen statt binär)
- Parallele Attribut-Extraktion (verhindert Fehlermultiplikation)
- Pattern-basiert + heuristische Regeln
- Später erweiterbar mit ML-Modellen

Architektur:
    1. Alle Attribute werden PARALLEL extrahiert
    2. Cross-Validation zwischen Attributen
    3. Confidence-Scores für jeden Schritt
    4. Klare Trennlinie: Product vs. Reporting Requirements
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from src.requirement_schema import (
    RequirementClass, Addressee, Modality, Topic, 
    ObligationType, Scope, StructuredRequirement
)


# ============================================================================
# Pattern Definitions (kontrollierte Mustersammlung)
# ============================================================================

class ExtractionPatterns:
    """Zentrale Pattern-Sammlung für alle Sprachen."""
    
    # ------------------------------------------------------------------------
    # Modality Patterns (Englisch)
    # ------------------------------------------------------------------------
    MODALITY_EN = {
        'shall': [
            r'\bshall\b',
            r'\bmust\b',
            r'\bis required to\b',
            r'\bis obliged to\b',
            r'\bhave to\b'
        ],
        'shall_not': [
            r'\bshall not\b',
            r'\bmust not\b',
            r'\bmay not\b',
            r'\bis prohibited\b',
            r'\bare prohibited\b'
        ],
        'may': [
            r'\bmay\b',
            r'\bis allowed to\b',
            r'\bare allowed to\b',
            r'\bcan\b'
        ],
        'should': [
            r'\bshould\b',
            r'\bis recommended\b'
        ]
    }
    # ------------------------------------------------------------------------
    # Modality Patterns (Deutsch)
    # ------------------------------------------------------------------------
    MODALITY_DE = {
            'must': [
                r'\bmüssen\b',
                r'\bmüssen[.,;:]?\b',
                r'\bmuss\b',
                r'\bmuss[.,;:]?\b',
                r'\bmuessen\b',
                r'\bmuessen[.,;:]?\b',
                r'\bhat zu\b',
                r'\bhaben zu\b',
                r'\bsind verpflichtet\b',
                r'\bist verpflichtet\b'
            ],
        'must_not': [
            r'\bdürfen nicht\b',
            r'\bdarf nicht\b',
            r'\bist untersagt\b',
            r'\bsind untersagt\b',
            r'\bist verboten\b'
        ],
        'may': [
            r'\bdürfen\b',
            r'\bdarf\b',
            r'\bkönnen\b',
            r'\bkann\b'
        ],
        'should': [
            r'\bsollten\b',
            r'\bsollte\b'
        ]
    }

# Zusätzliche Obligation-Signale ohne klassische Modalverben
    # ------------------------------------------------------------------------
    # Addressee Patterns
    # ------------------------------------------------------------------------
    ADDRESSEE_PATTERNS = {
        'undertaking': [
            r'\bundertakings?\b',
            r'\bcompan(?:y|ies)\b',
            r'\benterprise\b',
            r'\borganisations?\b',
            r'\bUnternehmen\b',
            r'\bGesellschaft\b'
        ],
        'member_state': [
            r'\bMember States?\b',
            r'\bMitgliedstaaten?\b',
            r'\bthe State\b',
            r'\bStaaten\b'
        ],
        'manufacturer': [
            r'\bmanufacturers?\b',
            r'\bHerstellers?\b',
            r'\bproducers?\b',
            r'\bErzeugers?\b'
        ],
        'auditor': [
            r'\bauditors?\b',
            r'\bPrüfers?\b',
            r'\bWirtschaftsprüfers?\b'
        ],
        'authority': [
            r'\bauthorit(?:y|ies)\b',
            r'\bcompetent authorit(?:y|ies)\b',
            r'\bBehörden?\b',
            r'\bzuständigen? Behörden?\b',
            r'\bregulator\b'
        ],
        'operator': [
            r'\boperators?\b',
            r'\bBetreibers?\b'
        ],
        'commission': [
            r'\bthe Commission\b',
            r'\bEuropean Commission\b',
            r'\bKommission\b'
        ]
    }
    
    # ------------------------------------------------------------------------
    # Topic Patterns
    # ------------------------------------------------------------------------
    TOPIC_PATTERNS = {
        'reporting': [
            r'\breport\b',
            r'\bdisclos',
            r'\bpubli[sc]h',
            r'\binformation',
            r'\bstatement',
            r'\bBericht',
            r'\bOffenlegung',
            r'\bVeröffentlichung'
        ],
        'product_design': [
            r'\bproduct',
            r'\bdesign',
            r'\btechnical',
            r'\bspecification',
            r'\bperformance',
            r'\benergy efficiency',
            r'\bProdukt',
            r'\bGestaltung',
            r'\btechnisch',
            r'\bEnergieeffizienz'
        ],
        'due_diligence': [
            r'\bdue diligence',
            r'\bsupply chain',
            r'\bsorgfaltspflicht',
            r'\bLieferkette',
            r'\brisk assessment',
            r'\bRisikobewertung'
        ],
        'labeling': [
            r'\blabel',
            r'\bmarking',
            r'\bKennzeichnung',
            r'\bEtikettierung'
        ],
        'environmental': [
            r'\bemission',
            r'\bcarbon',
            r'\bCO2',
            r'\bGHG\b',
            r'\bclimate',
            r'\bUmwelt',
            r'\bKlima'
        ],
        'monitoring': [
            r'\bmonitor',
            r'\bsupervi',
            r'\binspect',
            r'\bÜberwachung',
            r'\bKontrolle'
        ]
    }
    
    # ------------------------------------------------------------------------
    # Obligation Type Patterns
    # ------------------------------------------------------------------------
    PROHIBITION_PATTERNS = [
        r'shall not',
        r'must not',
        r'may not',
        r'prohibited',
        r'dürfen nicht',
        r'verboten',
        r'untersagt'
    ]
    
    CONDITIONAL_PATTERNS = [
        r'\bif\b',
        r'\bwhere\b',
        r'\bwhen\b',
        r'\bunless\b',
        r'\bexcept\b',
        r'\bwenn\b',
        r'\bfalls\b',
        r'\bsofern\b'
    ]
    
    EXEMPTION_PATTERNS = [
        r'by way of derogation',
        r'exemption',
        r'except when',
        r'unless',
        r'Ausnahme',
        r'abweichend'
    ]


# Zusätzliche Obligation-Signale ohne klassische Modalverben
OBLIGATION_SIGNAL_PATTERNS_EN = [
    re.compile(r'\bensure that\b', re.I),
    re.compile(r'\bare responsible for ensuring\b', re.I),
    re.compile(r'\bit is prohibited\b', re.I),
    re.compile(r'\bit is forbidden\b', re.I),
    re.compile(r'\bit is mandatory\b', re.I),
    re.compile(r'\bit is required\b', re.I),
    re.compile(r'\bprohibited to\b', re.I),
    re.compile(r'\bresponsible for compliance\b', re.I),
]

OBLIGATION_SIGNAL_PATTERNS_DE = [
    re.compile(r'\bhaben sicherzustellen\b', re.I),
    re.compile(r'\bhat sicherzustellen\b', re.I),
    re.compile(r'\bist sicherzustellen\b', re.I),
    re.compile(r'\bhaben dafür zu sorgen\b', re.I),
    re.compile(r'\bhat dafür zu sorgen\b', re.I),
    re.compile(r'\bsorgt dafür, dass\b', re.I),
    re.compile(r'\bes ist verboten\b', re.I),
    re.compile(r'\bes ist untersagt\b', re.I),
    re.compile(r'\bes besteht eine Pflicht\b', re.I),
    re.compile(r'\bes ist erforderlich,\s*dass\b', re.I),
]

OBLIGATION_SIGNAL_PATTERNS_GENERIC = [
    re.compile(r'\bensure compliance\b', re.I),
    re.compile(r'\bverboten\b', re.I),
    re.compile(r'\bverpflichtet\b', re.I),
]

# ============================================================================
# Parallel Attribute Extractor
# ============================================================================

@dataclass
class ExtractionResult:
    """Ergebnis einer Attribut-Extraktion mit Confidence."""
    value: str
    confidence: float
    method: str  # 'pattern', 'heuristic', 'ml_model'


class ParallelAttributeExtractor:
    """
    Extrahiert ALLE Attribute parallel aus einem Segment.
    Verhindert Fehlermultiplikation durch sequentielle Verarbeitung.
    """
    
    def __init__(self):
        self.patterns = ExtractionPatterns()
    
    # ------------------------------------------------------------------------
    # Modality Detection
    # ------------------------------------------------------------------------
    
    def extract_modality(self, text: str, language: str) -> ExtractionResult:
        """Erkennt Modalität (shall/must/may/etc.)"""
        text_lower = text.lower()
        
        patterns = (
            self.patterns.MODALITY_EN if language == 'en' 
            else self.patterns.MODALITY_DE
        )
        
        # Prüfe in Reihenfolge: shall_not vor shall (wegen Teilstring)
        for modality in ['shall_not', 'must_not', 'shall', 'must', 'may', 'should']:
            if modality in patterns:
                for pattern in patterns[modality]:
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        confidence = 0.9 if modality in ['shall', 'must', 'shall_not', 'must_not'] else 0.7
                        return ExtractionResult(modality, confidence, 'pattern')
        
        return ExtractionResult('unspecified', 0.3, 'default')
    
    # ------------------------------------------------------------------------
    # Addressee Detection
    # ------------------------------------------------------------------------
    
    def extract_addressee(self, text: str, requirement_class: str) -> ExtractionResult:
        """Erkennt Addressaten (undertaking/member_state/etc.)"""
        text_lower = text.lower()
        
        # Cross-Validation mit requirement_class
        if requirement_class == "requirement_undertaking":
            # Schaue zuerst nach Unternehmen-spezifischen Begriffen
            priority_order = ['undertaking', 'manufacturer', 'operator', 'auditor']
        elif requirement_class == "requirement_member_state":
            priority_order = ['member_state', 'authority']
        else:
            priority_order = list(self.patterns.ADDRESSEE_PATTERNS.keys())
        
        for addressee in priority_order:
            patterns = self.patterns.ADDRESSEE_PATTERNS.get(addressee, [])
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    # Höhere Confidence bei Übereinstimmung mit requirement_class
                    confidence = 0.95 if addressee in priority_order[:2] else 0.85
                    return ExtractionResult(addressee, confidence, 'pattern')
        
        # Fallback: Nutze requirement_class
        if requirement_class == "requirement_undertaking":
            return ExtractionResult('undertaking', 0.6, 'heuristic_from_class')
        elif requirement_class == "requirement_member_state":
            return ExtractionResult('member_state', 0.6, 'heuristic_from_class')
        
        return ExtractionResult('unspecified', 0.3, 'default')
    
    # ------------------------------------------------------------------------
    # Topic Detection
    # ------------------------------------------------------------------------
    
    def extract_topic(self, text: str, structure_type: str) -> ExtractionResult:
        """Erkennt Themenbereich (reporting/product/etc.)"""
        text_lower = text.lower()
        
        # Recitals sind meist non-requirements
        if structure_type == 'recital':
            return ExtractionResult('definition', 0.8, 'heuristic_structure')
        
        # Zähle Matches pro Topic
        topic_scores = {}
        for topic, patterns in self.patterns.TOPIC_PATTERNS.items():
            score = sum(1 for p in patterns if re.search(p, text_lower, re.IGNORECASE))
            if score > 0:
                topic_scores[topic] = score
        
        if topic_scores:
            best_topic = max(topic_scores, key=topic_scores.get)
            confidence = min(0.95, 0.6 + (topic_scores[best_topic] * 0.1))
            return ExtractionResult(best_topic, confidence, 'pattern')
        
        return ExtractionResult('other', 0.4, 'default')
    
    # ------------------------------------------------------------------------
    # Obligation Type Detection
    # ------------------------------------------------------------------------
    
    def extract_obligation_type(self, text: str, modality: str) -> ExtractionResult:
        """Erkennt Art der Verpflichtung (positive_duty/prohibition/etc.)"""
        text_lower = text.lower()
        
        # Prohibition
        if modality in ['shall_not', 'must_not']:
            return ExtractionResult('prohibition', 0.95, 'modality_based')
        
        for pattern in self.patterns.PROHIBITION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return ExtractionResult('prohibition', 0.9, 'pattern')
        
        # Exemption
        for pattern in self.patterns.EXEMPTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return ExtractionResult('exemption', 0.85, 'pattern')
        
        # Conditional
        for pattern in self.patterns.CONDITIONAL_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return ExtractionResult('conditional', 0.8, 'pattern')
        
        # Permission
        if modality in ['may', 'can']:
            return ExtractionResult('permission', 0.85, 'modality_based')
        
        # Reporting/Disclosure
        if re.search(r'\b(disclose|report|publish|provide information)\b', text_lower):
            duty_type = 'disclosure_duty' if 'disclose' in text_lower else 'reporting_duty'
            return ExtractionResult(duty_type, 0.88, 'pattern')
        
        # Default: positive_duty für shall/must
        if modality in ['shall', 'must']:
            return ExtractionResult('positive_duty', 0.75, 'modality_based')
        
        return ExtractionResult('unspecified', 0.4, 'default')
    
    # ------------------------------------------------------------------------
    # Scope Detection
    # ------------------------------------------------------------------------
    
    def extract_scope(self, text: str, topic: str, addressee: str) -> ExtractionResult:
        """Erkennt Anwendungsebene (entity/product/supply_chain/etc.)"""
        text_lower = text.lower()
        
        # Topic-basierte Heuristik
        if topic in ['product_design', 'labeling']:
            return ExtractionResult('product_level', 0.85, 'topic_based')
        
        if topic in ['due_diligence']:
            return ExtractionResult('supply_chain_level', 0.85, 'topic_based')
        
        if topic in ['reporting', 'governance']:
            return ExtractionResult('entity_level', 0.8, 'topic_based')
        
        # Pattern-basiert
        if re.search(r'\bproduct\b', text_lower):
            return ExtractionResult('product_level', 0.9, 'pattern')
        
        if re.search(r'\bsupply chain\b|\bLieferkette\b', text_lower):
            return ExtractionResult('supply_chain_level', 0.9, 'pattern')
        
        if re.search(r'\bprocess\b|\bVerfahren\b', text_lower):
            return ExtractionResult('process_level', 0.85, 'pattern')
        
        # Addressee-basiert
        if addressee == 'manufacturer':
            return ExtractionResult('product_level', 0.7, 'addressee_based')
        
        return ExtractionResult('entity_level', 0.6, 'default')
    
    # ------------------------------------------------------------------------
    # Product vs. Reporting Classification
    # ------------------------------------------------------------------------
    
    def classify_product_requirement(
        self, 
        text: str, 
        topic: str, 
        scope: str,
        addressee: str
    ) -> ExtractionResult:
        """
        Kritische Unterscheidung: Product-Requirement vs. Reporting-Requirement
        
        Product: Technische Design-Anforderungen an Produkte
        Reporting: Compliance/Disclosure-Anforderungen
        """
        text_lower = text.lower()
        
        # Starke Product-Indikatoren
        product_keywords = [
            'design', 'specification', 'performance', 'efficiency',
            'label', 'marking', 'capacity', 'durability', 'safety',
            'material', 'component', 'technical', 'shall contain',
            'shall be equipped', 'shall have'
        ]
        
        # Starke Reporting-Indikatoren
        reporting_keywords = [
            'disclose', 'report', 'publish', 'information', 'statement',
            'document', 'record', 'register', 'notify', 'submit'
        ]
        
        product_score = sum(1 for kw in product_keywords if kw in text_lower)
        reporting_score = sum(1 for kw in reporting_keywords if kw in text_lower)
        
        # Topic/Scope-basierte Entscheidung
        if topic in ['product_design', 'labeling', 'energy_efficiency', 'circular_economy']:
            is_product = True
            confidence = 0.9 + (product_score * 0.02)
        elif topic in ['reporting', 'transparency', 'disclosure']:
            is_product = False
            confidence = 0.9 + (reporting_score * 0.02)
        elif scope == 'product_level':
            is_product = True
            confidence = 0.85
        elif scope == 'entity_level':
            is_product = False
            confidence = 0.85
        else:
            # Keyword-Score entscheidet
            is_product = product_score > reporting_score
            confidence = 0.6 + (abs(product_score - reporting_score) * 0.05)
        
        confidence = min(0.99, confidence)
        return ExtractionResult(str(is_product), confidence, 'hybrid')
    
    # ------------------------------------------------------------------------
    # Main Extraction Method (PARALLEL)
    # ------------------------------------------------------------------------
    
    def extract_all_attributes(
        self,
        text: str,
        language: str,
        structure_type: str,
        requirement_class: str
    ) -> Dict[str, ExtractionResult]:
        """
        Extrahiert ALLE Attribute parallel.
        Verhindert Fehlerkaskaden durch sequentielle Verarbeitung.
        """
        
        # Step 1: Basisattribute (unabhängig)
        modality = self.extract_modality(text, language)
        topic = self.extract_topic(text, structure_type)
        
        # Step 2: Kontextabhängige Attribute
        addressee = self.extract_addressee(text, requirement_class)
        obligation_type = self.extract_obligation_type(text, modality.value)
        scope = self.extract_scope(text, topic.value, addressee.value)
        
        # Step 3: Product vs. Reporting
        is_product = self.classify_product_requirement(
            text, topic.value, scope.value, addressee.value
        )
        
        return {
            'modality': modality,
            'addressee': addressee,
            'topic': topic,
            'obligation_type': obligation_type,
            'scope': scope,
            'is_product_requirement': is_product
        }


# ============================================================================
# Multi-Class Requirement Classifier
# ============================================================================

class RequirementClassifier:
    """
    Klassifiziert Segmente in 4 Klassen statt binär.
    Nutzt Struktur-Metadaten + Modality + Addressee.
    """
    
    def __init__(self):
        self.extractor = ParallelAttributeExtractor()
    
    def _has_obligation_signal(self, text: str, language: str) -> bool:
        lang = (language or '').lower()
        patterns = list(OBLIGATION_SIGNAL_PATTERNS_GENERIC)
        if lang.startswith('de'):
            patterns += OBLIGATION_SIGNAL_PATTERNS_DE
        else:
            patterns += OBLIGATION_SIGNAL_PATTERNS_EN
        return any(pattern.search(text) for pattern in patterns)
    
    def classify(
        self,
        text: str,
        structure_type: str,
        language: str
    ) -> Tuple[RequirementClass, float]:
        """
        Multi-Class Classification: 4 Klassen
        
        Returns: (requirement_class, confidence)
        """
        
        obligation_signal = self._has_obligation_signal(text, language)

        # Recitals sind fast immer non_requirement (außer klaren Obligation-Signalen)
        if structure_type == 'recital' and not obligation_signal:
            return ('non_requirement', 0.95)
        
        # Definitionen sind non_requirement
        if re.search(r'\bmeans\b|\bfor the purposes of\b|\bDefinition\b', text, re.IGNORECASE):
            return ('non_requirement', 0.9)
        
        # Extrahiere Modality & Addressee
        modality = self.extractor.extract_modality(text, language)
        
        # Keine starke Modalität und kein Obligation-Signal → wahrscheinlich non_requirement
        if modality.value == 'unspecified' and not obligation_signal:
            return ('non_requirement', 0.7)
        
        # Provisorische Klassifikation für Addressee-Extraktion
        provisional_class = 'requirement_undertaking'  # Default
        addressee = self.extractor.extract_addressee(text, provisional_class)
        
        # Klassifizierung basierend auf Addressee
        if addressee.value in ['undertaking', 'manufacturer', 'operator']:
            return ('requirement_undertaking', 0.85 * addressee.confidence)
        
        elif addressee.value in ['member_state', 'authority']:
            return ('requirement_member_state', 0.85 * addressee.confidence)
        
        elif addressee.value in ['auditor', 'commission']:
            return ('requirement_other_actor', 0.8 * addressee.confidence)
        
        # Fallback: Hat Modalität aber unklarer Addressee
        if modality.value in ['shall', 'must', 'shall_not', 'must_not']:
            return ('requirement_undertaking', 0.6)

        # Obligation-Signal ohne klassisches Modalverb
        if obligation_signal:
            if addressee.value in ['member_state', 'authority']:
                return ('requirement_member_state', 0.55)
            elif addressee.value in ['auditor', 'commission']:
                return ('requirement_other_actor', 0.5)
            return ('requirement_undertaking', 0.5)
        
        return ('non_requirement', 0.5)


# ============================================================================
# Scope Labels Extractor (for ML Training)
# ============================================================================

class ScopeLabelsExtractor:
    """
    Extracts structured scope information for ML training.
    
    scope_labels = {
        "product_types": ["battery", "vehicle", "appliance"],
        "components": ["cell", "pack", "housing"],
        "materials": ["lithium", "cobalt", "plastic"],
        "processes": ["manufacturing", "assembly", "testing"],
        "thresholds": ["5kg", "2kWh", "80%"],
        "quantities": ["per unit", "annual", "quarterly"]
    }
    """
    
    # Product type patterns
    PRODUCT_TYPES = {
        'battery': r'\b(battery|batteries|portable battery|industrial battery|electric vehicle battery)\b',
        'vehicle': r'\b(vehicle|car|automobile|truck|bus|motorcycle)\b',
        'appliance': r'\b(appliance|washing machine|dishwasher|refrigerator|dryer)\b',
        'packaging': r'\b(packaging|package|container)\b',
        'electronics': r'\b(electronics|device|equipment|apparatus)\b',
        'machinery': r'\b(machinery|machine|equipment)\b',
        'textile': r'\b(textile|fabric|clothing|garment)\b',
        'chemical': r'\b(chemical|substance|compound|mixture)\b'
    }
    
    # Component patterns
    COMPONENTS = {
        'cell': r'\b(cell|cells|battery cell)\b',
        'pack': r'\b(pack|battery pack|module)\b',
        'housing': r'\b(housing|casing|enclosure)\b',
        'electrode': r'\b(electrode|cathode|anode)\b',
        'electrolyte': r'\b(electrolyte)\b',
        'separator': r'\b(separator)\b',
        'label': r'\b(label|marking|identifier)\b',
        'connector': r'\b(connector|terminal|connection)\b'
    }
    
    # Material patterns
    MATERIALS = {
        'lithium': r'\b(lithium|Li)\b',
        'cobalt': r'\b(cobalt|Co)\b',
        'nickel': r'\b(nickel|Ni)\b',
        'plastic': r'\b(plastic|polymer|PVC|polypropylene)\b',
        'metal': r'\b(metal|aluminum|steel|copper)\b',
        'recycled_material': r'\b(recycled|recyclate|secondary material)\b'
    }
    
    # Process patterns
    PROCESSES = {
        'manufacturing': r'\b(manufactur|produc|fabricat)\w*\b',
        'assembly': r'\b(assembl|construct)\w*\b',
        'testing': r'\b(test|verificat|validat)\w*\b',
        'recycling': r'\b(recycl|reprocess|recover)\w*\b',
        'disposal': r'\b(dispos|waste management)\w*\b',
        'transport': r'\b(transport|ship|deliver)\w*\b'
    }
    
    # Threshold/quantity patterns
    THRESHOLD_PATTERN = re.compile(r'\b(\d+(?:[.,]\d+)?)\s*(kg|g|kWh|Wh|%|percent|mm|cm|m|tonnes|units?)\b', re.IGNORECASE)
    QUANTITY_PATTERN = re.compile(r'\b(per\s+(?:unit|product|year|quarter|month)|annual|quarterly|monthly|daily)\b', re.IGNORECASE)
    
    def extract(self, text: str) -> Dict[str, List[str]]:
        """Extract scope labels from text."""
        text_lower = text.lower()
        
        scope_labels = {
            "product_types": [],
            "components": [],
            "materials": [],
            "processes": [],
            "thresholds": [],
            "quantities": []
        }
        
        # Extract product types
        for product_type, pattern in self.PRODUCT_TYPES.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                scope_labels["product_types"].append(product_type)
        
        # Extract components
        for component, pattern in self.COMPONENTS.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                scope_labels["components"].append(component)
        
        # Extract materials
        for material, pattern in self.MATERIALS.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                scope_labels["materials"].append(material)
        
        # Extract processes
        for process, pattern in self.PROCESSES.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                scope_labels["processes"].append(process)
        
        # Extract thresholds
        threshold_matches = self.THRESHOLD_PATTERN.findall(text)
        scope_labels["thresholds"] = [f"{val}{unit}" for val, unit in threshold_matches]
        
        # Extract quantities
        quantity_matches = self.QUANTITY_PATTERN.findall(text)
        scope_labels["quantities"] = list(set(quantity_matches))
        
        return scope_labels


# ============================================================================
# Action Labels Extractor (for ML Training)
# ============================================================================

class ActionLabelsExtractor:
    """
    Extracts structured action information for ML training.
    
    action_labels = {
        "action": "ensure",
        "actor": "manufacturer",
        "deadline": "2025-01-01",
        "document": "technical documentation",
        "references": ["Article 13", "Annex II"]
    }
    """
    
    # Action verb patterns (English and German)
    ACTION_VERBS = {
        'ensure': r'\b(ensure|ensur\w+|gewährleist\w+|sicherstell\w+)\b',
        'provide': r'\b(provide|provid\w+|bereitstell\w+)\b',
        'maintain': r'\b(maintain|maintaining|aufrechterhalten|pflegen)\b',
        'establish': r'\b(establish\w*|einricht\w+|schaf+\w+)\b',
        'implement': r'\b(implement\w*|umset\w+|durchführ\w+)\b',
        'disclose': r'\b(disclose|disclosing|offenleg\w+|angeb\w+)\b',
        'report': r'\b(report\w*|bericht\w+|meld\w+)\b',
        'publish': r'\b(publish\w*|veröffentlich\w+)\b',
        'notify': r'\b(notify|notif\w+|benachrichtig\w+|unterricht\w+)\b',
        'assess': r'\b(assess\w*|bewert\w+|prüf\w+)\b',
        'monitor': r'\b(monitor\w*|überwach\w+|kontrollier\w+)\b',
        'comply': r'\b(comply|complying|einhal\w+|entspr\w+)\b',
        'demonstrate': r'\b(demonstrate|demonstrat\w+|nachweis\w+)\b',
        'verify': r'\b(verif\w+|überprüf\w+|nachprüf\w+)\b'
    }
    
    # Actor patterns
    ACTORS = {
        'manufacturer': r'\b(manufacturer|producer|hersteller|produzent)\b',
        'operator': r'\b(operator|betreiber)\b',
        'undertaking': r'\b(undertaking|enterprise|unternehmen|wirtschaftsteilnehmer)\b',
        'auditor': r'\b(auditor|prüfer|wirtschaftsprüfer)\b',
        'authority': r'\b(authority|behörde|aufsichtsbehörde)\b',
        'member_state': r'\b(member state|mitgliedstaat)\b',
        'commission': r'\b(commission|kommission)\b'
    }
    
    # Document type patterns
    DOCUMENTS = {
        'technical_documentation': r'\b(technical documentation|technische dokumentation)\b',
        'report': r'\b(report|bericht)\b',
        'statement': r'\b(statement|erklärung)\b',
        'certificate': r'\b(certificate|bescheinigung|zertifikat)\b',
        'assessment': r'\b(assessment|bewertung)\b',
        'declaration': r'\b(declaration|deklaration)\b',
        'information': r'\b(information|informationen)\b'
    }
    
    # Date/deadline patterns
    DEADLINE_PATTERN = re.compile(
        r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s+\d{4}|\d{4}-\d{2}-\d{2}|by\s+\d{4}|until\s+\d{4}|bis\s+\d{4})\b',
        re.IGNORECASE
    )
    
    # Article reference patterns
    REFERENCE_PATTERN = re.compile(
        r'\b(Article|Art\.|Artikel|Art\.?)\s+(\d+[a-z]?(?:\(\d+\))?)|Annex\s+([IVX]+)|Anhang\s+([IVX]+)',
        re.IGNORECASE
    )
    
    def extract(self, text: str) -> Dict[str, any]:
        """Extract action labels from text."""
        text_lower = text.lower()
        
        action_labels = {
            "action": None,
            "actor": None,
            "deadline": None,
            "document": None,
            "references": []
        }
        
        # Extract action verb (first match wins)
        for action, pattern in self.ACTION_VERBS.items():
            if re.search(pattern, text, re.IGNORECASE):
                action_labels["action"] = action
                break
        
        # Extract actor (first match wins)
        for actor, pattern in self.ACTORS.items():
            if re.search(pattern, text, re.IGNORECASE):
                action_labels["actor"] = actor
                break
        
        # Extract document type (first match wins)
        for doc_type, pattern in self.DOCUMENTS.items():
            if re.search(pattern, text, re.IGNORECASE):
                action_labels["document"] = doc_type
                break
        
        # Extract deadline
        deadline_matches = self.DEADLINE_PATTERN.findall(text)
        if deadline_matches:
            action_labels["deadline"] = deadline_matches[0]
        
        # Extract references
        reference_matches = self.REFERENCE_PATTERN.findall(text)
        references = []
        for match in reference_matches:
            if match[0] and match[1]:  # Article X
                references.append(f"Article {match[1]}")
            elif match[2]:  # Annex X
                references.append(f"Annex {match[2]}")
            elif match[3]:  # Anhang X
                references.append(f"Annex {match[3]}")
        action_labels["references"] = list(set(references))
        
        return action_labels


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    classifier = RequirementClassifier()
    extractor = ParallelAttributeExtractor()
    scope_extractor = ScopeLabelsExtractor()
    action_extractor = ActionLabelsExtractor()
    
    # Test 1: Product Requirement
    text1 = "Manufacturers shall ensure that batteries are accompanied by technical documentation indicating the capacity."
    req_class1, conf1 = classifier.classify(text1, 'article', 'en')
    attrs1 = extractor.extract_all_attributes(text1, 'en', 'article', req_class1)
    scope1 = scope_extractor.extract(text1)
    action1 = action_extractor.extract(text1)
    
    print("="*80)
    print("TEST 1: Product Requirement (Battery Regulation)")
    print(f"Class: {req_class1} (confidence: {conf1:.2f})")
    print(f"Modality: {attrs1['modality'].value} ({attrs1['modality'].confidence:.2f})")
    print(f"Addressee: {attrs1['addressee'].value} ({attrs1['addressee'].confidence:.2f})")
    print(f"Topic: {attrs1['topic'].value} ({attrs1['topic'].confidence:.2f})")
    print(f"Is Product: {attrs1['is_product_requirement'].value} ({attrs1['is_product_requirement'].confidence:.2f})")
    print(f"Scope Labels: {scope1}")
    print(f"Action Labels: {action1}")
    
    # Test 2: Reporting Requirement
    text2 = "Undertakings shall disclose information necessary to understand the company's sustainability impacts."
    req_class2, conf2 = classifier.classify(text2, 'article', 'en')
    attrs2 = extractor.extract_all_attributes(text2, 'en', 'article', req_class2)
    scope2 = scope_extractor.extract(text2)
    action2 = action_extractor.extract(text2)
    
    print("\n" + "="*80)
    print("TEST 2: Reporting Requirement (CSRD)")
    print(f"Class: {req_class2} (confidence: {conf2:.2f})")
    print(f"Modality: {attrs2['modality'].value} ({attrs2['modality'].confidence:.2f})")
    print(f"Addressee: {attrs2['addressee'].value} ({attrs2['addressee'].confidence:.2f})")
    print(f"Topic: {attrs2['topic'].value} ({attrs2['topic'].confidence:.2f})")
    print(f"Is Product: {attrs2['is_product_requirement'].value} ({attrs2['is_product_requirement'].confidence:.2f})")
    print(f"Scope Labels: {scope2}")
    print(f"Action Labels: {action2}")
    
    # Test 3: Non-Requirement (Recital)
    text3 = "(12) This Regulation lays down harmonised rules for batteries."
    req_class3, conf3 = classifier.classify(text3, 'recital', 'en')
    
    print("\n" + "="*80)
    print("TEST 3: Non-Requirement (Recital)")
    print(f"Class: {req_class3} (confidence: {conf3:.2f})")
