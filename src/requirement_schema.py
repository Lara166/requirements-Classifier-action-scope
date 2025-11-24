"""
Requirement Schema Definition
=============================

Vollständiges, wissenschaftlich fundiertes Schema für strukturierte Requirements.
Verhindert inkonsistente Extraktion und ermöglicht saubere Gap-Analysis.
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field
from datetime import datetime


# ============================================================================
# Enumerations (kontrollierte Vokabulare)
# ============================================================================

RequirementClass = Literal[
    "requirement_undertaking",      # Pflicht für Unternehmen
    "requirement_member_state",     # Pflicht für Mitgliedstaaten
    "requirement_other_actor",      # Pflicht für Dritte (Auditor, Behörde)
    "non_requirement"               # Kein Requirement (Recital, Definition, etc.)
]

Addressee = Literal[
    "undertaking",                  # Unternehmen/Organisation
    "member_state",                 # EU-Mitgliedstaat
    "auditor",                      # Wirtschaftsprüfer/Auditor
    "authority",                    # Behörde/Regulator
    "manufacturer",                 # Hersteller
    "operator",                     # Betreiber
    "commission",                   # EU-Kommission
    "multiple",                     # Mehrere Addressaten
    "unspecified"                   # Nicht eindeutig
]

Modality = Literal[
    "shall",                        # Starke Verpflichtung
    "must",                         # Starke Verpflichtung (DE)
    "shall_not",                    # Verbot
    "must_not",                     # Verbot (DE)
    "may",                          # Erlaubnis/Option
    "should",                       # Empfehlung
    "can",                          # Möglichkeit
    "is_required_to",               # Implizite Verpflichtung
    "is_obliged_to",                # Implizite Verpflichtung
    "is_prohibited",                # Implizites Verbot
    "unspecified"                   # Keine klare Modalität
]

Topic = Literal[
    "reporting",                    # Berichterstattung/Disclosure
    "product_design",               # Produktanforderungen/Eco-Design
    "due_diligence",                # Sorgfaltspflichten/Supply Chain
    "governance",                   # Unternehmensführung/Management
    "environmental",                # Umweltschutz/Emissionen
    "social",                       # Soziale Verantwortung
    "labeling",                     # Kennzeichnung/Labeling
    "monitoring",                   # Überwachung/Kontrolle
    "enforcement",                  # Durchsetzung/Sanktionen
    "transparency",                 # Transparenz/Offenlegung
    "circular_economy",             # Kreislaufwirtschaft
    "energy_efficiency",            # Energieeffizienz
    "procedural",                   # Verfahren/Prozess
    "transitional",                 # Übergangsregelung
    "definition",                   # Definition/Begriffsbestimmung
    "other"                         # Sonstiges
]

ObligationType = Literal[
    "positive_duty",                # Aktive Handlungspflicht ("shall do X")
    "prohibition",                  # Verbot ("shall not do X")
    "permission",                   # Erlaubnis ("may do X")
    "conditional",                  # Bedingte Pflicht ("shall do X if Y")
    "exemption",                    # Ausnahme ("except when...")
    "delegation",                   # Übertragung ("Member States shall ensure...")
    "reporting_duty",               # Berichtspflicht
    "disclosure_duty",              # Offenlegungspflicht
    "unspecified"
]

Scope = Literal[
    "entity_level",                 # Unternehmensebene
    "product_level",                # Produktebene
    "supply_chain_level",           # Lieferkettenebene
    "process_level",                # Prozessebene
    "transaction_level",            # Transaktionsebene
    "sector_level",                 # Branchenebene
    "unspecified"
]

StructureType = Literal[
    "article",
    "recital",
    "annex",
    "section",
    "chapter",
    "paragraph",
    "definition",
    "unspecified"
]


# ============================================================================
# Main Requirement Model
# ============================================================================

class StructuredRequirement(BaseModel):
    """
    Vollständiges, strukturiertes Requirement-Objekt.
    
    Wissenschaftlich fundiert: Trennt Klassifikation, Attribute und Metadaten.
    Verhindert Fehlermultiplikation durch parallele Extraktion.
    """
    
    # -------------------------------------------------------------------------
    # Identifikation & Herkunft
    # -------------------------------------------------------------------------
    requirement_id: str = Field(
        description="Eindeutige ID: {law_id}#{article}#{chunk_index}"
    )
    
    doc_id: str = Field(
        description="Quelldokument-ID"
    )
    
    law_name: str = Field(
        description="Gesetzesname (z.B. CSRD, BatteryReg, KSG)"
    )
    
    celex_number: Optional[str] = Field(
        default=None,
        description="CELEX-Nummer für EU-Recht"
    )
    
    article_number: Optional[str] = Field(
        default=None,
        description="Artikelnummer (z.B. '19a', '5', 'Annex II')"
    )
    
    structure_type: StructureType = Field(
        description="Strukturtyp (article, recital, annex, etc.)"
    )
    
    # -------------------------------------------------------------------------
    # Text & Offsets
    # -------------------------------------------------------------------------
    text: str = Field(
        description="Original-Segmenttext"
    )
    
    char_start: int = Field(
        description="Start-Offset im Originaldokument"
    )
    
    char_end: int = Field(
        description="End-Offset im Originaldokument"
    )
    
    language: str = Field(
        description="Sprache (de/en)"
    )
    
    # -------------------------------------------------------------------------
    # CLASSIFICATION (Multi-Class)
    # -------------------------------------------------------------------------
    requirement_class: RequirementClass = Field(
        description="Hauptklassifikation (4 Klassen statt binär)"
    )
    
    classification_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Konfidenz der Klassifikation (0.0-1.0)"
    )
    
    # -------------------------------------------------------------------------
    # PARALLEL EXTRACTED ATTRIBUTES
    # -------------------------------------------------------------------------
    addressee: Addressee = Field(
        description="An wen richtet sich die Pflicht?"
    )
    
    addressee_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Konfidenz der Addressee-Extraktion"
    )
    
    modality: Modality = Field(
        description="Modalität (shall/must/may/etc.)"
    )
    
    modality_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Konfidenz der Modalitäts-Extraktion"
    )
    
    topic: Topic = Field(
        description="Themenbereich"
    )
    
    topic_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Konfidenz der Topic-Klassifikation"
    )
    
    obligation_type: ObligationType = Field(
        description="Art der Verpflichtung"
    )
    
    scope: Scope = Field(
        description="Anwendungsebene"
    )
    
    # -------------------------------------------------------------------------
    # PRODUCT vs. REPORTING
    # -------------------------------------------------------------------------
    is_product_requirement: bool = Field(
        description="Technisches Produkt-Requirement (true) vs. Compliance/Reporting (false)"
    )
    
    product_relevance_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Konfidenz der Product-Klassifikation"
    )
    
    # -------------------------------------------------------------------------
    # Zusätzliche Metadaten
    # -------------------------------------------------------------------------
    contains_exception: bool = Field(
        default=False,
        description="Enthält Ausnahmeregel ('unless', 'except')"
    )
    
    contains_definition: bool = Field(
        default=False,
        description="Enthält Definition ('means', 'for the purposes of')"
    )
    
    is_conditional: bool = Field(
        default=False,
        description="Ist bedingt ('if', 'where', 'when')"
    )
    
    references_other_articles: List[str] = Field(
        default_factory=list,
        description="Verweise auf andere Artikel"
    )
    
    # -------------------------------------------------------------------------
    # Temporal Validation
    # -------------------------------------------------------------------------
    validity_status: str = Field(
        description="active, amended, superseded, repealed"
    )
    
    last_amendment_date: Optional[str] = Field(
        default=None,
        description="Datum der letzten Änderung"
    )
    
    # -------------------------------------------------------------------------
    # Processing Metadata
    # -------------------------------------------------------------------------
    extracted_at: datetime = Field(
        default_factory=datetime.now,
        description="Zeitpunkt der Extraktion"
    )
    
    extraction_method: str = Field(
        default="rule_based",
        description="Extraktionsmethode (rule_based, ml_model, llm, hybrid)"
    )
    
    needs_manual_review: bool = Field(
        default=False,
        description="Flag für manuelle Überprüfung (bei niedriger Confidence)"
    )
    
    reviewer_notes: Optional[str] = Field(
        default=None,
        description="Notizen vom manuellen Review"
    )


# ============================================================================
# Beispiel-Validierung
# ============================================================================

if __name__ == "__main__":
    # Beispiel: Produkt-Requirement aus Battery Regulation
    example_product = StructuredRequirement(
        requirement_id="BatteryReg2023#Art-13#0042",
        doc_id="Battery_Regulation_2023_CELEX-32023R1542_EN",
        law_name="Battery Regulation",
        celex_number="32023R1542",
        article_number="13",
        structure_type="article",
        text="Manufacturers shall ensure that batteries are accompanied by technical documentation...",
        char_start=15420,
        char_end=15680,
        language="en",
        requirement_class="requirement_undertaking",
        classification_confidence=0.95,
        addressee="manufacturer",
        addressee_confidence=0.98,
        modality="shall",
        modality_confidence=0.99,
        topic="product_design",
        topic_confidence=0.92,
        obligation_type="positive_duty",
        scope="product_level",
        is_product_requirement=True,
        product_relevance_confidence=0.97,
        contains_exception=False,
        contains_definition=False,
        is_conditional=False,
        validity_status="active"
    )
    
    print("✅ Product Requirement Example:")
    print(example_product.model_dump_json(indent=2))
    
    # Beispiel: Reporting-Requirement aus CSRD
    example_reporting = StructuredRequirement(
        requirement_id="CSRD2022#Art-19a#0123",
        doc_id="CSRD_Corporate_Sustainability_Reporting_Directive_2022_EN",
        law_name="CSRD",
        celex_number="32022L2464",
        article_number="19a",
        structure_type="article",
        text="Undertakings shall include in the management report information necessary to understand...",
        char_start=45200,
        char_end=45580,
        language="en",
        requirement_class="requirement_undertaking",
        classification_confidence=0.93,
        addressee="undertaking",
        addressee_confidence=0.96,
        modality="shall",
        modality_confidence=0.99,
        topic="reporting",
        topic_confidence=0.98,
        obligation_type="disclosure_duty",
        scope="entity_level",
        is_product_requirement=False,
        product_relevance_confidence=0.88,
        contains_exception=False,
        contains_definition=False,
        is_conditional=False,
        validity_status="active"
    )
    
    print("\n✅ Reporting Requirement Example:")
    print(example_reporting.model_dump_json(indent=2))
