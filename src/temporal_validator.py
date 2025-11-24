"""
Temporal Validation für Regulierungstexte.
Prüft Gültigkeitsstatus basierend auf Textmustern.
"""
import re
from datetime import datetime
from typing import Dict, Optional


class TemporalValidator:
    """Validiert den zeitlichen Gültigkeitsstatus von Regulierungstexten."""
    
    def __init__(self):
        # Muster für Statuserkennung (Deutsch)
        self.patterns_de = {
            'aufgehoben': [
                r'(?:dieses?\s+Gesetz|diese\s+Verordnung)\s+(?:wurde?|ist)\s+aufgehoben',
                r'außer\s+Kraft\s+getreten',
                r'tritt.*außer\s+Kraft'
            ],
            'ersetzt': [
                r'ersetzt\s+durch',
                r'abgelöst\s+durch',
                r'wird\s+ersetzt'
            ],
            'geändert': [
                r'zuletzt\s+(?:ge)?ändert\s+durch',
                r'geändert\s+durch\s+(?:Gesetz|Verordnung|Artikel)',
                r'Änderung\s+vom\s+\d{1,2}\.\d{1,2}\.\d{4}',
                r'neugefasst\s+durch'
            ]
        }
        
        # Muster für Statuserkennung (Englisch)
        self.patterns_en = {
            'repealed': [
                r'(?:this|the)\s+(?:regulation|directive|act)\s+(?:is|was)\s+repealed',
                r'ceased\s+to\s+be\s+in\s+force',
                r'no\s+longer\s+in\s+force'
            ],
            'replaced': [
                r'replaced\s+by',
                r'superseded\s+by',
                r'succeeded\s+by'
            ],
            'amended': [
                r'(?:as\s+)?amended\s+by',
                r'last\s+amended\s+by',
                r'amendment\s+of\s+\d{1,2}[./]\d{1,2}[./]\d{4}'
            ]
        }
        
        # Datumsmuster
        self.date_patterns = [
            r'\d{1,2}\.\d{1,2}\.\d{4}',  # DD.MM.YYYY
            r'\d{4}-\d{2}-\d{2}',        # YYYY-MM-DD
            r'\d{1,2}/\d{1,2}/\d{4}'     # DD/MM/YYYY
        ]
    
    def validate_document(self, text: str, doc_metadata: Optional[Dict] = None) -> Dict:
        """
        Validiert ein Dokument und gibt Statusinformationen zurück.
        
        Args:
            text: Volltext des Dokuments
            doc_metadata: Optional metadata (doc_id, etc.)
        
        Returns:
            Dict mit validity_status, last_amendment_date, notes
        """
        # Standardannahme
        result = {
            'validity_status': 'active',
            'last_amendment_date': None,
            'amendment_count': 0,
            'notes': []
        }
        
        # Spracherkennung (vereinfacht)
        is_german = bool(re.search(r'\b(?:Gesetz|Verordnung|Absatz|Artikel)\b', text[:5000]))
        patterns = self.patterns_de if is_german else self.patterns_en
        
        # Text auf ersten 10000 Zeichen beschränken (Header/Präambel)
        check_text = text[:10000].lower()
        
        # Prüfe auf Aufhebung
        for pattern in patterns.get('aufgehoben' if is_german else 'repealed', []):
            if re.search(pattern, check_text, re.IGNORECASE):
                result['validity_status'] = 'repealed'
                result['notes'].append('Dokument wurde aufgehoben' if is_german else 'Document was repealed')
                break
        
        # Prüfe auf Ersetzung
        if result['validity_status'] == 'active':
            for pattern in patterns.get('ersetzt' if is_german else 'replaced', []):
                if re.search(pattern, check_text, re.IGNORECASE):
                    result['validity_status'] = 'superseded'
                    result['notes'].append('Dokument wurde ersetzt' if is_german else 'Document was superseded')
                    break
        
        # Prüfe auf Änderungen
        if result['validity_status'] == 'active':
            amendment_matches = []
            for pattern in patterns.get('geändert' if is_german else 'amended', []):
                matches = re.finditer(pattern, check_text, re.IGNORECASE)
                amendment_matches.extend(matches)
            
            if amendment_matches:
                result['validity_status'] = 'amended'
                result['amendment_count'] = len(amendment_matches)
                
                # Versuche letztes Änderungsdatum zu finden
                dates = self._extract_dates(text[:10000])
                if dates:
                    result['last_amendment_date'] = dates[-1]  # Letztes gefundenes Datum
                    result['notes'].append(f'{len(amendment_matches)} Änderung(en) gefunden - neueste vom {dates[-1]}')
                else:
                    result['notes'].append(f'{len(amendment_matches)} Änderung(en) gefunden')
        
        return result
    
    def _extract_dates(self, text: str) -> list:
        """Extrahiert Datumsangaben aus Text."""
        dates = []
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text)
            dates.extend(matches)
        return sorted(set(dates))  # Eindeutige Daten, sortiert
    
    def validate_segment(self, segment_text: str, doc_validity: Dict) -> Dict:
        """
        Validiert ein Textsegment im Kontext des Gesamtdokuments.
        
        Args:
            segment_text: Text des Segments
            doc_validity: Validitätsstatus des Gesamtdokuments
        
        Returns:
            Dict mit Segment-spezifischen Validitätsinformationen
        """
        # Segment erbt grundsätzlich Status des Dokuments
        result = {
            'validity_status': doc_validity['validity_status'],
            'inherits_from_document': True
        }
        
        # Prüfe auf segmentspezifische Übergangsbestimmungen
        if re.search(r'\b(?:Übergangsbestimmung|transitional provision)\b', segment_text, re.IGNORECASE):
            result['notes'] = ['Segment enthält Übergangsbestimmungen']
            result['requires_temporal_check'] = True
        
        return result
