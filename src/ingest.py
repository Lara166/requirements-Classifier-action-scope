import json
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from langdetect import detect, detect_langs
import pypdfium2 as pdfium
from src.temporal_validator import TemporalValidator


def _read_pdf(path: Path) -> str:
    """Read PDF using pypdfium2."""
    text = []
    try:
        pdf = pdfium.PdfDocument(str(path))
        for i in range(len(pdf)):
            try:
                page = pdf[i]
                textpage = page.get_textpage()
                page_text = textpage.get_text_range()
                text.append(page_text)
                textpage.close()
                page.close()
            except Exception as e:
                print(f"[warn]   Page {i+1} failed: {e}")
                continue
        pdf.close()
    except Exception as e:
        print(f"[error] Failed to read {path.name}: {e}")
        return ""
    return "\n".join(text)


def _read_any(path: Path) -> str:
    return _read_pdf(path) if path.suffix.lower() == ".pdf" else path.read_text(encoding="utf-8", errors="ignore")


def _compute_hash(text: str) -> str:
    """Compute SHA256 hash of text for deduplication."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]



def _normalize_doc_id(stem: str) -> str:
    """Normalize filenames to stable doc_ids by enforcing underscores and lowercase."""
    slug = stem.strip()
    slug = slug.replace(" ", "_")
    slug = re.sub(r"-+", "_", slug)
    slug = re.sub(r"_+", "_", slug)
    slug = slug.strip("_")
    return slug.lower()


PAGE_NOISE_KEYWORDS = [
    "amtsblatt der europ",
    "official journal of the european union",
    "www.gesetze-im-internet.de",
    "ein service des bundesministeriums der justiz",
    "bundesamt für justiz",
    "bundesamt fuer justiz"
]

PAGE_NOISE_REGEX = [
    re.compile(r'^[A-Z]{1,3}\s*\d+/\d+\s*(?:[A-Z]{2})?$', re.I),
    re.compile(r'^\d+\s*/\s*\d+$'),
    re.compile(r'^page\s+\d+(\s+of\s+\d+)?$', re.I),
    re.compile(r'^\(\+\+\+.*\+\+\+\)$')
]


def _clean_page_artifacts(text: str) -> str:
    """
    Entfernt typische Seitenköpfe/-füße (Amtsblatt, gesetze-im-internet,
    reine Seitenzahlen), um die Chunk-Qualität zu erhöhen.
    """
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        normalized = stripped.lower()
        if stripped:
            if any(keyword in normalized for keyword in PAGE_NOISE_KEYWORDS):
                continue
            if any(pattern.match(stripped) for pattern in PAGE_NOISE_REGEX):
                continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)



def _detect_structure_type(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Heading-based detection that only inspects the first lines of a chunk."""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
        if len(lines) >= 5:
            break

    if not lines:
        return (None, None)

    def _match(pattern: str, flags=0):
        for line in lines:
            m = re.match(pattern, line, flags)
            if m:
                return m
        return None

    if m := _match(r'^(?:Article|Art\.?|Artikel)\s+(\d+[a-z]?(?:\s*[a-z])?)', re.I):
        return ('article', m.group(1).strip())
    if m := _match(r'^§+\s*(\d+[a-z]?)', re.I):
        return ('paragraph', m.group(1))
    if m := _match(r'^(?:Recital|Erwägungsgrund)\s*\(?(\d+)\)?', re.I):
        return ('recital', m.group(1))
    if m := _match(r'^\((\d+)\)\s+[A-ZÄÖÜ]', re.I):
        return ('recital', m.group(1))
    if m := _match(r'^(?:Annex|Anhang)\s+([IVX]+|[A-Z]|\d+)', re.I):
        return ('annex', m.group(1).upper())
    if m := _match(r'^(?:Section|Abschnitt)\s+(\d+[a-z]?)', re.I):
        return ('section', m.group(1))
    if m := _match(r'^(?:Chapter|Kapitel)\s+(\d+[a-z]?|[IVX]+)', re.I):
        return ('chapter', m.group(1).upper())
    return (None, None)


def _detect_language_detailed(text: str) -> Dict[str, any]:
    """
    Detaillierte Spracherkennung mit Probabilities.
    Returns: {'primary': 'en', 'probabilities': {'en': 0.95, 'de': 0.05}, 'is_mixed': False}
    """
    try:
        langs = detect_langs(text[:2000])
        probs = {lang.lang: lang.prob for lang in langs}
        primary = langs[0].lang if langs else 'unknown'
        
        # Mixed language wenn zwei Sprachen > 0.3 Wahrscheinlichkeit
        high_prob_langs = [l for l, p in probs.items() if p > 0.3]
        is_mixed = len(high_prob_langs) > 1
        
        return {
            'primary': primary,
            'probabilities': probs,
            'is_mixed': is_mixed
        }
    except Exception as e:
        return {
            'primary': 'unknown',
            'probabilities': {},
            'is_mixed': False
        }


def _chunk_with_offsets(text: str, min_chars: int, max_chars: int, overlap: int) -> List[Tuple[str, int, int]]:
    """
    Intelligentes Chunking mit Source-Offsets:
    - Dynamische Größe basierend auf Absatz-Median
    - Trennt an Struktur-Keywords (Article/§) wenn möglich
    - Absätze/Satzenden als Fallback
    - Erzeugt überlappende Chunks für Kontext
    - Speichert (chunk_text, char_start, char_end) für Traceability
    """
    # Versuch 1: An Absätzen trennen
    paras_with_pos = []
    last_end = 0
    for para in text.split("\n\n"):
        para = para.strip()
        if para:
            start = text.find(para, last_end)
            end = start + len(para)
            paras_with_pos.append((para, start, end))
            last_end = end
    
    # Falls keine Absätze: An Satzenden trennen
    if len(paras_with_pos) == 1 and len(paras_with_pos[0][0]) > max_chars:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        paras_with_pos = []
        last_end = 0
        for sent in sentences:
            sent = sent.strip()
            if sent:
                start = text.find(sent, last_end)
                end = start + len(sent)
                paras_with_pos.append((sent, start, end))
                last_end = end
    
    # Dynamische Min-Größe: 30% der Median-Absatzlänge
    if paras_with_pos:
        para_lengths = sorted([len(p[0]) for p in paras_with_pos])
        median_len = para_lengths[len(para_lengths) // 2]
        dynamic_min = max(min_chars, int(median_len * 0.3))
    else:
        dynamic_min = min_chars
    
    chunks_with_offsets = []
    buf = ""
    buf_start = 0
    buf_end = 0
    
    for para, p_start, p_end in paras_with_pos:
        # Prüfe ob Paragraph Struktur-Keyword enthält (Article/§/Annex)
        has_structure_keyword = bool(re.search(
            r'\b(Article|Artikel|Annex|Anhang|Section|Abschnitt|Chapter|Kapitel)\s+\w+|§\s*\d+',
            para[:100], re.I
        ))
        
        # Wenn Absatz allein zu groß ist, hart schneiden
        if len(para) > max_chars:
            if buf:
                chunks_with_offsets.append((buf, buf_start, buf_end))
                buf = ""
            # Hart-Schnitt mit Überlappung
            char_offset = p_start
            start = 0
            while start < len(para):
                end = min(start + max_chars, len(para))
                chunk_text = para[start:end]
                chunks_with_offsets.append((chunk_text, char_offset + start, char_offset + end))
                start = end - overlap if end < len(para) else end
        # Bei Struktur-Keyword: Neuen Chunk starten (bessere Kohärenz)
        elif has_structure_keyword and buf and len(buf) >= dynamic_min:
            chunks_with_offsets.append((buf, buf_start, buf_end))
            buf = para
            buf_start = p_start
            buf_end = p_end
        # Absatz passt in Buffer
        elif len(buf) + len(para) + 2 <= max_chars:
            if buf:
                buf = buf + "\n\n" + para
                buf_end = p_end
            else:
                buf = para
                buf_start = p_start
                buf_end = p_end
        # Buffer voll, neuen Chunk starten
        else:
            if buf:
                chunks_with_offsets.append((buf, buf_start, buf_end))
            buf = para
            buf_start = p_start
            buf_end = p_end
    
    if buf:
        chunks_with_offsets.append((buf, buf_start, buf_end))
    
    # Nachbearbeitung: Zu kleine Chunks zusammenführen
    merged = []
    cur_text = ""
    cur_start = 0
    cur_end = 0
    
    for chunk_text, c_start, c_end in chunks_with_offsets:
        if len(cur_text) < dynamic_min:
            if cur_text:
                cur_text = cur_text + "\n\n" + chunk_text
                cur_end = c_end
            else:
                cur_text = chunk_text
                cur_start = c_start
                cur_end = c_end
        else:
            merged.append((cur_text, cur_start, cur_end))
            cur_text = chunk_text
            cur_start = c_start
            cur_end = c_end
    
    if cur_text:
        merged.append((cur_text, cur_start, cur_end))
    
    return merged if merged else chunks_with_offsets


def run_ingest(cfg) -> int:
    inp = Path(cfg.ingest.input_dir)
    outp = Path(cfg.ingest.processed_out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    
    # Initialisiere Temporal Validator
    validator = TemporalValidator()
    
    # Hash-Cache für Duplikatsvermeidung
    cache_file = outp.parent / ".ingest_cache.json"
    hash_cache = {}
    if cache_file.exists():
        try:
            with cache_file.open('r', encoding='utf-8') as f:
                hash_cache = json.load(f)
            print(f"[info] Loaded cache with {len(hash_cache)} entries")
        except Exception as e:
            print(f"[warn] Could not load cache: {e}")
    
    print(f"[info] Scanning {inp} for PDF/TXT files...")
    new_cache = {}
    
    with outp.open("w", encoding="utf-8") as w:
        for fp in sorted(inp.rglob("*")):
            if not fp.is_file() or fp.suffix.lower() not in {".pdf", ".txt"}:
                continue
            
            print(f"[info] Processing {fp.name}...")
            doc_id = _normalize_doc_id(fp.stem)
            raw_text = _read_any(fp)
            if not raw_text.strip():
                print(f"[warn] kein Text extrahiert: {fp.name}")
                continue
            raw = _clean_page_artifacts(raw_text)
            if not raw.strip():
                print(f"[warn] Dokument nach Cleanup leer: {fp.name}")
                continue
            
            # Hash-basierte Duplikatserkennung
            doc_hash = _compute_hash(raw)
            cache_key = f"{doc_id}#{doc_hash}"
            
            if cache_key in hash_cache:
                print(f"[info]   ⚡ Skipping (identical to cached version)")
                # Kopiere gecachte Segmente
                cached_segments = hash_cache[cache_key].get('segments', [])
                for seg in cached_segments:
                    w.write(json.dumps(seg, ensure_ascii=False) + "\n")
                    count += 1
                new_cache[cache_key] = hash_cache[cache_key]
                continue
            
            # Temporal Validation auf Dokumentebene
            doc_metadata = {'doc_id': doc_id, 'filename': fp.name}
            try:
                validity_info = validator.validate_document(raw_text, doc_metadata)
                print(f"[info]   → Validity: {validity_info['validity_status']}")
                if validity_info['notes']:
                    for note in validity_info['notes']:
                        print(f"[warn]   📝 {note}")
            except Exception as e:
                print(f"[warn]   Temporal validation failed: {e}")
                validity_info = {'validity_status': 'unknown', 'notes': []}
            
            # Detaillierte Spracherkennung
            lang_info = _detect_language_detailed(raw)
            
            # Chunking mit Offsets
            chunks_with_offsets = _chunk_with_offsets(
                raw, 
                cfg.ingest.min_section_chars, 
                cfg.ingest.max_section_chars, 
                cfg.ingest.overlap_chars
            )
            print(f"[info]   → {len(chunks_with_offsets)} chunks created")
            
            cached_segments = []
            
            last_structure_type = None
            last_structure_label = None

            for i, (chunk_text, char_start, char_end) in enumerate(chunks_with_offsets):
                # Struktur-Erkennung
                structure_type, structure_label = _detect_structure_type(chunk_text)
                if structure_type:
                    last_structure_type = structure_type
                    last_structure_label = structure_label
                else:
                    if last_structure_type:
                        structure_type = last_structure_type
                        structure_label = last_structure_label
                    else:
                        structure_type = 'preamble'
                        structure_label = None
                        last_structure_type = structure_type
                        last_structure_label = structure_label

                # Semantische Section-ID
                if structure_type and structure_label:
                    section_id = f"{doc_id}#{structure_type}-{structure_label}-chunk-{i:04d}"
                else:
                    section_id = f"{doc_id}#chunk-{i:04d}"
                
                rec = {
                    "doc_id": doc_id,
                    "section_id": section_id,
                    "doc_hash": doc_hash,
                    
                    # Sprache (erweitert)
                    "language": lang_info['primary'],
                    "language_probabilities": lang_info['probabilities'],
                    "is_mixed_language": lang_info['is_mixed'],
                    
                    # Text & Offsets
                    "text": chunk_text,
                    "char_start": char_start,
                    "char_end": char_end,
                    "chunk_index": i,
                    
                    # Struktur-Metadaten
                    "structure_type": structure_type,
                    "structure_label": structure_label,
                    
                    # Temporal Validation
                    "validity_status": validity_info['validity_status'],
                    "last_amendment_date": validity_info.get('last_amendment_date'),
                    "amendment_count": validity_info.get('amendment_count', 0)
                }
                
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cached_segments.append(rec)
                count += 1
            
            # Cache aktualisieren
            new_cache[cache_key] = {
                'doc_id': doc_id,
                'hash': doc_hash,
                'processed_at': datetime.now().isoformat(),
                'segments': cached_segments
            }
    
    # Cache speichern
    try:
        with cache_file.open('w', encoding='utf-8') as f:
            json.dump(new_cache, f, ensure_ascii=False, indent=2)
        print(f"[info] Cache updated with {len(new_cache)} documents")
    except Exception as e:
        print(f"[warn] Could not save cache: {e}")
    
    print(f"[info] Total segments: {count}")
    return count
