"""
Enhanced ingest.py with Extended Metadata Extraction
====================================================

Improvements:
1. Parse regulation info from filename (type, level, year, CELEX)
2. Detect section type from text (recital, article, section)
3. Extract section number
4. Detect addressee (undertaking, member_state, etc.)
5. Detect modality (shall, must, may, should)
6. Detect linguistic patterns (obligation, exception, definition)

Usage:
    python ingest.py --split train
    python ingest.py --split validation
    python ingest.py --split test
"""

import json
import re
from pathlib import Path
from langdetect import detect
import argparse

# Try multiple PDF libraries for robustness
try:
    import pypdfium2 as pdfium
    PYPDFIUM_AVAILABLE = True
except ImportError:
    PYPDFIUM_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


def _read_pdf_pypdfium(path: Path) -> str:
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
        raise Exception(f"pypdfium2 failed: {e}")
    return "\n".join(text)


def _read_pdf_pypdf2(path: Path) -> str:
    """Read PDF using PyPDF2 (fallback)."""
    text = []
    try:
        with open(path, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            for page_num in range(len(pdf.pages)):
                try:
                    page = pdf.pages[page_num]
                    page_text = page.extract_text()
                    text.append(page_text)
                except Exception as e:
                    print(f"[warn]   Page {page_num+1} failed: {e}")
                    continue
    except Exception as e:
        raise Exception(f"PyPDF2 failed: {e}")
    return "\n".join(text)


def _read_pdf_pymupdf(path: Path) -> str:
    """Read PDF using PyMuPDF (most robust)."""
    text = []
    try:
        doc = fitz.open(str(path))
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                page_text = page.get_text()
                text.append(page_text)
            except Exception as e:
                print(f"[warn]   Page {page_num+1} failed: {e}")
                continue
        doc.close()
    except Exception as e:
        raise Exception(f"PyMuPDF failed: {e}")
    return "\n".join(text)


def _read_pdf(path: Path) -> str:
    """Read PDF using available libraries (PyMuPDF first, then pypdfium2, then PyPDF2)."""
    
    # Try PyMuPDF first (most robust)
    if PYMUPDF_AVAILABLE:
        try:
            return _read_pdf_pymupdf(path)
        except Exception as e:
            print(f"[warn] PyMuPDF failed for {path.name}: {e}")
    
    # Try pypdfium2 second
    if PYPDFIUM_AVAILABLE:
        try:
            return _read_pdf_pypdfium(path)
        except Exception as e:
            print(f"[warn] pypdfium2 failed for {path.name}: {e}")
    
    # Fallback to PyPDF2
    if PYPDF2_AVAILABLE:
        try:
            return _read_pdf_pypdf2(path)
        except Exception as e:
            print(f"[warn] PyPDF2 failed for {path.name}: {e}")
    
    print(f"[error] No PDF library could read {path.name}")
    return ""


def _read_any(path: Path) -> str:
    return _read_pdf(path) if path.suffix.lower() == ".pdf" else path.read_text(encoding="utf-8", errors="ignore")


def _chunk(text: str, min_chars: int = 500, max_chars: int = 2000, overlap: int = 200):
    """
    Intelligentes Chunking:
    - Versucht zuerst an Absätzen zu trennen (\n\n)
    - Falls keine Absätze: Trennt an Satzenden (. ! ?)
    - Erzeugt überlappende Chunks für Kontext
    """
    # Versuch 1: An Absätzen trennen
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    # Falls keine klaren Absätze: An Satzenden trennen
    if len(paras) == 1 and len(paras[0]) > max_chars:
        # Teile an Satzenden (. ? ! gefolgt von Leerzeichen oder Zeilenumbruch)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        paras = sentences
    
    chunks = []
    buf = ""
    
    for p in paras:
        # Wenn Absatz allein zu groß ist, hart schneiden
        if len(p) > max_chars:
            if buf:
                chunks.append(buf)
                buf = ""
            # Hart-Schnitt mit Überlappung
            start = 0
            while start < len(p):
                end = min(start + max_chars, len(p))
                chunks.append(p[start:end])
                start = end - overlap if end < len(p) else end
        # Absatz passt in Buffer
        elif len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p) if buf else p
        # Buffer voll, neuen Chunk starten
        else:
            if buf:
                chunks.append(buf)
            buf = p
    
    if buf:
        chunks.append(buf)
    
    # Nachbearbeitung: Zu kleine Chunks zusammenführen
    merged = []
    cur = ""
    for c in chunks:
        if len(cur) < min_chars:
            cur = (cur + "\n\n" + c) if cur else c
        else:
            merged.append(cur)
            cur = c
    if cur:
        merged.append(cur)
    
    return merged if merged else chunks


# ============================================================================
# METADATA EXTRACTION - New functionality
# ============================================================================

def parse_regulation_info(doc_id: str) -> dict:
    """
    Extract regulation metadata from filename
    
    Examples:
        CSRD_Corporate_Sustainability_Reporting_Directive_2022_CELEX-32022L2464_EN
        LkSG_Lieferkettensorgfaltspflichtengesetz_2021_DE
    """
    parts = doc_id.split('_')
    
    # Extract regulation name (first part)
    regulation_name = parts[0]
    
    # Extract CELEX number if present
    celex_number = None
    for part in parts:
        if 'CELEX-' in part:
            celex_number = part.replace('CELEX-', '')
            break
    
    # Extract year (4-digit number)
    year = None
    for part in parts:
        if part.isdigit() and len(part) == 4:
            year = int(part)
            break
    
    # Determine regulation level
    regulation_level = 'eu' if celex_number else 'national'
    
    # Determine regulation type based on name
    regulation_type = classify_regulation_type(regulation_name)
    
    return {
        'regulation_name': regulation_name,
        'regulation_type': regulation_type,
        'regulation_level': regulation_level,
        'regulation_year': year,
        'celex_number': celex_number
    }


def classify_regulation_type(regulation_name: str) -> str:
    """Classify regulation by domain"""
    name_lower = regulation_name.lower()
    
    if name_lower in ['csrd', 'nfrd', 'csr']:
        return 'reporting'
    elif name_lower in ['csddd', 'lksg']:
        return 'due_diligence'
    elif name_lower in ['sfdr']:
        return 'finance'
    elif name_lower in ['battery', 'batt', 'battg', 'weee', 'elektrog']:
        return 'product'
    elif name_lower in ['energy', 'renewable', 'geg', 'kwkg']:
        return 'energy'
    elif name_lower in ['waste', 'krwg', 'verpackg', 'plastics']:
        return 'circular_economy'
    elif name_lower in ['cbam', 'behg', 'ksg']:
        return 'climate'
    elif name_lower in ['taxonomy']:
        return 'taxonomy'
    elif name_lower in ['conflict', 'minerals']:
        return 'supply_chain'
    elif name_lower in ['chemg', 'bimschg']:
        return 'environment'
    else:
        return 'other'


def detect_section_type(text: str) -> str:
    """Detect the type of legal segment"""
    text_start = text[:100].strip()
    
    # Recital (EU style)
    if re.match(r'^\(\d+\)', text_start):
        return 'recital'
    
    # Article (EU style)
    if re.match(r'^Article\s+\d+', text_start, re.IGNORECASE):
        return 'article'
    
    # Section (German style)
    if re.match(r'^§\s*\d+', text_start):
        return 'section'
    
    # Annex
    if re.match(r'^ANNEX|^Anlage', text_start, re.IGNORECASE):
        return 'annex'
    
    # Chapter/Title
    if any(text_start.upper().startswith(x) for x in ['CHAPTER', 'TITLE', 'KAPITEL', 'TITEL']):
        return 'header'
    
    # Reference (legal gazette)
    if re.search(r'^OJ [A-Z]|^\(\d+\)\s*OJ|^BGBl\.', text_start):
        return 'reference'
    
    return 'other'


def extract_section_number(text: str, section_type: str) -> str:
    """Extract section number from text"""
    text_start = text[:100].strip()
    
    if section_type == 'recital':
        match = re.match(r'^\((\d+)\)', text_start)
        return match.group(1) if match else None
    
    elif section_type == 'article':
        match = re.match(r'^Article\s+([\da-z]+)', text_start, re.IGNORECASE)
        return match.group(1) if match else None
    
    elif section_type == 'section':
        match = re.match(r'^§\s*(\d+[a-z]*)', text_start)
        return match.group(1) if match else None
    
    return None


def detect_addressee(text: str) -> str:
    """Detect who the requirement addresses"""
    text_lower = text.lower()
    
    # Priority order (most specific first)
    if any(x in text_lower for x in ['the undertaking', 'undertakings shall', 'companies shall', 'unternehmen']):
        return 'undertaking'
    if any(x in text_lower for x in ['member states shall', 'mitgliedstaaten']):
        return 'member_state'
    if any(x in text_lower for x in ['the auditor', 'auditors shall', 'prüfer']):
        return 'auditor'
    if any(x in text_lower for x in ['the operator', 'operators shall', 'betreiber']):
        return 'operator'
    if any(x in text_lower for x in ['the commission shall', 'kommission']):
        return 'commission'
    
    return None


def detect_modality(text: str) -> str:
    """Detect modal verb (strength of obligation)"""
    text_lower = text.lower()
    
    # Strong obligation
    if any(x in text_lower for x in [' shall ', ' must ', ' required to ', ' obliged to ', ' muss ', ' verpflichtet ']):
        return 'shall'
    
    # Prohibition
    if any(x in text_lower for x in [' shall not ', ' must not ', ' prohibited ', ' verboten ', ' darf nicht ']):
        return 'shall_not'
    
    # Permission
    if any(x in text_lower for x in [' may ', ' kann ', ' darf ']):
        return 'may'
    
    # Recommendation
    if any(x in text_lower for x in [' should ', ' encouraged to ', ' sollte ']):
        return 'should'
    
    return None


def detect_obligation(text: str) -> bool:
    """Check if text contains obligation language"""
    text_lower = text.lower()
    obligation_markers = [
        'shall', 'must', 'required', 'obliged', 'obligation', 'mandatory',
        'muss', 'verpflichtet', 'pflicht', 'erforderlich', 'hat sicherzustellen'
    ]
    return any(marker in text_lower for marker in obligation_markers)


def detect_exception(text: str) -> bool:
    """Check if text contains exception language"""
    text_lower = text.lower()
    exception_markers = [
        'unless', 'except when', 'by way of derogation', 'notwithstanding',
        'sofern nicht', 'ausgenommen', 'abweichend', 'es sei denn'
    ]
    return any(marker in text_lower for marker in exception_markers)


def detect_definition(text: str) -> bool:
    """Check if text is a definition"""
    text_lower = text.lower()[:200]  # Check first 200 chars
    definition_markers = [
        'means', 'for the purposes of', 'shall be understood as',
        'bezeichnet', 'bedeutet', 'versteht man', 'im sinne dieses'
    ]
    return any(marker in text_lower for marker in definition_markers)


def extract_metadata(segment: dict) -> dict:
    """Extract all metadata from a segment"""
    text = segment['text']
    doc_id = segment['doc_id']
    
    # Parse regulation info from filename
    regulation_info = parse_regulation_info(doc_id)
    
    # Detect section type and number
    section_type = detect_section_type(text)
    section_number = extract_section_number(text, section_type)
    
    # Detect addressee and modality
    addressee = detect_addressee(text)
    modality = detect_modality(text)
    
    # Detect linguistic patterns
    contains_obligation = detect_obligation(text)
    contains_exception = detect_exception(text)
    contains_definition = detect_definition(text)
    
    return {
        **segment,
        **regulation_info,
        'section_type': section_type,
        'section_number': section_number,
        'addressee': addressee,
        'modality': modality,
        'contains_obligation': contains_obligation,
        'contains_exception': contains_exception,
        'contains_definition': contains_definition
    }


# ============================================================================
# MAIN INGEST FUNCTION
# ============================================================================

def run_ingest(input_dir: str = None, output_file: str = None) -> int:
    """
    Main ingest function with metadata extraction
    
    Args:
        input_dir: Directory with PDFs (default: data/raw/)
        output_file: Output JSONL file (default: data/processed/segments.jsonl)
    
    Returns:
        Number of segments created
    """
    if input_dir is None:
        input_dir = "data/raw"
    if output_file is None:
        output_file = "data/processed/segments.jsonl"
    
    inp = Path(input_dir)
    outp = Path(output_file)
    outp.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    print(f"\n{'='*80}")
    print(f"INGESTING PDFs from {inp}")
    print(f"{'='*80}\n")
    
    with outp.open("w", encoding="utf-8") as w:
        for fp in sorted(inp.rglob("*.pdf")):
            if not fp.is_file():
                continue
            
            print(f"[info] Processing {fp.name}...")
            
            # Read PDF
            raw = _read_any(fp)
            if not raw.strip():
                print(f"[warn] No text extracted: {fp.name}")
                continue
            
            # Detect language
            try:
                lang = detect(raw[:2000])
            except Exception:
                lang = "unknown"
            
            # Create chunks
            chunks = _chunk(raw, min_chars=500, max_chars=2000, overlap=200)
            print(f"[info]   -> {len(chunks)} chunks created")
            
            # Process each chunk
            for i, chunk_text in enumerate(chunks):
                # Base segment
                segment = {
                    "doc_id": fp.stem,
                    "section_id": f"{fp.stem}#chunk-{i:04d}",
                    "language": lang,
                    "text": chunk_text
                }
                
                # Extract metadata
                segment_with_metadata = extract_metadata(segment)
                
                # Write to file
                w.write(json.dumps(segment_with_metadata, ensure_ascii=False) + "\n")
                count += 1
            
            print(f"[info]   Language: {lang}")
            print()
    
    print(f"\n{'='*80}")
    print(f"INGEST COMPLETE")
    print(f"{'='*80}")
    print(f"Total segments: {count}")
    print(f"Output: {outp}")
    print()
    
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ingest PDFs with metadata extraction')
    parser.add_argument('--input', type=str, default='data/raw',
                        help='Input directory with PDFs')
    parser.add_argument('--output', type=str, default='data/processed/segments.jsonl',
                        help='Output JSONL file')
    args = parser.parse_args()
    
    # Run
    run_ingest(args.input, args.output)
