# Detailed Pipeline Documentation

Complete step-by-step explanation of the requirement extraction pipeline with examples.

---

## 📋 Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Step 1: PDF Ingestion & Segmentation](#step-1-pdf-ingestion--segmentation)
3. [Step 2: Labeling Keywords & Criteria](#step-2-labeling-keywords--criteria)
4. [Step 3: ML Classification](#step-3-ml-classification)
5. [Step 4: Action Label Extraction](#step-4-action-label-extraction)
6. [Step 5: Scope Label Extraction](#step-5-scope-label-extraction)
7. [Complete Example Output](#complete-example-output)

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT: EU Regulation PDFs                   │
│           (Battery Regulation, REACH, RoHS, etc.)              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: PDF Ingestion & Intelligent Segmentation              │
│  Tool: ingest.py                                                │
│  Output: JSONL segments with metadata                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Manual Labeling (Training Data Creation)              │
│  Keywords: "shall", "must", "muss", "verpflichtet"             │
│  Output: labeled_train/validation/test.jsonl                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: ML Classification (XLM-RoBERTa-Large)                 │
│  Binary: requirement_undertaking / non_requirement              │
│  Output: Class label + confidence score                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Action Label Extraction (Rule-Based)                  │
│  Extracts: Actor, Action, Deadline, Document, References       │
│  Keywords: "shall ensure", "must provide", "submit by"         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: Scope Label Extraction (Rule-Based)                   │
│  Extracts: Products, Materials, Components, Thresholds         │
│  Keywords: "battery", "lithium", "waste", ">1000 tonnes"      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              OUTPUT: Structured Requirements JSON               │
│        {text, class, confidence, action_labels, scope_labels}  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: PDF Ingestion & Segmentation

### Implementation

**File:** `ingest.py`

**Libraries Used:**
- PyMuPDF (fitz) - Primary PDF reader
- pypdfium2 - Fallback option
- PyPDF2 - Secondary fallback

### Segmentation Strategy

**Goal:** Split PDF into coherent text chunks (500-2000 characters) that preserve semantic boundaries.

**Algorithm:**

```python
def _chunk(text: str, min_chars: int = 500, max_chars: int = 2000, overlap: int = 200):
    """
    Intelligent chunking with 3-step strategy:
    1. Try splitting at paragraph boundaries (\n\n)
    2. If no paragraphs: Split at sentence boundaries (. ! ?)
    3. If necessary: Hard cut with 200-char overlap for context
    """
```

**Why This Approach?**
- **Paragraph-first:** Preserves natural article/section boundaries
- **Sentence fallback:** Maintains grammatical completeness
- **Overlap:** Ensures requirements spanning chunk boundaries aren't lost
- **Size limits:** Fits transformer model context windows (512 tokens)

### Metadata Extraction

**Extracted Fields:**
- `doc_id`: Document identifier (e.g., `Battery_Regulation_2023_CELEX-32023R1542_EN`)
- `requirement_id`: Unique segment ID (e.g., `Battery#Article-10#0001`)
- `article_number`: Parsed from text (e.g., "Article 10", "§ 15")
- `structure_type`: `article`, `recital`, `annex`, `section`, `other`
- `language`: Auto-detected (`en`, `de`, `fr`)
- `char_start`, `char_end`: Position in original PDF

### Detection Patterns

**Article Detection:**
```python
if re.match(r'^Article\s+\d+', text_start, re.IGNORECASE):
    return 'article'
```

**Section Number Extraction:**
```python
# German style: § 15a
match = re.match(r'^§\s*(\d+[a-z]*)', text_start)

# EU style: Article 10
match = re.match(r'^Article\s+([\da-z]+)', text_start, re.IGNORECASE)
```

### Example Output (Segment)

```json
{
  "requirement_id": "Battery#Article-10#0001",
  "doc_id": "Battery_Regulation_2023_CELEX-32023R1542_EN",
  "law_name": "Battery",
  "celex_number": "32023R1542",
  "article_number": "10",
  "structure_type": "article",
  "text": "Article 10\nManufacturers shall ensure that batteries are designed and manufactured in such a way that they can be readily removed and replaced by the end-user at any time during their lifetime.",
  "char_start": 15420,
  "char_end": 15608,
  "language": "en"
}
```

---

## Step 2: Labeling Keywords & Criteria

### Manual Labeling Process

**Goal:** Create training data for ML classifier by identifying requirements vs. non-requirements.

### Classification Schema

**Binary Classes:**
1. **`requirement_undertaking`** - Obligations for companies/economic operators
2. **`non_requirement`** - Definitions, recitals, background information

### Labeling Keywords (Requirement Indicators)

**English:**
- **Strong obligation:** `shall`, `must`, `is required to`, `is obliged to`
- **Prohibition:** `shall not`, `must not`, `may not`, `is prohibited`
- **Permission:** `may`, `is allowed to`, `can`
- **Recommendation:** `should`, `is recommended`

**German:**
- **Strong obligation:** `muss`, `müssen`, `hat zu`, `sind verpflichtet`, `ist verpflichtet`
- **Prohibition:** `dürfen nicht`, `darf nicht`, `ist untersagt`, `ist verboten`
- **Permission:** `dürfen`, `darf`, `können`, `kann`
- **Recommendation:** `sollten`, `sollte`, `ist empfohlen`

### Addressee Detection

**Criteria for `requirement_undertaking`:**

```python
# Must contain addressee + obligation
if ('undertaking' in text or 'manufacturer' in text or 'operator' in text):
    if ('shall' in text or 'must' in text):
        label = 'requirement_undertaking'
```

**Keywords:**
- `the undertaking shall`
- `manufacturers shall`
- `economic operators must`
- `companies are required to`
- `Unternehmen müssen`
- `Hersteller haben sicherzustellen`

### Non-Requirement Indicators

**Definition patterns:**
- `means`, `for the purposes of`, `shall be understood as`
- `bezeichnet`, `bedeutet`, `versteht man`

**Recital markers:**
- Starts with `(1)`, `(2)`, etc.
- Contains background/justification language
- Past tense (`was`, `were`, `wurden`)

### Labeling Statistics

| Dataset | Total Samples | Requirements | Non-Requirements |
|---------|---------------|--------------|------------------|
| Train | 2,410 | 1,627 (67.5%) | 783 (32.5%) |
| Validation | 986 | 664 (67.3%) | 322 (32.7%) |
| Test | 2,934 | 1,440 (49.1%) | 1,494 (50.9%) |

### Example Labeled Segments

**Requirement Example:**
```json
{
  "text": "Economic operators shall ensure that batteries placed on the market are accompanied by a technical documentation demonstrating conformity with the requirements set out in Article 7.",
  "requirement_class": "requirement_undertaking",
  "addressee": "undertaking",
  "modality": "shall",
  "contains_obligation": true
}
```

**Non-Requirement Example:**
```json
{
  "text": "For the purposes of this Regulation, 'battery' means any source of electrical energy generated by direct conversion of chemical energy.",
  "requirement_class": "non_requirement",
  "contains_definition": true,
  "contains_obligation": false
}
```

---

## Step 3: ML Classification

### Model Architecture

**Model:** XLM-RoBERTa-Large (xlm-roberta-large)
- **Parameters:** 560 million
- **Size:** 2.1 GB (FP16 precision)
- **Input:** Max 512 tokens
- **Output:** Binary classification + confidence score

### Training Configuration

**Hyperparameters:**
```python
learning_rate = 2e-5
batch_size = 8
epochs = 3
warmup_steps = 500
weight_decay = 0.01
optimizer = AdamW
```

**Hardware:**
- Platform: Google Colab Pro
- GPU: NVIDIA A100 (80GB VRAM)
- Training Time: ~45 minutes

### Inference Process

```python
# 1. Tokenize input
inputs = tokenizer(
    text,
    max_length=512,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

# 2. Forward pass
with torch.no_grad():
    outputs = model(**inputs.to(device))
    logits = outputs.logits

# 3. Get prediction
probabilities = torch.softmax(logits, dim=-1)
predicted_class = torch.argmax(probabilities, dim=-1)
confidence = probabilities[0][predicted_class].item()
```

### Performance Metrics

**Test Set Results (2,934 samples):**
- **Accuracy:** 87.1%
- **Precision:** 80.4%
- **Recall:** 97.3% ← Critical for compliance!
- **F1 Score:** 88.1%

**Why High Recall Matters:**
- Missing a requirement (false negative) = compliance risk
- Flagging a non-requirement (false positive) = extra review time
- Trade-off: Optimized for recall (97.3%) to minimize compliance gaps

### Example Classification Output

```json
{
  "text": "Economic operators shall ensure that batteries placed on the market contain information on their composition.",
  "predicted_class": "requirement_undertaking",
  "confidence": 0.9823,
  "model": "xlm-roberta-large",
  "inference_time_ms": 145
}
```

---

## Step 4: Action Label Extraction

### Rule-Based Extraction

**File:** `src/requirement_extractor.py`
**Class:** `ActionLabelsExtractor`

### Extracted Fields

1. **Actor** - Who must act? (`manufacturer`, `commission`, `undertaking`)
2. **Action** - What must be done? (`ensure`, `provide`, `submit`, `report`)
3. **Deadline** - When? (`by 1 January 2025`, `within 6 months`, `annually`)
4. **Document** - What document? (`technical documentation`, `declaration of conformity`)
5. **References** - Legal references (`Article 7`, `Annex III`)

### Extraction Patterns

**Actor Patterns (English):**
```python
ACTOR_PATTERNS = [
    r'\b(manufacturer|manufacturers)\b',
    r'\b(economic operator|operators)\b',
    r'\b(undertaking|undertakings)\b',
    r'\b(the Commission)\b',
    r'\b(Member States?)\b',
    r'\b(distributor|distributors)\b'
]
```

**Actor Patterns (German):**
```python
ACTOR_PATTERNS_DE = [
    r'\b(Hersteller)\b',
    r'\b(Wirtschaftsakteur|Wirtschaftsakteure)\b',
    r'\b(Unternehmen)\b',
    r'\b(die Kommission)\b',
    r'\b(Mitgliedstaat|Mitgliedstaaten)\b'
]
```

**Action Patterns (English):**
```python
ACTION_PATTERNS = [
    r'\bshall (ensure|provide|submit|make available|place on the market)\b',
    r'\bmust (maintain|keep|demonstrate|verify)\b',
    r'\bis required to (assess|evaluate|implement)\b'
]
```

**Action Patterns (German):**
```python
ACTION_PATTERNS_DE = [
    r'\bmuss (sicherstellen|bereitstellen|vorlegen|nachweisen)\b',
    r'\bhat zu (gewährleisten|dokumentieren)\b',
    r'\bist verpflichtet (zu prüfen|zu bewerten)\b'
]
```

**Deadline Patterns:**
```python
DEADLINE_PATTERNS = [
    r'\bby (\d{1,2} \w+ \d{4})\b',  # by 1 January 2025
    r'\bwithin (\d+) (days?|months?|years?)\b',  # within 6 months
    r'\b(annually|monthly|quarterly)\b',
    r'\bbis zum (\d{1,2}\. \w+ \d{4})\b',  # German: bis zum 1. Januar 2025
    r'\binnerhalb (von )?\d+ (Tagen|Monaten|Jahren)\b'
]
```

**Reference Patterns:**
```python
REFERENCE_PATTERNS = [
    r'\bArticle \d+[a-z]?\b',
    r'\bAnnex [IVX]+\b',
    r'\b§ ?\d+[a-z]?\b',
    r'\bArtikel \d+\b'
]
```

### Example Extraction

**Input:**
```
"Economic operators shall ensure that batteries placed on the market are accompanied by a technical documentation demonstrating conformity with the requirements set out in Article 7."
```

**Output:**
```json
{
  "action_labels": {
    "actor": "economic operators",
    "action": "ensure",
    "deadline": null,
    "document": "technical documentation",
    "references": ["Article 7"]
  }
}
```

---

## Step 5: Scope Label Extraction

### Rule-Based Extraction

**File:** `src/requirement_extractor.py`
**Class:** `ScopeLabelsExtractor`

### Extracted Fields

1. **Product Types** - `battery`, `vehicle`, `industrial equipment`
2. **Materials** - `lithium`, `cobalt`, `lead`, `mercury`
3. **Components** - `cathode`, `anode`, `electrolyte`, `separator`
4. **Processes** - `manufacturing`, `recycling`, `disposal`, `treatment`
5. **Thresholds** - `>1000 tonnes`, `≥5%`, `<0.01% by weight`
6. **Quantities** - `per year`, `per unit`, `per kilogram`

### Product Type Patterns

**Battery-specific:**
```python
PRODUCT_TYPES = [
    r'\bportable batter(y|ies)\b',
    r'\bindustrial batter(y|ies)\b',
    r'\belectric vehicle batter(y|ies)\b',
    r'\bLMT batter(y|ies)\b',  # Light Means of Transport
    r'\bSLI batter(y|ies)\b',  # Starting, Lighting, Ignition
    r'\brechargeable batter(y|ies)\b'
]
```

**German:**
```python
PRODUCT_TYPES_DE = [
    r'\bGerätebatterie(n)?\b',
    r'\bIndustriebatterie(n)?\b',
    r'\bFahrzeugbatterie(n)?\b',
    r'\bwiederaufladbare Batterie(n)?\b'
]
```

### Material Patterns

```python
MATERIALS = [
    r'\blithium\b',
    r'\bcobalt\b',
    r'\bnickel\b',
    r'\bmanganese\b',
    r'\blead\b',
    r'\bcadmium\b',
    r'\bmercury\b',
    r'\bgraphite\b'
]
```

### Threshold Patterns

```python
THRESHOLD_PATTERNS = [
    r'(?:>|≥|more than|at least)\s*(\d+(?:[.,]\d+)?)\s*(%|percent|kg|tonnes?|g)',
    r'(?:<|≤|less than|below)\s*(\d+(?:[.,]\d+)?)\s*(%|percent|kg|tonnes?|g)',
    r'(\d+(?:[.,]\d+)?)\s*(%|percent)\s*by weight'
]
```

### Example Extraction

**Input:**
```
"Portable batteries containing more than 0.002% mercury or 0.004% cadmium shall be labeled accordingly and collected separately."
```

**Output:**
```json
{
  "scope_labels": {
    "product_types": ["portable batteries"],
    "materials": ["mercury", "cadmium"],
    "components": [],
    "processes": ["collected"],
    "thresholds": [
      ">0.002% mercury",
      ">0.004% cadmium"
    ],
    "quantities": []
  }
}
```

---

## Complete Example Output

### End-to-End Pipeline Result

**Input Text:**
```
"Article 10 - Design for Removability

Economic operators shall ensure that portable batteries incorporated into appliances are readily removable and replaceable by the end-user at any time during the lifetime of the appliance. The technical documentation referred to in Article 18 shall include information on the removability and replaceability of the battery."
```

### Complete JSON Output

```json
{
  "requirement_id": "Battery#Article-10#0001",
  "doc_id": "Battery_Regulation_2023_CELEX-32023R1542_EN",
  "law_name": "Battery Regulation",
  "celex_number": "32023R1542",
  "article_number": "10",
  "structure_type": "article",
  
  "text": "Economic operators shall ensure that portable batteries incorporated into appliances are readily removable and replaceable by the end-user at any time during the lifetime of the appliance. The technical documentation referred to in Article 18 shall include information on the removability and replaceability of the battery.",
  
  "char_start": 15420,
  "char_end": 15730,
  "language": "en",
  
  "requirement_class": "requirement_undertaking",
  "classification_confidence": 0.9823,
  "model_used": "xlm-roberta-large",
  
  "action_labels": {
    "actor": "economic operators",
    "action": "ensure",
    "deadline": null,
    "document": "technical documentation",
    "references": ["Article 18"]
  },
  
  "scope_labels": {
    "product_types": ["portable batteries", "appliances"],
    "materials": [],
    "components": [],
    "processes": ["removable", "replaceable"],
    "thresholds": [],
    "quantities": []
  },
  
  "extracted_at": "2025-11-24T14:23:15.123456",
  "extraction_method": "hybrid_ml_rule_based"
}
```

### Another Example: Threshold Requirement

**Input Text:**
```
"Industrial batteries with a capacity greater than 2 kWh shall be equipped with a Battery Management System (BMS) and shall have a minimum energy round-trip efficiency of 89% for systems up to 4 hours of discharge duration."
```

**Output:**
```json
{
  "requirement_id": "Battery#Article-7#0003",
  "doc_id": "Battery_Regulation_2023_CELEX-32023R1542_EN",
  "law_name": "Battery Regulation",
  "celex_number": "32023R1542",
  "article_number": "7",
  "structure_type": "article",
  
  "text": "Industrial batteries with a capacity greater than 2 kWh shall be equipped with a Battery Management System (BMS) and shall have a minimum energy round-trip efficiency of 89% for systems up to 4 hours of discharge duration.",
  
  "language": "en",
  
  "requirement_class": "requirement_undertaking",
  "classification_confidence": 0.9645,
  "model_used": "xlm-roberta-large",
  
  "action_labels": {
    "actor": "manufacturer",
    "action": "equip",
    "deadline": null,
    "document": null,
    "references": []
  },
  
  "scope_labels": {
    "product_types": ["industrial batteries"],
    "materials": [],
    "components": ["Battery Management System", "BMS"],
    "processes": [],
    "thresholds": [
      ">2 kWh capacity",
      "≥89% efficiency",
      "up to 4 hours discharge"
    ],
    "quantities": ["2 kWh", "89%", "4 hours"]
  },
  
  "extracted_at": "2025-11-24T14:25:42.789012",
  "extraction_method": "hybrid_ml_rule_based"
}
```

---

## Summary Statistics

### Processing Performance

| Metric | Value |
|--------|-------|
| **Documents Processed** | 12 PDFs (train/val/test) |
| **Total Segments** | 6,330 |
| **Avg Segment Length** | 850 characters |
| **Segmentation Time** | ~2 minutes (all PDFs) |
| **Classification Speed** | 145 ms/segment (GPU) |
| **Extraction Speed** | 5 ms/segment (rule-based) |
| **Total Pipeline Time** | ~15 minutes (6,330 segments) |

### Output Files

| File | Size | Samples | Format |
|------|------|---------|--------|
| `segments_train.jsonl` | 2.6 MB | 2,410 | JSONL |
| `segments_validation.jsonl` | 2.6 MB | 986 | JSONL |
| `segments_test.jsonl` | 7.5 MB | 2,934 | JSONL |
| `classifier_results.json` | 1.2 MB | 2,934 | JSON |

---

**Pipeline Version:** 1.0  
**Last Updated:** November 24, 2025  
**Thesis Project:** ML-Based Requirement Extraction from EU Regulations
