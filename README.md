# ML-Based Requirement Extraction from EU Regulations

Master Thesis Project: Automated extraction of compliance requirements from EU regulatory texts using Machine Learning.

## 🎯 Project Overview

This project implements a **hybrid ML/rule-based pipeline** for extracting structured requirements from EU regulations (Battery Regulation, REACH, RoHS). The system classifies text segments as requirements and extracts key information like actors, actions, deadlines, and scope.

### Key Results

| Component | Method | Performance |
|-----------|--------|-------------|
| **Requirement Classifier** | ML (XLM-RoBERTa-Large) | **F1: 88.1%** (Test Set) |
| **Baseline (Rule-Based)** | Pattern Matching | F1: 11.8% |
| **Improvement** | ML over Baseline | **+647%** |
| **Action Extraction** | Rule-Based | Functional |
| **Scope Extraction** | Rule-Based | Functional |

---

## 📁 Project Structure

```
sa2_v2/
├── README.md                          # This file
├── THESIS_RESULTS.md                  # Detailed metrics & analysis
├── PIPELINE_DETAILS.md                # Step-by-step pipeline explanation
├── ingest.py                          # Main data ingestion
├── hybrid_pipeline.py                 # End-to-end ML pipeline
├── rule_based_classifier.py           # Baseline classifier
│
├── colab_scripts/                     # Google Colab training
│   ├── train_extractors_colab.py
│   ├── evaluate_extractors_colab.py
│   └── colab_evaluation_script.py
│
├── src/                               # Core pipeline
│   ├── requirement_extractor.py       # Rule-based extractors
│   ├── requirement_schema.py
│   ├── requirement_pipeline.py
│   └── temporal_validator.py
│
├── data/
│   ├── raw/                           # PDFs (train/val/test)
│   └── processed/                     # JSONL files
│
├── outputs/
│   ├── labeled_train.jsonl            # 2,410 samples
│   ├── labeled_validation.jsonl       # 986 samples
│   ├── labeled_test.jsonl             # 2,934 samples
│   └── *_results.json                 # Evaluation results
│
├── models/
│   └── requirement_classifier/        # XLM-RoBERTa (2.1 GB)
│
└── configs/
    └── config.yaml
```

---

## 🚀 Quick Start

### Installation

```bash
pip install torch transformers scikit-learn pydantic pyyaml
```

### Run Demo

```bash
python hybrid_pipeline.py
```

### Evaluate Baseline

```bash
python rule_based_classifier.py
```

---

## 📊 Results Summary

### ML Classifier (XLM-RoBERTa-Large)

**Test Set (2,934 samples):**
- Accuracy: 87.1%
- Precision: 80.4%
- Recall: 97.3%
- **F1 Score: 88.1%**

**Validation Set (986 samples):**
- F1 Score: 90.3%

### Rule-Based Baseline

**Test Set:**
- F1 Score: 11.8%
- **ML Improvement: +647%**

### Training Details

- Model: XLM-RoBERTa-Large (560M parameters)
- Training Samples: 2,410
- Platform: Google Colab (A100 GPU)
- Training Time: ~45 minutes

---

## 🏗️ Architecture

For detailed step-by-step explanation with examples, see **[PIPELINE_DETAILS.md](PIPELINE_DETAILS.md)**.

```
PDF Input (data/raw/*.pdf)
  ↓
Ingestion & Chunking (ingest.py)
  • Intelligent segmentation (500-2000 chars)
  • Paragraph/sentence boundary detection
  • Metadata extraction (article #, type, language)
  ↓
JSONL Segments (data/processed/segments.jsonl)
  • 6,330 labeled segments
  • Fields: text, doc_id, article_number, structure_type
  ↓
ML Classifier (XLM-RoBERTa-Large)
  models/requirement_classifier/
  • Binary: requirement_undertaking / non_requirement
  • Input: 512 tokens max
  • Output: class + confidence (0-1)
  ├─→ Non-Requirement → Skip
  └─→ Requirement → Extract
            ↓
Rule-Based Extraction
  src/requirement_extractor.py
  ├─→ Action Labels
  │   • Actor: "manufacturer", "commission"
  │   • Action: "ensure", "provide", "submit"
  │   • Deadline: "by 1 Jan 2025", "within 6 months"
  │   • References: "Article 7", "Annex III"
  │
  └─→ Scope Labels
      • Product Types: "portable battery", "industrial battery"
      • Materials: "lithium", "cobalt", "mercury"
      • Thresholds: ">2 kWh", "≥89%", "<0.002%"
      • Components: "BMS", "cathode", "electrolyte"
            ↓
Structured JSON Output (outputs/*.json)
  • Complete requirement object
  • Classification + extraction results
  • Confidence scores + metadata
```

**Key Processing Steps:**

1. **Segmentation:** 500-2000 char chunks with 200-char overlap
2. **Labeling Keywords:** `shall`, `must`, `muss`, `verpflichtet` (EN/DE)
3. **Classification:** 88.1% F1, 97.3% recall (critical for compliance)
4. **Extraction:** Pattern matching for 20+ action/scope fields

---

## 📈 Dataset

| Split | Samples | Requirements | Non-Requirements |
|-------|---------|--------------|------------------|
| Train | 2,410 | 1,627 (67.5%) | 783 (32.5%) |
| Validation | 986 | 664 (67.3%) | 322 (32.7%) |
| Test | 2,934 | 1,440 (49.1%) | 1,494 (50.9%) |
| **Total** | **6,330** | **3,731** | **2,599** |

### Regulations Processed

**EU Regulations (18 documents, EN/DE):**

| Regulation | CELEX Number | Year | Split |
|------------|--------------|------|-------|
| Battery Regulation | 32023R1542 | 2023 | Validation |
| CBAM (Carbon Border Adjustment) | 32023R0956 | 2023 | Validation |
| CSDDD (Due Diligence Directive) | 32024L1760 | 2024 | Train |
| CSRD (Sustainability Reporting) | 32022L2464 | 2022 | Train |
| Conflict Minerals Regulation | 32017R0821 | 2017 | Train |
| Energy Efficiency Directive | 32023L1791 | 2023 | Train |
| EU Taxonomy Regulation | 32020R0852 | 2020 | Train/Test |
| EU Taxonomy Climate Delegated Act | 32021R2139 | 2021 | Test |
| NFRD (Non-Financial Reporting) | 32014L0095 | 2014 | Test |
| Renewable Energy Directive | 32018L2001 | 2018 | Train/Test |
| SFDR (Sustainable Finance) | 32019R2088 | 2019 | Test |
| Single-Use Plastics Directive | 32019L0904 | 2019 | Train/Validation |
| Waste Framework Directive | 32008L0098 | 2008 | Train/Validation |
| WEEE Directive | 32012L0019 | 2012 | Test |

**German Laws (11 documents, DE):**

| Law | Abbreviation | Year | Split |
|-----|--------------|------|-------|
| Batteriegesetz | BattG | 2009 | Validation |
| Brennstoffemissionshandelsgesetz | BEHG | 2019 | Test |
| Bundes-Immissionsschutzgesetz | BImSchG | 2021 | Train |
| Bundes-Klimaschutzgesetz | KSG | 2019 | Train |
| Chemikaliengesetz | ChemG | 2008 | Test |
| CSR-Richtlinie-Umsetzungsgesetz | CSR-RUG | 2017 | Test |
| Elektro- und Elektronikgerätegesetz | ElektroG | 2015 | Test |
| Gebäudeenergiegesetz | GEG | 2020 | Train |
| Kreislaufwirtschaftsgesetz | KrWG | 2012 | Train |
| Lieferkettensorgfaltspflichtengesetz | LkSG | 2021 | Train |
| Verpackungsgesetz | VerpackG | 2017 | Train |

**Total:** 29 PDF files (18 EU + 11 German)  
**Languages:** English (EN), German (DE)  
**Domains:** Sustainability, Due Diligence, Energy, Circular Economy, Climate, Finance

---

## 📝 Key Contributions

1. **ML-Based Classification**: 88% F1 (vs. 12% baseline)
2. **Comprehensive Dataset**: 6,330 labeled segments
3. **Hybrid Pipeline**: ML + rule-based extraction
4. **End-to-End System**: PDF → Structured Requirements

---

## 🔬 Technologies

- PyTorch 2.5.1
- Transformers 4.57.1 (HuggingFace)
- XLM-RoBERTa-Large
- scikit-learn
- Google Colab (A100 GPU)

---

## 📚 Citation

```bibtex
@mastersthesis{requirement_extraction_2025,
  title={ML-Based Requirement Extraction from EU Regulations},
  author={[Your Name]},
  school={Technical University of Munich},
  year={2025}
}
```

---

**Last Updated:** November 2025
