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

```
PDF Input (data/raw/*.pdf)
  ↓
Ingestion & Chunking (ingest.py)
  ↓
JSONL Segments (data/processed/segments.jsonl)
  ↓
ML Classifier (XLM-RoBERTa)
  models/requirement_classifier/
  ├─→ Non-Requirement → Skip
  └─→ Requirement → Extract
            ↓
Rule-Based Extraction
  src/requirement_extractor.py
  ├─→ Action (actor, action, deadline)
  └─→ Scope (products, materials, components)
            ↓
Structured JSON Output (outputs/*.json)
```

---

## 📈 Dataset

| Split | Samples | Requirements | Non-Requirements |
|-------|---------|--------------|------------------|
| Train | 2,410 | 1,627 (67.5%) | 783 (32.5%) |
| Validation | 986 | 664 (67.3%) | 322 (32.7%) |
| Test | 2,934 | 1,440 (49.1%) | 1,494 (50.9%) |
| **Total** | **6,330** | **3,731** | **2,599** |

**Sources:**
- Battery Regulation (EU) 2023/1542
- REACH Regulation (EC) 1907/2006
- RoHS Directive 2011/65/EU

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
