# HIGH-QUALITY TRAINING - Colab Setup
# Optimiert für beste Modell-Performance

print("="*70)
print("HIGH-QUALITY REQUIREMENT EXTRACTION TRAINING")
print("Estimated Total Time: 8-10 hours (Colab Pro T4)")
print("="*70)

# =============================================================================
# SETUP
# =============================================================================
print("\n[1/5] Installing dependencies...")
!pip install -q transformers torch datasets scikit-learn accelerate

import torch
print(f"\nGPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "")

if not torch.cuda.is_available():
    print("\n⚠️ WARNING: No GPU detected!")
    print("Go to Runtime → Change runtime type → GPU")
    exit()

# =============================================================================
# UPLOAD FILES
# =============================================================================
print("\n[2/5] Upload required files...")
from google.colab import files
uploaded = files.upload()

import os
required = ['labeled_train.jsonl', 'data_loader.py', 
            'train_requirement_classifier_colab.py',
            'train_scope_extraction_colab.py', 
            'train_action_extraction_colab.py']

for f in required:
    if not os.path.exists(f):
        print(f"❌ Missing: {f}")
        exit()
print("✅ All files uploaded!")

# =============================================================================
# DATA VALIDATION
# =============================================================================
print("\n[3/5] Validating training data...")
import json

with open('labeled_train.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

print(f"Total segments: {len(data)}")

# Check coverage
req_count = sum(1 for d in data if d['requirement_class'] != 'non_requirement')
scope_count = sum(1 for d in data if any(d['scope_labels'].values()))
action_count = sum(1 for d in data if any(v for v in d['action_labels'].values() if v))

print(f"Requirements: {req_count} ({req_count/len(data)*100:.1f}%)")
print(f"With scope_labels: {scope_count} ({scope_count/len(data)*100:.1f}%)")
print(f"With action_labels: {action_count} ({action_count/len(data)*100:.1f}%)")

# =============================================================================
# PHASE 1: REQUIREMENT CLASSIFIER (High Priority)
# =============================================================================
print("\n" + "="*70)
print("PHASE 1: REQUIREMENT CLASSIFIER")
print("="*70)
print("Model: xlm-roberta-large (560M parameters)")
print("Expected time: 2-3 hours")
print("Expected F1: 0.90-0.93")
print("\nThis is the foundation - train this first!")
print("-"*70)

# Uncomment to train:
# !python train_requirement_classifier_colab.py \
#     --data labeled_train.jsonl \
#     --epochs 10 \
#     --batch_size 8 \
#     --lr 2e-5 \
#     --max_length 512 \
#     --test_size 0.2 \
#     --patience 3

print("\n✅ After training, download the model:")
print("!zip -r requirement_classifier.zip requirement_classifier/")
print("files.download('requirement_classifier.zip')")

# =============================================================================
# PHASE 2: ACTION EXTRACTION (Best Coverage)
# =============================================================================
print("\n" + "="*70)
print("PHASE 2: ACTION EXTRACTION")
print("="*70)
print("Model: mt5-base (580M parameters)")
print("Expected time: 2.5-3.5 hours")
print("Expected BLEU: 60-75")
print("Data coverage: 92.6% (excellent!)")
print("-"*70)

# Uncomment to train:
# !python train_action_extraction_colab.py \
#     --data labeled_train.jsonl \
#     --epochs 12 \
#     --batch_size 4 \
#     --lr 3e-5 \
#     --max_source_length 512 \
#     --max_target_length 256 \
#     --num_beams 5 \
#     --patience 4

print("\n✅ After training, download the model:")
print("!zip -r action_extraction.zip action_extraction/")
print("files.download('action_extraction.zip')")

# =============================================================================
# PHASE 3: SCOPE EXTRACTION (Most Challenging)
# =============================================================================
print("\n" + "="*70)
print("PHASE 3: SCOPE EXTRACTION")
print("="*70)
print("Model: mt5-base (580M parameters)")
print("Expected time: 3-4 hours")
print("Expected BLEU: 45-60")
print("Data coverage: 19.8% (challenging!)")
print("\nTrain this last - it needs the most time and data")
print("-"*70)

# Uncomment to train:
# !python train_scope_extraction_colab.py \
#     --data labeled_train.jsonl \
#     --epochs 15 \
#     --batch_size 4 \
#     --lr 3e-5 \
#     --max_source_length 512 \
#     --max_target_length 256 \
#     --num_beams 5 \
#     --patience 5

print("\n✅ After training, download the model:")
print("!zip -r scope_extraction.zip scope_extraction/")
print("files.download('scope_extraction.zip')")

# =============================================================================
# MONITORING & TIPS
# =============================================================================
print("\n" + "="*70)
print("MONITORING DURING TRAINING")
print("="*70)
print("""
✅ Good signs:
   - eval_loss continuously decreasing
   - F1 / BLEU scores improving
   - Early stopping triggers (means model converged!)

⚠️ Warning signs:
   - eval_loss increasing (overfitting)
   - Training loss much lower than eval loss (overfitting)
   - No improvement after 5+ epochs (bad hyperparams)

💡 Tips:
   - Let it run overnight for best results
   - Early stopping is OK - saves time!
   - Test different checkpoints (epoch 7 vs 10)
   - Download checkpoints regularly (Colab can disconnect)

🔧 If CUDA OOM:
   - Reduce batch_size to 2
   - Increase gradient_accumulation_steps to 16
   - Or upgrade to Colab Pro (16GB VRAM)
""")

# =============================================================================
# POST-TRAINING EVALUATION
# =============================================================================
print("\n" + "="*70)
print("POST-TRAINING EVALUATION")
print("="*70)

def evaluate_classifier(model_dir="requirement_classifier"):
    """Quick evaluation of classifier."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from sklearn.metrics import classification_report
    import numpy as np
    
    print(f"Loading {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    
    # Test on sample
    test_texts = [
        "Manufacturers shall ensure that batteries are accompanied by technical documentation.",
        "Undertakings shall disclose information about sustainability impacts.",
        "This Regulation lays down harmonised rules for batteries.",
        "Article 5 defines the scope of application.",
        "(12) This recital provides background information."
    ]
    
    expected = [1, 1, 0, 0, 0]  # 1=requirement, 0=non-requirement
    
    predictions = []
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        pred = outputs.logits.argmax(dim=-1).item()
        predictions.append(pred)
        
        label = "✅ Requirement" if pred == 1 else "❌ Non-requirement"
        conf = torch.softmax(outputs.logits, dim=-1)[0][pred].item()
        print(f"\n{text[:80]}...")
        print(f"   → {label} (confidence: {conf:.2%})")
    
    # Simple accuracy
    accuracy = sum(p == e for p, e in zip(predictions, expected)) / len(expected)
    print(f"\n📊 Sample accuracy: {accuracy:.1%}")

print("\nAfter training, run:")
print("evaluate_classifier('requirement_classifier')")

# =============================================================================
# NEXT STEPS
# =============================================================================
print("\n" + "="*70)
print("NEXT STEPS AFTER TRAINING")
print("="*70)
print("""
1. Download all trained models
2. Test on validation data
3. Error analysis - where do models fail?
4. Label validation + test sets for proper evaluation
5. Fine-tune based on errors
6. Ensemble best checkpoints
7. Deploy to production

Quality Targets:
- Classifier F1: >0.90 ✅
- Action Extraction: >80% accuracy ✅
- Scope Extraction: >60% BLEU ⚠️ (challenging)

Time Investment: ~8-10 hours
Result: Production-ready models! 🎯
""")

print("\n" + "="*70)
print("Ready to start training!")
print("Uncomment the training commands above and run them sequentially.")
print("="*70)
