# ============================================================================
# ACTION & SCOPE EXTRACTION TRAINING - COLAB SCRIPT
# ============================================================================
# Trainiert beide Extractor-Modelle mit t5-small
# Runtime: GPU (A100 empfohlen)
# Voraussetzung: labeled_train.jsonl in Google Drive
# ============================================================================

# ============================================================================
# ZELLE 1: Setup
# ============================================================================
!nvidia-smi

# ============================================================================
# ZELLE 2: Drive mounten und Daten laden
# ============================================================================
from google.colab import drive
drive.mount('/content/drive')

# Kopiere Trainingsdaten (passe Pfad an!)
!cp /content/drive/MyDrive/labeled_train.jsonl /content/

# ============================================================================
# ZELLE 3: Action Extraction Training
# ============================================================================
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import torch

# Lade Daten
with open('labeled_train.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f if line.strip()]

# Filter für Action Extraction (nur mit action_labels)
action_samples = []
for rec in data:
    if rec.get('action_labels'):
        labels = rec['action_labels']
        # Kombiniere alle Felder zu einem Target
        parts = []
        if labels.get('actor'): parts.append(f"actor: {labels['actor']}")
        if labels.get('action'): parts.append(f"action: {labels['action']}")
        if labels.get('deadline'): parts.append(f"deadline: {labels['deadline']}")
        
        target = ' | '.join(parts) if parts else None
        
        # Nur wenn target existiert und länger als 5 Zeichen
        if target and len(target.strip()) > 5:
            action_samples.append({
                'input_text': 'extract action: ' + rec['text'],
                'target_text': target
            })

print(f"Action Samples: {len(action_samples)}")

# Erstelle Dataset
action_dataset = Dataset.from_list(action_samples)

# Tokenizer & Model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def tokenize_action(examples):
    model_inputs = tokenizer(examples['input_text'], max_length=512, truncation=True)
    labels = tokenizer(examples['target_text'], max_length=128, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_action = action_dataset.map(tokenize_action, batched=True, remove_columns=['input_text', 'target_text'])

# Training Args
training_args = Seq2SeqTrainingArguments(
    output_dir='./action_extraction',
    eval_strategy='no',
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy='epoch',
    fp16=False,  # Wichtig für Stabilität!
    logging_steps=50,
    save_total_limit=1,
    predict_with_generate=True
)

# Data Collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=tokenizer.pad_token_id
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_action,
    data_collator=data_collator
)

# Train!
print("\n" + "="*80)
print("TRAINING ACTION EXTRACTION")
print("="*80)
trainer.train()

# Save
trainer.save_model('./action_extraction')
tokenizer.save_pretrained('./action_extraction')
print("✓ Action Extraction Model gespeichert")

# ============================================================================
# ZELLE 4: Scope Extraction Training
# ============================================================================
# Filter für Scope Extraction
scope_samples = []
for rec in data:
    if rec.get('scope_labels'):
        labels = rec['scope_labels']
        parts = []
        if labels.get('product_types'): parts.append(f"products: {', '.join(labels['product_types'])}")
        if labels.get('materials'): parts.append(f"materials: {', '.join(labels['materials'])}")
        if labels.get('components'): parts.append(f"components: {', '.join(labels['components'])}")
        
        target = ' | '.join(parts) if parts else None
        
        if target and len(target.strip()) > 5:
            scope_samples.append({
                'input_text': 'extract scope: ' + rec['text'],
                'target_text': target
            })

print(f"Scope Samples: {len(scope_samples)}")

# Erstelle Dataset
scope_dataset = Dataset.from_list(scope_samples)

# Tokenizer & Model (neu laden)
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def tokenize_scope(examples):
    model_inputs = tokenizer(examples['input_text'], max_length=512, truncation=True)
    labels = tokenizer(examples['target_text'], max_length=128, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_scope = scope_dataset.map(tokenize_scope, batched=True, remove_columns=['input_text', 'target_text'])

# Training Args
training_args = Seq2SeqTrainingArguments(
    output_dir='./scope_extraction',
    eval_strategy='no',
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy='epoch',
    fp16=False,
    logging_steps=50,
    save_total_limit=1,
    predict_with_generate=True
)

# Data Collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=tokenizer.pad_token_id
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_scope,
    data_collator=data_collator
)

# Train!
print("\n" + "="*80)
print("TRAINING SCOPE EXTRACTION")
print("="*80)
trainer.train()

# Save
trainer.save_model('./scope_extraction')
tokenizer.save_pretrained('./scope_extraction')
print("✓ Scope Extraction Model gespeichert")

# ============================================================================
# ZELLE 5: Modelle als ZIP runterladen
# ============================================================================
!zip -r action_extraction.zip action_extraction/
!zip -r scope_extraction.zip scope_extraction/

from google.colab import files
files.download('action_extraction.zip')
files.download('scope_extraction.zip')

print("\n✅ Beide Modelle trainiert und heruntergeladen!")
