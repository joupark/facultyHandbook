# Fine-tuning Gemma2 on a University Faculty Handbook
### A Step-by-Step Tutorial Using Torchtune + QLoRA on Google Colab

---

## Overview

This tutorial walks you through the complete pipeline for fine-tuning Google's **Gemma2-2B** model on a university employee/faculty handbook PDF, using **QLoRA** (Quantized Low-Rank Adaptation) via the **Torchtune** library — all on a free Google Colab T4 GPU.

### What You'll Build
```
PDF Document → Text Extraction → Q&A Generation → QLoRA Fine-tuning → Evaluation
```

### Why This Approach?
- **Fine-tuning** teaches the model a specific tone, style, and response format
- **QLoRA** reduces memory requirements from 16GB+ to ~3GB, making it feasible on free Colab
- **Torchtune** provides a clean, config-based interface for PyTorch fine-tuning

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Google Colab | Free tier with T4 GPU enabled |
| Google Drive | ~20GB free space |
| HuggingFace account | Token + Gemma2 license approved |
| Faculty handbook | PDF format |

### Before You Start

1. **Enable GPU in Colab**: Runtime → Change runtime type → T4 GPU
2. **Approve Gemma2 access**: Visit [huggingface.co/google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) and accept the license agreement
3. **Get a HuggingFace token**: Visit [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → New token → Read access
4. **Upload your PDF** to Google Drive at: `My Drive/school-llm/data/your_handbook.pdf`

---

## Step 0: Install Packages

> ⚠️ **Run this cell FIRST, before importing torch.** This prevents version conflicts.

```python
import subprocess

print("Installing packages...")
subprocess.run(['pip', 'uninstall', 'torchao', 'torchtune', '-y'], capture_output=True)
subprocess.run(['pip', 'install', 'torchao==0.9.0', '--no-cache-dir', '-q'], capture_output=True)
subprocess.run(['pip', 'install', 'torchtune==0.5.0', '--no-cache-dir', '-q'], capture_output=True)
subprocess.run(['pip', 'install', 'pdfplumber', '-q'], capture_output=True)
subprocess.run(['pip', 'install', 'rouge-score', '-q'], capture_output=True)

print("Done! Now RESTART the session: Runtime → Restart session")
```

> After restarting, **skip this step** and start from Step 1.

---

## Step 1: Verify Environment

```python
import torch
import torchtune

print("PyTorch:", torch.__version__)
print("torchtune:", torchtune.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")

from torchtune.models.gemma2 import qlora_gemma2_2b
print("✅ All imports successful!")
```

Expected output:
```
PyTorch: 2.11.0+cu128
CUDA available: True
GPU: Tesla T4
VRAM: 15.6 GB
✅ All imports successful!
```

---

## Step 2: Mount Google Drive

```python
import os, shutil
from google.colab import drive

if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')
else:
    print("Drive already mounted!")

# Create folder structure
folders = [
    '/content/drive/MyDrive/school-llm/models',
    '/content/drive/MyDrive/school-llm/data',
    '/content/drive/MyDrive/school-llm/output',
    '/content/drive/MyDrive/school-llm/logs',
    '/content/output',
    '/content/logs',
]
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Check available Drive space (need at least 15GB)
total, used, free = shutil.disk_usage('/content/drive/MyDrive')
print(f"Drive - Total: {total/1e9:.1f}GB | Used: {used/1e9:.1f}GB | Free: {free/1e9:.1f}GB")
```

---

## Step 3: Download Gemma2-2B Model

> ⏱️ This takes 15–20 minutes. The model (~5GB) is saved to Drive so you only need to do this once.

```python
HF_TOKEN = "hf_YOUR_TOKEN_HERE"  # ← Replace with your token

# Better: use Colab Secrets (left sidebar 🔑 icon → Add OPENAI_API_KEY)
# from google.colab import userdata
# HF_TOKEN = userdata.get('HF_TOKEN')

MODEL_PATH = '/content/drive/MyDrive/school-llm/models/gemma-2-2b'
REQUIRED_FILES = ['config.json', 'tokenizer.model', 'model-00001-of-00002.safetensors']

if os.path.exists(MODEL_PATH) and all(
    os.path.exists(os.path.join(MODEL_PATH, f)) for f in REQUIRED_FILES
):
    print("✅ Model already downloaded!")
else:
    from huggingface_hub import snapshot_download
    print("Downloading Gemma2-2B...")
    snapshot_download(
        repo_id="google/gemma-2-2b-it",
        local_dir='/content/gemma-2-2b-tmp',
        token=HF_TOKEN,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*"]
    )
    print("Copying to Drive...")
    shutil.copytree('/content/gemma-2-2b-tmp', MODEL_PATH, dirs_exist_ok=True)
    print("✅ Model saved to Drive!")
```

After download, your model folder should contain:
```
gemma-2-2b/
├── config.json                           ← Required!
├── tokenizer.model
├── tokenizer_config.json
├── model-00001-of-00002.safetensors     (~4.9GB)
└── model-00002-of-00002.safetensors     (~0.2GB)
```

---

## Step 4: Extract Text from PDF

```python
import pdfplumber

PDF_PATH = '/content/drive/MyDrive/school-llm/data/school_rules.pdf'

def extract_pdf(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({"page": i + 1, "content": text.strip()})
    return pages

pages = extract_pdf(PDF_PATH)
print(f"✅ Extracted {len(pages)} pages")
print("\n--- Sample (Page 1) ---")
print(pages[0]['content'][:500])
```

> **Tip**: If Korean/non-English text looks garbled, try `pdfplumber` alternatives like `pymupdf` (`pip install pymupdf`).

---

## Step 5: Generate Q&A Training Data

We use rule-based extraction to generate Q&A pairs without needing an external API.

```python
import json, re, random

def generate_qa_from_text(pages):
    all_qa = []
    patterns = [
        {"trigger": ["policy", "policies", "procedure"],
         "questions": ["What is the policy regarding {topic}?",
                       "What are the procedures for {topic}?"]},
        {"trigger": ["eligible", "eligibility", "requirement"],
         "questions": ["Who is eligible for {topic}?",
                       "What are the requirements for {topic}?"]},
        {"trigger": ["deadline", "schedule", "annual"],
         "questions": ["When is the deadline for {topic}?"]},
        {"trigger": ["submit", "apply", "request"],
         "questions": ["How do I {topic}?",
                       "What is the process to {topic}?"]},
        {"trigger": ["benefit", "leave", "salary"],
         "questions": ["What benefits are available for {topic}?"]},
    ]

    for page in pages:
        content = page['content']
        page_qa = []
        used_questions = set()

        for pattern in patterns:
            for trigger in pattern['trigger']:
                if trigger.lower() in content.lower():
                    idx = content.lower().find(trigger)
                    topic = content[max(0,idx-20):idx+40].strip()[:50]
                    sentences = re.split(r'(?<=[.!?])\s+', content)
                    answer_sents = [s for s in sentences if trigger.lower() in s.lower()]
                    answer = ' '.join(answer_sents[:3]).strip() if answer_sents else content[:300]

                    if len(answer) < 20:
                        continue

                    question = random.choice(pattern['questions']).format(topic=topic)
                    if question not in used_questions:
                        used_questions.add(question)
                        page_qa.append({"instruction": question, "input": "", "output": answer})
                    if len(page_qa) >= 5:
                        break
            if len(page_qa) >= 5:
                break
        all_qa.extend(page_qa)
    return all_qa

all_qa = generate_qa_from_text(pages)
print(f"✅ Generated {len(all_qa)} Q&A pairs")

# Save
DATA_PATH = '/content/drive/MyDrive/school-llm/data/school_data.jsonl'
with open(DATA_PATH, 'w', encoding='utf-8') as f:
    for qa in all_qa:
        f.write(json.dumps(qa, ensure_ascii=False) + '\n')
print(f"✅ Saved to {DATA_PATH}")
```

> **Want better quality data?** Use Claude or GPT-4o-mini to generate Q&A pairs. Higher quality data = better fine-tuning results.

---

## Step 6: Create Custom Dataset Class

```python
dataset_code = '''
from torchtune.datasets import SFTDataset
from torchtune.data import InputOutputToMessages

def school_dataset(tokenizer, data_files="school_data.jsonl", packed=False):
    return SFTDataset(
        source="json",
        data_files=data_files,
        split="train",
        message_transform=InputOutputToMessages(
            train_on_input=False,
            column_map={"input": "instruction", "output": "output"}
        ),
        model_transform=tokenizer,
    )
'''

with open('/content/custom_dataset.py', 'w') as f:
    f.write(dataset_code)
print("✅ custom_dataset.py created")
```

---

## Step 7: Create Training Configuration

```python
MODEL_PATH = '/content/drive/MyDrive/school-llm/models/gemma-2-2b'

# Auto-detect model files
safetensors = sorted([f for f in os.listdir(MODEL_PATH) if f.endswith('.safetensors')])
checkpoint_files_yaml = '\n    - '.join(safetensors)

config = f"""
model:
  _component_: torchtune.models.gemma2.qlora_gemma2_2b
  lora_attn_modules: ['q_proj', 'k_proj', 'v_proj']
  apply_lora_to_mlp: True
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.05
  quantize_base: True            # QLoRA: 4-bit base model

tokenizer:
  _component_: torchtune.models.gemma.gemma_tokenizer
  path: {MODEL_PATH}/tokenizer.model
  max_seq_len: 2048

checkpointer:
  _component_: torchtune.training.FullModelHFCheckpointer
  checkpoint_dir: {MODEL_PATH}
  checkpoint_files:
    - {checkpoint_files_yaml}
  output_dir: /content/output
  model_type: GEMMA2
  recipe_checkpoint: null

resume_from_checkpoint: False
save_every_n_epochs: 1
save_adapter_weights_only: True  # Saves only 40MB/epoch instead of 5GB!

dataset:
  _component_: custom_dataset.school_dataset
  data_files: /content/drive/MyDrive/school-llm/data/school_data.jsonl
  packed: False

seed: 42
shuffle: True
epochs: 3
max_steps_per_epoch: null
batch_size: 2
gradient_accumulation_steps: 8  # Effective batch size = 16

optimizer:
  _component_: torch.optim.AdamW
  lr: 2e-4
  weight_decay: 0.01

lr_scheduler:
  _component_: torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup
  num_warmup_steps: 100

loss:
  _component_: torchtune.modules.loss.CEWithChunkedOutputLoss

device: cuda
dtype: bf16
enable_activation_checkpointing: True
clip_grad_norm: 1.0
compile: False

metric_logger:
  _component_: torchtune.training.metric_logging.DiskLogger
  log_dir: /content/logs

log_every_n_steps: 10
output_dir: /content/output
"""

with open('/content/school_gemma2_config.yaml', 'w') as f:
    f.write(config)
print("✅ Config created!")
```

### Key Config Parameters Explained

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `lora_rank` | 16 | LoRA rank — higher = more capacity but more memory |
| `lora_alpha` | 32 | LoRA scaling factor (usually rank × 2) |
| `quantize_base` | True | QLoRA: quantize base model to 4-bit |
| `batch_size` | 2 | Per-GPU batch size |
| `gradient_accumulation_steps` | 8 | Effective batch = 2 × 8 = 16 |
| `save_adapter_weights_only` | True | Save only 40MB adapter, not full 5GB model |

---

## Step 8: Run Fine-tuning

> ⏱️ Expected time: ~10 min/epoch × 3 epochs = **~30 minutes** on T4 GPU
>
> ⚠️ **Do NOT close the browser tab during training!**
>
> ⚠️ **Do NOT press Ctrl+C during checkpoint saving!**

```python
!tune run lora_finetune_single_device \
    --config /content/school_gemma2_config.yaml
```

### What to Watch For

Good training looks like this:
```
1|10|Loss: 10.461:  50% ████████▌         [05:00<05:00]
1|20|Loss:  8.234: 100% █████████████████ [10:00<00:00]
INFO: Adapter checkpoint saved to /content/output/epoch_0/
2|10|Loss:  6.123:  50% ████████▌         ...
2|20|Loss:  5.382: 100% █████████████████ ...
INFO: Adapter checkpoint saved to /content/output/epoch_1/
3|10|Loss:  3.891:  50% ████████▌         ...
3|20|Loss:  3.099: 100% █████████████████ ...
INFO: Adapter checkpoint saved to /content/output/epoch_2/
```

Loss should **decrease each step and each epoch**. Final loss around 3.0 is good for this dataset size.

---

## Step 9: Save to Google Drive

```python
shutil.copytree(
    '/content/output',
    '/content/drive/MyDrive/school-llm/output',
    dirs_exist_ok=True
)
print("✅ Saved to Google Drive!")
```

Your output folder will contain:
```
output/
├── epoch_0/
│   ├── adapter_model.safetensors   (40MB)
│   └── adapter_config.json
├── epoch_1/
│   └── ...
└── epoch_2/
    ├── adapter_model.safetensors   ← Use this for inference
    └── adapter_config.json
```

---

## Step 10: Evaluation

> ⚠️ **Restart session first** to free GPU memory: Runtime → Restart session

### 10A. Install Evaluation Packages

```python
# Run BEFORE importing torch!
import subprocess
subprocess.run(['pip', 'install', 'torchao>=0.16.0', '-q', '--no-cache-dir'], capture_output=True)
subprocess.run(['pip', 'install', 'transformers==4.47.0', 'peft', 'rouge-score', '-q'], capture_output=True)
print("✅ Done! Now restart session again.")
```

### 10B. Compare Base vs Fine-tuned Model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_PATH = '/content/drive/MyDrive/school-llm/models/gemma-2-2b'
ADAPTER_PATH = '/content/drive/MyDrive/school-llm/output/epoch_2'

hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def generate_answer(model, question, max_new_tokens=200):
    prompt = f"### Instruction:\n{question}\n\n### Response:\n"
    inputs = hf_tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.3, top_p=0.9, do_sample=True,
        )
    full = hf_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full.split("### Response:")[-1].strip() if "### Response:" in full else full.strip()

# Test questions — customize for your handbook!
test_questions = [
    "What is the policy on faculty office hours?",
    "How many sick days are faculty members entitled to?",
    "What is the procedure for requesting a leave of absence?",
]

# Base model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
)
base_model.eval()
base_answers = [generate_answer(base_model, q) for q in test_questions]
del base_model; torch.cuda.empty_cache()

# Fine-tuned model
base_hf = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
)
ft_model = PeftModel.from_pretrained(base_hf, ADAPTER_PATH)
ft_model.eval()
ft_answers = [generate_answer(ft_model, q) for q in test_questions]
```

### 10C. ROUGE Score + Chart

```python
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt

# Replace with actual answers from your handbook
ground_truth = [
    "Faculty members must hold office hours as specified in their contract.",
    "Faculty are entitled to sick leave per the university sick leave policy.",
    "Faculty must submit a leave request to their department chair for approval.",
]

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
base_scores = [scorer.score(gt, ba)['rougeL'].fmeasure for gt, ba in zip(ground_truth, base_answers)]
ft_scores   = [scorer.score(gt, fa)['rougeL'].fmeasure for gt, fa in zip(ground_truth, ft_answers)]

base_avg = sum(base_scores) / len(base_scores)
ft_avg   = sum(ft_scores)   / len(ft_scores)

print(f"Base Model   RougeL: {base_avg:.3f}")
print(f"Fine-tuned   RougeL: {ft_avg:.3f}")
print(f"Improvement:        +{ft_avg - base_avg:.3f} ({(ft_avg-base_avg)/base_avg*100:.1f}%)")
```

---

## Troubleshooting

### Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: torchao` | torchao not installed | Run Step 0, then restart |
| `torch.int1` AttributeError | torch version too old | Install torch 2.11+ |
| `Missing key: max_steps_per_epoch` | Config missing required key | Add `max_steps_per_epoch: null` |
| `Unknown target module o_proj` | Gemma2 uses `output_proj` not `o_proj` | Remove `o_proj` from `lora_attn_modules` |
| `FileNotFoundError: config.json` | Model download incomplete | Re-run Step 3 |
| `GatedRepoError` | No Gemma2 access | Accept license at HuggingFace |
| `OutOfMemoryError` during eval | Previous training session still in memory | Restart session before evaluation |
| `torchao version 0.9.0` for PEFT | PEFT needs 0.16.0+ | Run `pip install torchao>=0.16.0` then restart |

---

## Performance Notes

### Training (T4 GPU)
| Dataset Size | Epochs | Time |
|-------------|--------|------|
| ~300 Q&A | 3 | ~30 min |
| ~500 Q&A | 3 | ~50 min |
| ~1000 Q&A | 3 | ~100 min |

### Loss Progression (Expected)
```
Epoch 1: ~10.5 → ~8.0   (large initial drop)
Epoch 2: ~8.0  → ~5.4   (continued learning)
Epoch 3: ~5.4  → ~3.1   (refinement)
```

### How to Improve Results
1. **Better training data** — Use Claude/GPT to generate higher quality Q&A pairs (500+ recommended)
2. **More epochs** — Increase `epochs: 3` to `epochs: 5`
3. **Add RAG** — Combine fine-tuning with retrieval for accurate policy content

---

## Next Step: Add RAG

For production use, combine fine-tuning with RAG:

```python
# Fine-tuned model handles tone/style
# RAG handles accurate, up-to-date policy retrieval

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load PDF into vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

# At inference time:
# 1. Retrieve relevant policy sections
# 2. Pass to fine-tuned model as context
# 3. Model responds in trained tone/style
```

---

*This tutorial was developed for the CBU Faculty Handbook QA System.*
*Fine-tuning pipeline: Torchtune 0.5.0 + QLoRA + Gemma2-2B on Google Colab T4*
