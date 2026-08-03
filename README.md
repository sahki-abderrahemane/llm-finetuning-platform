# MentorAI LLM Fine-Tuning

> End-to-end domain-specific LLM fine-tuning project for Ai applications.
> End-to-end domain-specific LLM fine-tuning project for Ai applications.

## Overview

This repository documents the complete engineering process of building, training, evaluating, and deploying a domain-specific Large Language Model (LLM).

The primary objective is to understand every stage of the LLM fine-tuning lifecycle while producing a production-quality model that will be integrated into Ai applications.
The primary objective is to understand every stage of the LLM fine-tuning lifecycle while producing a production-quality model that will be integrated into Ai applications.

This repository focuses on engineering best practices, reproducibility, modularity, and experimentation rather than simply fine-tuning a model.
This repository focuses on engineering best practices, reproducibility, modularity, and experimentation rather than simply fine-tuning a model.

The final goal is to successfully fine-tune a **7B parameter instruction model** using **QLoRA** on an **My local gpu**.
The final goal is to successfully fine-tune a **7B parameter instruction model** using **QLoRA** on an **My local gpu**.

---

## Objectives

* Build a high-quality instruction dataset.
* Implement a complete dataset engineering pipeline.
* Support multiple conversational dataset formats.
* Design a reusable tokenization pipeline.
* Train models using:

  * Supervised Fine-Tuning (SFT)
  * LoRA
  * QLoRA
* Benchmark different training strategies.
* Evaluate model quality using automatic and LLM-as-a-Judge metrics.
* Export trained models for deployment.
* Integrate the final model into other ai powered apps.
* Integrate the final model into other ai powered apps.

---

## Technology Stack

### Core

* Python 3.12
* PyTorch
* Hugging Face Transformers
* Datasets
* Accelerate
* PEFT
* TRL
* BitsAndBytes
* Safetensors

### Configuration & Utilities

* OmegaConf
* Pydantic
* Loguru
* Rich

### Deployment

* FastAPI
* Docker

---

## Training Pipeline

```text
Knowledge Units
        │
        ▼
Dataset Engineering
        │
        ▼
Prompt Formatting
        │
        ▼
Tokenization
        │
        ▼
Supervised Fine-Tuning (SFT)
        │
        ▼
LoRA
        │
        ▼
QLoRA
        │
        ▼
Evaluation
        │
        ▼
Model Export
        │
        ▼
Inference API
        │
        ▼
MentorAI Integration
```

---

## Repository Structure

```text
configs/          Configuration files
data/             Raw, processed and tokenized datasets
docs/             Project documentation
experiments/      Experiment outputs
models/           Base models, adapters and exports
notebooks/        Research notebooks
reports/          Training and evaluation reports
scripts/          CLI entry points
src/              Source code
tests/            Unit and integration tests
```

---

## Command Line Interface

The package ships a single unified CLI, `mentorai-cli`, that organizes every
workflow in one place:

```text
mentorai-cli
├── train        # Fine-tune a model using SFT, LoRA, or QLoRA
├── serve        # Start the FastAPI inference server
├── data         # Dataset preparation utilities
│   └── prepare  # Download an HF dataset and write train/validation JSONL
├── infer        # Generate a response from a fine-tuned checkpoint
├── chat         # Interactively chat with a fine-tuned model
├── eval         # Evaluate predictions against reference answers
├── merge        # Merge a LoRA/QLoRA adapter into its base model
└── registry     # Manage the local model registry (not yet implemented)
```

### Examples

```bash
# Show the full command tree
mentorai-cli --help

# Train a QLoRA model on GPU (fp16, batch 1 for 8GB VRAM)
mentorai-cli train --dataset data/dolly/train.jsonl \
  --val-dataset data/dolly/validation.jsonl \
  --adapter alpaca --method qlora \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output-dir models/qwen-7b-qlora \
  --epochs 3 --batch-size 1 --eval-batch-size 1 \
  --max-length 1024 --fp16

# Prepare a Hugging Face dataset
mentorai-cli data prepare --hf-dataset databricks/databricks-dolly-15k \
  --adapter alpaca \
  --column-map instruction=instruction,input=context,output=response \
  --output-dir data/dolly

# Run one-off inference against a checkpoint
mentorai-cli infer "What is the capital of France?" --path models/qwen-7b-qlora

# Chat interactively
mentorai-cli chat --path models/qwen-7b-qlora

# Evaluate predictions against references (JSONL: {"prediction": "...", "reference": "..."})
mentorai-cli eval --predictions data/eval/predictions.jsonl \
  --path models/qwen-7b-qlora --metrics rouge,bleu

# Merge a LoRA/QLoRA adapter back into its base model (full precision)
mentorai-cli merge --adapter models/qwen-7b-qlora \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --output models/merged/qwen-7b

# Start the inference API server (see .env for HOST/PORT/DEPLOY_*)
mentorai-cli serve
```

The `train` command supports every flag from the underlying training pipeline
(`--adapter {alpaca,chatml,openai,sharegpt}`, `--method {sft,qlora}`, `--packing`,
`--lora-rank`, `--lora-alpha`, `--lora-dropout`, `--fp16`, `--bf16`, ...).

> **Note:** the thin wrappers `scripts/train.sh`, `scripts/infer.sh` and
> `scripts/chat.sh` set the CUDA runtime library path required by
> bitsandbytes (4-bit QLoRA models) and delegate to `mentorai-cli`. Use them
> instead of calling the CLI directly for 4-bit checkpoints.

The `eval` command reads a JSONL file where each line is a
`{"prediction": "...", "reference": "..."}` object; metrics are computed with
the existing evaluation framework (`BLEU`, `ROUGE-L`, `BERTScore`, and an
optional `llm_judge` when `OPENAI_API_KEY` is set). The `merge` command loads
the base model at full precision (defaulting to the `MODEL_NAME` in `.env`)
and writes a standard, deployable checkpoint.

---


## Final Deliverable

By the end of this project, this repository will contain:

* A complete dataset engineering pipeline
* Prompt formatting utilities
* Tokenization pipeline
* SFT training pipeline
* LoRA implementation
* QLoRA implementation
* Evaluation framework
* Experiment tracking utilities
* Inference API
* A fine-tuned **7B domain-specific instruction model**

