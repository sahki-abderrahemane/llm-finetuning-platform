# MentorAI LLM Fine-Tuning

> End-to-end domain-specific LLM fine-tuning project for MentorAI.

## Overview

This repository documents the complete engineering process of building, training, evaluating, and deploying a domain-specific Large Language Model (LLM).

The primary objective is to understand every stage of the LLM fine-tuning lifecycle while producing a production-quality model that will be integrated into **MentorAI**.

Unlike tutorial-based projects, this repository focuses on engineering best practices, reproducibility, modularity, and experimentation rather than simply fine-tuning a model.

The final goal is to successfully fine-tune a **7B parameter instruction model** using **QLoRA** on an **RTX 3070 Ti (8GB)**.

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
* Integrate the final model into MentorAI.

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

## Roadmap

* [x] Project Foundation
* [ ] Dataset Engineering
* [ ] Prompt Formatting
* [ ] Tokenization
* [ ] Supervised Fine-Tuning (SFT)
* [ ] LoRA
* [ ] QLoRA
* [ ] Hyperparameter Optimization
* [ ] Evaluation Framework
* [ ] Model Export
* [ ] Inference API
* [ ] MentorAI Integration

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
* Model registry
* Inference API
* A fine-tuned **7B domain-specific instruction model**
* Integration into MentorAI

---

## License

This project is licensed under the MIT License.
