"""
Centralized filesystem paths for the MentorAI Fine-Tuning project.
"""

from pathlib import Path

# =============================================================================
# Project Root
# =============================================================================

# Repository root:
# mentorai-finetuning/
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# =============================================================================
# Top-Level Directories
# =============================================================================

CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# =============================================================================
# Dataset Directories
# =============================================================================

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TOKENIZED_DATA_DIR = DATA_DIR / "tokenized"
EVALUATION_DATA_DIR = DATA_DIR / "evaluation"

# =============================================================================
# Configuration Directories
# =============================================================================

DATASET_CONFIG_DIR = CONFIGS_DIR / "dataset"
TRAINING_CONFIG_DIR = CONFIGS_DIR / "training"
LORA_CONFIG_DIR = CONFIGS_DIR / "lora"
QLORA_CONFIG_DIR = CONFIGS_DIR / "qlora"
INFERENCE_CONFIG_DIR = CONFIGS_DIR / "inference"

# =============================================================================
# Model Directories
# =============================================================================

BASE_MODELS_DIR = MODELS_DIR / "base"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
ADAPTERS_DIR = MODELS_DIR / "adapters"
MERGED_MODELS_DIR = MODELS_DIR / "merged"
EXPORTED_MODELS_DIR = MODELS_DIR / "exported"

# =============================================================================
# Report Directories
# =============================================================================

TRAINING_REPORTS_DIR = REPORTS_DIR / "training"
EVALUATION_REPORTS_DIR = REPORTS_DIR / "evaluation"
BENCHMARK_REPORTS_DIR = REPORTS_DIR / "benchmarks"
FIGURES_DIR = REPORTS_DIR / "figures"

# =============================================================================
# Common Files
# =============================================================================

ENV_FILE = PROJECT_ROOT / ".env"
README_FILE = PROJECT_ROOT / "README.md"
PYPROJECT_FILE = PROJECT_ROOT / "pyproject.toml"