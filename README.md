#  End-to-End MLOps Pipeline for Question Answering using BERT

## 👥 Team Members
- Deepthi Rajagopal Gajendra  
- Tengzhe Zhang  
- Mourad Reda  

---

## 1. 📌 Project Overview

This project implements a **complete end-to-end MLOps pipeline** for an **extractive Question Answering (QA)** task using a **BERT-based model fine-tuned on the SQuAD dataset**.

The goal of this project is not only to train a performant NLP model, but also to apply **MLOps best practices** across the entire machine learning lifecycle, including reproducibility, experiment tracking, model serving, containerisation, CI/CD, and monitoring.

---

## 2. 🎯 Problem Definition & Data

### Problem Statement
Given a **context paragraph** and a **question**, the system extracts the most relevant answer span from the context.  
If the context does not contain sufficient information, the model returns **“no answer”**.

This is an **extractive Question Answering** problem.

### Dataset
- **Name:** Stanford Question Answering Dataset (SQuAD v1.1 / v2.0)
- **Source:** https://rajpurkar.github.io/SQuAD-explorer/
- **Description:**
  - Wikipedia-based context passages
  - Human-annotated question–answer pairs
  - Includes both answerable and unanswerable questions (v2.0)

---

## 3. 🏗️ System Architecture

```text
├── data/
│   ├── raw/                # Original SQuAD dataset
│   └── processed/          # Tokenized & preprocessed data
│
├── src/
│   ├── data/               # Data loading & preprocessing
│   ├── training/           # BERT fine-tuning scripts
│   ├── evaluation/         # Evaluation metrics (EM, F1)
│   ├── inference/          # Inference utilities
│   └── api/                # FastAPI application
│
├── tests/                  # Unit tests (≥60% coverage)
│
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.api
│
├── mlruns/                 # MLflow experiment tracking
│
├── .github/workflows/      # CI/CD pipelines
│
├── pyproject.toml          # Dependency management (UV)
├── uv.lock                 # Reproducible environment
├── .pre-commit-config.yaml
└── README.md
```

 Here it is cleanly formatted in pure Markdown (.md), ready to paste directly into your README.md 👇

## 4. ⚙️ MLOps Practices

### Environment & Dependency Management
- Python environment managed using **UV**
- Dependencies defined in `pyproject.toml`
- Fully reproducible via `uv.lock`

```bash

 
 







