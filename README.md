

# Continuous Magnification Sampling

This repository contains the accompanying code for our work on continuous magnification sampling in pathology foundation models, along with instructions to obtain and evaluate the benchmark data (TCGA-MS, BRACS-MS).

> **Note:** This repository is under active development and will be updated continuously.

## Release Notes

**2025-01-09**
- Added code to create and evaluate the BRACS-MS dataset
**2025-01-22**
- Added code to create and evaluate the TCGA-MS dataset

## Getting Started

### Prerequisites

- Python 3.x
- Hugging Face account (for accessing certain models)
- Installed packages from requirements.txt (`pip install -r requirements.txt`)

### Authentication

Some public models require Hugging Face authentication. Log in before running evaluations:
```bash
huggingface-cli login --token YOUR_TOKEN_HERE
```

## TCGA-MS Dataset

### 1. Download the dataset from huggingface

[TCGA-MS](https://huggingface.co/datasets/bifold-pathomics/TCGA-MS)

## BRACS-MS Dataset

### 1. Download Source Data

Download the BRACS ROIs from the official source:
https://www.bracs.icar.cnr.it/download/

### 2. Create the Dataset

Adjust the source and output paths in `create_bracs_dataset.sh`, then run:
```bash
./create_bracs_dataset.sh
```

### Run Evaluation

Adjusts the necessary paths in the command and run: 

```bash
run_tcga_eval.sh
```

```bash
run_bracs_eval.sh
```
