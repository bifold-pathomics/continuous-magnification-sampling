# Continuous Magnification Sampling

This repository contains the accompanying code for our work on continuous magnification sampling in pathology foundation models, along with instructions to obtain and evaluate the benchmark data (TCGA-MS, BRACS-MS).

> **Note:** This repository is under active development and will be updated continuously.


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

### 1. Download Dataset

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

## Run Evaluation

Adjust the necessary data paths in the commands and run either of:

```bash
./run_tcga_eval.sh
```

```bash
./run_bracs_eval.sh
```

## Analysis

### RankMe Analysis

Generate RankMe plots to analyze embedding dimensionality across magnifications:

```bash
./create_rankme_plot.sh
```

### t-SNE Visualization

Generate t-SNE plots to visualize embedding distributions across magnifications:

```bash
./create_tsne.sh
```

## Models

The trained models from our paper are available on Hugging Face:

[bifold-pathomics/MultiScale_Models](https://huggingface.co/bifold-pathomics/MultiScale_Models)

### Available Models

Models are automatically downloaded when specified via the `--model_name` argument. The following models are available:

| Model Family | Model Names |
|--------------|-------------|
| CU MaxAvg Inf | `cu_maxavg_inf_s1`, `cu_maxavg_inf_s2`, `cu_maxavg_inf_s3` |
| ViT-S 0.25 MPP | `vits_025mpp_s1`, `vits_025mpp_s2`, `vits_025mpp_s3` |
| ViT-S 0.5 MPP | `vits_05mpp_s1`, `vits_05mpp_s2`, `vits_05mpp_s3` |
| ViT-S 1.0 MPP | `vits_1mpp_s1`, `vits_1mpp_s2`, `vits_1mpp_s3` |
| ViT-S 2.0 MPP | `vits_2mpp_s1`, `vits_2mpp_s2`, `vits_2mpp_s3` |
| ViT-S DU | `vits_du_s1`, `vits_du_s2`, `vits_du_s3` |
| ViT-S CU | `vits_cu_s1`, `vits_cu_s2`, `vits_cu_s3` |
| ViT-S CU MinMax Inf | `vits_cu_minmax_inf_s1`, `vits_cu_minmax_inf_s2`, `vits_cu_minmax_inf_s3` |


## Release Notes

**2025-02-11**
- Added RankMe analysis and t-SNE visualization scripts
- Released trained models: [bifold-pathomics/MultiScale_Models](https://huggingface.co/bifold-pathomics/MultiScale_Models)

**2025-01-22**
- Added code to create and evaluate the TCGA-MS dataset

**2025-01-09**
- Added code to create and evaluate the BRACS-MS dataset