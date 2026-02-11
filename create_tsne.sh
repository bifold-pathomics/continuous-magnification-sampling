#!/bin/bash

python magnification_analysis/plot_magnification_analysis_metrics.py.py \
    --model_name vits_du_s1 \
    --mpps 0.25 0.5 1.0 2.0 \
    --tcga_data_root /root/TCGA-MS/tcga_ms \
    --output_dir tsne_figures \
    --skip_rank \
    --batch_size 250 \
