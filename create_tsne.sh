#!/bin/bash

python magnification_analysis/compute_tsne.py \
    --model_name vits_du_s1 \
    --mpps 0.25 0.5 1.0 2.0 \
    --tcga_data_root /root/TCGA-MS/tcga_ms \
    --output_dir tsne_figures \
    --batch_size 250 \
    --token_mode cls \
    --gpu_ids 0