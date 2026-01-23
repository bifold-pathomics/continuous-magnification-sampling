python evaluate_benchmarks.py \
    --dataset tcga \
    --model_name Virchow \
    --train_mpps 0.25 0.5 1.0 2.0 \
    --test_mpps 0.25 0.375 0.5 0.75 1.0 1.5 2.0 \
    --folds 0 1 2 3 4 \
    --tcga_data_root /app/tcga_ms \
    --tcga_label_file tcga_ms/labels.csv \
    --gpu_ids "0" \