import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import argparse
import os
from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm
from model_loading.load_models import External
from model_loading.load_models import TokenWrapperModel
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from datasets import (
    create_bracs_dataloaders,
    create_tcga_dataloaders,
)

Image.MAX_IMAGE_PIXELS = 500_000_000


def train_and_evaluate_knn(
    X_train, y_train, X_val, y_val, X_test_dict, y_test_dict, n_jobs=4
):
    """Train kNN classifier with hyperparameter tuning on k."""

    k_values = [1, 3, 5, 10, 20, 40, 80]
    print(f"\nTuning kNN with k values: {k_values}")

    def evaluate_k(k):
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=1)
        knn.fit(X_train, y_train)
        y_val_pred = knn.predict(X_val)
        val_balanced_accuracy = balanced_accuracy_score(y_val, y_val_pred)
        return k, val_balanced_accuracy

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(evaluate_k)(k) for k in k_values
    )

    best_k = max(results, key=lambda x: x[1])[0]
    best_val_balanced_acc = max(results, key=lambda x: x[1])[1]

    print(f"\nBest k: {best_k} with Val balanced accuracy: {best_val_balanced_acc:.4f}")

    final_model = KNeighborsClassifier(
        n_neighbors=best_k, metric="cosine", n_jobs=n_jobs
    )
    final_model.fit(X_train, y_train)

    y_val_pred = final_model.predict(X_val)
    val_metrics = {
        "accuracy": accuracy_score(y_val, y_val_pred),
        "balanced_accuracy": balanced_accuracy_score(y_val, y_val_pred),
        "f1_macro": f1_score(y_val, y_val_pred, average="macro"),
        "f1_weighted": f1_score(y_val, y_val_pred, average="weighted"),
    }

    test_metrics = {}
    for mpp in sorted(X_test_dict.keys()):
        y_test_pred = final_model.predict(X_test_dict[mpp])

        test_metrics[mpp] = {
            "accuracy": accuracy_score(y_test_dict[mpp], y_test_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test_dict[mpp], y_test_pred),
            "f1_macro": f1_score(y_test_dict[mpp], y_test_pred, average="macro"),
            "f1_weighted": f1_score(y_test_dict[mpp], y_test_pred, average="weighted"),
        }

    return test_metrics, best_k, val_metrics, None


def train_and_evaluate_classifier(
    X_train, y_train, X_val, y_val, X_test_dict, y_test_dict
):
    """Train logistic regression classifier with hyperparameter tuning."""

    C_values = np.logspace(-4, 4, 9)
    print(f"\nTuning L2 penalty with C values: {C_values}")

    def evaluate_C(C):
        lr = LogisticRegression(
            C=C,
            max_iter=1000,
            solver="lbfgs",
            multi_class="multinomial",
            random_state=42,
            class_weight="balanced",
        )
        lr.fit(X_train, y_train)
        y_val_pred = lr.predict(X_val)
        val_balanced_accuracy = balanced_accuracy_score(y_val, y_val_pred)
        return C, val_balanced_accuracy

    results = Parallel(n_jobs=4, verbose=10)(delayed(evaluate_C)(C) for C in C_values)

    best_C = max(results, key=lambda x: x[1])[0]
    best_val_balanced_acc = max(results, key=lambda x: x[1])[1]

    print(
        f"\nBest C: {best_C:.6f} with Val balanced accuracy: {best_val_balanced_acc:.4f}"
    )

    final_model = LogisticRegression(
        C=best_C,
        max_iter=1000,
        solver="lbfgs",
        multi_class="multinomial",
        random_state=42,
        class_weight="balanced",
    )
    final_model.fit(X_train, y_train)

    y_val_pred = final_model.predict(X_val)
    val_metrics = {
        "accuracy": accuracy_score(y_val, y_val_pred),
        "balanced_accuracy": balanced_accuracy_score(y_val, y_val_pred),
        "f1_macro": f1_score(y_val, y_val_pred, average="macro"),
        "f1_weighted": f1_score(y_val, y_val_pred, average="weighted"),
    }

    test_metrics = {}
    for mpp in sorted(X_test_dict.keys()):
        y_test_pred = final_model.predict(X_test_dict[mpp])

        test_metrics[mpp] = {
            "accuracy": accuracy_score(y_test_dict[mpp], y_test_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test_dict[mpp], y_test_pred),
            "f1_macro": f1_score(y_test_dict[mpp], y_test_pred, average="macro"),
            "f1_weighted": f1_score(y_test_dict[mpp], y_test_pred, average="weighted"),
        }

    return test_metrics, best_C, val_metrics


def train_and_evaluate_classifiers(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test_dict,
    y_test_dict,
    use_knn=True,
    use_logreg=True,
):
    """Train and evaluate both logistic regression and kNN classifiers."""
    results = {}

    if use_logreg:
        print("\n" + "=" * 40)
        print("LOGISTIC REGRESSION")
        print("=" * 40)
        logreg_test, logreg_C, logreg_val = train_and_evaluate_classifier(
            X_train, y_train, X_val, y_val, X_test_dict, y_test_dict
        )
        results["logreg"] = {
            "test_metrics": logreg_test,
            "val_metrics": logreg_val,
            "best_param": logreg_C,
            "param_name": "C",
        }

    if use_knn:
        print("\n" + "=" * 40)
        print("K-NEAREST NEIGHBORS")
        print("=" * 40)
        knn_test, knn_k, knn_val, _ = train_and_evaluate_knn(
            X_train, y_train, X_val, y_val, X_test_dict, y_test_dict
        )
        results["knn"] = {
            "test_metrics": knn_test,
            "val_metrics": knn_val,
            "best_param": knn_k,
            "param_name": "k",
        }

    return results


def aggregate_classifier_results(all_results_by_classifier):
    """Aggregate results across folds for each classifier."""
    aggregated = {}

    for clf_name, fold_results in all_results_by_classifier.items():
        all_test_metrics = {}
        all_val_metrics = {
            "accuracy": [],
            "balanced_accuracy": [],
            "f1_macro": [],
            "f1_weighted": [],
        }
        all_best_params = []

        for fold, result in fold_results.items():
            if result is None:
                continue

            for metric_name, value in result["val_metrics"].items():
                all_val_metrics[metric_name].append(value)

            all_best_params.append(result["best_param"])

            for mpp, metrics_dict in result["test_metrics"].items():
                if mpp not in all_test_metrics:
                    all_test_metrics[mpp] = {
                        "accuracy": [],
                        "balanced_accuracy": [],
                        "f1_macro": [],
                        "f1_weighted": [],
                    }
                for metric_name, value in metrics_dict.items():
                    all_test_metrics[mpp][metric_name].append(value)

        avg_val = {k: np.mean(v) for k, v in all_val_metrics.items()}
        std_val = {k: np.std(v) for k, v in all_val_metrics.items()}

        avg_test = {}
        std_test = {}
        for mpp in all_test_metrics:
            avg_test[mpp] = {m: np.mean(v) for m, v in all_test_metrics[mpp].items()}
            std_test[mpp] = {m: np.std(v) for m, v in all_test_metrics[mpp].items()}

        aggregated[clf_name] = {
            "avg_val_metrics": avg_val,
            "std_val_metrics": std_val,
            "avg_test_metrics": avg_test,
            "std_test_metrics": std_test,
            "best_params": all_best_params,
            "avg_best_param": np.mean(all_best_params),
        }

    return aggregated


def extract_embeddings(model, dataloader, device):
    """Extract embeddings from model."""
    embeddings_list = []
    labels_list = []

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)
            features = model(images)
            embeddings_list.append(features.cpu())
            labels_list.append(labels)

    embeddings = torch.cat(embeddings_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    return embeddings, labels


def evaluate_dataset(args, model, device, img_normalization, fold, label_df=None):
    """Unified evaluation function for both BRACS and TCGA."""

    train_embeddings_list = []
    val_embeddings_list = []
    train_labels_list = []
    val_labels_list = []

    seed = int(args.seed)

    print(f"\nCollecting training/val embeddings from MPPs: {args.train_mpps}")

    if args.dataset == "bracs":
        for train_mpp in args.train_mpps:
            print(f"\nProcessing training MPP: {train_mpp}")

            try:
                train_loader, val_loader, _ = create_bracs_dataloaders(
                    args.bracs_data_root,
                    train_mpp,
                    args.target_patchsize,
                    args.batch_size,
                    args.num_workers,
                    excel_path=args.bracs_excel_path,
                    seed=seed,
                    fold=fold,
                    n_folds=len(args.folds),
                    transform_parameters=img_normalization,
                )

                with torch.no_grad():
                    train_emb, train_lab = extract_embeddings(
                        model, train_loader, device
                    )
                    val_emb, val_lab = extract_embeddings(model, val_loader, device)
                    print(f"Training data shape for MPP {train_mpp}: {train_emb.shape}")

                    train_embeddings_list.append(train_emb)
                    train_labels_list.append(train_lab)
                    val_embeddings_list.append(val_emb)
                    val_labels_list.append(val_lab)

            except Exception as e:
                print(f"Error processing training MPP {train_mpp}: {str(e)}")
                continue

    elif args.dataset == "tcga":
        for train_mpp in args.train_mpps:
            print(f"\nProcessing training MPP: {train_mpp}")

            train_loader, val_loader, _ = create_tcga_dataloaders(
                args.tcga_data_root,
                label_df,
                train_mpp,
                fold,
                args.batch_size,
                args.num_workers,
                img_normalization,
            )

            with torch.no_grad():
                train_emb, train_lab = extract_embeddings(model, train_loader, device)
                val_emb, val_lab = extract_embeddings(model, val_loader, device)
                print(f"Training data shape for MPP {train_mpp}: {train_emb.shape}")

                train_embeddings_list.append(train_emb)
                train_labels_list.append(train_lab)
                val_embeddings_list.append(val_emb)
                val_labels_list.append(val_lab)

    if not train_embeddings_list:
        raise RuntimeError(
            "No training embeddings were collected. Cannot proceed with evaluation."
        )

    train_embeddings = torch.cat(train_embeddings_list, dim=0)
    val_embeddings = torch.cat(val_embeddings_list, dim=0)
    train_labels = torch.cat(train_labels_list, dim=0)
    val_labels = torch.cat(val_labels_list, dim=0)

    print(f"\nTraining data shape: {train_embeddings.shape}")
    print(f"Validation data shape: {val_embeddings.shape}")

    X_train = train_embeddings.numpy()
    y_train = train_labels.numpy()
    X_val = val_embeddings.numpy()
    y_val = val_labels.numpy()

    print("\n" + "=" * 60)
    print("Collecting test embeddings")
    print("=" * 60)

    X_test_dict = {}
    y_test_dict = {}

    if args.dataset == "bracs":
        for test_mpp in args.test_mpps:
            print(f"\nProcessing test MPP: {test_mpp}")

            try:
                _, _, test_loader = create_bracs_dataloaders(
                    args.bracs_data_root,
                    test_mpp,
                    args.target_patchsize,
                    args.batch_size,
                    args.num_workers,
                    excel_path=args.bracs_excel_path,
                    fold=fold,
                    n_folds=len(args.folds),
                    seed=seed,
                    transform_parameters=img_normalization,
                )

                with torch.no_grad():
                    test_emb, test_lab = extract_embeddings(model, test_loader, device)

                X_test_dict[test_mpp] = test_emb.numpy()
                y_test_dict[test_mpp] = test_lab.numpy()

            except Exception as e:
                raise RuntimeError(f"Error processing test MPP {test_mpp}") from e

    elif args.dataset == "tcga":
        for test_mpp in args.test_mpps:
            print(f"\nProcessing test MPP: {test_mpp}")

            _, _, test_loader = create_tcga_dataloaders(
                args.tcga_data_root,
                label_df,
                test_mpp,
                fold,
                args.batch_size,
                args.num_workers,
                img_normalization,
            )

            with torch.no_grad():
                test_emb, test_lab = extract_embeddings(model, test_loader, device)

            X_test_dict[test_mpp] = test_emb.numpy()
            y_test_dict[test_mpp] = test_lab.numpy()

    print("\n" + "=" * 60)
    print("Training classifier and evaluating")
    print("=" * 60)

    use_logreg = "logreg" in args.classifier
    use_knn = "knn" in args.classifier

    classifier_results = train_and_evaluate_classifiers(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test_dict,
        y_test_dict,
        use_knn=use_knn,
        use_logreg=use_logreg,
    )

    return classifier_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SSL models on TCGA or BRACS dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["tcga", "bracs"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model to evaluate"
    )
    parser.add_argument(
        "--train_mpps",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0, 2.0],
        help="List of MPPs to use for training data",
    )
    parser.add_argument(
        "--test_mpps",
        type=float,
        nargs="+",
        default=[0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0],
        help="List of MPPs to evaluate on",
    )
    parser.add_argument(
        "--target_patchsize",
        type=int,
        default=224,
        help="Target patch size for evaluation",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for dataloader"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for dataloader"
    )
    parser.add_argument(
        "--gpu_ids", type=str, default="0", help="GPU IDs to use, comma separated"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Fold numbers to evaluate",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        nargs="+",
        default=["logreg", "knn"],
        choices=["logreg", "knn"],
        help="Classifier(s) to use: logreg, knn, or both",
    )
    parser.add_argument(
        "--bracs_data_root",
        type=str,
        default="BRACS_multimag",
        help="Root directory for BRACS dataset",
    )
    parser.add_argument(
        "--bracs_excel_path", type=str, default=None, help="Path to BRACS.xlsx"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument("--token_mode", type=str, default="cls+mean")
    parser.add_argument(
        "--tcga_data_root",
        type=str,
        default="/app/tcga_ms",
        help="Root directory for extracted TCGA patches",
    )
    parser.add_argument("--tcga_label_file", type=str, default="tcga_ms/labels.csv")

    return parser.parse_args()


def main():
    args = parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model_name}")
    if args.model_name in [
        "uni_vitl",
        "H-optimus-0",
        "conch_1_5_trunk",
        "Virchow2",
        "prov_gigapath",
        "phikon-v2",
        "kaiko_vit_l",
        "uni_vitl_2",
        "kaiko_midnight",
        "Virchow",
        "H0-mini",
        # Custom models
        "cu_maxavg_inf_s1",
        "cu_maxavg_inf_s2",
        "cu_maxavg_inf_s3",
        "vits_025mpp_s1",
        "vits_025mpp_s2",
        "vits_025mpp_s3",
        "vits_05mpp_s1",
        "vits_05mpp_s2",
        "vits_05mpp_s3",
        "vits_1mpp_s1",
        "vits_1mpp_s2",
        "vits_1mpp_s3",
        "vits_2mpp_s1",
        "vits_2mpp_s2",
        "vits_2mpp_s3",
        "vits_du_s1",
        "vits_du_s2",
        "vits_du_s3",
        "vits_cu_s1",
        "vits_cu_s2",
        "vits_cu_s3",
        "vits_cu_minmax_inf_s1",
        "vits_cu_minmax_inf_s2",
        "vits_cu_minmax_inf_s3",
    ]:
        model_eval = External(model_id=args.model_name)
        model, image_transform_params = model_eval.get_model_and_parameters(
            device=device
        )
        img_normalization = (
            image_transform_params.normalization_mean,
            image_transform_params.normalization_std,
        )

    else:
        raise ValueError(f"Model {args.model_name} not recognized or supported.")

    model = TokenWrapperModel(
        model,
        token_mode=args.token_mode,
        call_mode="forward_features",
    )
    model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device).eval()

    # Load TCGA metadata and labels once if needed
    label_df = None
    if args.dataset == "tcga":
        label_df = pd.read_csv(args.tcga_label_file)
        print(f"Labels shape: {label_df.shape}")
  
    print("\n" + "=" * 60)
    print(f"EVALUATING {args.dataset.upper()} - K-FOLD CROSS-VALIDATION")
    print("=" * 60)

    folds_to_run = args.folds
    all_results_by_classifier = {}

    for fold in folds_to_run:
        print(f"\n{'=' * 60}")
        print(f"Processing fold {fold}")
        print(f"{'=' * 60}")

        classifier_results = evaluate_dataset(
            args, model, device, img_normalization, fold, label_df=label_df
        )

        if classifier_results is None:
            raise RuntimeError(f"Evaluation failed for fold {fold}")

        for clf_name, result in classifier_results.items():
            if clf_name not in all_results_by_classifier:
                all_results_by_classifier[clf_name] = {}
            all_results_by_classifier[clf_name][fold] = result

    aggregated = aggregate_classifier_results(all_results_by_classifier)

    for clf_name, results in aggregated.items():
        print("\n" + "=" * 60)
        print(f"FINAL RESULTS - {clf_name.upper()}")
        print("=" * 60)
        print(f"Dataset: {args.dataset.upper()}")
        print(f"Model: {args.model_name}")
        print(f"Training MPPs: {args.train_mpps}")
        print(f"Split method: {len(folds_to_run)}-fold cross-validation")
        print(
            f"Average best {results['avg_best_param']:.4f} (param values: {results['best_params']})"
        )
        print("\nTest metrics per MPP:")

        avg_test = results["avg_test_metrics"]
        std_test = results["std_test_metrics"]
        for mpp in sorted(avg_test.keys()):
            print(f"\n  MPP {mpp}:")
            for metric_name in [
                "accuracy",
                "balanced_accuracy",
                "f1_macro",
                "f1_weighted",
            ]:
                avg = avg_test[mpp][metric_name]
                std = std_test[mpp][metric_name]
                print(f"    {metric_name}: {avg:.4f} ± {std:.4f}")

        split_method = f"{len(folds_to_run)}fold"
        results_path = (
            Path(args.output_dir)
            / f"{args.dataset.upper()}_{args.model_name.replace(' ', '_')}_{clf_name}_trainMPPs_{'_'.join(map(str, args.train_mpps))}_{split_method}_results.csv"
        )

        rows = []
        for mpp in sorted(avg_test.keys()):
            row = {"Test_MPP": mpp}
            for metric_name in [
                "accuracy",
                "balanced_accuracy",
                "f1_macro",
                "f1_weighted",
            ]:
                row[f"Avg_{metric_name}"] = avg_test[mpp][metric_name]
                row[f"Std_{metric_name}"] = std_test[mpp][metric_name]
            rows.append(row)

        results_df = pd.DataFrame(rows)
        results_df.to_csv(results_path, index=False)

        print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
