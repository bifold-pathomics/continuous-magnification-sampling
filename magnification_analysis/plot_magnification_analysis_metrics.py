import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from model_loading.load_models import External, TokenWrapperModel
from dataloaders import TCGADataset
from evaluate_benchmarks import extract_embeddings
from magnification_analysis.rankme import RankMeMetric


def plot_tsne(embeddings_dict, output_path):
    """Generate and save t-SNE visualization."""
    all_embeddings = []
    all_mpp_labels = []
    for mpp in sorted(embeddings_dict.keys()):
        emb = embeddings_dict[mpp]
        all_embeddings.append(emb)
        all_mpp_labels.extend([mpp] * emb.shape[0])

    all_embeddings = np.vstack(all_embeddings)
    all_mpp_labels = np.array(all_mpp_labels)

    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    fig, ax = plt.subplots(figsize=(6, 5))
    unique_mpps = sorted(np.unique(all_mpp_labels), reverse=True)
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(unique_mpps)))
    color_map = {mpp: cmap[i] for i, mpp in enumerate(unique_mpps)}

    for mpp in unique_mpps:
        mask = all_mpp_labels == mpp
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[color_map[mpp]],
            label=f"{mpp} MPP",
            alpha=0.6,
            s=13,
            edgecolors="none",
        )

    ax.legend(
        title="MPP",
        loc="best",
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
        framealpha=0.95,
        fontsize=14,
        title_fontsize=14,
    )
    ax.set_xlabel("t-SNE 1", fontsize=16)
    ax.set_ylabel("t-SNE 2", fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", dpi=600)
    plt.close()
    print(f"Saved t-SNE plot to: {output_path}")


def plot_effective_rank(rank_per_mpp, output_path):
    """Generate and save effective rank line plot."""
    mpps = sorted(rank_per_mpp.keys())
    ranks = [rank_per_mpp[mpp] for mpp in mpps]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mpps, ranks, marker="o", linewidth=2, markersize=8, color="#2E86AB")
    ax.set_xlabel("MPP", fontsize=16)
    ax.set_ylabel("Effective Rank", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", dpi=600)
    plt.close()
    print(f"Saved effective rank plot to: {output_path}")


def main():
    args = parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model_eval = External(model_id=args.model_name)
    model, image_transform_params = model_eval.get_model_and_parameters(device=device)
    img_normalization = (
        image_transform_params.normalization_mean,
        image_transform_params.normalization_std,
    )

    model = TokenWrapperModel(model, token_mode="cls", call_mode="forward_features")
    model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device).eval()

    # Build transform
    mean, std = img_normalization
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )

    # Extract embeddings for each MPP
    embeddings_dict = {}
    for mpp in args.mpps:
        print(f"Extracting embeddings for MPP {mpp}")
        mpp_dir = Path(args.tcga_data_root) / f"mpp_{mpp}"
        data_loader = DataLoader(
            TCGADataset(mpp_dir, slide_uuids=None, label_map=None, transform=transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        with torch.no_grad():
            emb, _ = extract_embeddings(model, data_loader, device)
        embeddings_dict[mpp] = emb.cpu().numpy()

    # Compute and plot effective rank
    if not args.skip_rank:
        print("\nComputing effective rank per MPP...")
        rank_metric = RankMeMetric()
        rank_per_mpp = {}
        for mpp in sorted(embeddings_dict.keys()):
            emb_tensor = torch.from_numpy(embeddings_dict[mpp])
            effective_rank, _ = rank_metric.compute(emb_tensor)
            rank_per_mpp[mpp] = effective_rank
            print(f"  MPP {mpp}: Effective Rank = {effective_rank:.2f}")

        rank_output_path = os.path.join(
            args.output_dir, f"effective_rank_{args.model_name}.png"
        )
        plot_effective_rank(rank_per_mpp, rank_output_path)

    # t-SNE visualization
    if not args.skip_tsne:
        # Subsample for t-SNE if needed
        tsne_embeddings = {}
        total_points = sum(emb.shape[0] for emb in embeddings_dict.values())
        if total_points > args.max_points_tsne:
            np.random.seed(42)
            sample_ratio = args.max_points_tsne / total_points
            for mpp, emb in embeddings_dict.items():
                n_samples = max(1, int(emb.shape[0] * sample_ratio))
                idx = np.random.choice(emb.shape[0], n_samples, replace=False)
                tsne_embeddings[mpp] = emb[idx]
        else:
            tsne_embeddings = embeddings_dict

        tsne_output_path = os.path.join(args.output_dir, f"tsne_{args.model_name}.png")
        plot_tsne(tsne_embeddings, args.mpps, tsne_output_path)
        plot_tsne(tsne_embeddings, args.mpps, tsne_output_path)

    print("\nAnalysis complete!")


def parse_args():
    parser = argparse.ArgumentParser(
        description="MPP analysis: t-SNE visualization and effective rank"
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--mpps", type=float, nargs="+", required=True)
    parser.add_argument("--tcga_data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="mpp_analysis_figures")
    parser.add_argument("--batch_size", type=int, default=250)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--max_points_tsne", type=int, default=5000)
    parser.add_argument(
        "--skip_tsne", action="store_true", help="Skip t-SNE computation"
    )
    parser.add_argument(
        "--skip_rank", action="store_true", help="Skip effective rank computation"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
