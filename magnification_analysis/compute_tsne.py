from magnification_analysis.utils import setup_journal_style
from evaluate_benchmarks import extract_embeddings


def parse_args():
    parser = argparse.ArgumentParser(description='MPP TSNE ')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--mpps', type=float, nargs='+', required=True)
    parser.add_argument('--tcga_label_file', type=str, required=True)
    parser.add_argument('--tcga_data_root', type=str, required=True)')
    parser.add_argument('--output_dir', type=str, default='heatmap_figures')
    parser.add_argument('--batch_size', type=int, default=250)
    parser.add_argument('--token_mode', type=str, default='cls')
    return parser.parse_args()


    args = parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_eval = External(model_id=args.model_name)
        model, image_transform_params = model_eval.get_model_and_parameters(
            device=device
        )
        img_normalization = (
            image_transform_params.normalization_mean,
            image_transform_params.normalization_std,
        )
    
     model = TokenWrapperModel(
        model,
        token_mode=args.token_mode,
        call_mode="forward_features",
    )
    model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device).eval()




    mean, std = transform_parameters
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )
    
    embeddings_dict = {}
    for mpp in args.mpps:
            mpp_dir = Path(data_root) / f"mpp_{mpp}"
            data_loader = DataLoader(
                TCGADataset(mpp_dir, slide_uuid=None, label_map=None, transform=transform),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            with torch.no_grad():
                emb, _ = extract_embeddings(model, data_loader, device)
            embeddings_dict[mpp] = torch.cat(emb, dim=0)
    
     # Combine all embeddings with MPP labels
    all_embeddings = []
    all_mpp_labels = []
    for mpp in mpps_sorted:
        emb = embeddings_dict[mpp].cpu().numpy()
        all_embeddings.append(emb)
        all_mpp_labels.extend([mpp] * emb.shape[0])
    
    all_embeddings = np.vstack(all_embeddings)
    all_mpp_labels = np.array(all_mpp_labels)
    
    # Subsample if too many points for faster t-SNE
    max_points = 5000
    if len(all_embeddings) > max_points:
        np.random.seed(42)
        idx = np.random.choice(len(all_embeddings), max_points, replace=False)
        all_embeddings = all_embeddings[idx]
        all_mpp_labels = all_mpp_labels[idx]
    
    print("\nComputing t-SNE...")
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # Plot t-SNE
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Color palette for MPPs - using colorblind-friendly palette
    unique_mpps = sorted(np.unique(all_mpp_labels), reverse=True)
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(unique_mpps)))
    color_map = {mpp: cmap[i] for i, mpp in enumerate(unique_mpps)}
    
    for mpp in unique_mpps:
        mask = all_mpp_labels == mpp
        ax.scatter(
            embeddings_2d[mask, 0], 
            embeddings_2d[mask, 1],
            c=[color_map[mpp]],
            label=f'{mpp} MPP',
            alpha=0.6,
            s=13,
            edgecolors='none'
        )
    
    ax.legend(title='MPP', loc='best', frameon=True, fancybox=False, 
              edgecolor='#cccccc', framealpha=0.95, fontsize=14, title_fontsize=14)
    #ax.set_title(f't-SNE Visualization: {args.model_name}', fontsize=18, fontweight='medium')
    ax.set_xlabel('t-SNE 1', fontsize=16)
    ax.set_ylabel('t-SNE 2', fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    
    tsne_path = os.path.join(output_dir, f'tsne_{token_mode_safe}.png')
    plt.savefig(tsne_path, dpi=600, bbox_inches='tight')
    plt.savefig(tsne_path.replace('.png', '.pdf'), bbox_inches='tight', dpi=600)
    print(f"Saved t-SNE plot to: {tsne_path}")







       
    
