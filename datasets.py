from pathlib import Path
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class TCGADataset(Dataset):
    def __init__(
        self, image_dir: Path, slide_uuids: set, label_map: dict, transform=None
    ):
        self.image_dir = image_dir
        self.transform = transform
        self.label_map = label_map

        all_files = list(image_dir.glob("*.png"))
        self.image_paths = [
            f for f in all_files if f.stem.rsplit("_", 2)[0] in slide_uuids
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        slide_uuid = img_path.stem.rsplit("_", 2)[0]
        label = self.label_map[slide_uuid]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class BRACSDataset(Dataset):
    """Dataset for BRACS images with labels extracted from filenames."""

    LABEL_MAP = {"N": 0, "PB": 1, "UDH": 2, "FEA": 3, "ADH": 4, "DCIS": 5, "IC": 6}

    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        filename = img_path.stem
        parts = filename.split("_")
        subtype = parts[2]
        label = self.LABEL_MAP[subtype]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def create_bracs_dataloaders(
    data_root,
    mpp,
    target_patchsize,
    batch_size,
    num_workers,
    fold,
    excel_path=None,
    n_folds=5,
    seed=42,
    transform_parameters=None,
):
    """Create BRACS dataloaders with k-fold splits."""
    if excel_path is None:
        excel_path = Path(data_root) / "BRACS.xlsx"
        if not excel_path.exists():
            raise FileNotFoundError(f"BRACS.xlsx not found at {excel_path}")

    train_paths, val_paths, test_paths = get_bracs_wsi_splits(
        excel_path, data_root, mpp, fold=fold, n_folds=n_folds, seed=seed
    )

    print(
        f"  Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}"
    )

    mean, std = transform_parameters
    transform = transforms.Compose(
        [
            transforms.Resize((target_patchsize, target_patchsize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_dataset = BRACSDataset(train_paths, transform=transform)
    val_dataset = BRACSDataset(val_paths, transform=transform)
    test_dataset = BRACSDataset(test_paths, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def get_bracs_wsi_splits(excel_path, data_root, mpp, fold, n_folds=5, seed=42):
    """Get BRACS train/val/test splits at WSI level."""
    df = pd.read_excel(excel_path, sheet_name="WSI_Information")

    data_path = Path(data_root) / f"BRACS_{mpp}mpp"
    if not data_path.exists():
        raise FileNotFoundError(f"BRACS dataset not found at {data_path}")

    all_image_paths = sorted(list(data_path.glob("BRACS_*.png")))
    if fold < 0 or fold >= n_folds:
        raise ValueError(
            f"Fold {fold} is out of range for {n_folds} folds. Must be in [0, {n_folds - 1}]"
        )

    wsi_to_images = {}
    wsi_to_label = {}
    wsi_to_patient = {}

    for img_path in all_image_paths:
        filename = img_path.stem
        parts = filename.split("_")
        wsi_id = f"{parts[0]}_{parts[1]}"
        subtype = parts[2]
        label = BRACSDataset.LABEL_MAP[subtype]

        if wsi_id not in wsi_to_images:
            wsi_to_images[wsi_id] = []
            wsi_to_label[wsi_id] = label

            wsi_match = df[df["WSI Filename"] == wsi_id]
            if not wsi_match.empty:
                wsi_to_patient[wsi_id] = wsi_match.iloc[0]["Patient Id"]
            else:
                wsi_to_patient[wsi_id] = wsi_id

        wsi_to_images[wsi_id].append(img_path)

    wsi_ids = list(wsi_to_images.keys())

    patient_to_wsis = {}
    patient_labels = {}

    for wsi_id in wsi_ids:
        patient_id = wsi_to_patient[wsi_id]
        if patient_id not in patient_to_wsis:
            patient_to_wsis[patient_id] = []
            patient_labels[patient_id] = wsi_to_label[wsi_id]
        patient_to_wsis[patient_id].append(wsi_id)

    patients = list(patient_to_wsis.keys())
    labels = [patient_labels[p] for p in patients]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(skf.split(patients, labels))
    train_val_idx, test_idx = splits[fold]

    train_val_patients = [patients[i] for i in train_val_idx]
    train_val_labels = [labels[i] for i in train_val_idx]

    skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    inner_splits = list(skf_inner.split(train_val_patients, train_val_labels))
    train_idx_inner, val_idx_inner = inner_splits[0]

    train_patients = [train_val_patients[i] for i in train_idx_inner]
    val_patients = [train_val_patients[i] for i in val_idx_inner]
    test_patients = [patients[i] for i in test_idx]

    train_wsis = [wsi for p in train_patients for wsi in patient_to_wsis[p]]
    val_wsis = [wsi for p in val_patients for wsi in patient_to_wsis[p]]
    test_wsis = [wsi for p in test_patients for wsi in patient_to_wsis[p]]

    train_paths = [img for wsi in train_wsis for img in wsi_to_images.get(wsi, [])]
    val_paths = [img for wsi in val_wsis for img in wsi_to_images.get(wsi, [])]
    test_paths = [img for wsi in test_wsis for img in wsi_to_images.get(wsi, [])]

    return train_paths, val_paths, test_paths


def create_tcga_dataloaders(
    data_root, label_df, mpp, fold, batch_size, num_workers, transform_parameters
):
    fold_col = f"fold_{fold}"

    train_slides = set(label_df[label_df[fold_col] == "train"]["slide_uuid"])
    val_slides = set(label_df[label_df[fold_col] == "dev"]["slide_uuid"])
    test_slides = set(label_df[label_df[fold_col] == "test"]["slide_uuid"])

    label_map = dict(zip(label_df["slide_uuid"], label_df["label_id"]))
    mpp_dir = Path(data_root) / f"mpp_{mpp}"

    mean, std = transform_parameters
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )

    train_loader = DataLoader(
        TCGADataset(mpp_dir, train_slides, label_map, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        TCGADataset(mpp_dir, val_slides, label_map, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        TCGADataset(mpp_dir, test_slides, label_map, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
