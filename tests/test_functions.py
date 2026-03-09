import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import MagicMock
from PIL import Image

from magnification_analysis.rankme import RankMeMetric
from model_loading.utils import ImageTransformParams, get_img_normalization_statistics
from model_loading.load_models import TokenWrapperModel, TokenWrapperConfig
from dataloaders import BRACSDataset


# ---------------------------------------------------------------------------
# RankMeMetric
# ---------------------------------------------------------------------------

class TestRankMeMetric:
    def test_returns_positive_effective_rank(self):
        metric = RankMeMetric()
        X = torch.randn(64, 16)
        effective_rank, _ = metric.compute(X)
        assert effective_rank.item() > 0

    def test_effective_rank_bounded_by_dimensions(self):
        """Effective rank must be <= number of features (columns)."""
        metric = RankMeMetric()
        n_features = 16
        X = torch.randn(64, n_features)
        effective_rank, _ = metric.compute(X)
        assert effective_rank.item() <= n_features + 1e-5

    def test_rank_one_matrix_has_effective_rank_one(self):
        """A matrix with a single dominant singular value should have effective rank close to 1.
        We build this by making one dimension have very high variance and the rest near zero,
        using SVD to construct a controlled near-rank-1 covariance structure."""
        metric = RankMeMetric()
        # One strong direction, rest are tiny noise → effective rank ≈ 1
        dominant = torch.zeros(64, 32)
        dominant[:, 0] = torch.linspace(-10, 10, 64)  # large variance in dim 0
        dominant[:, 1:] = torch.randn(64, 31) * 1e-6   # negligible variance elsewhere
        effective_rank, _ = metric.compute(dominant)
        assert effective_rank.item() == pytest.approx(1.0, abs=0.01)

    def test_eigenvalues_are_sorted(self):
        metric = RankMeMetric()
        X = torch.randn(32, 8)
        _, (eigenvalues, _) = metric.compute(X)
        assert torch.all(eigenvalues[1:] >= eigenvalues[:-1])

    def test_custom_eps(self):
        metric = RankMeMetric(eps=1e-6)
        X = torch.randn(32, 8)
        effective_rank, _ = metric.compute(X)
        assert effective_rank.item() > 0


# ---------------------------------------------------------------------------
# get_img_normalization_statistics
# ---------------------------------------------------------------------------

class TestGetImgNormalizationStatistics:
    def test_imagenet_values(self):
        mean, std = get_img_normalization_statistics("imagenet")
        assert mean == pytest.approx((0.485, 0.456, 0.406))
        assert std == pytest.approx((0.229, 0.224, 0.225))

    def test_optimus_values(self):
        mean, std = get_img_normalization_statistics("optimus")
        assert len(mean) == 3
        assert len(std) == 3

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="not a valid normalization name"):
            get_img_normalization_statistics("nonexistent_norm")

    def test_returns_three_channel_tuples(self):
        mean, std = get_img_normalization_statistics("imagenet")
        assert len(mean) == 3
        assert len(std) == 3


# ---------------------------------------------------------------------------
# ImageTransformParams
# ---------------------------------------------------------------------------

class TestImageTransformParams:
    def test_instantiation(self):
        params = ImageTransformParams(
            mini_patch_size=16,
            pretraining_image_size=224,
            normalization_mean=(0.5, 0.5, 0.5),
            normalization_std=(0.5, 0.5, 0.5),
        )
        assert params.mini_patch_size == 16
        assert params.pretraining_image_size == 224
        assert params.transforms_profile == "MB-transforms"  # default

    def test_custom_transforms_profile(self):
        params = ImageTransformParams(
            mini_patch_size=14,
            pretraining_image_size=224,
            normalization_mean=(0.485, 0.456, 0.406),
            normalization_std=(0.229, 0.224, 0.225),
            transforms_profile="custom-profile",
        )
        assert params.transforms_profile == "custom-profile"


# ---------------------------------------------------------------------------
# TokenWrapperConfig validation
# ---------------------------------------------------------------------------

class TestTokenWrapperConfig:
    def _make_model(self):
        model = MagicMock(spec=nn.Module)
        model.forward_features = MagicMock(return_value=torch.randn(2, 10, 8))
        model.num_reg_tokens = 0
        return model

    def test_valid_cls_mode(self):
        cfg = TokenWrapperConfig(
            token_mode="cls",
            call_mode="forward_features",
            model=self._make_model(),
        )
        assert cfg.token_mode == "cls"

    def test_valid_mean_mode(self):
        cfg = TokenWrapperConfig(
            token_mode="mean",
            call_mode="forward_features",
            model=self._make_model(),
        )
        assert cfg.token_mode == "mean"

    def test_invalid_token_mode_raises(self):
        with pytest.raises(Exception):
            TokenWrapperConfig(
                token_mode="patch",
                call_mode="forward_features",
                model=self._make_model(),
            )

    def test_missing_forward_features_raises(self):
        bare_model = MagicMock(spec=[])  # no forward_features attribute
        with pytest.raises(Exception):
            TokenWrapperConfig(
                token_mode="cls",
                call_mode="forward_features",
                model=bare_model,
            )


# ---------------------------------------------------------------------------
# TokenWrapperModel – forward pass
# ---------------------------------------------------------------------------

class _DummyViT(nn.Module):
    """Minimal ViT-like model: returns (batch, seq, dim) from forward_features."""
    def __init__(self, seq_len=10, dim=16):
        super().__init__()
        self.seq_len = seq_len
        self.dim = dim
        self.num_reg_tokens = 0

    def forward_features(self, x):
        B = x.shape[0]
        return torch.randn(B, self.seq_len, self.dim)


class TestTokenWrapperModel:
    def setup_method(self):
        self.vit = _DummyViT(seq_len=10, dim=16)
        self.batch = torch.randn(4, 3, 224, 224)

    def test_cls_output_shape(self):
        model = TokenWrapperModel(self.vit, token_mode="cls")
        out = model(self.batch)
        assert out.shape == (4, 16)

    def test_mean_output_shape(self):
        model = TokenWrapperModel(self.vit, token_mode="mean")
        out = model(self.batch)
        assert out.shape == (4, 16)

    def test_concat_cls_mean_output_shape(self):
        model = TokenWrapperModel(self.vit, token_mode="cls+mean")
        out = model(self.batch)
        # each token is dim=16, concatenated → 32
        assert out.shape == (4, 32)

    def test_concat_no_norm_output_shape(self):
        model = TokenWrapperModel(self.vit, token_mode="cls+mean_no_norm")
        out = model(self.batch)
        assert out.shape == (4, 32)

    def test_invalid_token_mode_raises(self):
        with pytest.raises(Exception):
            TokenWrapperModel(self.vit, token_mode="patch")


# ---------------------------------------------------------------------------
# BRACSDataset
# ---------------------------------------------------------------------------

class TestBRACSDataset:
    def _make_paths(self, subtypes, tmp_path):
        paths = []
        for i, subtype in enumerate(subtypes):
            p = tmp_path / f"BRACS_001_{subtype}_{i:03d}.png"
            Image.new("RGB", (32, 32), color=(i * 30, i * 30, i * 30)).save(p)
            paths.append(p)
        return paths

    def test_len(self, tmp_path):
        paths = self._make_paths(["N", "PB", "UDH"], tmp_path)
        ds = BRACSDataset(paths)
        assert len(ds) == 3

    def test_correct_labels(self, tmp_path):
        subtypes = ["N", "PB", "UDH", "FEA", "ADH", "DCIS", "IC"]
        paths = self._make_paths(subtypes, tmp_path)
        ds = BRACSDataset(paths)
        for i, subtype in enumerate(subtypes):
            _, label = ds[i]
            assert label == BRACSDataset.LABEL_MAP[subtype]

    def test_returns_pil_image_without_transform(self, tmp_path):
        paths = self._make_paths(["N"], tmp_path)
        ds = BRACSDataset(paths)
        img, _ = ds[0]
        assert isinstance(img, Image.Image)

    def test_transform_applied(self, tmp_path):
        from torchvision import transforms
        paths = self._make_paths(["IC"], tmp_path)
        t = transforms.Compose([transforms.ToTensor()])
        ds = BRACSDataset(paths, transform=t)
        img, _ = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape[0] == 3  # RGB channels