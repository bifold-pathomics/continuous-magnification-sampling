from typing import Any, Literal
from dataclasses import dataclass
from torch import nn

@dataclass
class ImageTransformParams:

    mini_patch_size: int
    pretraining_image_size: int
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]
    transforms_profile: str = "MB-transforms"

    @classmethod
    def create_from_model(cls, model: nn.Module) -> "ImageTransformParams":
        return ImageTransformParams(
            mini_patch_size=model.patch_size,
            pretraining_image_size=model.pretrained_cfg["input_size"][
                -1
            ],  # input size example: (3, 224, 224)
            normalization_mean=model.pretrained_cfg["mean"],
            normalization_std=model.pretrained_cfg["std"],
        )


_IMAGE_NORMALIZATIONS = {
    # ImageNet default
    "imagenet": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "optimus": {
        "mean": (0.707223, 0.578729, 0.703617),
        "std": (0.211883, 0.230117, 0.177517),
    },
}

def get_img_normalization_statistics(
    norm_name: Literal["imagenet", "ihc_proxy", "kang", "hibou", "kaiko", "optimus"] = "imagenet",
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Get the mean and standard deviation statistics for image normalization.

    These values are commonly used for normalizing RGB images before feeding them
    into neural networks. The function provides access to several predefined
    normalization configurations.

    Args:
      norm_name: The name of the normalization statistics to retrieve.
      must be one of the following:
        - "imagenet"
        - "ihc_proxy"
        - "kang"
        - "hibou"
        - "kaiko"
        - "optimus"

    Returns:
      A tuple containing:
        - RGB mean values as tuple of three floats
        - RGB standard deviation values as tuple of three floats

    Raises:
        ValueError: If the provided norm_name is not one of the valid options
    """
    if norm_name not in _IMAGE_NORMALIZATIONS:
        raise ValueError(f"< {norm_name} > is not a valid normalization name.")

    stats = _IMAGE_NORMALIZATIONS[norm_name]

    return stats["mean"], stats["std"]
