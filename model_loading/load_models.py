from typing import Literal
import torch
import torch.nn.functional as F
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from model_loading.utils import get_img_normalization_statistics, ImageTransformParams
import timm
from timm.layers import SwiGLUPacked
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import json


def load_model(name):
    image_size = 224
    patch_size = 14
    normalization = get_img_normalization_statistics("imagenet")
    if name == "uni_vitl_2":
        timm_kwargs = {
            # 'model_name': 'vit_giant_patch14_224',
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }
        model = timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs
        )
        patch_size = 14
    elif name == "prov_gigapath":
        model = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath", pretrained=True, dynamic_img_size=True
        )
        patch_size = 16

    elif name == "Virchow":
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        patch_size = 14
    elif name == "Virchow2":
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        model = model.eval()
        patch_size = 14
    else:
        patch_size = 14
        repo_id = "bifold-pathomics/MultiScale_Models"
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        with open(config_path) as f:
            config = json.load(f)
        weights_path = hf_hub_download(repo_id=repo_id, filename=f"{name}.safetensors")
        state_dict = load_file(weights_path)
        model = timm.create_model(
            config["architecture"], pretrained=False, img_size=config["img_size"]
        )
        model.load_state_dict(state_dict)
        model.eval()
    return model, normalization, image_size, patch_size


class BaseModelEvaluation:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def get_model_and_parameters(self):
        """Get model and image parameters based on experiment config."""
        raise NotImplementedError()

    def get_image_transform_params(self, config):
        """Read image transform parameters from config and create ImageTransformParams."""
        norm_name = (
            config.img_normalization
            if hasattr(config, "img_normalization")
            else "imagenet"
        )
        norm_mean, norm_std = get_img_normalization_statistics(norm_name=norm_name)

        print(f"Using norm {norm_name}:")
        print("norm mean", norm_mean)
        print("norm std", norm_std)

        return ImageTransformParams(
            mini_patch_size=config.student.patch_size,
            pretraining_image_size=config.crops.global_crops_size,
            normalization_mean=norm_mean,
            normalization_std=norm_std,
            transforms_profile=config.transforms_profile
            if hasattr(config, "transforms_profile")
            else "MB-transforms",
        )


class External(BaseModelEvaluation):
    """Model Selection Benchmark evaluation for external models."""

    def __init__(self, model_id: str):
        """
        Args:
            model_id: WandB run id of the model to evaluate.
        """
        # For external models the model id is just the name
        self.model_id = model_id
        self.model_name = self.model_id

        self.run_name = f"{self.model_name}"
        self.artifact_name = f"{self.model_name}"

        super().__init__(model_name=self.model_name)

    def get_model_and_parameters(self, device=None):
        """Get model and image parameters based on experiment config."""
        model, normalization, image_size, patch_size = load_model(self.model_name)
        image_transform_params = ImageTransformParams(
            mini_patch_size=patch_size,
            pretraining_image_size=image_size,
            normalization_mean=normalization[0],
            normalization_std=normalization[1],
        )
        model = model.cuda()
        model = model.eval()
        return model, image_transform_params

    def get_output_subfolder(self) -> str:
        return self.model_name


class TokenWrapperConfig(BaseModel):
    """Configuration for the TokenWrapperModel.

    Attributes:
        token_mode (str): The mode for extracting tokens. See `extract_tokens` for details.
        call_mode (str): The mode for calling the model. Can be 'forward_features' or 'fsdp_call'.
        model (torch.nn.Module): The model to wrap.
    """

    class Config:
        arbitrary_types_allowed = True

    token_mode: str = Field(default="cls")
    call_mode: Literal["forward_features", "fsdp_call"]
    model: torch.nn.Module

    @field_validator("token_mode")
    @classmethod
    def check_token_mode(cls, token_mode: str) -> str:
        """Validate the token mode"""
        valid_modes = {"cls", "mean"}

        modes = token_mode.replace("_no_norm", "").split("+")
        if unsupported_modes := set(modes) - set(valid_modes):
            raise ValueError(
                f"Unsupported token mode(s): {', '.join(unsupported_modes)}. "
                f"Valid modes are: {', '.join(valid_modes)}."
            )

        return token_mode

    @model_validator(mode="after")
    def validate_model_compatibility(self):
        """Validate model compatibility with call_mode"""
        if self.call_mode == "forward_features" and not hasattr(
            self.model, "forward_features"
        ):
            raise ValueError(
                f"The model does not have forward_features method: {self.model}"
            )
        return self


class TokenWrapperModel(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        token_mode: str,
        call_mode: Literal["forward_features", "fsdp_call"] = "forward_features",
    ):
        super().__init__()

        config = TokenWrapperConfig(
            model=model,
            token_mode=token_mode,
            call_mode=call_mode,
        )
        self.token_mode = config.token_mode
        self.call_mode = config.call_mode
        self.model = config.model

    def forward(self, x):
        features = (
            self.model.forward_features(x)
            if self.call_mode == "forward_features"
            else self.model(x, is_training=True)
        )

        if isinstance(features, dict):
            return self.extract_tokens(
                self.token_mode,
                cls_token=features["x_norm_clstoken"],
                patch_tokens=features["x_norm_patchtokens"],
            )
        else:
            reg_token_idx = getattr(self.model, "num_reg_tokens", 0) + 1
            return self.extract_tokens(
                self.token_mode,
                cls_token=features[:, 0, :],
                patch_tokens=features[:, reg_token_idx:, :],
            )

    def extract_tokens(
        self,
        modes: str,
        cls_token: torch.Tensor,
        patch_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Extract a single embedding from all tokens, optionally combining several tokens.

        Modes:
        * 'cls': class token
        * 'mean': mean of patch tokens

        It is also possible to combine several tokens:
        * '<token_a>+<token_b>[+<token_c>...]': Concatenate the tokens and normalize the concatenation.
        * '<token_a>+<token_b>[+<token_c>...]_no_norm': Concatenate the tokens but don't normalize.

        Returns:
        ---
        A tensor of shape `batch_size x num_dimensions` with the extracted token or all patch
        tokens (shape `batch_size x num_mini_patches x num_dimensions`).
        """

        if "+" in modes:
            # Concatenate several tokens, optionally with normalization (trailing '_no_norm')
            tokens = [
                self.extract_tokens(mode, cls_token, patch_tokens)
                for mode in modes.replace("_no_norm", "").split("+")
            ]
            return (
                torch.cat(tokens, dim=1)
                if "_no_norm" in modes
                else torch.cat([F.normalize(t, dim=-1) for t in tokens], dim=1)
            )

        # Process individual token modes
        match modes:
            case "cls":
                return cls_token
            case "mean":
                return torch.mean(patch_tokens, dim=1)
            case _:
                raise ValueError(f"Unknown token mode: {modes}")
