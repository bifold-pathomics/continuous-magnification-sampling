from huggingface_hub import HfApi, create_repo
from safetensors.torch import save_file
import torch
import json
import os

LOCAL_DIR = "/app/release"
HF_ORG = "bifold-pathomics"
REPO_NAME = "MultiScale_Models"
COLLECTION_SLUG = "bifold-pathomics/multiscale-models"

model_groups = [
    "cu_maxavg_inf",
    "vits_025mpp",
    "vits_05mpp",
    "vits_1mpp",
    "vits_2mpp",
    "vits_du",
    "vits_cu",
    "vits_cu_minmax_inf",
]

api = HfApi()
repo_id = f"{HF_ORG}/{REPO_NAME}"

# Create private repo
print(f"Creating private repo: {repo_id}")
create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)

# Model config
config = {
    "architecture": "vit_small_patch14_reg4_dinov2",
    "num_classes": 0,
    "img_size": 224,
    "patch_size": 14,
    "embed_dim": 384,
    "num_heads": 6,
    "depth": 12,
    "num_reg_tokens": 4,
}

# Upload main config
print("Uploading config.json...")
config_path = "/tmp/config.json"
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

api.upload_file(
    path_or_fileobj=config_path,
    path_in_repo="config.json",
    repo_id=repo_id,
    repo_type="model",
)

# Convert and upload each model as safetensors
for model_name in model_groups:
    for seed in ["s1", "s2", "s3"]:
        if model_name == "vits_du":
            local_filename = f"vits_mmpp_{seed}.pth"
        else:
            local_filename = f"{model_name}_{seed}.pth"
        
        remote_name = f"{model_name}_{seed}"
        local_path = os.path.join(LOCAL_DIR, local_filename)
        
        # Load .pth and convert to safetensors
        print(f"Converting {local_filename} to safetensors...")
        state_dict = torch.load(local_path, map_location='cpu', weights_only=True)
        
        # Save as safetensors
        safetensors_path = f"/tmp/{remote_name}.safetensors"
        save_file(state_dict, safetensors_path)
        
        # Upload safetensors
        print(f"Uploading {remote_name}.safetensors...")
        api.upload_file(
            path_or_fileobj=safetensors_path,
            path_in_repo=f"{remote_name}.safetensors",
            repo_id=repo_id,
            repo_type="model",
        )
        
        # Cleanup
        os.remove(safetensors_path)

# Add to collection
print(f"Adding to collection: {COLLECTION_SLUG}")
try:
    from huggingface_hub import add_collection_item
    add_collection_item(
        collection_slug=COLLECTION_SLUG,
        item_id=repo_id,
        item_type="model",
    )
except Exception as e:
    print(f"Collection note: {e}")

print(f"\nDone! All models uploaded to: https://huggingface.co/{repo_id}")