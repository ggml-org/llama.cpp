#!/usr/bin/env python3

from huggingface_hub import HfApi
import argparse
import os

def upload_gguf_file(local_file_path, repo_id, filename_in_repo=None):
    """
    Upload a GGUF file to a Hugging Face model repository

    Args:
        local_file_path: Path to your local GGUF file
        repo_id: Your repository ID (e.g., "username/model-name")
        filename_in_repo: Optional custom name for the file in the repo
    """

    if not os.path.exists(local_file_path):
        print(f"❌ File not found: {local_file_path}")
        return False

    if filename_in_repo is None:
        filename_in_repo = os.path.basename(local_file_path)

    if filename_in_repo is None or filename_in_repo == "":
        filename_in_repo = os.path.basename(local_file_path)

    print(f"📤 Uploading {local_file_path} to {repo_id}/{filename_in_repo}")

    api = HfApi()

    try:
        api.upload_file(
            path_or_fileobj=local_file_path,
            path_in_repo=filename_in_repo,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {filename_in_repo}"
        )

        print("✅ Upload successful!")
        print(f"🔗 File available at: https://huggingface.co/{repo_id}/blob/main/{filename_in_repo}")
        return True

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

# This script requires that the environment variable HF_TOKEN is set with your
# Hugging Face API token.
api = HfApi()

parser = argparse.ArgumentParser(description='Upload a GGUF model to a Huggingface model repository')
parser.add_argument('--gguf-model-path', '-m', help='The GGUF model file to upload', required=True)
parser.add_argument('--repo-id', '-r', help='The repository to upload to', required=True)
parser.add_argument('--name', '-o', help='The name in the model repository', required=False)
args = parser.parse_args()

upload_gguf_file(args.gguf_model_path, args.repo_id, args.name)
